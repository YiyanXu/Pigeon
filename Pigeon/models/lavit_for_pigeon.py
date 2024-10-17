import random
import torch
import numpy as np
import contextlib
from torch import nn, einsum
import torch.nn.functional as F
import math
import os
import logging

from packaging import version
from collections import OrderedDict
from functools import partial
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from models.modeling_decoder import build_tokenizer_decoder
from models.modeling_visual_tokenzier import build_dynamic_tokenizer, VectorQuantizer
from models.transform import LaVITImageProcessor, LaVITQuestionProcessor

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import CLIPImageProcessor
from transformers.modeling_utils import get_parameter_device
from transformers.trainer_pt_utils import nested_detach
import PIL
from PIL import Image
from tqdm import tqdm
from IPython import embed
import data_utils
from models.mutual_mask_generator import MutualMaskGenerator, MLP

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

logger = logging.getLogger(__name__)

class LaVITforPigeon(nn.Module):
    def __init__(
        self,
        model_path="",
        model_dtype="bf16",
        lora_config=None,
        lora_weights=None,
        use_xformers=False,
        check_safety=False,
        pixel_decoding='highres',
        mask_type='mutual',
        num_heads=4,
        num_layers=1,
        drop_prob=0.2,
        img_folder_path=None,
        all_image_paths=None,
        data_path=None,
        load_in_8bit=True,
        device_map=None,
        training=True,
        pretrained=False,
        **kwargs
    ):
        """
        model_path: The pre-trained model checkpoint path, the local path for downloaded LaVIT weight
        model_dtype: The precision of model weight during inference, should be set bf16 or fp16, default is bf16.
        load_tokenizer: Whether to load the tokenizer encoder during the image generation. For text-to-image generation,
        The visual tokenizer is not needed, so the default is False for saving the GPU memory. When using for the 
        multi-modal synthesis (the input image needs to be tokenizd to dircrete ids), the load_tokenizer must be set to True.
        check_safety: load the stable diffusion safety checker to check the safety of generated image, if not safe, output a black image
        pixel_decoding: can be set to `highres` or `lowres`, default is `highres`: using the high resolution decoding 
            for generating high-quality images, if set to `lowres`, using the origin decoder to generate 512 x 512 image
        """
        super().__init__()

        self.visual_vocab_size = 16384   # The visual vocab size of LaVIT is 16384

        logger.info(f"Loading LaVIT tokenizer...")
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(model_path, subfolder='language_model', use_fast=False)
        self.llama_tokenizer.padding_side = "left"
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.add_tokens('<img>')  # add special token <img>: 32000 as the start of an image
        self.llama_tokenizer.add_tokens('<emb>')  # add special token <emb>: 32001 as a placeholder for target clip embedding

        logger.info(f"Loading LaVIT Model Weight from {model_path}, model precision: {model_dtype}")
    
        self.llama_model = LlamaForCausalLM.from_pretrained(
            model_path,
            subfolder='language_model',
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.bfloat16 if model_dtype=="bf16" else torch.float16, 
            device_map=device_map,
        )

        if training:
            if load_in_8bit:
                self.llama_model = prepare_model_for_kbit_training(self.llama_model)

            logger.info(f"Finetuning Llama with LoRA:\n{lora_config}")
            self.llama_model = get_peft_model(self.llama_model, lora_config)
            
            logger.info("***** Llama Trainable Parameters Statistics *****")
            self.llama_model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
        else:  # for inference
            if pretrained:
                if lora_weights:
                    raise ValueError(f"`pretrained=True` and `lora_weights` is provided. Please specify the inference is conducted over pretrained model or finetuned ckpt.")
            else:
                if lora_weights is None:
                    raise ValueError(f"`lora_weights` is expected for finetuned llama model during inference.")
                logger.info(f"Loading LoRA weights of Llama from {lora_weights} for inference...")
                self.llama_model = PeftModel.from_pretrained(
                    self.llama_model,
                    lora_weights,
                    torch_dtype=torch.bfloat16 if model_dtype=="bf16" else torch.float16,
                    device_map=device_map,
                )
        
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

        self.model_dtype = model_dtype
        self.mask_type = mask_type
        if self.mask_type == 'mutual':
            self.mask_generator = MutualMaskGenerator(
                emb_dim=self.llama_model.config.hidden_size,  # 4096
                num_heads=num_heads,
                num_layers=num_layers,
                drop_prob=drop_prob,
            )
        elif self.mask_type == 'random':
            self.mask_generator = None
        else:
            self.mask_generator = None

        logger.info(f"Loading LaVIT visual tokenizer...")
        self.visual_tokenizer = build_dynamic_tokenizer(model_path, use_xformers=use_xformers, for_understanding=False)
        logger.info(f"Loading LaVIT tokenizer decoder...")
        self.tokenizer_decoder = build_tokenizer_decoder(model_path, pixel_decoding=pixel_decoding)
        img_size = 224
        self.processor = LaVITImageProcessor(image_size=img_size)
        self.check_safety = check_safety

        self.adapter = MLP(
            in_dim=2048,
            hid_dim=512,
            out_dim=4096,
            drop_prob=drop_prob,
        )

        self.img_dataset = data_utils.ImagePathDataset(img_folder_path, all_image_paths, self.processor)

        # The diffusion related parameters
        self.pixel_decoding = pixel_decoding

        if not training:  # during inference phase, loading diffusion models to generate images.
            if pixel_decoding == 'lowres':
                diff_model_dir = os.path.join(model_path, 'pixel_decoding')
                self.register_buffer('uncond_embeddings', torch.load(os.path.join(diff_model_dir, 'uncond_embeddings.bin'), map_location='cpu'))
            else:
                diff_model_dir = os.path.join(model_path, 'highres_pixel_decoding')
            
            self.model_path = model_path
            self.diff_model_dir = diff_model_dir
            self.use_xformers = use_xformers
            self.check_safety = check_safety

            self.vae = None
            self.unet = None

        self.visual_tokenizer.requires_grad_(False)
        self.tokenizer_decoder.requires_grad_(False)
        
        self.kwargs = kwargs
    
    def prepare_dm_to_generate_image(self):
        logger.info(f"Loading dm-related modules...")
        self.vae = AutoencoderKL.from_pretrained(self.diff_model_dir, subfolder="vae", 
                torch_dtype=torch.bfloat16 if self.model_dtype=="bf16" else torch.float16)
        self.scheduler = DDIMScheduler.from_pretrained(self.diff_model_dir, subfolder="scheduler")
        self.unet = UNet2DConditionModel.from_pretrained(self.diff_model_dir, subfolder="unet", use_safetensors=False, 
                torch_dtype=torch.bfloat16 if self.model_dtype=="bf16" else torch.float16)

        if self.use_xformers:
            logger.info("You are using XFormers ops, please make sure your device install and support xformers")
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.info(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly or set `use_xformers=False`")
        
        if self.check_safety:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                os.path.join(self.model_path, 'pixel_decoding'), subfolder="feature_extractor",
            )
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                os.path.join(self.model_path, 'pixel_decoding'), subfolder="safety_checker",
            )
        
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.vae = self.vae.to(self.device)
        self.unet = self.unet.to(self.device)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)
    
    @property
    def dtype(self):
        if self.model_dtype == 'fp16':
            dtype = torch.float16
        elif self.model_dtype == 'bf16':
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        return dtype

    def maybe_autocast(self, dtype=None):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")
        dtype = self.dtype

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def prepare_input_for_lavit(
        self, batch, mask_ratio, hist_mask_ratio=None,
        add_special=True, with_mask=True, inference=False,
    ):
        """
        with_mask: during inference phase, mask target image or not
        """
        if not inference and not with_mask:
            raise ValueError(f"During training phase, `with_mask=True` is expected.")
        
        if inference:
            text_input_ids = batch["text_input_ids"][:, :-1]  # remove the target output <img> token
            text_attn_mask = batch["text_attn_mask"][:, :-1]  # remove the target output <img> token attn mask
            image_list = batch["image"]
        else:
            text_input_ids = batch["text_input_ids"]  # [bsz, seq_len]
            text_attn_mask = batch["text_attn_mask"]
            image_list = batch["image"]

        text_input_ids = text_input_ids.to(self.device)
        text_attn_mask = text_attn_mask.to(self.device)

        if self.mask_type == "random":
            image_tokens, labels, masked_labels = self.tokenize_image_with_random_mask(image_list, mask_ratio, add_special, with_mask, inference)
        elif self.mask_type == "mutual":
            if hist_mask_ratio is None:
                raise ValueError(f"mask_type=`mutual` expects the value of hist_mask_ratio.")
            image_tokens, labels, masked_labels = self.tokenize_image_with_mutual_mask(image_list, mask_ratio, hist_mask_ratio, add_special, with_mask, inference)
        else:
            image_tokens, labels, masked_labels = self.tokenize_image_wo_mask(image_list, add_special, inference) 

        with self.maybe_autocast():
            tgt_embs = self.generate_image_embeds(masked_labels)[0].mean(dim=1)  # [bsz, 256, 2048] --> [bsz, 2048]

        merged_input_tokens, merged_attention_masks, merged_labels = self.merge_batch_inputs(
            image_tokens, text_input_ids, text_attn_mask,
            inference, padding=self.llama_tokenizer.padding_side,
        )

        if inference:
            labels = self.pad_labels(labels)
            return merged_input_tokens, merged_attention_masks, labels, tgt_embs
        else:
            return merged_input_tokens, merged_attention_masks, merged_labels, tgt_embs

    def pad_labels(self, labels):
        """
        labels: List of target image tokens with different token nums
        """
        bsz = len(labels)
        label_num = [label.numel() for label in labels]
        max_len = max(label_num)
        padded_labels = torch.empty((bsz, max_len), dtype=torch.long, device=labels[0].device).fill_(-100)
        for i in range(bsz):
            padded_labels[i, :label_num[i]] = labels[i]
        
        return padded_labels

    def tokenize_image_wo_mask(self, image_list, add_special, inference=False):
        bsz, _ = image_list.shape

        image_tokens = self.tokenize_image(image_list, add_special)

        labels = []
        masked_labels = []
        for i in range(bsz):
            target_token = image_tokens[i][-1]
            labels.append(target_token.clone())
            masked_labels.append(target_token.clone())
        
        if inference:
            image_tokens = [image_token[:-1] for image_token in image_tokens]  # remove the target output visual tokens
        
        return image_tokens, labels, masked_labels

    def tokenize_image_with_random_mask(self, image_list, mask_ratio, add_special, with_mask, inference):
        """
        Randomly mask the target input image: replace some visual tokens with specific ratio 'mask_ratio'
        """
        bsz, _ = image_list.shape

        image_tokens = self.tokenize_image(image_list, add_special)
        tgt_idx = -1

        labels = []
        masked_labels = []
        if with_mask:  # inference=True/False
            for i in range(bsz):
                target_token = image_tokens[i][tgt_idx]
                
                target_token_num = len(target_token)
                mask_num = math.ceil(target_token_num * mask_ratio)

                if inference:
                    labels.append(target_token.clone())
                
                if add_special:
                    # can't mask image start token and end token
                    replaced_indices = random.sample(range(1, target_token_num-1), mask_num)
                else:
                    replaced_indices = random.sample(range(target_token_num), mask_num)
                
                masked_label = target_token.clone()
                masked_label[replaced_indices] = -1
                masked_labels.append(masked_label)
        else:  # inference=True: ONLY WHEN `inference=True`, `with_mask=False`
            for i in range(bsz):
                target_token = image_tokens[i][tgt_idx]
                labels.append(target_token.clone())
                masked_labels.append(target_token.clone())
        
        if inference:
            image_tokens = [image_token[:-1] for image_token in image_tokens]  # remove the target output visual tokens
        
        return image_tokens, labels, masked_labels
    
    def tokenize_image_with_mutual_mask(self, image_list, mask_ratio, hist_mask_ratio, add_special=True, with_mask=True, inference=False):
        """
        Generate mutual mask for user interacted image and target image.
        `mask_ratio`: the mask ratio of target image.
        `hist_mask_ratio`: the mask ratio of history images.
        `add_special`: whether add image start token and end token to history images
        `inference`: inference phase or not.
        `with_mask`: during inference phase, mask target or not.
        """
        bsz, _ = image_list.shape
        
        # 1. Tokenize image into visual tokens. 
        # During the generation process of mutual mask, image start token and end token should not be involved.
        # Set `add_special=False` to tokenize image
        image_tokens = self.tokenize_image(image_list, add_special=False)
        tgt_idx = -1

        # 2. Prepare input_ids, position_ids and type_ids for MutualMaskGenerator
        batch_input_ids = []
        batch_position_ids = []
        batch_type_ids = []
        batch_token_nums = []
        labels = []
        masked_labels = []
        for i in range(bsz):
            target_token = image_tokens[i][tgt_idx]
            hist_tokens = image_tokens[i][:tgt_idx]

            if inference:
                labels.append(target_token.clone())
            
            masked_labels.append(target_token.clone())

            target_num = target_token.numel()
            hist_nums = [hist_token.numel() for hist_token in hist_tokens]
            token_nums = hist_nums + [target_num]

            input_ids = torch.cat(hist_tokens, dim=0)
            input_ids = torch.cat((input_ids, target_token), dim=0)

            position_ids = [torch.arange(1, token_num+1) for token_num in token_nums]
            position_ids = torch.cat(position_ids, dim=0).to(input_ids.device)

            type_ids = torch.tensor([1] * sum(hist_nums) + [2] * target_num).to(input_ids.device)

            batch_input_ids.append(input_ids)
            batch_position_ids.append(position_ids)
            batch_type_ids.append(type_ids)
            batch_token_nums.append(token_nums)
        
        # 3. Pad all the inputs into a batch
        batch_input_ids, batch_position_ids, batch_type_ids, attention_mask, batch_token_nums = self.pad_inputs_for_mutual_mask(
            batch_input_ids, batch_position_ids, batch_type_ids, batch_token_nums
        )

        # 4. Generate mask for history and target
        token_embeds = self.llama_model.get_input_embeddings()(batch_input_ids)
        keep_decision = self.mask_generator(
            token_embeds,
            attention_mask,
            batch_position_ids,
            batch_type_ids,
            mask_ratio,
            hist_mask_ratio,
            inference,
        )

        # 5. Mask history and target tokens
        if add_special:
            # Add image start and end tokens into image tokens
            special_tokens = torch.tensor([32000, 32001], dtype=torch.long, device=self.device)

        for i in range(bsz):
            # split history mask
            split_masks = torch.split(keep_decision[i], batch_token_nums[i])
            hist_masks = split_masks[:-2]
            for j,hist_mask in enumerate(hist_masks):
                mask_indices = (hist_mask == 0.)
                image_tokens[i][j][mask_indices] = -1  # -1 indicates the hist token need to be replaced during embedding look-up

                if add_special:
                    image_tokens[i][j] = torch.cat([special_tokens[:1], image_tokens[i][j], special_tokens[1:]], dim=-1)

            if with_mask:
                target_mask = split_masks[-2]  # split_masks[-1] is mask for pad tokens
                replace_indices = (target_mask == 0.)
                masked_labels[i][replace_indices] = -1

                if add_special:
                    masked_labels[i] = torch.cat([special_tokens[:1], masked_labels[i], special_tokens[1:]], dim=-1)

        if inference:
            image_tokens = [image_token[:-1] for image_token in image_tokens]  # remove the target output visual tokens

        return image_tokens, labels, masked_labels
    
    def pad_inputs_for_mutual_mask(self, input_ids, position_ids, type_ids, batch_token_nums):
        """
        Padding side: right.
        """
        device = input_ids[0].device
        seq_lens = [input_id.numel() for input_id in input_ids]
        maxlen = max(seq_lens)
        bsz = len(seq_lens)
        padded_input_ids = torch.empty((bsz, maxlen), dtype=torch.long, device=device).fill_(self.llama_tokenizer.pad_token_id)
        padded_position_ids = torch.empty((bsz, maxlen), dtype=torch.long, device=device).fill_(0)
        padded_type_ids = torch.empty((bsz, maxlen), dtype=torch.long, device=device).fill_(0)
        attention_mask = torch.empty((bsz, maxlen), dtype=torch.bool, device=device).fill_(True)
        for i in range(bsz):
            attention_mask[i, :seq_lens[i]] = False  # In TransformerEncoder, True means pad position, which will be converted to '-inf' within encoder
            padded_input_ids[i, :seq_lens[i]] = input_ids[i]
            padded_position_ids[i, :seq_lens[i]] = position_ids[i]
            padded_type_ids[i, :seq_lens[i]] = type_ids[i]
            batch_token_nums[i].append(maxlen - seq_lens[i])  # append pad num
        
        return padded_input_ids, padded_position_ids, padded_type_ids, attention_mask, batch_token_nums

    def merge_batch_inputs(self, all_image_tokens, text_tokens, text_attn_mask, inference=False, padding="left"):
        
        # 1. Create a mask to know where special tokens <img> are
        pad_token_id = self.llama_tokenizer.pad_token_id
        special_token_mask = (text_tokens == 32000).long()  # special_img_token: <img>=32000
        special_token_num = torch.sum(special_token_mask, dim=-1)
        batch_indices, non_img_indices = torch.where(text_tokens != 32000)
        
        # 2. Compute the max text&image sequence length
        bsz, ori_seq_len = text_tokens.shape
        ori_pad_num = torch.sum(text_tokens == pad_token_id, dim=1)
        all_image_token_nums = []
        batch_added_image_token_nums = []
        for image_tokens in all_image_tokens:
            # image_tokens.dtype: tuple
            image_token_nums = torch.tensor([image_token.numel() for image_token in image_tokens])
            batch_added_image_token_nums.append(sum(image_token_nums) - len(image_token_nums))  # additional img tokens num added to seq
            all_image_token_nums.append(image_token_nums)
        all_image_token_nums = torch.cat(all_image_token_nums, dim=0).to(device=text_tokens.device)  # [bsz*6]
        max_seq_len = ori_seq_len + max(batch_added_image_token_nums)
        
        # 3. Compute the positions where text and image should be written
        special_batch_indices, img_indices = torch.where(text_tokens == 32000)
        special_token_mask[special_batch_indices, img_indices] = all_image_token_nums - 1  # special_token_mask with added token num
        new_text_token_positions = torch.cumsum((special_token_mask + 1), -1) - 1
        new_pad_num = max_seq_len - 1 - new_text_token_positions[:, -1]
        if padding == "left":
            new_text_token_positions += new_pad_num[:, None]  # offset for left padding
        
        text_to_overwrite = new_text_token_positions[batch_indices, non_img_indices].to(text_tokens.device)
        
        # 4. Create merged tokens, already padded to the maximum sequence length
        merged_tokens = torch.empty((bsz, max_seq_len), dtype=text_tokens.dtype, device=text_tokens.device).fill_(pad_token_id)
        merged_attn_mask = torch.zeros((bsz, max_seq_len), dtype=text_attn_mask.dtype, device=text_attn_mask.device)
        if inference:
            merged_labels = None
            # during inference, the labels are output image tokens without input tokenids
        else:
            merged_labels = torch.empty((bsz, max_seq_len), dtype=text_tokens.dtype, device=text_tokens.device).fill_(-100)
            # -100 is the default ignore index in cross-entropy loss

        # 5. Fill the text_tokens into new text_tokens based on the mask
        merged_tokens[batch_indices, text_to_overwrite] = text_tokens[batch_indices, non_img_indices]
        merged_tokens[merged_tokens == 32001] = -2  # placeholder for target clip embeds
        merged_attn_mask[batch_indices, text_to_overwrite] = text_attn_mask[batch_indices, non_img_indices]

        # 6. Fill the image tokens into new text_tokens
        image_to_overwrite = (merged_tokens == pad_token_id)
        if padding == "left":
            image_to_overwrite &= (image_to_overwrite.cumsum(-1) - 1 >= (new_pad_num[:, None] + ori_pad_num[:, None]))
        else:
            image_to_overwrite &= (image_to_overwrite.cumsum(-1) - 1 <= (max_seq_len - 1 - new_pad_num[:, None] - ori_pad_num[:, None]))
        
        if image_to_overwrite.sum() != all_image_token_nums.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of images is {special_token_num} while"
                f" the number of image given to the model is {len(all_image_token_nums)}. This prevents correct indexing and breaks batch generation."
            )
        
        for i,image_tokens in enumerate(all_image_tokens):
            flattened_image_tokens = torch.cat(image_tokens, dim=-1).to(dtype=text_tokens.dtype)
            img_indices = image_to_overwrite[i].nonzero(as_tuple=True)
            merged_tokens[i][img_indices] = flattened_image_tokens
            merged_attn_mask[i][img_indices] = torch.ones_like(flattened_image_tokens).to(device=merged_attn_mask.device)
            output_token_num = len(image_tokens[-1])
            if not inference:
                merged_labels[i][-output_token_num+1:] = merged_tokens[i][-output_token_num+1:].clone()
                # Set target img start token as the last token of the model input
                # The rest tokens (img visual tokens + img end token) as the model output to compute loss
        
        # 7. Remove redundant pad_token_id added by data_collator
        redundant_num = max_seq_len - max([batch_added_image_token_nums[i] + ori_seq_len - ori_pad_num[i] for i in range(bsz)])
        merged_tokens = merged_tokens[:, redundant_num:]
        merged_attn_mask = merged_attn_mask[:, redundant_num:]
        if not inference:
            merged_labels = merged_labels[:, redundant_num:]

        if inference:
            # During inference, we remove the output target image in the merged_tokens, so we need to
            # add an image start token to merged_tokens and merged_attn_mask
            image_start_token = torch.tensor([32000], dtype=torch.long).to(text_tokens.device)
            image_start_token = image_start_token.expand(bsz, -1)
            image_start_attn = torch.ones((bsz, 1), dtype=torch.long).to(text_tokens.device)   # [bsz, 1]

            merged_tokens = torch.cat((merged_tokens, image_start_token), dim=-1)
            merged_attn_mask = torch.cat((merged_attn_mask, image_start_attn), dim=-1)

        return merged_tokens, merged_attn_mask, merged_labels
    
    def tokenize_image(self, image_list, add_special=True):
        bsz, num = image_list.shape
        image_tensor = []
        for iids in image_list:
            for iid in iids:
                image_tensor.append(self.img_dataset[iid.item()])
        image_tensor = torch.stack(image_tensor, dim=0).to(self.device)

        with self.maybe_autocast():
            image_tokens = self.visual_tokenizer.tokenize_image(image_tensor, add_special=add_special)  # token seq tuples

        image_tokens = split_tuple(image_tokens, num)
        if not add_special:
            # add image start token and end token to target output image
            special_tokens = torch.tensor([32000, 32001], dtype=torch.long, device=self.device)
            for i,image_token in enumerate(image_tokens):
                image_tokens[i][-1] = torch.cat([special_tokens[:1], image_token[-1], special_tokens[1:]], dim=-1)
        assert len(image_tokens) == bsz

        return image_tokens

    def forward(self, batch, hist_mask_ratio, add_special, training=True, label_smooth_factor=0, use_cache=None):
        mask_ratio = random.uniform(0, 1)
        input_ids, attention_mask, labels, tgt_embs = self.prepare_input_for_lavit(
            batch, mask_ratio, hist_mask_ratio, add_special=add_special,
            with_mask=True, inference=False,
        )

        if not training:  # detach labels for evaluation in PigeonTrainer
            labels = nested_detach(labels)
        if label_smooth_factor != 0:
            input_labels = None
        else:
            input_labels = labels
        
        batch_indices, replace_indices = torch.where(input_ids == -1)
        batch_clip_indices, replace_clip_indices = torch.where(input_ids == -2)
        replace_num = len(replace_indices)
        if replace_num > 0:
            input_ids[batch_indices, replace_indices] = self.llama_tokenizer.pad_token_id

        input_ids[batch_clip_indices, replace_clip_indices] = self.llama_tokenizer.pad_token_id
        temp_inputs_embeds = self.llama_model.get_input_embeddings()(input_ids)
        inputs_embeds = temp_inputs_embeds.clone()
        if replace_num > 0:
            inputs_embeds[batch_indices, replace_indices] = torch.zeros(
                (replace_num, self.llama_model.config.hidden_size), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        with self.maybe_autocast():
            tgt_embs = self.adapter(tgt_embs).to(dtype=inputs_embeds.dtype)
            inputs_embeds[batch_clip_indices, replace_clip_indices] = tgt_embs

            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=input_labels,
                use_cache=use_cache,
            )
            # Here we only compute the loss of target output tokens

        return outputs, labels
    
    @torch.no_grad()
    def generate_image_tokenids(self, batch, mask_ratio, 
        hist_mask_ratio, generation_config=None, add_special=True, with_mask=True,
    ):
        input_ids, attention_mask, labels, tgt_embs = self.prepare_input_for_lavit(
            batch, mask_ratio, hist_mask_ratio, add_special=add_special,
            with_mask=with_mask, inference=True,
        )

        batch_indices, replace_indices = torch.where(input_ids == -1)
        batch_clip_indices, replace_clip_indices = torch.where(input_ids == -2)
        replace_num = len(replace_indices)
        if replace_num > 0:
            input_ids[batch_indices, replace_indices] = self.llama_tokenizer.pad_token_id

        input_ids[batch_clip_indices, replace_clip_indices] = self.llama_tokenizer.pad_token_id
        temp_inputs_embeds = self.llama_model.get_input_embeddings()(input_ids)
        inputs_embeds = temp_inputs_embeds.clone()
        if replace_num > 0:
            inputs_embeds[batch_indices, replace_indices] = torch.zeros(
                (replace_num, self.llama_model.config.hidden_size), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        # supress the text tokens
        supress_range = range(3, 32000)
        supress_tokens = [x for x in supress_range]

        with self.maybe_autocast():
            tgt_embs = self.adapter(tgt_embs)
            inputs_embeds[batch_clip_indices, replace_clip_indices] = tgt_embs
            
            outputs = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                suppress_tokens=supress_tokens,
            )
        
        return outputs, labels
    
    @torch.no_grad()
    def generate_image(self, batch=None,
        image_tokens=None,
        mask_ratio=0.2,
        hist_mask_ratio=0.2,
        add_special=True,
        with_mask=True,
        llama_generation_config=None,
        num_return_images=1,  # num_return_images for each token_sequence
        width=512,
        height=512,
        original_size=None,
        crops_coords_top_left=None,
        num_inference_steps=25, 
        guidance_scale_for_diffusion=7.0,
        generator=None,
    ):
        if batch is None and image_tokens is None:
            raise ValueError("Generated image requires `batch` input or `image_tokens` input.")

        if image_tokens is None:
            image_tokens = self.generate_image_tokenids(
                batch, mask_ratio, hist_mask_ratio, llama_generation_config, add_special, with_mask
            )

        if self.unet is None:
            self.prepare_dm_to_generate_image()
            
        if self.pixel_decoding == 'lowres':
            return self.pixel_decoding_origin(image_tokens, width, height, num_inference_steps, guidance_scale_for_diffusion, generator)
        
        # Perform pixel decoding from tokenids to RGB pixel values
        with self.maybe_autocast():
            # Take the token id as input, generate the decoded embeddings
            # The negative prompt embeddings shall be forced to always be set to 0
            prompt_embeds, pooled_prompt_embeds = self.generate_image_embeds(image_tokens)
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

            batch_size = len(prompt_embeds) * num_return_images
            torch_device = self.device

            latents = torch.randn(
                (batch_size, self.unet.config.in_channels, height // 8, width // 8), generator=generator, device=torch_device
            )
            latents = latents * self.scheduler.init_noise_sigma
            self.scheduler.set_timesteps(num_inference_steps, device=torch_device)

            # Prepare added time ids & embeddings
            add_text_embeds = pooled_prompt_embeds
            target_size = (width, height)
            original_size = original_size if original_size is not None else target_size
            crops_coords_top_left = crops_coords_top_left if crops_coords_top_left is not None else (0,0)
            add_time_ids = self._get_add_time_ids(original_size, crops_coords_top_left, target_size, prompt_embeds.dtype)
            add_time_ids = add_time_ids.repeat(batch_size, 1)
            negative_add_time_ids = add_time_ids

            input_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            input_embeds = input_embeds.repeat_interleave(num_return_images, dim=0)
            add_text_embeds = add_text_embeds.repeat_interleave(num_return_images, dim=0)

            input_embeds = input_embeds.to(torch_device)
            add_text_embeds = add_text_embeds.to(torch_device)
            add_time_ids = add_time_ids.to(torch_device)

            assert input_embeds.shape[0] == add_text_embeds.shape[0] == add_time_ids.shape[0] == latents.shape[0] * 2

            for t in tqdm(self.scheduler.timesteps, leave=False):
            # for t in self.scheduler.timesteps:
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=input_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale_for_diffusion * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if needs_upcasting:
            self.upcast_vae()
        
        with torch.cuda.amp.autocast(enabled=False):
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            output_image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        output_image = output_image.float()
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        output_image = output_image.detach().cpu().permute(0, 2, 3, 1).numpy()

        if self.check_safety:
            output_image, _ = self.run_safety_checker(output_image)
        
        output_images = self.numpy_to_pil(output_image)

        return output_images, prompt_embeds, pooled_prompt_embeds
    
    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids
    
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)
    
    def run_safety_checker(self, image_array):
        safety_checker_input = self.feature_extractor(self.numpy_to_pil(image_array), return_tensors="pt").to(self.device)
        image_array, has_nsfw_concept = self.safety_checker(
            images=image_array, clip_input=safety_checker_input.pixel_values.to(self.dtype)
        )
        return image_array, has_nsfw_concept

    @torch.no_grad()
    def pixel_decoding_origin(self, image_tokens, 
        width=512,
        height=512,
        num_inference_steps=50, 
        guidance_scale_for_decoder=5.0,
        generator=None,
    ):
        """
        For the origin LaVIT pixel decoder (based on SD v1.5), we have updated the pixel decoder. The new
        decoder supports to generate high resolution and aesthetics images. We strongly recommond you to use
        our new decoder for image synthesis.
        """
        with self.maybe_autocast():
            # Take the token id as input, generate the decoded embeddings
            xrec = self.generate_image_embeds(image_tokens)
            batch_size = len(xrec)
            torch_device = self.device

            # To prepare the neative condition
            _, num_tokens, C = xrec.shape
            encoder_hidden_uncond = torch.zeros(batch_size, num_tokens, C, dtype=xrec.dtype).to(torch_device)
            uncond_embeddings = self.uncond_embeddings[0].to(xrec.dtype)
            encoder_hidden_uncond[:,:len(uncond_embeddings)] = uncond_embeddings
            
            # To set the mask
            encoder_mask = torch.ones(batch_size, num_tokens, dtype=torch.long).to(torch_device)
            uncond_encoder_mask = torch.zeros(batch_size, num_tokens, dtype=torch.long).to(torch_device)
            uncond_encoder_mask[:, :len(uncond_embeddings)] = 1
            encoder_mask = encoder_mask.bool()
            uncond_encoder_mask = uncond_encoder_mask.bool()
            
            # text_embeddings, uncond_embeddings, encoder_mask, uncond_encoder_mask = self.generate_prompt_embeds(xrec)
            text_embeddings = torch.cat([encoder_hidden_uncond, xrec])
            text_embeddings_mask = torch.cat([uncond_encoder_mask, encoder_mask])
    
            latents = torch.randn(
                (batch_size, self.unet.config.in_channels, height // 8, width // 8), generator=generator
            )
            latents = latents * self.scheduler.init_noise_sigma
            latents = latents.to(torch_device)

            self.scheduler.set_timesteps(num_inference_steps, device=torch_device)

            for t in tqdm(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, 
                        encoder_attention_mask=text_embeddings_mask).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale_for_decoder * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            latents = latents / self.vae.config.scaling_factor
            output_image = self.vae.decode(latents).sample

        output_image = output_image.float()
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        output_image = output_image.detach().cpu().permute(0, 2, 3, 1).numpy()

        if self.check_safety:
            output_image, _ = self.run_safety_checker(output_image)
        
        output_images = self.numpy_to_pil(output_image)

        return output_images, xrec
    
    def generate_image_embeds(self, image_tokens):
        # Generate the image embeddings, that can be input to decoder to rendering pixel
        batch_size = len(image_tokens)
        tokens_prune = []; token_nums = []
        device = self.device
        
        for i_b in range(batch_size):
            image_token = image_tokens[i_b]
            image_token = image_token - 32002
            image_token = image_token[image_token >= 0]
            if len(image_token) > 256:
                image_token = image_token[:256]  # remove redundant image tokens. The token num of each image should no more than 256.
            token_nums.append(len(image_token))
            tokens_prune.append(image_token)

        tokens_prune = torch.cat(tokens_prune, dim=0)
        token_nums = torch.as_tensor(token_nums, dtype=torch.long).to(device)
        torch_dtype = self.dtype

        token_quantize = self.visual_tokenizer.quantize.embedding(tokens_prune)  # [np, d]

        token_quantize = token_quantize.to(torch_dtype)

        return self.tokenizer_decoder(token_quantize, token_nums)
    
    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

def split_tuple(tup, size):
    return [list(tup[i:i + size]) for i in range(0, len(tup), size)]
