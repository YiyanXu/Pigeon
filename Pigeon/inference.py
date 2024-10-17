import os
import sys
import math
from typing import List, Tuple
import logging
from tqdm import tqdm
import shutil
import random

import inspect
import numpy as np 
import fire
import torch
import safetensors
from PIL import Image
import torch.utils
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback
from typing import Optional, Union
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.data.data_collator import default_data_collator
from models.lavit_utils import get_rank
from models.lavit_for_pigeon import LaVITforPigeon
from models.trainer.trainer_for_pigeon import PigeonTrainer, PigeonTrainingArguments
import data_utils

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)

setup_logger()
logger = logging.getLogger(__name__)

RATIO_DICT = {
    '1:1' : (1024, 1024),
    '4:3' : (896, 1152),
    '3:2' : (832, 1216),
    '16:9' : (768, 1344),
    '2:3' : (1216, 832),
    '3:4' : (1152, 896),
}

def main(
    model_path: str = "/path/to/pre-trained/LaVIT/",
    model_dtype: str = "bf16",
    output_dir: str = "/path/to/output/",
    pre_ckpt: str = None,
    mode: str = "test",
    # model-specific hyperparams
    use_xformers: bool = True,
    load_in_8bit: bool = True,
    check_safety: bool = False,
    pixel_decoding: str = "highres",
    mask_type: str = "mutual",
    with_mask: bool = False,
    mask_ratio: float = 0.2,  # the mask ratio could be changed during inference
    hist_mask_ratio: float = 0.2, 
    add_special: bool = False,
    # dataset info
    scenario: str = "sticker",
    img_folder_path: str = "/path/to/data/",
    data_path: str = "/path/to/data/",
    # inference hyperparams
    batch_size: int = 4,
    dm_batch_size: int = 4,
    seed: int = 123,
    resume_from_checkpoint: Optional[Union[int]] = None,  # either training checkpoint or final adapter
    # llama generation config
    use_nucleus_sampling: bool = True,
    top_p: float = 1.0,
    top_k: int = 50,
    temperature: float = 1,
    num_beams: int = 4,
    min_length: int = 20,
    length_penalty: int = 1,  # TODO: what ??
    num_return_sequences: int = 1,
    guidance_scale_for_llm: float = 5.0,
    # dm generation config
    ratio: str = "2:3",
    crops_coords_top_left=None,
    guidance_scale_for_dm: float = 7.0,
    num_inference_steps: int = 25,
    num_return_images: int = 1,
):
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    params = {arg:values[arg] for arg in args}

    set_random_seed(seed=seed)

    logger.info(f"***** Inference LaVIT with params:")
    for k,v in params.items():
        logger.info(f"{k}: {v}")
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    all_image_paths = np.load(os.path.join(data_path, "all_image_paths.npy"), allow_pickle=True)
    
    eval_name = "eval" if mode == "valid" else "eval-tes"
    if not with_mask:
        eval_name += "-wo-mask"
        
    if resume_from_checkpoint:
        resume_from_checkpoint = f"checkpoint-{resume_from_checkpoint}"
        checkpoint_path = os.path.join(output_dir, resume_from_checkpoint)
        eval_save_path = os.path.join(output_dir, eval_name, f"{resume_from_checkpoint}-scale{guidance_scale_for_llm}-mask{mask_ratio}")
        if not os.path.exists(checkpoint_path):
            logger.info(f"{resume_from_checkpoint} doesn't exist in {output_dir}. Please check the checkpoint carefully.")
            return
        else:
            os.makedirs(eval_save_path, exist_ok=True)
            logger.info(f"***** Resume from checkpoint {resume_from_checkpoint}")
            lora_weights = os.path.join(checkpoint_path, "llama")
            
            # Prepare LaVIT for Pigeon and load lora weights for llama model.
            lavit = LaVITforPigeon(
                model_path=model_path,
                model_dtype=model_dtype,
                lora_config=None,
                lora_weights=lora_weights,
                use_xformers=use_xformers,
                check_safety=check_safety,
                pixel_decoding=pixel_decoding,
                mask_type=mask_type,
                img_folder_path=img_folder_path,
                all_image_paths=all_image_paths,
                data_path=data_path,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
                training=False,
            )
            lavit = lavit.to(device)

            if lavit.mask_generator is not None:
                # Load mask generator for model
                mask_generater_path = os.path.join(pre_ckpt, "mask_generator")
                mask_generator_ckpt = os.listdir(mask_generater_path)[0]
                if mask_generator_ckpt.endswith(".bin"):
                    state_dict = torch.load(os.path.join(mask_generater_path, mask_generator_ckpt))
                elif mask_generator_ckpt.endswith(".safetensors"):
                    state_dict = safetensors.torch.load_file(os.path.join(mask_generater_path, mask_generator_ckpt))
                else:
                    raise ValueError(f"Unexpected mask generator checkpoint {mask_generator_ckpt} in PATH:{mask_generater_path}.")

                lavit.mask_generator.load_state_dict(state_dict, False)
                del state_dict
            
            adapter_path = os.path.join(pre_ckpt, "adapter")
            adapter_ckpt = os.listdir(adapter_path)[0]
            if adapter_ckpt.endswith(".bin"):
                state_dict = torch.load(os.path.join(adapter_path, adapter_ckpt))
            elif adapter_ckpt.endswith(".safetensors"):
                state_dict = safetensors.torch.load_file(os.path.join(adapter_path, adapter_ckpt))
            else:
                raise ValueError(f"Unexpected mask generator checkpoint {adapter_ckpt} in PATH:{adapter_path}.")

            lavit.adapter.load_state_dict(state_dict, False)
            del state_dict

    else:
        resume_from_checkpoint = f"checkpoint-{resume_from_checkpoint}"  # pretrained LaVIT
        eval_save_path = os.path.join(output_dir, eval_name, f"{resume_from_checkpoint}-scale{guidance_scale_for_llm}")
        os.makedirs(eval_save_path, exist_ok=True)
        # Prepare pretrained LaVIT for Pigeon.
        logger.info("Inference on pretrained LaVIT...")
        lavit = LaVITforPigeon(
            model_path=model_path,
            model_dtype=model_dtype,
            lora_config=None,
            lora_weights=None,
            use_xformers=use_xformers,
            check_safety=check_safety,
            pixel_decoding=pixel_decoding,
            mask_type=mask_type,
            img_folder_path=img_folder_path,
            all_image_paths=all_image_paths,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            training=False,
            pretrained=True,
        )
        lavit = lavit.to(device)

    save_params_to_txt(os.path.join(eval_save_path, f"args_{resume_from_checkpoint}.txt"), **params)

    if not ddp and torch.cuda.device_count() > 1:
        lavit.llama_model.is_parallelizable = True
        lavit.llama_model.model_parallel = True

    if mode == "test":
        test_data_path = os.path.join(data_path, "data_ready", "test_ready.npy")
    else:
        test_data_path = os.path.join(data_path, "data_ready", "valid_ready.npy")

    if os.path.exists(test_data_path):
        logger.info(f"Loading processed test dataset from {os.path.join(data_path, 'data_ready')}")
        test_data = np.load(test_data_path, allow_pickle=True).item()
    else:
        logger.info(f"Processed dataset doesn't exist. Start data processing...")
        if scenario == "sticker":
            item_info = np.load(os.path.join(data_path, "mapped_anno_dict.npy"), allow_pickle=True).item()
            semantics = np.load(os.path.join(data_path, "sticker_semantics.npy"), allow_pickle=True).tolist()
        elif scenario == "movie":
            item_info = np.load(os.path.join(data_path, "mapped_movies.npy"), allow_pickle=True).item()
            semantics = np.load(os.path.join(data_path, "movie_semantics.npy"), allow_pickle=True).tolist()

        test_seq = np.load(os.path.join(data_path, "test.npy"), allow_pickle=True).item()

        test_data = data_utils.process_data(scenario, test_seq, item_info, semantics, lavit.llama_tokenizer)

        if not os.path.exists(os.path.join(data_path, "data_ready")):
            os.makedirs(os.path.join(data_path, "data_ready"))

        np.save(test_data_path, np.array(test_data))
        logger.info(f"Saving processed test dataset into {os.path.join(data_path, 'processed_seq', 'data_ready')}")

    test_dataset = data_utils.PigeonDataset(test_data)
    data_collator = data_utils.PigeonCollator(
        pad_token_id=lavit.llama_tokenizer.pad_token_id,
        padding_side=lavit.llama_tokenizer.padding_side,
        return_tensors="pt"
    )
    test_sampler = SequentialSampler(test_dataset)
    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": data_collator,
        "sampler": test_sampler,
        "drop_last": False
    }
    test_dataloader = DataLoader(test_dataset, **dataloader_params)

    generation_config = transformers.GenerationConfig(
        do_sample=use_nucleus_sampling,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=num_beams,
        min_new_tokens=min_length,
        max_new_tokens=300,
        bos_token_id=32000,
        eos_token_id=32001,
        pad_token_id=lavit.llama_tokenizer.pad_token_id,
        length_penalty=length_penalty,
        num_return_sequences=num_return_sequences,
        guidance_scale=guidance_scale_for_llm,
    )

    lavit.eval()

    # DM config
    height, width = RATIO_DICT[ratio]
    original_size = None

    temp_folder = os.path.join(eval_save_path, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    if os.path.exists(os.path.join(temp_folder, f"image_tokens_{num_return_sequences}seq.npy")):
        # Already infered image tokens for image generation
        image_tokens = np.load(os.path.join(temp_folder, f"image_tokens_{num_return_sequences}seq.npy"), allow_pickle=True)
        image_tokens = [torch.tensor(image_token) for image_token in image_tokens]

        labels = np.load(os.path.join(temp_folder, "labels.npy"), allow_pickle=True)
        labels = [torch.tensor(label) for label in labels]

        uids = np.load(os.path.join(temp_folder, "uids.npy"), allow_pickle=True)
        uids = torch.tensor(uids)

        genres = np.load(os.path.join(temp_folder, "genres.npy"), allow_pickle=True)
        genres = torch.tensor(genres)

        target_image_iid = np.load(os.path.join(temp_folder, "target_image_iid.npy"), allow_pickle=True)
        target_image_iid = torch.tensor(target_image_iid)

        history = np.load(os.path.join(temp_folder, "history.npy"), allow_pickle=True)
        history = torch.tensor(history)
    else:
        image_tokens, labels, uids, genres, target_image_iid, history = generate_image_tokens(
            lavit, test_dataloader, mask_ratio, hist_mask_ratio, 
            add_special, with_mask, generation_config, temp_folder,
        )

    generator = torch.Generator(device=device).manual_seed(seed)
    dm_batch_size = dm_batch_size // (num_return_sequences * num_return_images)
    generate_images(
        lavit=lavit, 
        image_tokens=image_tokens,
        labels=labels, 
        uids=uids,
        genres=genres,
        target_image_iid=target_image_iid, 
        history=history,
        num_return_sequences=num_return_sequences,
        batch_size=dm_batch_size,
        save_folder=eval_save_path,
        eval_name=eval_name,
        img_folder_path=img_folder_path,
        all_image_paths=all_image_paths,
        num_return_images=num_return_images,
        width=width,
        height=height,
        original_size=original_size,
        crops_coords_top_left=crops_coords_top_left,
        num_inference_steps=num_inference_steps,
        guidance_scale_for_dm=guidance_scale_for_dm,
        generator=generator,
    )

    logger.info(f"The {resume_from_checkpoint} has been inferenced for evaluation in {eval_save_path}.")

def generate_images(
    lavit: LaVITforPigeon = None,
    image_tokens: List[torch.Tensor] = None,
    labels: List[torch.Tensor] = None,
    uids: torch.Tensor = None,
    genres: torch.Tensor = None,
    target_image_iid: torch.Tensor = None,
    history: torch.Tensor = None,
    num_return_sequences: int = 1,
    batch_size: int = 15,
    save_folder: str = None,
    eval_name: str = None,
    img_folder_path: str = None,
    all_image_paths: List[str] = None,
    num_return_images: int = None,
    width: int = 512,
    height: int = 512,
    original_size: Tuple[int] = None,
    crops_coords_top_left: Tuple[int] = None,
    num_inference_steps: int = 25,
    guidance_scale_for_dm: float = 7.0,
    generator: torch.Generator = None,
):
    total_token_num = len(image_tokens)
    step_num = batch_size * num_return_sequences
    total_steps = math.ceil(total_token_num / step_num)

    total_task_num = len(uids)

    all_res = {}

    token_start = 0
    for step in tqdm(range(total_steps), total=total_steps, dynamic_ncols=True):
        token_end = min(total_token_num, token_start + step_num)
        batch_tokens = [(image_tokens[i]).to(lavit.device) for i in range(token_start, token_end)]

        start = step * batch_size
        end = min(total_task_num, start + batch_size)
        batch_uids = uids[start:end]
        batch_genres = genres[start:end]
        batch_labels = labels[start:end]
        batch_target_image_iid = target_image_iid[start:end]
        batch_history = history[start:end]

        batch_images, _, _ = lavit.generate_image(
            image_tokens=batch_tokens,
            num_return_images=num_return_images,
            width=width,
            height=height,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            num_inference_steps=num_inference_steps,
            guidance_scale_for_diffusion=guidance_scale_for_dm,
            generator=generator,
        )  # ACTUAL batch size = batch_size * num_return_sequences * num_return_images

        image_save_folder = os.path.join(save_folder, f"dm_scale{guidance_scale_for_dm}")
        os.makedirs(image_save_folder, exist_ok=True)
        all_res = save_batch_results(all_res, batch_uids, batch_genres, batch_history, batch_target_image_iid,
                                     batch_labels, batch_tokens,
                                     batch_images, num_return_sequences, num_return_images, image_save_folder,
                                     eval_name, img_folder_path, all_image_paths)
        
        np.save(os.path.join(save_folder, f"all_res_dm_scale{guidance_scale_for_dm}.npy"), np.array(all_res))
        token_start += batch_size * num_return_sequences
    
    return

def save_batch_results(all_res, batch_uids, batch_genres, batch_history, batch_target_image_iid, batch_labels,
                       batch_tokens, batch_images,
                       num_return_sequences, num_return_images, save_folder,
                       eval_name, img_folder_path, all_image_paths):
    for i,uid in enumerate(batch_uids):
        uid = uid.item()
        genre_id = batch_genres[i].item()

        if uid not in all_res:
            all_res[uid] = {}

        target_iid = batch_target_image_iid[i].item()

        if target_iid not in all_res[uid]:
            all_res[uid][target_iid] = {}

        all_res[uid][target_iid][genre_id] = {}
        all_res[uid][target_iid][genre_id]["history"] = batch_history[i].clone()
        all_res[uid][target_iid][genre_id]["labels"] = batch_labels[i].clone()
        all_res[uid][target_iid][genre_id]["seqs"] = {}

        hist_img_save_path = os.path.join(save_folder, str(uid), str(target_iid), str(genre_id))
        os.makedirs(hist_img_save_path, exist_ok=True)
        for j,iid in enumerate(batch_history[i]):
            shutil.copy(os.path.join(img_folder_path, all_image_paths[iid]), os.path.join(hist_img_save_path, f"{j}-{iid}.jpg"))

        for seq_id in range(num_return_sequences):
            all_res[uid][target_iid][genre_id]["seqs"][seq_id] = {}
            all_res[uid][target_iid][genre_id]["seqs"][seq_id]["tokens"] = batch_tokens[i * num_return_sequences + seq_id].cpu()

            img_save_folder = os.path.join(save_folder, str(uid), str(target_iid), str(genre_id), str(seq_id))
            os.makedirs(img_save_folder, exist_ok=True)

            all_res[uid][target_iid][genre_id]["seqs"][seq_id][f"images"] = []
            image_start_idx = (i * num_return_sequences + seq_id) * num_return_images
            for j,image in enumerate(batch_images[image_start_idx:image_start_idx + num_return_images]):
                img_save_path = os.path.join(img_save_folder, f"{j}.jpg")
                image.save(img_save_path)
                img_save_path = img_save_path.split(f"/{eval_name}/")[-1]
                all_res[uid][target_iid][genre_id]["seqs"][seq_id][f"images"].append(img_save_path)
            
        # copy the target image for viewing
        tgt_img_save_path = os.path.join(save_folder, str(uid), str(target_iid), str(genre_id), f"target.jpg")
        shutil.copy(os.path.join(img_folder_path, all_image_paths[target_iid]), tgt_img_save_path)

    return all_res
    
def generate_image_tokens(
    lavit: LaVITforPigeon = None,
    dataloader: DataLoader = None,
    mask_ratio: float = 0.2,
    hist_mask_ratio: float = 0.2,
    add_special: bool = True,
    with_mask: bool = True,
    generation_config: transformers.GenerationConfig = None,
    temp_folder: str = None,
):
    image_tokens = []
    labels = []
    uids = []
    genres = []
    target_image_iid = []
    history = []
    postfix = f"{generation_config.num_return_sequences}seq"
    if os.path.exists(os.path.join(temp_folder, "uids.npy")):
        save_uids = False
    else:
        save_uids = True

    for i,batch in tqdm(enumerate(dataloader), total=len(dataloader), dynamic_ncols=True):
        batch_image_tokens, batch_labels = lavit.generate_image_tokenids(
            batch, mask_ratio, hist_mask_ratio,
            generation_config, add_special, with_mask
        )

        image_tokens.extend(torch.unbind(batch_image_tokens.cpu(), dim=0))
        np.save(os.path.join(temp_folder, f"image_tokens_{postfix}.npy"), np.array(image_tokens, dtype=object))

        if save_uids:
            batch_uids = batch["uids"]
            batch_genres = batch["genres"]
            batch_target_image_iid = batch["image"][:, -1]
            batch_history = batch["image"][:, :-1]

            labels.extend(torch.unbind(batch_labels.cpu(), dim=0))
            uids.append(batch_uids)
            genres.append(batch_genres)
            target_image_iid.append(batch_target_image_iid)
            history.append(batch_history)
            
            np.save(os.path.join(temp_folder, "labels.npy"), np.array(labels, dtype=object))

        # if i > 10:  # for debug
        #     break
    
    if save_uids:
        uids = torch.cat(uids, dim=0)  # [bsz]
        genres = torch.cat(genres, dim=0)
        target_image_iid = torch.cat(target_image_iid, dim=0)  # [bsz]
        history = torch.cat(history, dim=0)  # [bsz, 5]

        np.save(os.path.join(temp_folder, "uids.npy"), np.array(uids))
        np.save(os.path.join(temp_folder, "genres.npy"), np.array(genres))
        np.save(os.path.join(temp_folder, "target_image_iid.npy"), np.array(target_image_iid))
        np.save(os.path.join(temp_folder, "history.npy"), np.array(history))
    
    return image_tokens, labels, uids, genres, target_image_iid, history


def save_params_to_txt(file_path, **kwargs):
    with open(file_path, 'w') as file:
        for key, value in kwargs.items():
            file.write(f'{key}: {value}\n')

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_free_space(path):
    statvfs = os.statvfs(path)
    free_space_bytes = statvfs.f_bavail * statvfs.f_frsize
    return free_space_bytes

if __name__ == "__main__":
    fire.Fire(main)

