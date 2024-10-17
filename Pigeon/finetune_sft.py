import os
import sys
from typing import List
import logging
import random

import inspect
import numpy as np 
import fire
import torch
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
from models.lavit_utils import convert_weights_to_bf16, convert_weights_to_fp16
import data_utils

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno <= logging.INFO

def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    info_filter = InfoFilter()
    console_handler.addFilter(info_filter)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)

setup_logger()
logger = logging.getLogger(__name__)

def train(
    model_path: str = "/path/to/pre-trained/LaVIT/",
    model_dtype: str = "bf16",
    output_dir: str = "/path/to/output/",
    # model-specific hyperparams
    use_xformers: bool = True,
    load_in_8bit: bool = True,
    check_safety: bool = False,
    pixel_decoding: str = "highres",
    # mask generator hyperparams
    mask_type: str = "random",
    num_heads: int = 4,
    num_layers: int = 1,
    drop_prob: int = 0.2,
    hist_mask_ratio: float = 0.2, 
    add_special: bool = False,
    # dataset info
    scenario: str = "sticker",
    img_folder_path: str = "/path/to/data/",
    data_path: str = "/path/to/data/",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    group_by_length: bool = False,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    lr_schedule_type: str = "cosine",
    min_learning_rate: float = 1e-6,
    seed: int = 123,
    logging_steps: int = 25,
    save_steps: int = 25,
    eval_steps: int = 25,
    eval_num: int = 1000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    resume_from_checkpoint: Optional[Union[str, bool]] = None,  # either training checkpoint or final adapter
):
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    params = {arg:values[arg] for arg in args}
    
    os.makedirs(output_dir, exist_ok=True)
    if resume_from_checkpoint:
        ckpts = os.listdir(output_dir)
        ckpts = [ckpt for ckpt in ckpts if ckpt.startswith("checkpoint") and os.path.isdir(os.path.join(output_dir, ckpt))]
        if len(ckpts) == 0:
            resume_from_checkpoint = False
            params["resume_from_checkpoint"] = False
    
    logger.info(f"***** Finetuning Llama model with params:")
    for k,v in params.items():
        logger.info(f"{k}: {v}")
    
    save_params_to_txt(os.path.join(output_dir, "args.txt"), **params)
    set_random_seed(seed=seed)

    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    
    all_image_paths = np.load(os.path.join(data_path, "all_image_paths.npy"), allow_pickle=True)

    llama_lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    lavit = LaVITforPigeon(
        model_path=model_path,
        model_dtype=model_dtype,
        lora_config=llama_lora_config,
        use_xformers=use_xformers,
        check_safety=check_safety,
        pixel_decoding=pixel_decoding,
        mask_type=mask_type,
        num_heads=num_heads,
        num_layers=num_layers,
        drop_prob=drop_prob,
        img_folder_path=img_folder_path,
        all_image_paths=all_image_paths,
        data_path=data_path,
        load_in_8bit=load_in_8bit,
        device_map=device_map,
    )

    if not ddp and torch.cuda.device_count() > 1:
        lavit.llama_model.is_parallelizable = True
        lavit.llama_model.model_parallel = True


    # Load dataset
    train_data_path = os.path.join(data_path, "data_ready", "train_ready.npy")
    valid_data_path = os.path.join(data_path, "data_ready", "valid_ready.npy")
    if os.path.exists(train_data_path) and os.path.exists(valid_data_path):
        logger.info(f"Loading processed train dataset and valid dataset from {os.path.join(data_path, 'data_ready')}")
        train_data = np.load(train_data_path, allow_pickle=True).item()
        valid_data = np.load(valid_data_path, allow_pickle=True).item()
    else:
        logger.info(f"Processed datasets don't exist. Start data processing...")
        if scenario == "sticker":
            item_info = np.load(os.path.join(data_path, "mapped_anno_dict.npy"), allow_pickle=True).item()
            semantics = np.load(os.path.join(data_path, "sticker_semantics.npy"), allow_pickle=True).tolist()
        elif scenario == "movie":
            item_info = np.load(os.path.join(data_path, "mapped_movies.npy"), allow_pickle=True).item()
            semantics = np.load(os.path.join(data_path, "movie_semantics.npy"), allow_pickle=True).tolist()
            
        train_seq = np.load(os.path.join(data_path, "train.npy"), allow_pickle=True).item()
        valid_seq = np.load(os.path.join(data_path, "valid.npy"), allow_pickle=True).item()

        train_data = data_utils.process_data(scenario, train_seq, item_info, semantics, lavit.llama_tokenizer)
        valid_data = data_utils.process_data(scenario, valid_seq, item_info, semantics, lavit.llama_tokenizer)

        if not os.path.exists(os.path.join(data_path, "data_ready")):
            os.makedirs(os.path.join(data_path, "data_ready"))

        np.save(train_data_path, np.array(train_data))
        np.save(valid_data_path, np.array(valid_data))
        logger.info(f"Saving processed train dataset and valid dataset into {os.path.join(data_path, 'processed_seq', 'data_ready')}")

    train_dataset = data_utils.PigeonDataset(train_data)
    valid_dataset = data_utils.PigeonDataset(valid_data)

    trainer = PigeonTrainer(
        model=lavit,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=PigeonTrainingArguments(
            # model-specific arguments
            hist_mask_ratio=hist_mask_ratio,
            add_special=add_special,
            # training arguments
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            group_by_length=group_by_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_schedule_type,
            min_lr_ratio=min_learning_rate,
            fp16=False,
            logging_steps=logging_steps,
            optim="adamw_bnb_8bit",
            eval_num=eval_num,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            report_to=None,
            seed=seed,
            log_level="info",
        ),
        data_collator=data_utils.PigeonCollator(pad_token_id=lavit.llama_tokenizer.pad_token_id, padding_side=lavit.llama_tokenizer.padding_side, return_tensors="pt"),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    logger.info(f"BEST MODEL PATH: {trainer.state.best_model_checkpoint}")

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

if __name__ == "__main__":
    fire.Fire(train)

