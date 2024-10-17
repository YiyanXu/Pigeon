import copy
import os
import random
import re
from fileinput import filename

import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from typing import Optional, Union, Mapping, List, Dict, Any
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from transformers import LlamaTokenizer
from transformers.utils import PaddingStrategy

class sticker_template:
    instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    task_description = "Instruction: You are a helpful personalized assistant. You will be provided with a list of stickers that the user likes to analyze the user's preference over stickers. Based on this analysis, please design a personalized sticker that aligns with target sticker info and the user's sticker tastes."
    prompt_start = "Input: "
    history_start = "The user likes the following stickers: "
    target_start = "The target sticker info: a sticker with the emotion of <emo>. "
    prompt_end = "Response: "

class movie_template:
    instruction = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    task_description = "Instruction: You are a helpful personalized assistant. You will be provided with a list of movies that the user likes, along with the movie posters, to analyze the user's preference over posters. Based on this analysis, please design a personalized poster that aligns with target movie and the user's poster tastes."
    prompt_start = "Input: "
    history_start = "The user likes the following movies: "
    target_start = "The target movie is titled '<title>'. "
    prompt_end = "Response: "

SEP = "\n"
IMG_TOKEN = "<img>"
EMB_TOKEN = "<emb>"

class ImagePathDataset(Dataset):
    def __init__(self, folder_path, paths, trans=None):
        self.folder_path = folder_path
        self.paths = paths
        self.trans = trans
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.paths[idx])
        img = Image.open(path).convert('RGB')
        if self.trans is not None:
            img = self.trans(img)
        
        return img.to(memory_format=torch.contiguous_format)

class DiffusionImageDataset(Dataset):
    def __init__(self, start, end):
        self.data = list(range(start, end))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class PigeonDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data["uids"])
    
    def __getitem__(self, idx):
        text_input_ids = self.data["text_input_ids"][idx]
        text_attn_mask = self.data["text_attn_mask"][idx]
        image = self.data["image"][idx]
        uids = self.data["uids"][idx]
        genres = self.data["genres"][idx]

        return {
            "text_input_ids": text_input_ids,
            "text_attn_mask": text_attn_mask,
            "image": image,
            "uids": uids,
            "genres": genres
        }
    
    def shuffle(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        data_len = len(self.data["uids"])
        indices = torch.randperm(data_len).tolist()
        self.data = {key:[self.data[key][i] for i in indices] for key in self.data}

        return self
    
    def select(self, indices):
        selected_data = {key:[self.data[key][i] for i in indices] for key in self.data}
        return PigeonDataset(selected_data)

@dataclass
class PigeonCollator:
    pad_token_id: int = 2
    padding_side: Optional[str] = "left"
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(features, (list, tuple)) and isinstance(features[0], Mapping):
            features = {key: [example[key] for example in features] for key in features[0].keys()}

        max_length = max(len(input_ids) for input_ids in features["text_input_ids"])
        bsz = len(features["text_input_ids"])
        for i in range(bsz):
            pad_num = max_length - len(features["text_input_ids"][i])

            if self.padding_side == "left":
                features["text_input_ids"][i] = [self.pad_token_id] * pad_num + features["text_input_ids"][i]
                features["text_attn_mask"][i] = [0] * pad_num + features["text_attn_mask"][i]
            else:
                features["text_input_ids"][i] = features["text_input_ids"][i] + [self.pad_token_id] * pad_num
                features["text_attn_mask"][i] = features["text_attn_mask"][i] + [0] * pad_num
        
        if self.return_tensors == "pt":
            features = {key: torch.tensor(features[key], dtype=torch.long) for key in features.keys()}

        return features

def process_data(
    scenario: str="sticker",
    data: dict=None,
    item_info: dict=None,
    semantics: list=None,
    tokenizer: LlamaTokenizer=None,
):
    text_input_ids = []
    text_attn_mask = []
    image = []
    uids = []
    genres = []
    for uid in data:
        for genre in data[uid]:
            hist_seqs = data[uid][genre]
            for hist_seq in hist_seqs:
                if scenario == "sticker":
                    text_prompt = generate_prompt_for_stickers(sticker_template, hist_seq, item_info, semantics)
                elif scenario == "movie":
                    text_prompt = generate_prompt_for_movies(movie_template, hist_seq, item_info, semantics)
                prompt_tokens = tokenizer(
                    text_prompt, padding="longest", return_tensors="pt", add_special_tokens=False
                )
                text_input_ids.append(prompt_tokens.input_ids.squeeze(0).tolist())
                text_attn_mask.append(prompt_tokens.attention_mask.squeeze(0).tolist())
                # the last image is the target image, to be the input and output at the same time
                image.append(hist_seq)
                uids.append(uid)
                genres.append(genre)
    
    return {"text_input_ids": text_input_ids, "text_attn_mask": text_attn_mask,
            "image": image, "uids": uids, "genres": genres}

def generate_prompt_for_movies(
    template,
    hist_seq: dict=None,
    movies_info: dict=None,
    semantics: list=None,
):
    hist_prompt = template.history_start
    for iid in hist_seq[:-1]:
        title = movies_info[iid]["title"]
        try:
            title = re.findall(r'^(.*) \(\d+\) *$', title)[0]
        except:
            title = title

        hist_prompt += SEP
        hist_prompt += title
        hist_prompt += " "
        hist_prompt += IMG_TOKEN
    
    tgt_iid = hist_seq[-1]
    target_title = movies_info[tgt_iid]["title"]
    try:
        target_title = re.findall(r'^(.*) \(\d+\) *$', target_title)[0]
    except:
        target_title = target_title

    target_prompt = template.target_start.("<title>", target_title)
    target_prompt = target_prompt + semantics[tgt_iid] + " " + EMB_TOKEN

    text_prompt = template.instruction + SEP + template.task_description + SEP + template.prompt_start + hist_prompt + SEP + target_prompt + SEP + template.prompt_end
    text_prompt += IMG_TOKEN
    
    return text_prompt

def generate_prompt_for_stickers(
    template,
    hist_seq: dict=None,
    anno_dict: dict=None,
    semantics: list=None,
):
    hist_prompt = template.history_start
    for iid in hist_seq[:-1]:
        hist_prompt += SEP
        hist_prompt += IMG_TOKEN
    
    tgt_iid = hist_seq[-1]
    emo = anno_dict[tgt_iid]["emo"]

    target_prompt = template.target_start.replace("<emo>", emo)
    target_prompt = target_prompt + semantics[tgt_iid] + " " + EMB_TOKEN

    text_prompt = template.instruction + SEP + template.task_description + SEP + template.prompt_start + hist_prompt + SEP + target_prompt + SEP + template.prompt_end
    text_prompt += IMG_TOKEN
    
    return text_prompt


