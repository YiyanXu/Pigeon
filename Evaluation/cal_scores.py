import os
import ast
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
import fire
import inspect

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import open_clip

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

import eval_utils

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

def main(
    output_dir: str = "",
    scenario: str = "sticker",
    data_path: str = "",
    img_folder_path: str = "",
    batch_size: int = 64,
    mode: str = "test",
    ckpt: int = None,
    height: int = 512,
    width: int = 512,
    scale_for_llm: float = 1.0,
    scale_for_dm: float = 7.0,
    seed: int = 123,
):
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    params = {arg:values[arg] for arg in args}

    set_random_seed(seed=seed)

    logger.info(f"##### Calculate preference score and semantic alignment score #####")
    for k,v in params.items():
        logger.info(f"{k}: {v}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    if mode == "valid":
        hist_path = os.path.join(data_path, "data_ready", "valid_hist_embeds.npy")
    else:
        hist_path = os.path.join(data_path, "data_ready", "test_hist_embeds.npy")
    if os.path.exists(hist_path):
        logger.info(f"Loading history embeds for {mode} from {hist_path}...")
        history = np.load(hist_path, allow_pickle=True).item()
    else:
        all_clip_feats_path = os.path.join(data_path, "all_clip_feats.npy")
        if os.path.exists(all_clip_feats_path):
            logger.info(f"Loading clip feats for all images from {all_clip_feats_path}...")
            all_clip_feats = np.load(all_clip_feats_path, allow_pickle=True)
        else:
            logger.info("Extracting clip feats for all images...")
            all_clip_feats = eval_utils.extract_clip_feats(
                args.img_folder_path, all_image_paths, args.clip_batch_size,
                num_workers, device
            )
            np.save(all_clip_feats_path, np.array(all_clip_feats))
            logger.info(f"Successfully saved clip feats for all images into {all_clip_feats_path}.")
        
        all_clip_feats = torch.tensor(all_clip_feats)
        logger.info(f"clip feats shape: {all_clip_feats.shape}")
        logger.info("Processing history embeds for evaluation...")
        history = eval_utils.process_hist_embs(all_res, all_clip_feats)
        np.save(hist_path, np.array(history))
        logger.info(f"Successfully saved ori history embeds for {mode} set into {hist_path}.")
    
    all_image_paths = np.load(os.path.join(data_path, "all_image_paths.npy"), allow_pickle=True)

    if scenario == "sticker":
        semantics = np.load(os.path.join(data_path, "sticker_semantics.npy"), allow_pickle=True).tolist()
    elif scenario == "movie":
        semantics = np.load(os.path.join(data_path, "movie_semantics.npy"), allow_pickle=True).tolist()

    default_shape = (height, width)
    resize = transforms.Resize(default_shape, interpolation=transforms.InterpolationMode.BILINEAR)
    _, _, clip_trans = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")

    mode_name = "eval" if mode == "valid" else "eval-test"
    eval_path = os.path.join(output_dir, mode_name)

    mask_ratios = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    
    for i,mask_ratio in enumerate(mask_ratios):
        res_folder = os.path.join(eval_path, f"checkpoint-{ckpt}-scale{scale_for_llm}-mask{mask_ratio}")
        res_path = os.path.join(res_folder, f"all_res_dm_scale{scale_for_dm}.npy")
        logger.info(f"##### CURRENT RESULTS: {res_path} #####")
        all_res = np.load(res_path, allow_pickle=True).item()

        if i == 0:
            text4eval = []
            hist4clip = []

            seq_num, img_num = None, None
            for uid in all_res: 
                for tgt_iid in all_res[uid]:
                    for genre_id in all_res[uid][tgt_iid]:
                        text4eval.append(semantics[tgt_iid])
                        for i in range(5):  # hist_num=5
                            hist4clip.append(history[uid][tgt_iid][genre_id][i])
                        
                        if seq_num is None:
                            seq_num = len(all_res[uid][tgt_iid][genre_id]["seqs"])
                            img_num = len(all_res[uid][tgt_iid][genre_id]["seqs"][0]["images"])
        
        cur_score_save_path = os.path.join(res_folder, f"select_scores_dm_scale{scale_for_dm}.npy")
        if os.path.exists(cur_score_save_path):
            logger.info(f"The select scores for {res_folder} has already been calculated. Skip.")
            continue

        select_scores = {}
        for i in range(seq_num):
            for j in range(img_num):
                gen4clip = []
                for uid in all_res:
                    for tgt_iid in all_res[uid]:
                        for genre_id in all_res[uid][tgt_iid]:
                            img_path = all_res[uid][tgt_iid][genre_id]["seqs"][i]["images"][j]
                            img_path = os.path.join(eval_path, img_path)
                            im = Image.open(img_path).convert('RGB')
                            im = resize(im)
                            gen4clip.append(clip_trans(im))
            
                gen_dataset = EvalDataset(gen4clip)
                txt_dataset = EvalDataset(text4eval)

                _, clip_grd_scores = eval_utils.calculate_clip_score_given_data(
                    gen_dataset,
                    txt_dataset,
                    batch_size=batch_size,
                    device=device,
                )

                gen4clip_hist = [ele for ele in gen4clip for _ in range(5)]
                gen4personal_sim = {}
                gen4personal_sim["gen"] = gen4clip_hist
                gen4personal_sim["hist"] = hist4clip

                gen_dataset_personal_sim = PersonalSimDataset(gen4personal_sim)
                _, clip_hist_scores = eval_utils.evaluate_personalization_given_data_sim(
                    gen4eval=gen_dataset_personal_sim,
                    batch_size=batch_size,
                    device=device,
                )
                clip_hist_scores = clip_hist_scores.view(-1, 5)
                clip_hist_scores = torch.mean(clip_hist_scores, dim=1)

                del gen4clip
                del gen4clip_hist

                cur_idx = 0
                for uid in all_res:
                    if uid not in select_scores:
                        select_scores[uid] = {}
                    
                    for tgt_iid in all_res[uid]:
                        if tgt_iid not in select_scores[uid]:
                            select_scores[uid][tgt_iid] = {}
                        
                        for genre_id in all_res[uid][tgt_iid]:
                            if genre_id not in select_scores[uid][tgt_iid]:
                                select_scores[uid][tgt_iid][genre_id] = {}
                            
                            if i not in select_scores[uid][tgt_iid][genre_id]:
                                select_scores[uid][tgt_iid][genre_id][i] = {}
                                select_scores[uid][tgt_iid][genre_id][i]["tgt_score"] = []
                                select_scores[uid][tgt_iid][genre_id][i]["hist_score"] = []
                            
                            select_scores[uid][tgt_iid][genre_id][i]["tgt_score"].append(clip_grd_scores[cur_idx].item())
                            select_scores[uid][tgt_iid][genre_id][i]["hist_score"].append(clip_hist_scores[cur_idx].item())
                            cur_idx += 1

        np.save(cur_score_save_path, np.array(select_scores))
        logger.info(f"Select score for {res_folder} has been calculated successfully and saved in {cur_score_save_path}.")
    
    logger.info(f"Select scores for all the mask_ratio {mask_ratio} has been calculated successfully.")

class EvalDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class PersonalSimDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data["gen"])
    
    def __getitem__(self, index):
        gen = self.data["gen"][index]
        hist = self.data["hist"][index]

        return gen, hist

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
    fire.Fire(main)
