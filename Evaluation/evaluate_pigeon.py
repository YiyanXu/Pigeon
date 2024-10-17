import os
import ast
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import open_clip
from transformers import AutoImageProcessor

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

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--clip_batch_size', type=int, default=512,
                    help='Batch size to use')
parser.add_argument('--num_workers', type=int, default=1,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--dims4fid', type=int, default=2048,
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--data_path', type=str, default='/dataset/SER-30K/processed_seq')
parser.add_argument('--img_folder_path', type=str, default="/dataset/SER-30K/Images")
parser.add_argument('--dino_model_path', type=str, default='')
parser.add_argument('--dataset', type=str, default="SER-30K")
parser.add_argument('--output_dir', type=str, default="/checkpoints/sticker/DPO/")
parser.add_argument('--with_mask', action='store_true')
parser.add_argument('--eval_version', type=str, default=None)
parser.add_argument('--scenario', type=str, default="sticker")
parser.add_argument('--hist_mask_ratio', type=float, default=None)
parser.add_argument('--scale_for_llama', type=float, default=5.0)
parser.add_argument('--scale_for_dm', type=float, default=7.0)
parser.add_argument('--mask_ratio', type=float, default=0.0)
parser.add_argument('--ckpt', type=int, default=0)
parser.add_argument('--sim_func', type=str, default="cosine")
parser.add_argument('--lpips_net', type=str, default="vgg")
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--log_name', type=str, default="log")
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--height', type=int, default=512)
parser.add_argument('--mode', type=str, default="valid")

METRIC_LIST = ["FID", 
               "DINO Image Score", "CLIP Score", "CLIP Image Score",
               "Hist LPIPS",
               "Hist DINO Image Score",
               "Hist CLIP Score",
               "Hist CLIP Image Score",
               "Hist SSIM",
               "Hist MS-SSIM"]

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

def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    eval_mode_name = "eval" if args.mode == "valid" else "eval-test"
    if not args.with_mask:
        eval_mode_name += "-wo-mask"

    if args.hist_mask_ratio is not None:
        eval_mode_name += f"_hist{args.hist_mask_ratio}"

    if args.eval_version:
        eval_path = os.path.join(args.output_dir, args.eval_version, eval_mode_name)
    else:
        eval_path = os.path.join(args.output_dir, eval_mode_name)

    ckpt = args.ckpt
    scale_for_llama = f"scale{args.scale_for_llama}"
    scale_for_dm = f"dm_scale{args.scale_for_dm}"
    mask_ratio = f"mask{args.mask_ratio}"

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU:{torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    num_workers = args.num_workers

    eval_save_path = os.path.join(eval_path, f"eval_results.npy")
    logger.info(f"save_path:{eval_save_path}")

    if not os.path.exists(eval_save_path):
        all_eval_metrics = {}
    else:
        all_eval_metrics = np.load(eval_save_path, allow_pickle=True).item()

    if ckpt not in all_eval_metrics:
        all_eval_metrics[ckpt] = {}
        all_eval_metrics[ckpt][args.mask_ratio] = {}
        all_eval_metrics[ckpt][args.mask_ratio][args.scale_for_llama] = {}
        all_eval_metrics[ckpt][args.mask_ratio][args.scale_for_llama][args.scale_for_dm] = {}
    else:
        if args.mask_ratio not in all_eval_metrics[ckpt]:
            all_eval_metrics[ckpt][args.mask_ratio] = {}
        if args.scale_for_llama not in all_eval_metrics[ckpt][args.mask_ratio]:
            all_eval_metrics[ckpt][args.mask_ratio][args.scale_for_llama] = {}
        if args.scale_for_dm in all_eval_metrics[ckpt][args.mask_ratio][args.scale_for_llama]:
            logger.info(f"checkpoint-{ckpt} under llama-scale-{args.scale_for_llama} and dm-scale-{args.scale_for_dm} has already been evaluated. Skip.")
            return
        else:
            all_eval_metrics[ckpt][args.mask_ratio][args.scale_for_llama][args.scale_for_dm] = {}
    
    all_image_paths = np.load(os.path.join(args.data_path, "all_image_paths.npy"), allow_pickle=True)
    if scenario == "sticker":
        semantics = np.load(os.path.join(data_path, "sticker_semantics.npy"), allow_pickle=True).tolist()
    elif scenario == "movie":
        semantics = np.load(os.path.join(data_path, "movie_semantics.npy"), allow_pickle=True).tolist()
    res_folder = os.path.join(eval_path, f"checkpoint-{ckpt}-{scale_for_llama}-{mask_ratio}")
    all_res = np.load(os.path.join(res_folder, f"all_res_{scale_for_dm}.npy"), allow_pickle=True).item()

    if args.mode == "valid":
        hist_path = os.path.join(args.data_path, "data_ready", "valid_hist_embeds.npy")
    else:
        hist_path = os.path.join(args.data_path, "data_ready", "test_hist_embeds.npy")
    if os.path.exists(hist_path):
        logger.info(f"Loading history embeds for {args.mode} from {hist_path}...")
        history = np.load(hist_path, allow_pickle=True).item()
    else:
        all_clip_feats_path = os.path.join(args.data_path, "all_clip_feats.npy")
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
        logger.info(f"Successfully saved ori history embeds for {args.mode} set into {hist_path}.")

    default_shape = (args.height, args.width)
    resize = transforms.Resize(default_shape, interpolation=transforms.InterpolationMode.BILINEAR)
    _, _, clip_trans = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
    trans = transforms.ToTensor()

    dino_processor = AutoImageProcessor.from_pretrained(args.dino_model_path)
    all_dino_feats = np.load(os.path.join(args.data_path, "all_dino_feats.npy"))
    all_dino_feats = torch.tensor(all_dino_feats)

    grd4eval = []
    grd4clip = []
    text4eval = []
    grd4dino = []

    hist4eval = []
    hist4clip = []
    hist4lpips = []
    hist4ssim = []
    hist_text4eval = []
    hist4dino = []

    seq_num, img_num = None, None
    for uid in all_res:
        for tgt_iid in all_res[uid]:
            for genre_id in all_res[uid][tgt_iid]:
                tgt_img_path = os.path.join(args.img_folder_path, all_image_paths[tgt_iid])
                tgt_im = Image.open(tgt_img_path).convert('RGB')
                tgt_im = resize(tgt_im)
                grd4eval.append(trans(tgt_im))
                grd4clip.append(clip_trans(tgt_im))
                text4eval.append(semantics[tgt_iid])
                grd4dino.append(all_dino_feats[tgt_iid])

                for i in range(5):  # hist_num=5
                    hist4clip.append(history[uid][tgt_iid][genre_id][i])

                hist_iids = all_res[uid][tgt_iid][genre_id]["history"]
                for hist_iid in hist_iids:
                    hist_img_path = os.path.join(args.img_folder_path, all_image_paths[hist_iid])
                    hist_im = Image.open(hist_img_path).convert('RGB')
                    hist_im = resize(hist_im)
                    hist4eval.append(trans(hist_im))
                    hist4lpips.append(eval_utils.im2tensor_lpips(hist_im))
                    hist4ssim.append(np.array(hist_im, dtype=np.float32).transpose((2, 0, 1)))
                    hist_text4eval.append(semantics[hist_iid])
                    hist4dino.append(all_dino_feats[hist_iid])

                if seq_num is None:
                    seq_num = len(all_res[uid][tgt_iid][genre_id]["seqs"])
                    img_num = len(all_res[uid][tgt_iid][genre_id]["seqs"][0]["images"])

    cur_metrics = {}
    for metric in METRIC_LIST:
        cur_metrics[metric] = [[] for _ in range(seq_num)]

    cur_eval_save_path = os.path.join(res_folder, f"eval_res_{scale_for_dm}.npy")
    resume = False
    if resume and os.path.exists(cur_eval_save_path):
        cur_metrics = np.load(cur_eval_save_path, allow_pickle=True).item()
        for metric in METRIC_LIST:
            if metric not in cur_metrics:
                cur_metrics[metric] = [[] for _ in range(seq_num)]
            all_eval_metrics[ckpt][args.mask_ratio][args.scale_for_llama][args.scale_for_dm][metric] = cur_metrics[metric]
        
        logger.info(f"Found eval results for checkpoint-{ckpt} under llama-scale-{args.scale_for_llama} and dm-scale-{args.scale_for_dm}.")
        logger.info(f"UPDATE all evaluation results in {eval_save_path} with {cur_eval_save_path}.")
        np.save(eval_save_path, np.array(all_eval_metrics))
        logger.info(f"UPDATED all evaluation results: {all_eval_metrics}")
        return

    for i in range(seq_num):
        for j in range(img_num):
            gen4eval = []
            gen4clip = []
            gen4lpips = []
            gen4ssim = []
            gen4dino = []

            for uid in all_res:
                for tgt_iid in all_res[uid]:
                    for genre_id in all_res[uid][tgt_iid]:
                        img_path = all_res[uid][tgt_iid][genre_id]["seqs"][i]["images"][j]
                        img_path = os.path.join(eval_path, img_path)
                        im = Image.open(img_path).convert('RGB')
                        im = resize(im)
                        gen4eval.append(trans(im))

                        im_clip_tensor = clip_trans(im)
                        gen4clip.append(im_clip_tensor)

                        im_lpips_tensor = eval_utils.im2tensor_lpips(im)
                        gen4lpips.append(im_lpips_tensor)

                        im_ssim_arr = np.array(im, dtype=np.float32).transpose((2, 0, 1))
                        gen4ssim.append(im_ssim_arr)
                        
                        dino_tensor = dino_processor(im, return_tensors="pt")["pixel_values"].squeeze(0)
                        gen4dino.append(dino_tensor)

            gen_dataset = EvalDataset(gen4eval)
            grd_dataset = EvalDataset(grd4eval)
        
            # -------------------------------------------------------------- #
            #                       Calculating FID                          #
            # -------------------------------------------------------------- #
            logger.info("Calculating FID Value...")
            fid_value = eval_utils.calculate_fid_given_data(
                gen_dataset,
                grd_dataset,
                batch_size=args.batch_size,
                dims=args.dims4fid,
                device=device,
                num_workers=num_workers
            )
            cur_metrics["FID"][i].append(fid_value)
            np.save(cur_eval_save_path, np.array(cur_metrics))

            del grd4eval
            torch.cuda.empty_cache()

            ######################################## Semantic Alignment ########################################
            # -------------------------------------------------------------- #
            #                 Calculating CLIP Image Score                   #
            # -------------------------------------------------------------- #
            gen_dataset = EvalDataset(gen4clip)
            grd_dataset = EvalDataset(grd4clip)

            logger.info("Calculating CLIP Image Score...")
            clip_img_score, _ = eval_utils.calculate_clip_img_score_given_data(
                gen_dataset,
                grd_dataset,
                batch_size=args.batch_size,
                device=device,
                num_workers=num_workers,
                similarity_func=args.sim_func
            )
            cur_metrics["CLIP Image Score"][i].append(clip_img_score)
            np.save(cur_eval_save_path, np.array(cur_metrics))

            del grd4clip
            torch.cuda.empty_cache()

            # -------------------------------------------------------------- #
            #                    Calculating CLIP Score                      #
            # -------------------------------------------------------------- #
            gen_dataset = EvalDataset(gen4clip)
            txt_dataset = EvalDataset(text4eval)

            logger.info("Calculating CLIP Score...")
            clip_score, _ = eval_utils.calculate_clip_score_given_data(
                gen_dataset,
                txt_dataset,
                batch_size=args.batch_size,
                device=device,
                num_workers=num_workers,
            )
            cur_metrics["CLIP Score"][i].append(clip_score)
            np.save(cur_eval_save_path, np.array(cur_metrics))

            del text4eval
            torch.cuda.empty_cache()

            # -------------------------------------------------------------- #
            #                 Calculating DINO Image Score                   #
            # -------------------------------------------------------------- #
            gen_dataset = EvalDataset(gen4dino)
            grd_dataset = EvalDataset(grd4dino)

            logger.info("Calculating DINO Image Score...")
            dino_img_score, _ = eval_utils.calculate_dino_img_score_given_data(
                args.dino_model_path,
                gen_dataset,
                grd_dataset,
                batch_size=args.batch_size,
                device=device,
                num_workers=num_workers,
            )
            cur_metrics["DINO Image Score"][i].append(dino_img_score)
            np.save(cur_eval_save_path, np.array(cur_metrics))

            del grd4dino
            torch.cuda.empty_cache()

            ######################################## History Similarity ########################################
            # -------------------------------------------------------------- #
            #             Calculating History DINO Image Score               #
            # -------------------------------------------------------------- #
            gen4dino = [tensor for tensor in gen4dino for _ in range(5)]
            gen_dataset = EvalDataset(gen4dino)
            hist_dataset = EvalDataset(hist4dino)

            logger.info("Calculating History DINO Image Score...")
            hist_dino_img_score, _ = eval_utils.calculate_dino_img_score_given_data(
                args.dino_model_path,
                gen_dataset,
                hist_dataset,
                batch_size=args.batch_size,
                device=device,
                num_workers=num_workers,
            )
            cur_metrics["Hist DINO Image Score"][i].append(hist_dino_img_score)
            np.save(cur_eval_save_path, np.array(cur_metrics))

            del hist4dino
            del gen4dino
            torch.cuda.empty_cache()

            # -------------------------------------------------------------- #
            #                Calculating History CLIP Score                  #
            # -------------------------------------------------------------- #
            gen4clip = [tensor for tensor in gen4clip for _ in range(5)]
            gen_dataset = EvalDataset(gen4clip)
            txt_dataset = EvalDataset(hist_text4eval)

            logger.info("Calculating History CLIP Score...")
            hist_clip_score, _ = eval_utils.calculate_clip_score_given_data(
                gen_dataset,
                txt_dataset,
                batch_size=args.batch_size,
                device=device,
                num_workers=num_workers,
            )
            cur_metrics["Hist CLIP Score"][i].append(hist_clip_score)
            np.save(cur_eval_save_path, np.array(cur_metrics))

            del hist_text4eval
            torch.cuda.empty_cache()

            # -------------------------------------------------------------- #
            #             Calculating History CLIP Image Score               #
            # -------------------------------------------------------------- #
            hist_sim = {}
            hist_sim["gen"] = gen4clip
            hist_sim["hist"] = hist4clip
            gen_dataset_personal_sim = PersonalSimDataset(hist_sim)
            hist_clip_img_score, _ = eval_utils.evaluate_personalization_given_data_sim(
                gen4eval=gen_dataset_personal_sim,
                batch_size=args.batch_size,
                device=device,
                num_workers=num_workers,
                similarity_func=args.sim_func
            )
            cur_metrics["Hist CLIP Image Score"][i].append(hist_clip_img_score)
            np.save(cur_eval_save_path, np.array(cur_metrics))

            del hist4clip
            del gen4clip
            del gen4eval
            torch.cuda.empty_cache()

            # -------------------------------------------------------------- #
            #                  Calculating History LPIPS                     #
            # -------------------------------------------------------------- #
            hist_sim = {}
            hist_sim["gen"] = [tensor for tensor in gen4lpips for _ in range(5)]
            hist_sim["hist"] = hist4lpips
            gen_dataset_personal_sim = PersonalSimDataset(hist_sim)
            hist_lpips = eval_utils.evaluate_personalization_given_data_lpips(
                gen4eval=gen_dataset_personal_sim,
                batch_size=args.batch_size // 4,
                device=device,
                num_workers=num_workers,
            )
            cur_metrics["Hist LPIPS"][i].append(hist_lpips)
            np.save(cur_eval_save_path, np.array(cur_metrics))

            del gen4lpips
            del hist4lpips
            torch.cuda.empty_cache()

            # -------------------------------------------------------------- #
            #                   Calculating History SSIM                     #
            # -------------------------------------------------------------- #
            hist_sim = {}
            hist_sim["gen"] = [tensor for tensor in gen4ssim for _ in range(5)]
            hist_sim["hist"] = hist4ssim
            gen_dataset_personal_sim = PersonalSimDataset(hist_sim)
            hist_ssim, hist_ms_ssim = eval_utils.evaluate_personalization_given_data_ms_ssim(
                gen4eval=gen_dataset_personal_sim,
                batch_size=args.batch_size,
                device=device,
                num_workers=num_workers,
            )
            cur_metrics["Hist SSIM"][i].append(hist_ssim)
            cur_metrics["Hist MS-SSIM"][i].append(hist_ms_ssim)
            np.save(cur_eval_save_path, np.array(cur_metrics))

            del gen4ssim
            del hist4ssim
            torch.cuda.empty_cache()


            logger.info("-" * 10 + f"{args.eval_version}-checkpoint-{str(ckpt)}-{scale_for_llama}-{scale_for_dm}-seq{str(i)}-img{str(j)}" + "-" * 10)
            logger.info("")
            logger.info("## Fidelity ##")
            logger.info(" " * 2 + f"[FID Value]: {fid_value:.2f}")
            logger.info("")

            logger.info("## Target Consistency ##")
            logger.info(" " * 2 + f"[CLIP Score]: {clip_score:.2f}")
            logger.info(" " * 2 + f"[CLIP Image Score]: {clip_img_score:.2f}")
            logger.info(" " * 2 + f"[DINO Image Score]: {dino_img_score:.2f}")
            logger.info("")

            logger.info("## History Similarity ##")
            logger.info(" " * 2 + f"[Hist CLIP Score]: {hist_clip_score:.2f}")
            logger.info(" " * 2 + f"[Hist CLIP Image Score]: {hist_clip_img_score:.2f}")
            logger.info(" " * 2 + f"[Hist DINO Image Score]: {hist_dino_img_score:.2f}")
            logger.info(" " * 2 + f"[Hist LPIPS]: {hist_lpips:.4f}")
            logger.info(" " * 2 + f"[Hist SSIM]: {hist_ssim:.4f}")
            logger.info(" " * 2 + f"[Hist MS-SSIM]: {hist_ms_ssim:.4f}")
            logger.info("")
            logger.info("")

    for metric in METRIC_LIST:
        all_eval_metrics[ckpt][args.mask_ratio][args.scale_for_llama][args.scale_for_dm][metric] = cur_metrics[metric]
    np.save(eval_save_path, np.array(all_eval_metrics))
    logger.info(f"Successfully saved evaluation results of {args.eval_version} checkpoint-{ckpt} under llama-scale-{args.scale_for_llama} and dm-scale-{args.scale_for_dm} to {eval_save_path}.")


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    main()
