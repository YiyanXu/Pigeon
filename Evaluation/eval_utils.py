import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.models import inception_v3
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import fid_inception_v3
from pytorch_msssim import ssim, ms_ssim
import lpips
import open_clip
from transformers import AutoModel

class ImagePathDataset(Dataset):
    def __init__(self, folder_path, paths, trans=None, do_normalize=True):
        self.folder_path = folder_path
        self.paths = paths
        self.trans = trans
        self.do_normalize = do_normalize
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = os.path.join(self.folder_path, self.paths[idx])
        img = Image.open(path).convert('RGB')

        if self.trans is not None:
            img = self.trans(img)
        if self.do_normalize:
            img = 2 * img - 1
        return img

class InceptionV3(nn.Module):
    def __init__(self):  # , model_path, num_classes):
        super(InceptionV3, self).__init__()
        self.model = inception_v3(weights='DEFAULT')
        self.num_classes = self.model.fc.out_features
        # num_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_features, num_classes)
        # num_features_aux = self.model.AuxLogits.fc.in_features
        # self.model.AuxLogits.fc = nn.Linear(num_features_aux, num_classes)

        # self.model.load_state_dict(torch.load(model_path))
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
    
    def forward(self, x):
        x = self.model(x)
        x = nn.Softmax(dim=1)(x)
        return x
    
    @torch.no_grad()
    def feature_extract(self, x):  # N x 3 x 299 x 299
        x = self.model._transform_input(x)
        features = self.extractor(x)  # N x 2048
        return features
    
    def extractor(self, x):
        # N x 3 x 299 x 299
        x = self.model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.model.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x

class CLIPScore:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def calculate_clip_score(self, images, texts):
        img_features = self.model.encode_image(images)
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        txt_features = self.model.encode_text(texts)
        txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

        clip_score = 100 * F.cosine_similarity(img_features, txt_features)

        del img_features
        del txt_features
        torch.cuda.empty_cache()

        return clip_score
    
    @torch.no_grad()
    def calculate_clip_img_score(self, images1, images2, similarity_func="cosine"):
        img_features1 = self.model.encode_image(images1)
        img_features1 = img_features1 / img_features1.norm(p=2, dim=-1, keepdim=True)

        img_features2 = self.model.encode_image(images2)
        img_features2 = img_features2 / img_features2.norm(p=2, dim=-1, keepdim=True)

        if similarity_func == "cosine":
            img_score = 100 * F.cosine_similarity(img_features1, img_features2)
        elif similarity_func == "euclidean":
            img_score = torch.norm(img_features1 - img_features2, p=2)  # L2-norm, Euclidean Distance
        else:
            raise ValueError(f"Unrecognized similarity function {similarity_func}.")

        del img_features1
        del img_features2
        torch.cuda.empty_cache()

        return img_score

class FIDInceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(FIDInceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        inception = fid_inception_v3()

        self.blocks = nn.ModuleList()

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

def get_activations(model, data4eval, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    model.eval()

    dataloader = DataLoader(data4eval,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=num_workers)
    
    pred_arr = np.empty((len(data4eval), dims))

    start_idx = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr

def calculate_activation_statistics(model, data4eval, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    act = get_activations(model, data4eval, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_fid_given_data(gen4eval, grd4eval,
                             batch_size, dims, device, num_workers=1):
    block_idx = FIDInceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = FIDInceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(model, gen4eval, batch_size,
                                            dims, device, num_workers)
    m2, s2 = calculate_activation_statistics(model, grd4eval, batch_size,
                                            dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    # model.to('cpu')
    del model

    return fid_value

def calculate_clip_score_given_data(img4eval, txt4eval, batch_size, device, num_workers=1):
    clip = CLIPScore(device=device)

    img_dataloader = DataLoader(img4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
        
    txt_dataloader = DataLoader(txt4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    clip_scores = []
    for img_batch, txt_batch in zip(img_dataloader, txt_dataloader):
        tokenized_txt_batch = clip.tokenizer(txt_batch).to(device)
        img_batch = img_batch.to(device)
        score = clip.calculate_clip_score(img_batch, tokenized_txt_batch)
        clip_scores.append(score)
    
    clip_scores = torch.cat(clip_scores, dim=0)

    # clip.model.to('cpu')
    del clip.model

    return torch.mean(clip_scores).item(), clip_scores

def calculate_clip_img_score_given_data(gen4eval, grd4eval, batch_size, device,
                                        num_workers=1, similarity_func="cosine"):
    clip = CLIPScore(device=device)

    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    grd_dataloader = DataLoader(grd4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    clip_img_scores = []
    for gen_batch, grd_batch in zip(gen_dataloader, grd_dataloader):
        gen_batch = gen_batch.to(device)
        grd_batch = grd_batch.to(device)
        score = clip.calculate_clip_img_score(gen_batch, grd_batch, similarity_func=similarity_func)
        clip_img_scores.append(score)
    
    clip_img_scores = torch.cat(clip_img_scores, dim=0)

    # clip.model.to('cpu')
    del clip.model

    return torch.mean(clip_img_scores).item(), clip_img_scores

@torch.no_grad()
def calculate_dino_img_score_given_data(model_path, gen4eval, grd4eval, batch_size, device,
                                        num_workers=1):
    model = AutoModel.from_pretrained(model_path)
    model = model.to(device)
    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    grd_dataloader = DataLoader(grd4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    dino_img_scores = []
    for gen_batch, grd_feats in zip(gen_dataloader, grd_dataloader):
        gen_batch = gen_batch.to(device)
        grd_feats = grd_feats.to(device)
        grd_feats = grd_feats / grd_feats.norm(p=2, dim=-1, keepdim=True)

        gen_feats = model(pixel_values=gen_batch).pooler_output
        gen_feats = gen_feats / gen_feats.norm(p=2, dim=-1, keepdim=True)

        score = 100 * F.cosine_similarity(gen_feats, grd_feats)
        dino_img_scores.append(score)
    
    dino_img_scores = torch.cat(dino_img_scores, dim=0)

    del model

    return torch.mean(dino_img_scores).item(), dino_img_scores

@torch.no_grad()
def calculate_extracted_dino_img_score_given_data(model_path, gen4eval, grd4eval, batch_size, device,
                                        num_workers=1):
    model = AutoModel.from_pretrained(model_path)
    model = model.to(device)
    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    grd_dataloader = DataLoader(grd4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    dino_img_scores = []
    for gen_feats, grd_feats in zip(gen_dataloader, grd_dataloader):
        gen_feats = gen_feats.to(device)
        grd_feats = grd_feats.to(device)

        score = 100 * F.cosine_similarity(gen_feats, grd_feats)
        dino_img_scores.append(score)
    
    dino_img_scores = torch.cat(dino_img_scores, dim=0)

    del model

    return torch.mean(dino_img_scores).item(), dino_img_scores

def im2tensor_lpips(im, cent=1., factor=255./2.):
    im_np = np.array(im)
    im_tensor = torch.Tensor((im_np / factor - cent).transpose((2, 0, 1)))
    return im_tensor

@torch.no_grad()
def calculate_lpips_given_data(gen4eval, grd4eval, batch_size, device, num_workers=1, use_net="vgg"):
    model = lpips.LPIPS(net=use_net).to(device)

    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    grd_dataloader = DataLoader(grd4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    lpip_scores = []
    for gen_batch, grd_batch in zip(gen_dataloader, grd_dataloader):
        gen_batch = gen_batch.to(device)
        grd_batch = grd_batch.to(device)
        with torch.no_grad():
            score = model(gen_batch, grd_batch)
        lpip_scores.append(score)
    
    lpip_scores = torch.cat(lpip_scores, dim=0)

    # model.to('cpu')
    del model

    return torch.mean(lpip_scores).item()

def evaluate_personalization_given_data_sim(gen4eval, batch_size,
                                        device, num_workers=1, similarity_func="cosine"):
    # evaluate the similarity between history and generated images
    clip = CLIPScore(device=device)
    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    scores = []
    with torch.no_grad():
        for gen, hist in gen_dataloader:
            gen = gen.to(device)
            hist = hist.to(device)

            gen_emb = clip.model.encode_image(gen)
            gen_emb = gen_emb / gen_emb.norm(p=2, dim=-1, keepdim=True)

            hist = hist / hist.norm(p=2, dim=-1, keepdim=True)

            if similarity_func == "cosine":
                sim_score = 100 * F.cosine_similarity(gen_emb, hist)
            elif similarity_func == "eiclidean":
                sim_score = torch.norm(gen_emb - hist, p=2)
            else:
                raise ValueError(f"Unrecognized similarity function {similarity_func}.")
            
            scores.append(sim_score)
    
    scores = torch.cat(scores, dim=0)

    # clip.model.to('cpu')
    del clip.model

    return torch.mean(scores).item(), scores

@torch.no_grad()
def extract_clip_feats(img_folder_path, img_paths, batch_size, num_workers, device):
    model, _, img_trans = open_clip.create_model_and_transforms('ViT-H-14', pretrained="laion2b-s32b-b79K")
    model = model.to(device)
    model.eval()

    img_dataset = ImagePathDataset(img_folder_path, img_paths, trans=img_trans, do_normalize=False)
    img_dataloader = DataLoader(
        dataset=img_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    clip_feats = []
    for batch in tqdm(img_dataloader):
        batch = batch.to(device)
        feats = model.encode_image(batch)
        clip_feats.extend(feats.cpu().numpy())
    
    clip_feats = np.array(clip_feats)
    return clip_feats

def process_hist_embs(all_res, clip_feats):
    hist_embs = {}
    for uid in all_res:
        if uid not in hist_embs:
            hist_embs[uid] = {}

        for tgt_iid in all_res[uid]:
            if tgt_iid not in hist_embs[uid]:
                hist_embs[uid][tgt_iid] = {}

            for genre_id in all_res[uid][tgt_iid]:
                hist_iids = all_res[uid][tgt_iid][genre_id]["history"]
                hist_img_embs = clip_feats[hist_iids]
                hist_embs[uid][tgt_iid][genre_id] = hist_img_embs.clone()
    
    return hist_embs

def evaluate_personalization_given_data_lpips(gen4eval, batch_size,
                                        device, num_workers=1, use_net="vgg"):
    # evaluate the similarity between history and generated images
    model = lpips.LPIPS(net=use_net).to(device)
    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    scores = []
    with torch.no_grad():
        for gen, hist in gen_dataloader:
            gen = gen.to(device)
            hist = hist.to(device)

            score = model(gen, hist)
            
            scores.append(score)
    
    scores = torch.cat(scores, dim=0)

    del model

    return torch.mean(scores).item()

def calculate_ms_ssim_given_data(gen4eval, grd4eval, batch_size, device, num_workers=1):
    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    
    grd_dataloader = DataLoader(grd4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)

    ssim_scores = []
    ms_ssim_scores = []
    for gen_batch, grd_batch in zip(gen_dataloader, grd_dataloader):
        gen_batch = gen_batch.to(device)
        grd_batch = grd_batch.to(device)

        ssim_score = ssim(gen_batch, grd_batch, data_range=255, size_average=False)
        ms_ssim_score = ms_ssim(gen_batch, grd_batch, data_range=255, size_average=False)

        ssim_scores.append(ssim_score)
        ms_ssim_scores.append(ms_ssim_score)
    
    ssim_scores = torch.cat(ssim_scores, dim=0)
    ms_ssim_scores = torch.cat(ms_ssim_scores, dim=0)

    return torch.mean(ssim_scores).item(), torch.mean(ms_ssim_scores).item()

def evaluate_personalization_given_data_ms_ssim(gen4eval, batch_size, device, num_workers=1):
    gen_dataloader = DataLoader(gen4eval,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=num_workers)
    ssim_scores = []
    ms_ssim_scores = []
    for gen, hist in gen_dataloader:
        gen = gen.to(device)
        hist = hist.to(device)

        ssim_score = ssim(gen, hist, data_range=255, size_average=False)
        ms_ssim_score = ms_ssim(gen, hist, data_range=255, size_average=False)

        ssim_scores.append(ssim_score)
        ms_ssim_scores.append(ms_ssim_score)
    
    ssim_scores = torch.cat(ssim_scores, dim=0)
    ms_ssim_scores = torch.cat(ms_ssim_scores, dim=0)

    return torch.mean(ssim_scores).item(), torch.mean(ms_ssim_scores).item()
