from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=257):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        return self.pe[pos]

class MutualMaskGenerator(nn.Module):
    def __init__(self, emb_dim, num_heads, num_layers, drop_prob=0.2, eps=1e-5):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.position_embedding = PositionalEncoding(emb_dim, max_len=257)
        self.type_embedding = nn.Embedding(3, emb_dim)  # three types: history, target, pad
        self.dense = nn.Linear(emb_dim, emb_dim)
        self.activation = nn.Tanh()

        self.LayerNorm = nn.LayerNorm(emb_dim, eps=eps)
        self.dropout = nn.Dropout(drop_prob)
    
    def get_input_embeds(
        self,
        token_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        type_ids: Optional[torch.LongTensor] = None,
    ):
        position_embeds = self.position_embedding(position_ids)
        type_embeds = self.type_embedding(type_ids)
        embeds = token_embeds + position_embeds + type_embeds
        embeds = self.LayerNorm(embeds)
        embeds = self.dropout(embeds)

        return embeds
    
    def get_output_embeds(
        self,
        input_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        output_embeds = self.encoder(input_embeds, src_key_padding_mask=attention_mask)
        output_embeds = self.LayerNorm(output_embeds)
        output_embeds = self.dense(output_embeds)
        output_embeds = self.activation(output_embeds)
        output_embeds = self.dropout(output_embeds)

        return output_embeds

    def forward(self, token_embeds, attention_mask, position_ids, type_ids, mask_ratio, hist_mask_ratio, inference):
        input_embeds = self.get_input_embeds(token_embeds, position_ids, type_ids)
        output_embeds = self.get_output_embeds(input_embeds, attention_mask)

        # Compute the cosine similarity between token_embeds and output_embeds
        cos_sim = F.cosine_similarity(token_embeds, output_embeds, dim=-1)

        # Keep history tokens with higher cosine similarity as target-relevant user preference
        hist_mask = (type_ids == 1)
        hist_indices = Grouped_indices(hist_mask)
        hist_keep_prob = Masked_softmax(cos_sim, hist_mask, dim=-1)
        hist_hard_keep_decision = Gumbel_softmax_topk(hist_keep_prob, hist_indices, hist_mask_ratio)

        
        target_mask = (type_ids == 2)
        target_indices = Grouped_indices(target_mask)
        if inference:  # Mask target tokens with lower cosine similarity for inference
            target_keep_prob = Masked_softmax(cos_sim, target_mask, dim=-1)
        else:  # Mask target tokens with higher cosine similarity for training
            target_keep_prob = 1 - Masked_softmax(cos_sim, target_mask, dim=-1)
        target_hard_keep_decision = Gumbel_softmax_topk(target_keep_prob, target_indices, mask_ratio)

        return hist_hard_keep_decision + target_hard_keep_decision
        
def Gumbel_softmax_topk(keep_prob, valid_pos, mask_ratio, tau=1.0):
    bsz, _ = keep_prob.shape
    mask = torch.zeros_like(keep_prob)

    for i in range(bsz):
        prob = keep_prob[i]
        pos = valid_pos[i]
        n = len(pos)
        if n == 0:
            continue
        keep_num = max(1, int(n * (1 - mask_ratio)))

        logits = prob[pos].log()
        gumbels = -torch.empty_like(logits).exponential_().log() # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau
        y_soft = F.softmax(gumbels, dim=-1)

        # Return topk hard keep decision
        topk_indices = y_soft.topk(keep_num, dim=-1).indices
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[topk_indices] = 1.0
        ret = y_hard - y_soft.detach() + y_soft

        mask[i, pos] = ret
    
    return mask

def Masked_softmax(score, mask, dim=-1):
    score = torch.exp(score)
    mask = mask.to(dtype=score.dtype)
    masked_score = score * mask

    return masked_score / masked_score.sum(dim, keepdim=True)

def Grouped_indices(mask):
    indices = mask.nonzero(as_tuple=False)
    grouped_indices = []
    for i in range(mask.size(0)):
        row_indices = indices[indices[:, 0] == i][:, 1]
        grouped_indices.append(row_indices)
    
    return grouped_indices

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, drop_prob=0.1):
        super().__init__()
        
        self.encoder = nn.Linear(in_dim, hid_dim)
        self.drop = nn.Dropout(drop_prob)
        self.decoder = nn.Linear(hid_dim, out_dim)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.drop(x)
        x = self.decoder(x)
        x = self.activation(x)
        x = self.drop(x)

        return x
