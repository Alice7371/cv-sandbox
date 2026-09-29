"""Dinomaly unified training adapted for the competition Train layout (official data only).

Changes vs dinomaly_realiad_uni.py:
- categories read from the Train dir (50 competition cats)
- checkpoints saved every --save_every iters (model_<it>.pth)
- no test-set evaluation during training (offline eval is handled separately)
- iters configurable via --iters
"""
import argparse
import copy
import itertools
import logging
import os
import random
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_data_transforms
from dataset import RealIADDataset
from models.uad import ViTill
from models import vit_encoder
from torch.nn.init import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from torch.utils.data import DataLoader, ConcatDataset
from optimizers import StableAdamW
from utils import global_cosine_hm_percent, WarmCosineScheduler

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train(args, item_list, print_fn):
    setup_seed(1)

    total_iters = args.iters
    batch_size = int(os.environ.get("AD_BS", "16"))
    image_size = 448
    crop_size = 392

    device = 'cuda'
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    for item in item_list:
        train_data = RealIADDataset(root=args.data_path, category=item, transform=data_transform,
                                    gt_transform=gt_transform, phase='train')
        train_data.classes = item
        train_data.class_to_idx = {item: 0}
        train_data_list.append(train_data)

    train_data = ConcatDataset(train_data_list)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                   num_workers=8, drop_last=True, persistent_workers=True)
    print_fn('train image number:{}'.format(len(train_data)))

    encoder_name = 'dinov2reg_vit_base_14'
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads = 768, 12

    bottleneck = [bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4)]
    bottleneck = nn.ModuleList(bottleneck)

    decoder = []
    for i in range(8):
        blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                       qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                       attn=LinearAttention2)
        decoder.append(blk)
    decoder = nn.ModuleList(decoder)

    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=0, fuse_layer_encoder=fuse_layer_encoder,
                   fuse_layer_decoder=fuse_layer_decoder)
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4,
                                       total_iters=total_iters, warmup_iters=100)

    save_dir = os.path.join(args.save_dir, args.save_name)
    os.makedirs(save_dir, exist_ok=True)

    print_fn('train image number:{}'.format(len(train_data)))

    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)) + 1)):
        model.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device, non_blocking=True)
            label = label.to(device)

            en, de = model(img)
            p_final = 0.9
            p = min(p_final * it / 1000, p_final)
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % args.save_every == 0 or (it + 1) == total_iters:
                torch.save(model.state_dict(), os.path.join(save_dir, f'model_{it + 1}.pth'))
                print_fn(f'checkpoint saved: model_{it + 1}.pth', )

            it += 1
            if it == total_iters:
                break
            if (it + 1) % 100 == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []
        if it == total_iters:
            break

    return


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='dinomaly_data')
    parser.add_argument('--train_dir', type=str, default='data/Train')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='dinomaly_uni_ours')
    parser.add_argument('--iters', type=int, default=50000)
    parser.add_argument('--save_every', type=int, default=10000)
    args = parser.parse_args()

    item_list = sorted(d.name for d in Path(args.train_dir).iterdir() if d.is_dir())
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    print_fn(f'categories: {len(item_list)}')

    train(args, item_list, print_fn)


if __name__ == '__main__':
    main()
