"""Dinomaly inference: validation set or Test_B -> submission folder.

Single joint model (no memory banks). Per sample: 5 views -> per-view anomaly map +
top-1% image score; sample score = max over views; mask = EACH view's own map padded
into its own 448 canvas (center 392 crop region).

Usage: python dinomaly/dinomaly_infer.py \
  --ckpt saved_results/dinomaly_uni_ours/model_50000.pth --test data/val --out work/dino_infer_val
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

from models.uad import ViTill
from models import vit_encoder
from torch.nn.init import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(device):
    encoder = vit_encoder.load('dinov2reg_vit_base_14')
    embed_dim, num_heads = 768, 12
    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4)])
    decoder = []
    for i in range(8):
        decoder.append(VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                                attn_drop=0., attn=LinearAttention2))
    decoder = nn.ModuleList(decoder)
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=[2, 3, 4, 5, 6, 7, 8, 9], mask_neighbor_size=0,
                   fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                   fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]])
    model = model.to(device).eval()
    return model


def cal_anomaly_map(fs_list, ft_list, out_size):
    a_maps = []
    for fs, ft in zip(fs_list, ft_list):
        a = 1 - F.cosine_similarity(fs, ft)
        a = torch.unsqueeze(a, dim=1)
        a = F.interpolate(a, size=(out_size, out_size), mode='bilinear', align_corners=True)
        a_maps.append(a)
    amap = torch.cat(a_maps, dim=1).mean(dim=1, keepdim=True)
    return amap


def topk_mean(x, frac=0.01):
    k = max(1, int(x.size * frac))
    return float(np.sort(x.reshape(-1))[-k:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--test', default='data/val')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    device = 'cuda'
    model = build_model(device)
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    print(f'loaded {args.ckpt}', flush=True)

    tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.CenterCrop(392),
        transforms.Normalize(mean=MEAN, std=STD)])

    test_dir = Path(args.test)
    out = Path(args.out)
    (out / 'predicted_masks').mkdir(parents=True, exist_ok=True)

    samples = []
    for cat in sorted(d.name for d in test_dir.iterdir() if d.is_dir()):
        for sdir in sorted((test_dir / cat).iterdir()):
            if (sdir / 'label.txt').exists() or list(sdir.glob('*.png')):
                samples.append((cat, sdir))
    print(f'{len(samples)} samples', flush=True)

    rows = []
    with torch.no_grad():
        for si, (cat, sdir) in enumerate(samples):
            view_maps = []
            view_scores = []
            imgs = [tf(Image.open(p).convert('RGB')) for p in sorted(sdir.glob('*.png'))]
            for bi in range(0, len(imgs), args.batch):
                x = torch.stack(imgs[bi:bi + args.batch]).to(device)
                en, de = model(x)
                amap = cal_anomaly_map(en, de, out_size=392)[:, 0].cpu().numpy()
                for m in amap:
                    view_maps.append(m)
                    view_scores.append(topk_mean(m))
            score = max(view_scores)
            # one canvas per view (center 392 region matches the training crop);
            # scale by the sample's map max so uint8 quantisation does not clip
            # the anomaly tail of high-distance maps all to 255
            vmax = max(float(m.max()) for m in view_maps) or 1.0
            off = (448 - 392) // 2
            mdir = out / 'predicted_masks' / cat / sdir.name
            mdir.mkdir(parents=True, exist_ok=True)
            for v, m in enumerate(view_maps):
                canvas = np.zeros((448, 448), np.float32)
                canvas[off:off + 392, off:off + 392] = m.astype(np.float32) / vmax
                cv2.imwrite(str(mdir / f'{v}_mask.png'),
                            (np.clip(canvas, 0, 1) * 255).astype(np.uint8))
            rows.append((f'{cat}/{sdir.name}', score))
            if (si + 1) % 200 == 0:
                print(f'{si + 1}/{len(samples)}', flush=True)

    rows.sort()
    with open(out / 'submission.csv', 'w') as f:
        f.write('group_folder,anomaly_score\n')
        for g, s in rows:
            f.write(f'{g},{s:.6f}\n')
    print(f'DONE {len(rows)} rows -> {out}', flush=True)


if __name__ == '__main__':
    main()
