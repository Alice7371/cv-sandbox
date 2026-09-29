"""Shared utilities: DINOv2 features, foreground detection, multi-scale tiling, kNN scoring."""
import os
from pathlib import Path

import cv2
import numpy as np
import torch

IM_SIZE = 448
PATCH = 14
GRID = IM_SIZE // PATCH  # 32
N_PATCH = GRID * GRID
LAYERS = [9, 10, 11]  # last three blocks of ViT-B/14 (depth 12)
CANVAS = 1792         # fusion canvas (object resolved at ~1800px effective)
TILE_OVERLAP = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_state = {}
CUR_LAYERS = [9, 10, 11]  # set by load_model from the loaded backbone's depth
CUR_GRID = GRID           # patch grid size for the current model (14px->32, 16px->28)


def get_grid():
    return CUR_GRID


def load_model(model_name="vit_base_patch14_dinov2.lvd142m"):
    if model_name in _state:
        return _state[model_name]
    import timm
    try:
        model = timm.create_model(model_name, pretrained=True, img_size=IM_SIZE)
    except Exception as e:  # HF unreachable -> mirror
        print(f"[warn] pretrained download failed ({e}); retrying via hf-mirror")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        model = timm.create_model(model_name, pretrained=True, img_size=IM_SIZE)
    cfg = timm.data.resolve_model_data_config(model)
    mean = torch.tensor(cfg["mean"], dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
    model = model.to(DEVICE).eval()
    global CUR_LAYERS, CUR_GRID
    depth = len(model.blocks)
    CUR_LAYERS = [depth - 3, depth - 2, depth - 1]
    ps = model.patch_embed.patch_size
    CUR_GRID = IM_SIZE // (ps[0] if isinstance(ps, (tuple, list)) else ps)
    _state[model_name] = (model, mean, std)
    return _state[model_name]


def _intermediates(model, x, layers):
    """get_intermediate_layers with a forward_intermediates fallback (timm EVA-style
    models like DINOv3 lack the legacy API)."""
    if hasattr(model, "get_intermediate_layers"):
        return model.get_intermediate_layers(x, n=layers, norm=True)
    out = model.forward_intermediates(x, indices=layers, norm=True,
                                      output_fmt="NLC", intermediates_only=True)
    if isinstance(out, tuple):
        out = out[0]
    return list(out)


def _compute_dtype():
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        os.environ.get("AD_DTYPE", "fp16")
    ]


@torch.no_grad()
def feats_from_448(model, mean, std, batch_448):
    """batch_448: list/array of HxWx3 uint8 RGB images already resized to IM_SIZE.
    Returns (B, N_PATCH, C) fp16 cuda patch features (mean of last 3 layer outputs)."""
    x = torch.from_numpy(np.ascontiguousarray(np.stack(batch_448))).to(DEVICE)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    x = (x - mean) / std
    dt = _compute_dtype()
    with torch.autocast("cuda", dtype=dt, enabled=DEVICE == "cuda" and dt != torch.float32):
        outs = _intermediates(model, x, CUR_LAYERS)
    f = torch.stack(outs, 0).mean(0)
    return f.half()


def imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_pipeline(tasks, worker_fn, n_workers=6, maxsize=6):
    """Parallel CPU decode pipeline. tasks: iterable; worker_fn(task) -> result.
    Yields (i, result) as completed, unordered. Results may be ("err", exception)."""
    import queue as _q
    import threading as _th
    tasks = list(tasks)
    if not tasks:
        return
    in_q, out_q = _q.Queue(), _q.Queue(maxsize=maxsize)
    for i, t in enumerate(tasks):
        in_q.put((i, t))
    for _ in range(n_workers):
        in_q.put(None)

    def work():
        while True:
            it = in_q.get()
            if it is None:
                return
            i, t = it
            try:
                r = worker_fn(t)
            except Exception as e:  # keep the pipeline alive on single bad file
                r = ("err", e)
            out_q.put((i, r))

    threads = [_th.Thread(target=work, daemon=True) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for _ in range(len(tasks)):
        yield out_q.get()


def fg_mask_128(img):
    """Foreground mask (128x128 uint8) via border-color deviation."""
    small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    border = np.concatenate([
        small[:4].reshape(-1, 3), small[-4:].reshape(-1, 3),
        small[:, :4].reshape(-1, 3), small[:, -4:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    diff = np.abs(small - bg[None, None, :]).max(axis=2)
    m = (diff > 14).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m


def fg_bbox(img):
    """Returns (mask128, bbox) with bbox=(x0,y0,x1,y1) in original coords, or None bbox."""
    m = fg_mask_128(img)
    ys, xs = np.nonzero(m)
    if len(xs) == 0 or m.mean() < 0.004:
        return m, None
    H, W = img.shape[:2]
    mx, my = int(0.03 * 128), int(0.03 * 128)
    bx0, bx1 = max(0, xs.min() - mx), min(128, xs.max() + 1 + mx)
    by0, by1 = max(0, ys.min() - my), min(128, ys.max() + 1 + my)
    sx, sy = W / 128.0, H / 128.0
    return m, (int(bx0 * sx), int(by0 * sy), int(bx1 * sx), int(by1 * sy))


def tiles_from_bbox(bbox, W, H):
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if os.environ.get("AD_TILES", "4") == "9":
        xs = [x0 + int(w * f) for f in (0.0, 1 / 3, 2 / 3)]
        ys = [y0 + int(h * f) for f in (0.0, 1 / 3, 2 / 3)]
        span_x, span_y = int(w / 3 * 1.3), int(h / 3 * 1.3)
        rects = [(cx, cy, min(cx + span_x, x1), min(cy + span_y, y1))
                 for cy in ys for cx in xs]
    else:
        ox, oy = int(w * TILE_OVERLAP), int(h * TILE_OVERLAP)
        mx_, my_ = x0 + w // 2, y0 + h // 2
        rects = [
            (x0 - ox, y0 - oy, mx_ + ox, my_ + oy),
            (mx_ - ox, y0 - oy, x1 + ox, my_ + oy),
            (x0 - ox, my_ - oy, mx_ + ox, y1 + oy),
            (mx_ - ox, my_ - oy, x1 + ox, y1 + oy),
        ]
    out = []
    for r in rects:
        cx0, cy0, cx1, cy1 = max(0, r[0]), max(0, r[1]), min(W, r[2]), min(H, r[3])
        if cx1 - cx0 >= 64 and cy1 - cy0 >= 64:
            out.append((cx0, cy0, cx1, cy1))
    return out


def compute_passes(img):
    """Split one view into (rect, img448) passes: global + 2x2 zoom tiles over the object.
    Returns (passes, mask128). rect=(x0,y0,x1,y1) in original image coords."""
    H, W = img.shape[:2]
    passes = [((0, 0, W, H), cv2.resize(img, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_AREA))]
    m, bbox = fg_bbox(img)
    if bbox is not None:
        for r in tiles_from_bbox(bbox, W, H):
            cx0, cy0, cx1, cy1 = r
            crop = img[cy0:cy1, cx0:cx1]
            passes.append((r, cv2.resize(crop, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_AREA)))
    return passes, m


def patch_fg(mask128, rect, W, H, grid=None):
    """Patch-level foreground flags (grid x grid bool) for one pass region."""
    g = grid if grid is not None else CUR_GRID
    x0, y0, x1, y1 = rect
    if rect[2] - rect[0] == W and rect[3] - rect[1] == H:
        crop = mask128
    else:
        cx0, cy0 = int(x0 / W * 128), int(y0 / H * 128)
        cx1, cy1 = max(cx0 + 1, int(np.ceil(x1 / W * 128))), max(cy0 + 1, int(np.ceil(y1 / H * 128)))
        crop = mask128[cy0:cy1, cx0:cx1]
    small = cv2.resize(crop, (g, g), interpolation=cv2.INTER_AREA)
    return small > 0.3


@torch.no_grad()
def knn_dists(feat, bank, bank32=None, topk=3, chunk=None):
    """feat: (N,C) fp16 cuda; bank: (M,C) fp16 cuda. fp16 matmul (fp32 accumulate)
    keeps VRAM low; returns (N,) fp32: mean of topk smallest L2 distances."""
    N, M = feat.shape[0], bank.shape[0]
    if chunk is None:
        chunk = max(128, int(6e8 / (2 * M)))
    qn = (feat.float() ** 2).sum(1)
    mn = (bank.float() ** 2).sum(1)
    out = torch.empty(N, device=DEVICE, dtype=torch.float32)
    k = min(topk, M)
    for i in range(0, N, chunk):
        q = feat[i:i + chunk]
        sim = (q @ bank.T).float()
        d = qn[i:i + chunk, None] + mn[None, :] - 2.0 * sim
        d.clamp_(min=0)
        topv = torch.topk(d, k=k, dim=1, largest=False).values
        out[i:i + chunk] = topv.float().mean(1).sqrt()
        del sim, d, topv
    return out
