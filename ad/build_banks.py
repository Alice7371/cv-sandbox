"""Build per-category patch memory banks + a global (zero-shot) bank from normal Train data.

Usage: .venv/Scripts/python.exe work/ad/build_banks.py --train data/Train --out work/banks
Outputs per category c:
  banks/<c>.npy          fp16 (BANK_N, C) patch features
  banks/<c>_stats.json   {"med_g","med_t","n_g","n_t"}
  banks/_global/<c>.npy  fp16 global-pool samples for the unseen-category bank
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (DEVICE, compute_passes, decode_pipeline, feats_from_448,
                    get_grid, imread_rgb, knn_dists, load_model, patch_fg)

DIM = 768  # overwritten from the loaded model's embed dim
RES_CAP = 160_000       # per-category reservoir of patch features
BANK_N = 60_000         # final bank size per category
GLOBAL_PER_CAT = 6_000  # samples kept per category for the global bank
FWD_CHUNK = 10
SEED = 0
AUGMENT = False


class Reservoir:
    def __init__(self, cap, dim):
        self.cap, self.n, self.total = cap, 0, 0
        self.data = np.empty((cap, dim), np.float16)
        self.scale = np.empty(cap, np.uint8)  # 0=global pass, 1=tile pass
        self.fg = np.empty(cap, np.uint8)

    def add(self, feats, scale, fg):
        n = feats.shape[0]
        t0 = self.total
        fill = max(0, min(self.cap - t0, n))
        if fill:
            sl = slice(self.n, self.n + fill)
            self.data[sl] = feats[:fill]
            self.scale[sl] = scale
            self.fg[sl] = fg[:fill]
            self.n += fill
        if fill < n:
            ts = t0 + fill + np.arange(n - fill)
            js = np.random.randint(0, ts + 1)
            keep = js < self.cap
            self.data[js[keep]] = feats[fill:][keep]
            self.scale[js[keep]] = scale
            self.fg[js[keep]] = fg[fill:][keep]
        self.total += n


def iter_view_paths(train_dir, cat):
    for sdir in sorted((train_dir / cat).iterdir()):
        for imgf in sorted(sdir.glob("*.png")):
            yield imgf, sdir.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/Train")
    ap.add_argument("--out", default="work/banks")
    ap.add_argument("--cats", default="", help="comma-separated subset of categories")
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--augment", action="store_true",
                    help="also bank hflip + rot90 + rot270 variants (x4 coverage)")
    args = ap.parse_args()
    global AUGMENT
    AUGMENT = args.augment
    train_dir, out = Path(args.train), Path(args.out)
    (out / "_global").mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    cats = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    if args.cats:
        cats = [c for c in cats if c in set(args.cats.split(","))]
    model, mean, std = load_model(args.model)
    global DIM
    DIM = model.num_features
    print(f"{len(cats)} categories, model={args.model} dim={DIM} on {DEVICE}", flush=True)

    for cat in cats:
        stats_f = out / f"{cat}_stats.json"
        bank_f = out / f"{cat}.npy"
        if bank_f.exists() and stats_f.exists():
            print(f"[skip] {cat}")
            continue
        res = Reservoir(RES_CAP, DIM)
        gres = Reservoir(GLOBAL_PER_CAT, DIM)
        views = list(iter_view_paths(train_dir, cat))

        batch_imgs, batch_meta = [], []

        def flush():
            if not batch_imgs:
                return
            feats = feats_from_448(model, mean, std, batch_imgs)
            f_np = feats.cpu().numpy()
            g = get_grid()
            npx = g * g
            idx = np.random.randint(0, npx, size=max(1, npx // 512))
            for i, (scale, fgflags) in enumerate(batch_meta):
                fg8 = fgflags.astype(np.uint8)
                res.add(f_np[i], scale, fg8)
                gres.add(f_np[i][idx], scale, fg8[idx])
            if AUGMENT:
                aug_imgs, aug_flags = [], []
                for im, (scale, fgflags) in zip(batch_imgs, batch_meta):
                    fggrid = fgflags.reshape(g, g)
                    for v, fv in ((im[:, ::-1], fggrid[:, ::-1]),
                                  (np.rot90(im, 1), np.rot90(fggrid, 1)),
                                  (np.rot90(im, 3), np.rot90(fggrid, 3))):
                        aug_imgs.append(np.ascontiguousarray(v))
                        aug_flags.append((scale, np.ascontiguousarray(fv.reshape(-1)).astype(np.uint8)))
                fa_np = feats_from_448(model, mean, std, aug_imgs).cpu().numpy()
                for j, (scale, fgf) in enumerate(aug_flags):
                    res.add(fa_np[j], scale, fgf)
                    gres.add(fa_np[j][idx], scale, fgf[idx])
            batch_imgs.clear()
            batch_meta.clear()

        def decode_view(task):
            path, sname = task
            return ("ok", (sname, imread_rgb(path)))

        done = 0
        for i, out_ in decode_pipeline([(p, sn) for p, sn in views], decode_view,
                                       n_workers=int(os.environ.get("AD_WORKERS", "8")), maxsize=8):
            status, payload = out_
            if status == "err":
                print(f"[warn] {payload}")
                continue
            sname, img = payload
            passes, mask128 = compute_passes(img)
            H, W = img.shape[:2]
            del img
            for rect, p448 in passes:
                scale = 0 if (rect[2] - rect[0] == W and rect[3] - rect[1] == H) else 1
                batch_imgs.append(p448)
                batch_meta.append((scale, patch_fg(mask128, rect, W, H).reshape(-1)))
                if len(batch_imgs) >= FWD_CHUNK:
                    flush()
            done += 1
            if done % 500 == 0:
                print(f"  {cat}: {done}/{len(views)} views", flush=True)
        flush()

        # disjoint split of the reservoir: bank half vs stats half
        n = res.n
        perm = np.random.permutation(n)
        bank = res.data[perm[:BANK_N]]
        np.save(bank_f, bank)
        np.save(out / "_global" / f"{cat}.npy", gres.data[:gres.n])

        bank_g = torch.from_numpy(bank).to(DEVICE)
        stat = torch.from_numpy(res.data[perm[BANK_N:]]).to(DEVICE)
        with torch.no_grad():
            d = knn_dists(stat, bank_g, bank_g.float())
        d = d.cpu().numpy()
        s_g = res.scale[perm[BANK_N:]] == 0
        f_ok = res.fg[perm[BANK_N:]] == 1
        stats = {}
        for name, sm in (("g", s_g), ("t", ~s_g)):
            sel = sm & f_ok
            vals = d[sel] if sel.sum() >= 1000 else d[sm]
            pos = vals[vals > 0]
            stats[f"med_{name}"] = float(np.median(pos)) if len(pos) >= 100 else float(np.median(vals))
            stats[f"n_{name}"] = int(sel.sum())
        stats_f.write_text(json.dumps(stats))
        del bank_g, stat, d
        torch.cuda.empty_cache()
        print(f"[done] {cat}: bank={bank.shape} stats={stats}", flush=True)


if __name__ == "__main__":
    main()
