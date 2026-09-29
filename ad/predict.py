"""Run inference on a test set and produce a competition submission folder.

Seen categories (bank exists): PatchCore-style max patch distance vs per-category bank.
Unseen categories (no bank): global bank + per-pass self-normalized ratio score (zero-shot).

Usage: .venv/Scripts/python.exe work/ad/predict.py --test data/Test_A --out work/sub_A --banks work/banks
Outputs: <out>/submission.csv and <out>/predicted_masks/<cat>/<Sxxxx>/<v>_mask.png
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (CANVAS, DEVICE, GRID, compute_passes, decode_pipeline,
                    feats_from_448, get_grid, imread_rgb, knn_dists, load_model, patch_fg)

FWD_CHUNK = 10


def load_bank(path_or_arr):
    arr = path_or_arr if isinstance(path_or_arr, np.ndarray) else np.load(path_or_arr)
    b = torch.from_numpy(np.ascontiguousarray(arr)).to(DEVICE)
    return b, b.float()


def fuse_dmap(canvas, dmap32, rect, W, H):
    """Max-fuse a GRIDxGRID normalized score map into the canvas at rect coords."""
    if rect[2] - rect[0] == W and rect[3] - rect[1] == H:
        full = cv2.resize(dmap32, (CANVAS, CANVAS), interpolation=cv2.INTER_LINEAR)
        np.maximum(canvas, full, out=canvas)
        return
    sx, sy = CANVAS / W, CANVAS / H
    x0, y0 = int(rect[0] * sx), int(rect[1] * sy)
    x1, y1 = max(x0 + 1, int(round(rect[2] * sx))), max(y0 + 1, int(round(rect[3] * sy)))
    tile = cv2.resize(dmap32, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    np.maximum(canvas[y0:y1, x0:x1], tile, out=canvas[y0:y1, x0:x1])


def process_sample(model, mean, std, views, bank, bank32, med, unseen):
    """views: list of (passes, mask128, W, H). Returns (scores_per_view, masks_per_view)."""
    scores, masks = [], []
    g = get_grid()
    for passes, mask128, W, H in views:
        canvas = np.zeros((CANVAS, CANVAS), np.float32)
        pools = []  # patch score values for the image-level score
        for pi in range(0, len(passes), FWD_CHUNK):
            chunk = passes[pi:pi + FWD_CHUNK]
            feats = feats_from_448(model, mean, std, [p448 for _, p448 in chunk])
            for i, (rect, _) in enumerate(chunk):
                d = knn_dists(feats[i], bank, bank32).cpu().numpy()
                fg = patch_fg(mask128, rect, W, H).reshape(-1)
                scale_is_global = rect[2] - rect[0] == W and rect[3] - rect[1] == H
                if not unseen:
                    m = med["med_g"] if scale_is_global else med["med_t"]
                    norm = d / max(m, 1e-6)
                    pools.append(norm)
                    unit = (norm / 3.0).reshape(g, g).astype(np.float32)
                else:
                    fgvals = d[fg] if fg.sum() >= 8 else d
                    medv = float(np.median(fgvals)) if len(fgvals) else 0.0
                    if medv < 1e-3:  # uninformative pass (no object / flat)
                        continue
                    ratio = d / medv
                    pools.append(ratio[fg] if fg.sum() >= 8 else ratio)
                    unit = ((ratio - 1.0) / 3.0).reshape(g, g).astype(np.float32)
                fuse_dmap(canvas, unit, rect, W, H)
        allp = np.concatenate(pools) if pools else np.zeros(1, np.float32)
        if unseen:
            k = max(1, int(0.01 * len(allp)))
            s_view = float(np.mean(np.sort(allp)[-k:]))
        else:
            s_view = float(np.max(allp))
        scores.append(s_view)
        masks.append(canvas)
    return scores, masks


def decode_sample(task):
    cat, sdir = task
    views = []
    for imgf in sorted(sdir.glob("*.png")):
        img = imread_rgb(imgf)
        passes, mask128 = compute_passes(img)
        H, W = img.shape[:2]
        del img
        views.append((passes, mask128, W, H))
    return "ok", (cat, sdir.name, views)


def musc_cat(model, mean, std, cat, samples, out, FWD_CHUNK=FWD_CHUNK):
    """MuSc-style leave-one-out mutual scoring for one unseen category.
    samples: list of (sname, views). Writes masks; returns [(sname, score)]."""
    from concurrent.futures import ThreadPoolExecutor

    feats_all, fg_list, sid_list, pass_list = [], [], [], []
    pass_meta = []  # per pass: (rect, W, H, view_idx)
    n_views = []
    for si, (sname, views) in enumerate(samples):
        n_views.append(len(views))
        for vi, (passes, mask128, W, H) in enumerate(views):
            for pi in range(0, len(passes), FWD_CHUNK):
                chunk = passes[pi:pi + FWD_CHUNK]
                feats = feats_from_448(model, mean, std, [p448 for _, p448 in chunk])
                for i, (rect, _) in enumerate(chunk):
                    feats_all.append(feats[i].cpu())
                    fg_list.append(patch_fg(mask128, rect, W, H).reshape(-1))
                    sid_list.append(np.full(feats[i].shape[0], si))
                    pass_list.append(np.full(feats[i].shape[0], len(pass_meta)))
                    pass_meta.append((rect, W, H, vi))
    F = torch.cat(feats_all).to(DEVICE)
    del feats_all
    if os.environ.get("MUSC_MEAN_SHIFT", "0") == "1":
        mu = F.float().mean(0, keepdim=True)
        F = (F.float() - mu).half()
        print(f"  [musc] mean-shift applied", flush=True)
    fgn = np.concatenate(fg_list)
    sidn = np.concatenate(sid_list)
    passn = np.concatenate(pass_list)
    g = get_grid()
    print(f"  [musc] {cat}: {F.shape[0]} patches / {len(samples)} samples", flush=True)

    results = []
    for si, (sname, views) in enumerate(samples):
        qm_np = sidn == si
        q = F[torch.from_numpy(qm_np).to(DEVICE)]
        b = F[torch.from_numpy(~qm_np).to(DEVICE)]
        d = knn_dists(q, b, None).cpu().numpy()
        del b
        torch.cuda.empty_cache()
        qm = fgn[qm_np]
        pn = passn[qm_np]
        ratio = np.zeros_like(d)
        for p in np.unique(pn):
            m = pn == p
            fgv = d[m & qm] if (m & qm).sum() >= 8 else d[m]
            medv = float(np.median(fgv)) if len(fgv) else 0.0
            if medv > 1e-3:
                ratio[m] = d[m] / medv
        vals = ratio[qm] if qm.sum() >= 8 else ratio
        k = max(1, int(0.01 * len(vals)))
        score = float(np.mean(np.sort(vals)[-k:]))

        vidx = np.full(len(d), -1)
        for p, (rect, W, H, vi) in enumerate(pass_meta):
            vidx[pn == p] = vi
        for vi in range(n_views[si]):
            canvas = np.zeros((CANVAS, CANVAS), np.float32)
            sel = vidx == vi
            if not sel.any():
                canvas_masks_out = canvas
            for pi, (rect, W, H, vi2) in enumerate(pass_meta):
                if vi2 != vi:
                    continue
                m = (pn == pi)
                if not m.any():
                    continue
                fgv = d[m & qm] if (m & qm).sum() >= 8 else d[m]
                medv = float(np.median(fgv)) if len(fgv) else 0.0
                if medv <= 1e-3:
                    continue
                r = d[m] / medv
                unit = ((r - 1.0) / 3.0).reshape(g, g).astype(np.float32)
                fuse_dmap(canvas, unit, rect, W, H)
            mdir = out / "predicted_masks" / cat / sname
            mdir.mkdir(parents=True, exist_ok=True)
            m448 = cv2.resize(canvas, (448, 448), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(mdir / f"{vi}_mask.png"),
                        (np.clip(m448, 0.0, 1.0) * 255.0).astype(np.uint8))
            del canvas
        results.append((sname, score))
        torch.cuda.empty_cache()
    return results



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/Test_A")
    ap.add_argument("--out", default="work/sub_A")
    ap.add_argument("--banks", default="work/banks")
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--cats", default="", help="comma-separated subset of categories")
    ap.add_argument("--musc", action="store_true",
                    help="unseen categories: leave-one-out mutual scoring instead of global bank")
    args = ap.parse_args()
    test_dir, out, banks = Path(args.test), Path(args.out), Path(args.banks)
    (out / "predicted_masks").mkdir(parents=True, exist_ok=True)

    model, mean, std = load_model(args.model)
    cats = sorted(d.name for d in test_dir.iterdir() if d.is_dir())
    if args.cats:
        cats = [c for c in cats if c in set(args.cats.split(","))]
    seen = {c: json.loads((banks / f"{c}_stats.json").read_text())
            for c in cats if (banks / f"{c}.npy").exists()}
    unseen_cats = [c for c in cats if c not in seen]
    print(f"{len(cats)} cats: {len(seen)} seen, {len(unseen_cats)} unseen", flush=True)

    gbank = gbank32 = None
    if unseen_cats:
        gsrc = banks / "_global_all.npy"
        if gsrc.exists():
            gbank, gbank32 = load_bank(np.load(gsrc))
        else:
            gbank, gbank32 = load_bank(np.concatenate(
                [np.load(p) for p in sorted((banks / "_global").glob("*.npy"))]))
        print(f"global bank: {gbank.shape[0]} patches", flush=True)

    t0 = time.time()
    rows, dbg = [], []
    n_done = 0
    total = sum(len(list((test_dir / c).iterdir())) for c in cats)

    for cat in cats:
        unseen = cat not in seen
        if unseen and args.musc:
            tasks = [(cat, sdir) for sdir in sorted((test_dir / cat).iterdir()) if sdir.is_dir()]
            samples = []
            for i, out_ in decode_pipeline(tasks, decode_sample,
                                           n_workers=int(os.environ.get("AD_WORKERS", "8")), maxsize=2):
                status, payload = out_
                if status == "err":
                    print(f"[warn] {payload}")
                    continue
                _, sname, views = payload
                samples.append((sname, views))
            done_musc = 0
            for sname, sc in musc_cat(model, mean, std, cat, samples, out):
                rows.append((f"{cat}/{sname}", sc))
                dbg.append((cat, sname, "musc", f"{sc:.4f}"))
                n_done += 1
                done_musc += 1
                with open(out / "submission_musc_partial.csv", "w", newline="") as f:
                    f.write("group_folder,anomaly_score\n")
                    for g2, s2 in rows:
                        f.write(f"{g2},{s2:.6f}\n")
            del samples
            torch.cuda.empty_cache()
            continue
        if unseen:
            bank, bank32 = gbank, gbank32
            med = None
        else:
            bank, bank32 = load_bank(banks / f"{cat}.npy")
            med = seen[cat]
        tasks = [(cat, sdir) for sdir in sorted((test_dir / cat).iterdir()) if sdir.is_dir()]
        for i, out_ in decode_pipeline(tasks, decode_sample, n_workers=int(os.environ.get("AD_WORKERS", "8")), maxsize=4):
            status, payload = out_
            if status == "err":
                print(f"[warn] {payload}")
                continue
            cat_, sname, views = payload
            scores, canvas_masks = process_sample(model, mean, std, views, bank, bank32, med, unseen)
            mdir = out / "predicted_masks" / cat_ / sname
            mdir.mkdir(parents=True, exist_ok=True)
            for v, cm in enumerate(canvas_masks):
                m448 = cv2.resize(cm, (448, 448), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(mdir / f"{v}_mask.png"),
                            (np.clip(m448, 0.0, 1.0) * 255.0).astype(np.uint8))
            rows.append((f"{cat_}/{sname}", max(scores)))
            dbg.append((cat_, sname, "unseen" if unseen else "seen",
                        ";".join(f"{s:.4f}" for s in scores)))
            del views, canvas_masks
            n_done += 1
            if n_done % 100 == 0:
                el = time.time() - t0
                print(f"  {n_done}/{total} samples, {el / n_done:.2f} s/sample, "
                      f"eta {(total - n_done) * el / n_done / 60:.0f} min", flush=True)
        if not unseen:
            del bank, bank32
        torch.cuda.empty_cache()

    rows.sort(key=lambda r: r[0])
    with open(out / "submission.csv", "w", newline="") as f:
        f.write("group_folder,anomaly_score\n")
        for g, s in rows:
            f.write(f"{g},{s:.6f}\n")
    with open(out / "debug_scores.csv", "w", newline="") as f:
        f.write("category,sample,mode,view_scores\n")
        for r in dbg:
            f.write(",".join(r[:3]) + ',"' + r[3] + '"\n')
    print(f"DONE {n_done} samples in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
