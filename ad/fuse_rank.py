"""Rank-fusion experiment: ViT-B + DINOv3 seen scores, per-category rank normalization.

Usage: .venv/Scripts/python.exe work/ad/fuse_rank.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(p):
    return {r["group_folder"]: float(r["anomaly_score"]) for r in csv.DictReader(open(p))}


def rank_norm(scores_by_cat):
    """Per-category: map scores to [0,1] by rank (ties averaged)."""
    out = {}
    for cat, items in scores_by_cat.items():
        keys = sorted(items)
        vals = np.array([items[k] for k in keys])
        order = vals.argsort().argsort().astype(np.float64)
        denom = max(1, len(vals) - 1)
        out.update({k: order[i] / denom for i, k in enumerate(keys)})
    return out


def main():
    vb = load("work/eval_vitb/submission.csv")
    d3 = load("work/eval_d3_val/submission.csv")
    seen = set(p.stem for p in Path("work/banks").glob("*.npy") if not p.stem.startswith("_"))

    by_cat_vb, by_cat_d3 = defaultdict(dict), defaultdict(dict)
    for k in set(vb) & set(d3):
        cat = k.split("/")[0]
        by_cat_vb[cat][k] = vb[k]
        by_cat_d3[cat][k] = d3[k]
    rb, rd = rank_norm(by_cat_vb), rank_norm(by_cat_d3)

    out_dir = Path("work/eval_fusion")
    (out_dir / "predicted_masks").mkdir(parents=True, exist_ok=True)
    import shutil
    for src in ("work/eval_d3_val/predicted_masks",):
        if Path(src).exists():
            for cat in Path(src).iterdir():
                dst = out_dir / "predicted_masks" / cat.name
                if not dst.exists():
                    shutil.copytree(cat, dst)

    rows = []
    for k in sorted(rb):
        cat = k.split("/")[0]
        if cat in seen:
            s = 0.5 * rb[k] + 0.5 * rd[k]
        else:
            s = rd[k]  # unseen: DINOv3-MuSc only (ViT-B unseen is random)
        rows.append((k, s))
    with open(out_dir / "submission.csv", "w", newline="") as f:
        f.write("group_folder,anomaly_score\n")
        for k, s in rows:
            f.write(f"{k},{s:.6f}\n")
    print(f"fusion rows: {len(rows)} (seen rank-avg, unseen dino-only)")

    import subprocess
    r = subprocess.run([sys.executable, "work/ad/eval_valset.py", "--pred", str(out_dir),
                        "--banks", "work/banks", "--out", "work/eval_fusion.json"],
                       capture_output=True, text=True)
    print(r.stdout[-700:])


if __name__ == "__main__":
    main()
