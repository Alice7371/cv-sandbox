"""Evaluate a prediction folder against a real-GT validation set (data/val).

Metrics per category: I-AUROC, P-AUROC, P-AUPR, P-F1max; macro-averaged separately
over seen / unseen categories (seen = category has a memory bank).

Usage: python ad/eval_valset.py --test data/val --pred work/sub_eval_d3 --banks work/banks_D3
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def f1_max(y, s):
    thr = np.unique(s)
    if len(thr) > 256:
        thr = np.percentile(s, np.linspace(0, 100, 256))
    best = 0.0
    for t in thr[::-1]:
        pred = s >= t
        tp = np.logical_and(pred, y).sum()
        fp = np.logical_and(pred, ~y).sum()
        fn = np.logical_and(~pred, y).sum()
        if tp:
            best = max(best, 2 * tp / (2 * tp + fp + fn))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/val")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--banks", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    test_dir, pred = Path(args.test), Path(args.pred)

    scores = {}
    with open(pred / "submission.csv") as f:
        for row in csv.DictReader(f):
            scores[row["group_folder"]] = float(row["anomaly_score"])

    banks = Path(args.banks)
    seen = {p.name[:-4] for p in banks.glob("*.npy")
            if not p.name.startswith("_") and p.name.endswith(".npy")}

    per_cat = {}
    for cat in sorted(d.name for d in test_dir.iterdir() if d.is_dir()):
        y_img, s_img, pix_y, pix_s = [], [], [], []
        for sdir in sorted((test_dir / cat).iterdir()):
            if not (sdir / "label.txt").exists():
                continue
            y_img.append(int((sdir / "label.txt").read_text()))
            key = f"{cat}/{sdir.name}"
            if key not in scores:
                per_cat[cat] = None
                break
            s_img.append(scores[key])
            for gtf in sorted((sdir / "gt").glob("*.png")):
                v = gtf.name.split(".")[0]
                pm = cv2.imread(str(pred / "predicted_masks" / cat / sdir.name / f"{v}_mask.png"), 0)
                if pm is None:
                    pm = np.zeros((448, 448), np.uint8)
                gt = cv2.resize(cv2.imread(str(gtf), 0), (448, 448),
                                interpolation=cv2.INTER_NEAREST) > 0
                pix_y.append(gt.reshape(-1))
                pix_s.append(pm.reshape(-1).astype(np.float32) / 255.0)
        if sum(y_img) in (0, len(y_img)):
            per_cat[cat] = None
            continue
        py, ps = np.concatenate(pix_y), np.concatenate(pix_s)
        per_cat[cat] = dict(
            seen=cat in seen,
            iauroc=float(roc_auc_score(y_img, s_img)),
            iap=float(average_precision_score(y_img, s_img)),
            pauroc=float(roc_auc_score(py, ps)),
            paupr=float(average_precision_score(py, ps)),
            pf1=float(f1_max(py, ps)),
            n=len(y_img), pos=int(sum(y_img)),
        )

    rows = []
    for split in ("seen", "unseen"):
        vals = [v for v in per_cat.values() if v and v["seen"] == (split == "seen")]
        if not vals:
            continue
        m = {k: float(np.mean([v[k] for v in vals]))
             for k in ("iauroc", "iap", "pauroc", "paupr", "pf1")}
        m["n_cats"] = len(vals)
        rows.append((split, m))
    print(f"{'split':<8}{'cats':>5}{'I-AUROC':>9}{'I-AP':>9}{'P-AUROC':>9}{'P-AUPR':>8}{'P-F1max':>9}")
    for split, m in rows:
        print(f"{split:<8}{m['n_cats']:>5}{m['iauroc']:>9.4f}{m['iap']:>9.4f}{m['pauroc']:>9.4f}{m['paupr']:>8.4f}{m['pf1']:>9.4f}")
    if rows and args.out:
        Path(args.out).write_text(json.dumps({"per_cat": per_cat, "macro": {s: m for s, m in rows}},
                                             indent=1))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
