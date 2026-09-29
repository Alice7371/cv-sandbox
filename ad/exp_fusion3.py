"""Fusion gate experiment on validation-set seen classes.

1) 3-way image-score rank fusion (ViT-B + DINOv3 + Dinomaly) vs the incumbent
   2-way fusion — with competition-style MACRO metrics and a paired bootstrap.
2) exploratory pixel-map fusion (DINOv3 mask + Dinomaly mask), weight sweep.

Gate design: the baseline is the live 2-way incumbent, NOT the best single model;
PASS = macro3 >= macro2 + 0.6pt AND paired P(delta>0) >= 0.95. Image metrics are
per-category macro (competition formula). All inputs are asserted loudly: missing
CSV/mask dirs abort with a message that says what to produce. Weight-sweep labels:
70/30 means 0.7*DINOv3 + 0.3*Dinomaly.

Usage:
  python ad/exp_fusion3.py --dinomaly_dir work/dino50k_val
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from eval_valset import f1_max

GATE_MARGIN = 0.006   # +0.6 pt over the incumbent, with paired bootstrap significance
GATE_P = 0.95
BOOT_REPS = 2000


def load(p):
    return {r["group_folder"]: float(r["anomaly_score"]) for r in csv.DictReader(open(p))}


def rank_norm_by_cat(sd):
    """Per-category ordinal rank normalisation to [0,1] — production formula."""
    by_cat = defaultdict(dict)
    for k, v in sd.items():
        by_cat[k.split("/")[0]][k] = v
    n = {}
    for cat, items in by_cat.items():
        keys = sorted(items)
        vals = np.array([items[k] for k in keys])
        order = vals.argsort().argsort().astype(np.float64)
        denom = max(1, len(vals) - 1)
        n.update({k: order[i] / denom for i, k in enumerate(keys)})
    return n


def macro_metrics(keys, gt, scores):
    per = defaultdict(list)
    for k in keys:
        per[k.split("/")[0]].append(k)
    aucs, aps = [], []
    for cat, ks in per.items():
        yy = [gt[k] for k in ks]
        if len(set(yy)) < 2:
            continue
        ss = [scores[k] for k in ks]
        aucs.append(roc_auc_score(yy, ss))
        aps.append(average_precision_score(yy, ss))
    return float(np.mean(aucs)), float(np.mean(aps))


def paired_bootstrap(per_cat, reps=BOOT_REPS, seed=0):
    rng = np.random.default_rng(seed)
    deltas = np.empty(reps)
    for b in range(reps):
        inc, cand = [], []
        for cat, (y, si, sc) in per_cat.items():
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if len(set(yy.tolist())) < 2:
                continue
            inc.append(roc_auc_score(yy, si[idx]))
            cand.append(roc_auc_score(yy, sc[idx]))
        deltas[b] = np.mean(cand) - np.mean(inc)
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dinomaly_dir", default="work/dino50k_val",
                    help="output dir of dinomaly_infer.py on the validation set "
                         "(needs submission.csv + predicted_masks/)")
    ap.add_argument("--val_root", default="data/val")
    ap.add_argument("--banks_dir", default="work/banks_D3")
    ap.add_argument("--pix_cats", type=int, default=12)
    args = ap.parse_args()

    inputs = {
        "work/eval_vitb/submission.csv": Path("work/eval_vitb/submission.csv"),
        "work/eval_d3_val/submission.csv": Path("work/eval_d3_val/submission.csv"),
        f"{args.dinomaly_dir}/submission.csv": Path(args.dinomaly_dir) / "submission.csv",
    }
    missing = [str(p) for p in inputs.values() if not p.exists()]
    assert not missing, (
        "missing fusion inputs:\n  " + "\n  ".join(missing) +
        "\nproduce the Dinomaly arm first: python dinomaly/dinomaly_infer.py "
        "--ckpt <model_50000.pth> --test data/val --out " + args.dinomaly_dir)

    vb = load("work/eval_vitb/submission.csv")
    d3 = load("work/eval_d3_val/submission.csv")
    dn = load(str(Path(args.dinomaly_dir) / "submission.csv"))
    seen = set(p.stem for p in Path(args.banks_dir).glob("*.npy")
               if not p.stem.startswith("_"))

    gt = {}
    for cat in sorted(seen):
        cdir = Path(args.val_root, cat)
        if cdir.exists():
            for sdir in cdir.iterdir():
                lf = sdir / "label.txt"
                if lf.exists():
                    gt[f"{cat}/{sdir.name}"] = int(lf.read_text())

    keys = [k for k in gt if k in vb and k in d3 and k in dn]
    assert len(keys) > 800, f"only {len(keys)} common seen keys — Dinomaly arm incomplete?"
    n_vb = rank_norm_by_cat({k: vb[k] for k in keys})
    n_d3 = rank_norm_by_cat({k: d3[k] for k in keys})
    n_dn = rank_norm_by_cat({k: dn[k] for k in keys})
    f2 = {k: 0.5 * (n_vb[k] + n_d3[k]) for k in keys}
    f3 = {k: (n_vb[k] + n_d3[k] + n_dn[k]) / 3.0 for k in keys}

    print(f"{len(keys)} seen keys, {sum(gt[k] for k in keys)} positives")
    for tag, sc in (("ViT-B solo", n_vb), ("DINOv3 solo", n_d3), ("Dinomaly solo", n_dn),
                    ("2-way incumbent", f2), ("3-way candidate", f3)):
        auc, ap = macro_metrics(keys, gt, sc)
        print(f"  {tag:16s} macro I-AUROC={auc:.4f}  macro I-AP={ap:.4f}")

    a2, _ = macro_metrics(keys, gt, f2)
    a3, ap3 = macro_metrics(keys, gt, f3)
    per_cat = {}
    for cat in sorted(set(k.split("/")[0] for k in keys)):
        ks = [k for k in keys if k.split("/")[0] == cat]
        yy = [gt[k] for k in ks]
        if len(set(yy)) < 2:
            continue
        per_cat[cat] = (np.array(yy), np.array([f2[k] for k in ks]),
                        np.array([f3[k] for k in ks]))
    deltas = paired_bootstrap(per_cat)
    p_gt0 = float((deltas > 0).mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"\npaired 3-way vs 2-way: delta={100*(a3-a2):+.2f}pt  "
          f"95% CI [{100*lo:+.2f}, {100*hi:+.2f}]pt  P(delta>0)={p_gt0:.3f}")
    gate = (a3 >= a2 + GATE_MARGIN) and (p_gt0 >= GATE_P)
    print(f"GATE (baseline = live 2-way {a2:.4f}): "
          f"{'PASS' if gate else 'FAIL'}  "
          f"(needs macro3 >= {a2 + GATE_MARGIN:.4f} AND P >= {GATE_P})")
    if gate:
        ap2 = macro_metrics(keys, gt, f2)[1]
        approx_b = 0.3 * ((100 * (a3 - a2)) + (100 * (ap3 - ap2))) / 2
        print("  -> assemble v4 (seen image scores = 3-way rank fusion, unseen = MuSc,")
        print("     masks unchanged: DINOv3 + per-source smoothing), then A board x3 first;")
        print("     B shot only if A confirms >= +0.5 pt.")
        print(f"  rough B impact if it transfers: ~{approx_b:+.2f} pt (masks/S_seg unchanged)")

    # ---- exploratory pixel fusion (12-cat subset, CPU) ----
    pix_cats = sorted(seen)[:args.pix_cats]
    pm_d3 = Path("work/eval_d3_val/predicted_masks")
    if not pm_d3.exists():
        pm_d3 = Path("work/eval_d3/predicted_masks")
    assert pm_d3.exists(), "no DINOv3 predicted_masks found (eval_d3_val nor eval_d3)"
    pm_dn = Path(args.dinomaly_dir) / "predicted_masks"
    assert pm_dn.exists(), (
        f"{pm_dn} missing — dinomaly_infer.py must write predicted_masks/ "
        "(fixed 09-10: per-view masks) before the pixel arm can run")

    py, p_base, p_73, p_55, p_37 = [], [], [], [], []
    skipped = 0
    for cat in pix_cats:
        for sdir in sorted(Path(args.val_root, cat).iterdir()):
            if not (sdir / "label.txt").exists() or (sdir / "label.txt").read_text() != "1":
                continue
            for v in range(5):
                gt_m = cv2.resize(cv2.imread(str(sdir / "gt" / f"{v}.png"), 0), (448, 448),
                                  interpolation=cv2.INTER_NEAREST) > 0
                m1 = cv2.imread(str(pm_d3 / cat / sdir.name / f"{v}_mask.png"), 0)
                m2 = cv2.imread(str(pm_dn / cat / sdir.name / f"{v}_mask.png"), 0)
                if m1 is None or m2 is None:
                    skipped += 1
                    continue
                m1 = m1.astype(np.float32) / 255.0
                m2 = m2.astype(np.float32) / 255.0

                def rn(m):
                    order = m.reshape(-1).argsort().argsort().astype(np.float64)
                    return (order / (m.size - 1)).reshape(m.shape)

                r1, r2 = rn(m1), rn(m2)
                py.append(gt_m.reshape(-1))
                p_base.append(r1.reshape(-1))
                p_73.append((0.7 * r1 + 0.3 * r2).reshape(-1))   # 70/30 = DINOv3/Dinomaly
                p_55.append((0.5 * r1 + 0.5 * r2).reshape(-1))
                p_37.append((0.3 * r1 + 0.7 * r2).reshape(-1))
    assert py, "pixel arm collected ZERO maps — inputs missing (v1 failed silently here)"
    if skipped:
        print(f"[warn] pixel arm skipped {skipped} maps (missing files) — numbers below "
              f"cover only the found subset")
    py = np.concatenate(py)
    for name, ps in (("DINOv3 only", p_base), ("70/30 DINOv3/Dinomaly", p_73),
                     ("50/50", p_55), ("30/70", p_37)):
        ps = np.concatenate(ps)
        print(f"pixel {name:22s}: P-AUROC={roc_auc_score(py, ps):.4f} "
              f"P-AUPR={average_precision_score(py, ps):.4f} P-F1max={f1_max(py, ps):.4f}")
    print("[note] shipped v4 masks stay DINOv3+sigma4 (unchanged by the 3-way swap-in); "
          "pixel rows above are exploratory only")


if __name__ == "__main__":
    main()
