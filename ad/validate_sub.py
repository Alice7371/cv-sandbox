"""Validate a submission folder against the expected format and the test set.

Usage: .venv/Scripts/python.exe work/ad/validate_sub.py --test data/Test_A --sub work/sub_A
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/Test_A")
    ap.add_argument("--sub", default="work/sub_A")
    args = ap.parse_args()
    test_dir, sub = Path(args.test), Path(args.sub)

    truth = {}
    for cat in sorted(p.name for p in test_dir.iterdir() if p.is_dir()):
        for sdir in sorted((test_dir / cat).iterdir()):
            n_views = len(list(sdir.glob("*.png")))
            truth[f"{cat}/{sdir.name}"] = n_views

    errs = []
    with open(sub / "submission.csv", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != ["group_folder", "anomaly_score"]:
            errs.append(f"bad header: {r.fieldnames}")
        rows = list(r)
    got = {row["group_folder"]: row["anomaly_score"] for row in rows}
    if len(got) != len(rows):
        errs.append("duplicate group_folder rows")
    missing = set(truth) - set(got)
    extra = set(got) - set(truth)
    if missing:
        errs.append(f"missing {len(missing)} rows, e.g. {sorted(missing)[:3]}")
    if extra:
        errs.append(f"extra {len(extra)} rows, e.g. {sorted(extra)[:3]}")
    bad_vals = [v for v in got.values() if not v.replace(".", "", 1).replace("-", "", 1).isdigit()]
    if bad_vals:
        errs.append(f"non-numeric scores: {bad_vals[:3]}")

    n_masks, bad_shape, bad_mode, missing_masks = 0, 0, 0, []
    pm = sub / "predicted_masks"
    for gf, n_views in truth.items():
        sdir = pm / gf
        for v in range(n_views):
            f = sdir / f"{v}_mask.png"
            if not f.exists():
                missing_masks.append(str(f))
                continue
            im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if im is None:
                bad_mode += 1
                continue
            if im.ndim != 2:
                bad_mode += 1
            if im.shape != (448, 448):
                bad_shape += 1
            n_masks += 1

    print(f"rows={len(rows)} expected={len(truth)} masks={n_masks}/{sum(truth.values())}")
    if missing_masks:
        errs.append(f"missing masks: {len(missing_masks)} e.g. {missing_masks[:3]}")
    if bad_shape:
        errs.append(f"wrong-size masks: {bad_shape}")
    if bad_mode:
        errs.append(f"non-grayscale masks: {bad_mode}")
    vals = np.array([float(v) for v in got.values()])
    print(f"score range [{vals.min():.4f}, {vals.max():.4f}] mean={vals.mean():.4f}")
    if errs:
        print("FAIL:")
        for e in errs:
            print(" -", e)
        raise SystemExit(1)
    print("OK: submission format valid")


if __name__ == "__main__":
    main()
