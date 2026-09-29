"""Merge multiple predict output dirs into one submission folder (CSV concat + mask copy).

Usage: .venv/Scripts/python.exe work/ad/merge_subs.py --out work/sub_final --parts work/testb_seen work/testb_musc1 work/testb_musc2
"""
import argparse
import csv
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "predicted_masks").mkdir(parents=True, exist_ok=True)

    rows = []
    for part in args.parts:
        part = Path(part)
        csvf = None
        for name in ("submission.csv", "submission_musc_partial.csv"):
            if (part / name).exists():
                csvf = part / name
                break
        if csvf is None:
            print(f"[warn] no csv in {part}")
            continue
        with open(csvf, newline="") as f:
            for row in csv.DictReader(f):
                rows.append((row["group_folder"], row["anomaly_score"]))
        pm = part / "predicted_masks"
        if not pm.exists():
            continue
        for cat in pm.iterdir():
            dst = out / "predicted_masks" / cat.name
            dst.mkdir(parents=True, exist_ok=True)
            for sdir in cat.iterdir():
                dstd = dst / sdir.name
                if dstd.exists():
                    continue
                shutil.copytree(sdir, dstd)
        print(f"[merge] {part.name}: ok")

    seen_rows = {}
    for g, s in rows:
        seen_rows.setdefault(g, s)
    rows = sorted(seen_rows.items())
    with open(out / "submission.csv", "w", newline="") as f:
        f.write("group_folder,anomaly_score\n")
        for g, s in rows:
            f.write(f"{g},{s}\n")
    print(f"[merge] total rows {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
