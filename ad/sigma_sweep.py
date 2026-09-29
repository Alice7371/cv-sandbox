"""Sigma post-processing sweep: Gaussian-blur predicted masks, re-evaluate pixel metrics.

Usage: .venv/Scripts/python.exe work/ad/sigma_sweep.py --src work/eval_musc_ms --base work/eval_d3_musc.json
"""
import argparse
import cv2
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-base", default="work/sigma_eval")
    ap.add_argument("--sigmas", default="0.8,1.5,2.5")
    args = ap.parse_args()
    src = Path(args.src)

    import numpy as np
    sigmas = [float(s) for s in args.sigmas.split(",")]
    masks = sorted(src.glob("predicted_masks/*/*/*_mask.png"))
    print(f"total masks: {len(masks)}")
    for sigma in sigmas:
        out = Path(f"{args.out_base}_s{sigma}")
        if out.exists():
            print(f"[skip] sigma {sigma}")
            continue
        print(f"[sigma {sigma}] blurring...")
        n = 0
        for m in masks:
            rel = m.relative_to(src)
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(m), 0)
            cv2.imwrite(str(dst), cv2.GaussianBlur(img, (0, 0), sigma))
            n += 1
            if n % 1000 == 0:
                print(f"  {n}")
        shutil.copy(src / "submission.csv", out / "submission.csv")
        r = subprocess.run([sys.executable, "work/ad/eval_valset.py",
                            "--pred", str(out), "--banks", "work/banks_D3",
                            "--out", f"{out}.json"], capture_output=True, text=True)
        tail = "\n".join(r.stdout.strip().split("\n")[-3:])
        print(tail)
    print("SWEEP_DONE")


if __name__ == "__main__":
    main()
