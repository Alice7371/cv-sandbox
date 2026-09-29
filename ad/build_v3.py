"""Assemble B_final v3: v2 CSV/structure, masks re-rendered with per-source optimal smoothing.
Seen cats (DINOv3 per-cat banks): sigma 4. Unseen cats (MuSc): sigma 11.
"""
import csv
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

SIG_SEEN, SIG_UNSEEN = 4.0, 11.0
OUT = Path("work/sub_B_final_v3")

def main():
    seen = set(Path("work/seen_cats.txt").read_text().strip().split(","))
    parts = ["work/testb_seen", "work/testb_musc1", "work/testb_musc2", "work/testb_musc2a"]
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "predicted_masks").mkdir(parents=True)
    shutil.copy("work/sub_B_final_v2/submission.csv", OUT / "submission.csv")

    n = 0
    for part in parts:
        pm = Path(part) / "predicted_masks"
        if not pm.exists():
            continue
        for cat in pm.iterdir():
            sigma = SIG_SEEN if cat.name in seen else SIG_UNSEEN
            for sdir in cat.iterdir():
                for m in sdir.glob("*_mask.png"):
                    rel = m.relative_to(pm)
                    dst = OUT / "predicted_masks" / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    img = cv2.imread(str(m), 0)
                    cv2.imwrite(str(dst), cv2.GaussianBlur(img, (0, 0), sigma))
                    n += 1
        print(f"part {part}: done")
    print(f"masks blurred: {n} -> {OUT}")


if __name__ == "__main__":
    main()
