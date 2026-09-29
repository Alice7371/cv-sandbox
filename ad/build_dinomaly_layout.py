"""Build a Dinomaly-compatible data layout from the OFFICIAL competition Train data.

Creates (under --root):
  realiad_1024/<cat>/OK/<Sxxxx>/<v>.png   symlinks to official Train images
  realiad_jsons/realiad_jsons/<cat>.json  {"train": [...OK entries...], "test": [small subset]}
Uses official data only (Train split, all OK, 5 views as independent images).
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/Train")
    ap.add_argument("--root", default="dinomaly_data")
    ap.add_argument("--test-samples", type=int, default=2, help="held-out samples per cat for the test split")
    args = ap.parse_args()

    train_dir = Path(args.train)
    root = Path(args.root)
    (root / "realiad_jsons" / "realiad_jsons").mkdir(parents=True, exist_ok=True)

    cats = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    for cat in cats:
        img_root = root / "realiad_1024" / cat / "OK"
        train_entries, test_entries = [], []
        sdirs = sorted((train_dir / cat).iterdir())
        for si, sdir in enumerate(sdirs):
            phase_test = si < args.test_samples  # small held-out tail for the test split
            for v in range(5):
                rel = f"OK/{sdir.name}/{v}.png"
                dst = root / "realiad_1024" / cat / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.symlink_to((train_dir / cat / sdir.name / f"{v}.png").resolve())
                entry = {"category": cat, "anomaly_class": "OK", "image_path": rel, "mask_path": None}
                (test_entries if phase_test else train_entries).append(entry)
        jp = root / "realiad_jsons" / "realiad_jsons" / f"{cat}.json"
        jp.write_text(json.dumps({"train": train_entries, "test": test_entries}))
        print(f"{cat}: train={len(train_entries)} test={len(test_entries)}")


if __name__ == "__main__":
    main()
