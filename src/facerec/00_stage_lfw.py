import argparse, shutil, random
from pathlib import Path
import cv2

# project paths
ROOT = Path(__file__).resolve().parents[2]
RAW_FACE = ROOT / "data" / "raw" / "face"

def log(m): print(f"[stage] {m}")

def main():
    p = argparse.ArgumentParser(description="Stage a subset of local LFW (deepfunneled) into data/raw/face/")
    p.add_argument("--src", required=True, help="Path to lfw-deepfunneled folder (the one with person subfolders)")
    p.add_argument("--identities", type=int, default=50, help="How many people to include (0=all)")
    p.add_argument("--per-person", type=int, default=10, help="Max images per person (0=all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    assert src_root.exists(), f"Source not found: {src_root}"

    people = sorted([d for d in src_root.iterdir() if d.is_dir()])
    if args.identities > 0:
        random.seed(args.seed); random.shuffle(people)
        people = people[:args.identities]

    RAW_FACE.mkdir(parents=True, exist_ok=True)
    copied = 0
    for person in people:
        imgs = sorted(person.glob("*.jpg"))
        if args.per_person > 0 and len(imgs) > args.per_person:
            random.seed(args.seed); imgs = random.sample(imgs, args.per_person)
        out_dir = RAW_FACE / person.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in imgs:
            img = cv2.imread(str(src))
            if img is None:
                log(f"skip unreadable: {src}")
                continue
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst); copied += 1
    log(f"Staged {copied} images into {RAW_FACE}")

if __name__ == "__main__":
    main()
