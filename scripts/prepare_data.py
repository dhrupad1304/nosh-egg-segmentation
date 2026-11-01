import argparse, json, random, shutil
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def group_key(path):
    # Roboflow augmented files commonly use: original.rf.<hash>.jpg
    stem = path.stem
    if ".rf." in stem:
        stem = stem.split(".rf.", 1)[0]
    return stem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Root directory of unzipped dataset")
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-ratio", type=float, default=0.8)
    args = ap.parse_args()

    src, out = Path(args.source), Path(args.out)
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"No images found under {src}")

    groups = {}
    for p in images:
        groups.setdefault(group_key(p), []).append(p)

    keys = list(groups)
    random.Random(args.seed).shuffle(keys)
    n_dev = max(1, round(len(keys) * args.dev_ratio))
    dev_groups, eval_groups = set(keys[:n_dev]), set(keys[n_dev:])

    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    split = {"dev": [], "eval": [], "grouping": "stem before .rf.<hash>", "seed": args.seed}

    for p in images:
        # Stable unique output name
        name = p.name
        dest = img_out / name
        if dest.exists():
            dest = img_out / f"{p.parent.name}_{name}"
        shutil.copy2(p, dest)
        key = group_key(p)
        split["dev" if key in dev_groups else "eval"].append(dest.name)

    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(split, indent=2))
    print(f"Images: {len(images)} | groups: {len(groups)}")
    print(f"Dev: {len(split['dev'])} | Eval: {len(split['eval'])}")
    print(f"Split saved to {out / 'split.json'}")

if __name__ == "__main__":
    main()
