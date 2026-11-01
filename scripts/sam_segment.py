import argparse, json
from pathlib import Path
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--output", default="outputs/sam")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--point-mode", default="center", choices=["center", "grid"])
    args = ap.parse_args()

    device = args.device
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image_dir, out = Path(args.images), Path(args.output)
    masks_dir, overlays_dir = out / "masks", out / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    names = split.get("dev", []) + split.get("eval", [])

    for name in names:
        path = image_dir / name
        if not path.exists():
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            print("Skipping unreadable:", path)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        predictor.set_image(rgb)

        if args.point_mode == "center":
            points = np.array([[w / 2, h / 2]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=points, point_labels=labels, multimask_output=True
            )
        else:
            points = np.array([[w*.35,h*.5],[w*.5,h*.5],[w*.65,h*.5]], dtype=np.float32)
            labels = np.ones(len(points), dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=points, point_labels=labels, multimask_output=True
            )

        # Select plausible mask: score weighted by area preference; this is a heuristic.
        candidates = []
        for mask, score in zip(masks, scores):
            area = float(mask.mean())
            if 0.02 <= area <= 0.75:
                candidates.append((float(score), mask))
        selected = max(candidates, key=lambda x: x[0])[1] if candidates else masks[int(np.argmax(scores))]
        mask_u8 = (selected.astype(np.uint8) * 255)
        cv2.imwrite(str(masks_dir / f"{path.stem}.png"), mask_u8)

        overlay = bgr.copy()
        tint = np.zeros_like(bgr); tint[:, :, 1] = 255
        overlay[selected] = cv2.addWeighted(bgr, .55, tint, .45, 0)[selected]
        cv2.imwrite(str(overlays_dir / f"{path.stem}.jpg"), overlay)
        print(f"Saved mask: {path.name}")

if __name__ == "__main__":
    main()
