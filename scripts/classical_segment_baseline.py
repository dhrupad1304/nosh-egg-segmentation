import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np

def fill_holes(mask):
    h, w = mask.shape
    flood = mask.copy()
    floodfill = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, floodfill, (0,0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)

def segment(image):
    """Starter classical baseline. Tune parameters on dev only."""
    h, w = image.shape[:2]
    # Combine Otsu cues from LAB chroma and HSV saturation.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)

    # Otsu on chroma distance from neutral; egg appearance varies, so this is a baseline.
    chroma = cv2.absdiff(A, np.full_like(A, 128))
    _, chroma_mask = cv2.threshold(chroma, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, sat_mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Eggs are often low saturation; retain low-S pixels and chromatic candidates.
    low_sat = cv2.bitwise_not(sat_mask)
    candidate = cv2.bitwise_or(low_sat, chroma_mask)

    # Smooth noise and close small gaps.
    k = max(3, (min(h, w) // 80) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: use a central ellipse prior when no useful contour is found.
        mask = np.zeros((h,w), np.uint8)
        cv2.ellipse(mask, (w//2,h//2), (max(1,w//4),max(1,h//4)), 0, 0, 360, 255, -1)
        return mask

    # Prefer large, reasonably egg-like contour, not necessarily just largest if frame has clutter.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    chosen = contours[0]
    mask = np.zeros((h,w), np.uint8)
    cv2.drawContours(mask, [chosen], -1, 255, thickness=cv2.FILLED)
    mask = fill_holes(mask)
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True, help="SAM masks directory; required for dev scoring")
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", default="outputs/classical")
    ap.add_argument("--experiments", default="experiments.csv")
    args = ap.parse_args()

    image_dir, gt_dir, out = Path(args.images), Path(args.masks), Path(args.output)
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path(args.split).read_text())
    dev_names = split.get("eval", [])

    rows = []
    for name in dev_names:
        p = image_dir / name
        image = cv2.imread(str(p))
        if image is None:
            continue
        pred = segment(image)
        cv2.imwrite(str(pred_dir / f"{p.stem}.png"), pred)

        gt_path = gt_dir / f"{p.stem}.png"
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 0
            pr = pred > 0
            inter = np.logical_and(gt, pr).sum()
            union = np.logical_or(gt, pr).sum()
            iou = inter / union if union else 1.0
            dice = (2*inter) / (gt.sum()+pr.sum()) if (gt.sum()+pr.sum()) else 1.0
            rows.append([name, iou, dice])
        print("Processed", name)

    if rows:
        mean_iou = float(np.mean([r[1] for r in rows]))
        mean_dice = float(np.mean([r[2] for r in rows]))
        print(f"DEV mean IoU={mean_iou:.4f}, Dice={mean_dice:.4f}")
    else:
        print("No matching SAM masks found; predictions saved without scores.")

    exp_path = Path(args.experiments)
    exists = exp_path.exists()
    with exp_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["experiment","change","reason","parameters","dev_iou","dev_dice","decision","notes"])
        writer.writerow(["baseline","LAB/HSV Otsu + morphology + largest contour",
                         "Initial reproducible classical baseline",
                         "kernel=max(3,min_dim//80 odd); close iterations=2",
                         np.mean([r[1] for r in rows]) if rows else "",
                         np.mean([r[2] for r in rows]) if rows else "",
                         "review","Starter baseline; tune on dev only"])

if __name__ == "__main__":
    main()
