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
  

    """Experiment 2: GrabCut with a central ellipse initialization."""
    h, w = image.shape[:2]

    # Initialize GrabCut using a rectangle around the central object.
    margin_x = max(1, int(w * 0.25))
    margin_y = max(1, int(h * 0.25))

    rect = (
        margin_x,
        margin_y,
        w - 2 * margin_x,
        h - 2 * margin_y
    )

    mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image,
            mask,
            rect,
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_RECT
        )
    except cv2.error:
        return np.zeros((h, w), np.uint8)

    # Convert GrabCut labels into a binary foreground mask.
    result = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    # Keep the largest connected foreground component.
    contours, _ = cv2.findContours(
        result,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = np.zeros((h, w), np.uint8)

    if not contours:
        return output

    chosen = max(contours, key=cv2.contourArea)
    cv2.drawContours(output, [chosen], -1, 255, thickness=cv2.FILLED)

    return fill_holes(output)
  

    
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
    dev_names = split.get("dev", [])

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
