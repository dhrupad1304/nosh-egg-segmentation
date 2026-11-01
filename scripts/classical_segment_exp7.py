
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def fill_holes(mask):
    h, w = mask.shape
    flood = mask.copy()
    floodfill = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, floodfill, (0, 0), 255)
    holes = cv2.bitwise_not(flood)

    return cv2.bitwise_or(mask, holes)


def K(size):
    return np.ones((size, size), np.uint8)


def clean_mask(mask):
    mask = (mask > 0).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K(5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K(9))

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = np.zeros_like(mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(
            output,
            [largest],
            -1,
            255,
            thickness=cv2.FILLED
        )

    return output


def grabcut_rectangle(image, margin, iterations=5):
    h, w = image.shape[:2]

    x = int(w * margin)
    y = int(h * margin)
    rect_w = w - 2 * x
    rect_h = h - 2 * y

    if rect_w <= 0 or rect_h <= 0:
        return np.zeros((h, w), np.uint8)

    mask = np.zeros((h, w), np.uint8)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image,
            mask,
            (x, y, rect_w, rect_h),
            bg_model,
            fg_model,
            iterations,
            cv2.GC_INIT_WITH_RECT
        )
    except cv2.error:
        return np.zeros((h, w), np.uint8)

    result = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    return result


def generate_candidates(image, lab, hsv):
    candidates = []

    h, w = image.shape[:2]
    border_size = 12

    border_pixels = np.concatenate([
        lab[:border_size].reshape(-1, 3),
        lab[-border_size:].reshape(-1, 3),
        lab[:, :border_size].reshape(-1, 3),
        lab[:, -border_size:].reshape(-1, 3)
    ])

    border_color = np.median(border_pixels, axis=0)

    # Candidate 1: distance from border color in Lab space.
    distance = np.linalg.norm(
        lab - border_color,
        axis=2
    )

    channels = [
        distance,
        lab[:, :, 0],
        lab[:, :, 1],
        lab[:, :, 2],
        hsv[:, :, 1].astype(np.float32)
    ]

    for index, channel in enumerate(channels):
        normalized = cv2.normalize(
            channel,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        blurred = cv2.GaussianBlur(
            normalized,
            (7, 7),
            0
        )

        threshold = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        if index == 0:
            candidates.append(threshold)
        else:
            candidates.append(threshold)
            candidates.append(255 - threshold)

    # Candidates 2-4: GrabCut with different rectangle margins.
    for margin in (0.10, 0.25, 0.35):
        candidates.append(
            grabcut_rectangle(image, margin)
        )

    return candidates


def score_candidate(mask, lab, gradient):
    h, w = mask.shape

    foreground = mask > 0
    area_fraction = foreground.mean()

    # Reject masks that are implausibly small or large.
    if not 0.02 < area_fraction < 0.65:
        return -1.0

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return -1.0

    contour = max(contours, key=cv2.contourArea)

    if len(contour) < 5:
        return -1.0

    area = cv2.contourArea(contour)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = area / (hull_area + 1e-6)

    # Compare the mask area with its fitted ellipse.
    try:
        (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(contour)

        ellipse_area = np.pi * major_axis * minor_axis / 4

        ellipse_fit = (
            min(area, ellipse_area) /
            (max(area, ellipse_area) + 1e-6)
        )

    except cv2.error:
        ellipse_fit = 0.0

    # Boundary strength.
    edge = cv2.morphologyEx(
        mask,
        cv2.MORPH_GRADIENT,
        K(3)
    ) > 0

    if edge.any():
        edge_strength = min(
            gradient[edge].mean() /
            (np.percentile(gradient, 99) + 1e-6),
            1.0
        )
    else:
        edge_strength = 0.0

    # Contrast between foreground and its surrounding ring.
    dilated_large = cv2.dilate(mask, K(25))
    dilated_small = cv2.dilate(mask, K(5))

    ring = (
        (dilated_large > 0) &
        (dilated_small == 0)
    )

    if foreground.any() and ring.any():
        foreground_color = lab[foreground].mean(axis=0)
        ring_color = lab[ring].mean(axis=0)

        contrast = min(
            np.linalg.norm(foreground_color - ring_color) / 40,
            1.0
        )
    else:
        contrast = 0.0

    # Penalize candidates touching the image border.
    touches = sum([
        bool(mask[0].any()),
        bool(mask[-1].any()),
        bool(mask[:, 0].any()),
        bool(mask[:, -1].any())
    ])

    score = (
        0.30 * solidity
        + 0.25 * ellipse_fit
        + 0.25 * edge_strength
        + 0.20 * contrast
        - 0.15 * touches
    )

    return float(score)


def refine_candidate(image, mask):
    h, w = image.shape[:2]

    grabcut_mask = np.full(
        (h, w),
        cv2.GC_BGD,
        dtype=np.uint8
    )

    # Around the candidate: probable background.
    grabcut_mask[
        cv2.dilate(mask, K(31)) > 0
    ] = cv2.GC_PR_BGD

    # Candidate pixels: probable foreground.
    grabcut_mask[
        mask > 0
    ] = cv2.GC_PR_FGD

    # Eroded interior: definite foreground.
    grabcut_mask[
        cv2.erode(mask, K(21)) > 0
    ] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image,
            grabcut_mask,
            None,
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )

        result = np.where(
            (grabcut_mask == cv2.GC_FGD) |
            (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

        result = clean_mask(result)

    except cv2.error:
        result = mask.copy()

    contours, _ = cv2.findContours(
        result,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return mask

    largest = max(contours, key=cv2.contourArea)

    output = np.zeros_like(result)

    # Convex hull refinement.
    hull = cv2.convexHull(largest)

    cv2.fillConvexPoly(
        output,
        hull,
        255
    )

    return fill_holes(output)


def segment(image):
    """
    Experiment 7:
    Multiple candidates + egg-likeness scoring + GrabCut refinement.
    """

    h, w = image.shape[:2]

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    ).astype(np.float32)

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    ).astype(np.float32)

    grad_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0
    )

    grad_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1
    )

    gradient = cv2.magnitude(
        grad_x,
        grad_y
    )

    candidates = generate_candidates(
        image,
        lab,
        hsv
    )

    best_mask = None
    best_score = -1.0

    for candidate in candidates:
        cleaned = clean_mask(candidate)

        score = score_candidate(
            cleaned,
            lab,
            gradient
        )

        if score > best_score:
            best_score = score
            best_mask = cleaned

    if best_mask is None:
        return np.zeros((h, w), np.uint8)

    return refine_candidate(
        image,
        best_mask
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--images", required=True)
    ap.add_argument(
        "--masks",
        required=True,
        help="SAM masks directory; required for dev scoring"
    )
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", default="outputs/classical")
    ap.add_argument("--experiments", default="experiments.csv")

    args = ap.parse_args()

    image_dir = Path(args.images)
    gt_dir = Path(args.masks)
    out = Path(args.output)

    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(
        Path(args.split).read_text()
    )

    # Use development images only.
    dev_names = split.get("dev", [])

    rows = []

    for name in dev_names:
        image_path = image_dir / name

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        prediction = segment(image)

        cv2.imwrite(
            str(pred_dir / f"{image_path.stem}.png"),
            prediction
        )

        gt_path = gt_dir / f"{image_path.stem}.png"

        if gt_path.exists():
            gt = (
                cv2.imread(
                    str(gt_path),
                    cv2.IMREAD_GRAYSCALE
                ) > 0
            )

            pred = prediction > 0

            intersection = np.logical_and(
                gt,
                pred
            ).sum()

            union = np.logical_or(
                gt,
                pred
            ).sum()

            iou = (
                intersection / union
                if union
                else 1.0
            )

            dice = (
                2 * intersection /
                (gt.sum() + pred.sum())
                if (gt.sum() + pred.sum())
                else 1.0
            )

            rows.append([
                name,
                float(iou),
                float(dice)
            ])

        print("Processed", name)

    if rows:
        mean_iou = float(
            np.mean([row[1] for row in rows])
        )

        mean_dice = float(
            np.mean([row[2] for row in rows])
        )

        print(
            f"DEV mean IoU={mean_iou:.4f}, "
            f"Dice={mean_dice:.4f}"
        )

    else:
        mean_iou = ""
        mean_dice = ""

        print(
            "No matching SAM masks found; "
            "predictions saved without scores."
        )

    # Append the experiment result.
    experiment_path = Path(args.experiments)

    exists = experiment_path.exists()

    with experiment_path.open(
        "a",
        newline=""
    ) as file:
        writer = csv.writer(file)

        if not exists:
            writer.writerow([
                "experiment",
                "change",
                "reason",
                "parameters",
                "dev_iou",
                "dev_dice",
                "decision",
                "notes"
            ])

        writer.writerow([
            "exp7",
            "Multiple candidates + scoring + GrabCut refinement",
            "Handle diverse scenes using candidate selection",
            "Lab/Otsu/Saturation/GrabCut candidates; "
            "solidity + ellipse + edge + contrast score",
            mean_iou,
            mean_dice,
            "review",
            "Heuristic candidate scoring; compare against Exp 3"
        ])


if __name__ == "__main__":
    main()
