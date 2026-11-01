# NOSH Egg Segmentation & Video Tracking

A computer-vision project developed for the NOSH CV Internship Task. The project covers egg segmentation using pretrained Segment Anything Model (SAM) masks, classical OpenCV-based segmentation, quantitative evaluation, and video-based egg detection and tracking.

The project is organized around four tasks:

1. Generate egg segmentation pseudo-labels using SAM.
2. Develop and experiment with classical OpenCV segmentation methods.
3. Evaluate the final segmentation pipeline against SAM-generated reference masks.
4. Detect and track eggs in a video using a classical computer-vision pipeline.

> **Important:** SAM-generated masks are pseudo-ground-truth labels, not manually verified ground truth. Some reference masks contain defects, including inverted masks. Therefore, reported metrics measure agreement with the generated references and should not be interpreted as definitive segmentation accuracy.

## Project Links

- **GitHub Repository:** [nosh-egg-segmentation](https://github.com/dhrupad1304/nosh-egg-segmentation)
- **Final Project Report:** [View PDF Report](reports/final_report.pdf)
- **Video Reference:** [Google Drive Video](https://drive.google.com/file/d/1MYhljXmk2cjCzZFSpQ-AB04evRMmb1pN/view?usp=drive_link)

## Project Structure

```text
nosh-egg-segmentation/
│
├── data/
│   └── Dataset and prepared data
│
├── outputs/
│   ├── classical_baseline/
│   ├── classical_eval/
│   ├── classical_exp1/
│   ├── classical_exp2/
│   ├── classical_exp3/
│   ├── classical_exp3_dev/
│   ├── classical_exp3_eval/
│   ├── classical_exp4_dev/
│   ├── classical_exp5_dev/
│   ├── classical_exp6_dev/
│   ├── classical_exp7_dev/
│   ├── classical_exp7_eval/
│   ├── classical_exp8_dev/
│   ├── exp3_exp4_comparison/
│   ├── exp4_exp8_comparison/
│   ├── exp7_eval_run.csv
│   └── exp8_run_log.csv
│
├── reports/
│   ├── evaluation_baseline/
│   ├── evaluation_exp3/
│   ├── final_report.pdf
│   └── report_template.md
│
├── scripts/
│   ├── classical_segment.py
│   ├── classical_segment_baseline.py
│   ├── classical_segment_exp3.py
│   ├── classical_segment_exp3_dev.py
│   ├── classical_segment_exp4.py
│   ├── classical_segment_exp5.py
│   ├── classical_segment_exp6.py
│   ├── classical_segment_exp7.py
│   ├── classical_segment_exp8.py
│   ├── evaluate.py
│   ├── prepare_data.py
│   ├── sam_segment.py
│   └── track_video.py
│
├── experiments.csv
├── egg_final_data.zip
├── prepared.zip
├── prepared_data.zip
├── requirements.txt
├── README.md
└── .gitignore
```

The repository preserves the segmentation scripts, experiment iterations, evaluation outputs, and final report. Dataset archives and generated outputs are included as available in the repository.

## Technologies Used

- Python
- OpenCV
- Segment Anything Model (SAM)
- NumPy
- Pandas
- Matplotlib
- Scikit-image
- SciPy
- Git and GitHub

## Task 1 — SAM-Based Egg Segmentation

The first task focuses on generating segmentation masks for egg images using a pretrained Segment Anything Model.

The SAM mask-generation workflow includes:

- Loading the pretrained SAM model.
- Generating candidate segmentation masks.
- Filtering candidates using area and shape characteristics.
- Selecting egg-like regions based on geometric properties.
- Removing duplicate or unsuitable masks.
- Saving segmentation masks for subsequent evaluation.

The mask-generation process uses characteristics such as:

- Area
- Solidity
- Ellipse-fit quality
- Border contact
- Overlap with other candidate masks

### SAM Configuration

The report documents the following SAM ViT-B generator settings:

| Parameter | Value |
|---|---:|
| Points per side | 32 |
| Predicted IoU threshold | 0.90 |
| Stability score threshold | 0.92 |
| Minimum region area | 500 pixels |

SAM-generated masks were used as reference labels for the classical segmentation experiments.

### Limitations

SAM can select incorrect objects or generate unsuitable masks in cluttered scenes. Some reference masks were inverted or otherwise defective.

These issues affect the reliability of the evaluation metrics.

## Task 2 — Classical OpenCV Segmentation

The second task focuses on developing a classical computer-vision segmentation pipeline without relying on a trained egg-segmentation model.

Several experiments were conducted to investigate different segmentation strategies and improve the quality of the predicted masks.

### Experiment 7 — Final Segmentation Approach

Experiment 7 uses candidate generation, egg-likeness scoring, and GrabCut refinement.

The pipeline includes:

1. Generate threshold-based candidates using color information from Lab and HSV color spaces.
2. Evaluate different thresholding polarities.
3. Apply morphological operations to clean candidate masks.
4. Score candidates using shape and appearance characteristics.
5. Select suitable egg-like regions.
6. Refine the selected region using GrabCut.
7. Apply final mask cleanup.

The candidate scoring process considers:

- Solidity
- Ellipse-fit quality
- Edge strength
- Ring contrast
- Border penalties

The final refinement uses foreground and probable-background regions to improve the segmentation boundary.

### Experimentation

Multiple experiments were conducted to investigate:

- Color thresholding
- Otsu thresholding
- Morphological operations
- Contour selection
- Candidate scoring
- GrabCut refinement
- Mask cleanup

The experiment scripts and CSV logs are preserved in the repository.

Development data was used for experimentation. The evaluation split was kept separate from development tuning.

## Task 3 — Quantitative Evaluation

The final segmentation pipeline was evaluated against SAM-generated pseudo-ground-truth masks.

### Final Results

| Split | Images | Mean IoU | Mean Dice |
|---|---:|---:|---:|
| Development (DEV) | 88 | 0.4636 | 0.5679 |
| Evaluation (EVAL) | 22 | 0.5394 | 0.6449 |
| **Total** | **110** | — | — |

### Evaluation Metrics

The project uses the following metrics:

**Intersection over Union (IoU)**

Measures the overlap between the predicted mask and the reference mask.

\[
IoU = \frac{|P \cap G|}{|P \cup G|}
\]

**Dice Coefficient**

Measures the similarity between the predicted mask and the reference mask.

\[
Dice = \frac{2|P \cap G|}{|P| + |G|}
\]

Where:

- \(P\) represents the predicted segmentation mask.
- \(G\) represents the reference mask.

### Evaluation Interpretation

The evaluation results represent agreement with SAM-generated pseudo-labels.

Some reference masks contain defects, including inverted masks. These defects can reduce the measured IoU and Dice scores even when the predicted mask identifies the egg region reasonably well.

The evaluation results should therefore be interpreted alongside qualitative inspection of the masks.

### Observed Segmentation Challenges

The final pipeline has limitations in:

- Low-contrast egg boundaries
- Shadows and reflections
- Multiple eggs in a single image
- Eggs touching image borders
- Defective reference masks
- Cases where the largest-component assumption is unsuitable

The current implementation may miss multiple eggs when they are present in the same image.

## Task 4 — Video Egg Detection and Tracking

The fourth task focuses on detecting and tracking eggs in a video using classical computer-vision techniques.

The input video contains a fixed-camera view of eggs moving through the scene.

### Video Information

| Property | Value |
|---|---|
| Duration | 60 seconds |
| Frame rate | 24 FPS |
| Total frames | 1,440 |
| Resolution | 640 × 360 |
| Camera | Fixed camera |

Eggs are visible approximately between 9.5 seconds and 41.5 seconds in the video.

### Detection Pipeline

The detection process uses:

- Brightness and saturation thresholds
- Morphological opening and closing
- Connected-component analysis
- Area filtering
- Solidity filtering
- Ellipse-fit filtering
- Reflection rejection

The pipeline also uses fallback thresholding and filtering methods to handle challenging lighting conditions.

### Tracking Pipeline

The tracker uses greedy nearest-centroid matching to associate detections across frames.

The process includes:

1. Detect egg candidates in each frame.
2. Calculate the centroid of each detected object.
3. Associate detections with existing tracks based on centroid distance.
4. Assign persistent IDs to tracked objects.
5. Display bounding boxes and tracking labels.
6. Retain tracks temporarily when detections are missed.

### Tracking Results and Limitations

The pipeline detects eggs in many frames, but the tracking results are not fully stable.

Observed limitations include:

- Reflections can produce false detections.
- Missed detections can interrupt tracking.
- Occlusion can affect object association.
- Tracking IDs may change over time.
- The tracker may retain an object's last position during temporary detection loss.

The current tracker is a classical baseline and is not intended to provide robust multi-object tracking under all conditions.

## Dataset and Data Preparation

The segmentation dataset contains 110 images with a resolution of 640 × 640 pixels.

The project uses a development and evaluation split:

| Split | Images |
|---|---:|
| Development | 88 |
| Evaluation | 22 |
| Total | 110 |

The data preparation workflow groups augmented Roboflow image variants to reduce the possibility of data leakage between development and evaluation data.

The grouping heuristic is not guaranteed to identify every near-duplicate image.

## Reproducibility

The repository contains the source scripts and experiment outputs used throughout the project.

The main scripts are:

| Script | Description |
|---|---|
| `prepare_data.py` | Dataset preparation |
| `sam_segment.py` | SAM-based segmentation |
| `classical_segment.py` | Classical segmentation |
| `classical_segment_baseline.py` | Baseline implementation |
| `classical_segment_exp3.py` – `classical_segment_exp8.py` | Experiment iterations |
| `evaluate.py` | Evaluation metrics |
| `track_video.py` | Video detection and tracking |

The experiment history is documented in `experiments.csv` and the output CSV files.

## Environment Setup

Python 3.10 or 3.11 is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For SAM-based processing, install PyTorch compatible with the available hardware and obtain the required pretrained SAM checkpoint.

Large model weights are not included in the repository.

## Deliverables

The project deliverables include:

- Public GitHub repository containing the source code.
- Final PDF report.
- Segmentation experiment scripts.
- Evaluation results and experiment logs.
- Video detection and tracking implementation.
- Documentation of methods, results, and limitations.

### Final Report

The complete report contains the methodology, experimental analysis, quantitative evaluation, qualitative observations, tracking results, and limitations.

[**View the Final Project Report (PDF)**](reports/final_report.pdf)

### Video

[**Open the Google Drive Video**](https://drive.google.com/file/d/1MYhljXmk2cjCzZFSpQ-AB04evRMmb1pN/view?usp=drive_link)

## AI Tools Disclosure

AI assistance was used during the development and documentation process. The implementation, experiments, and reported results were reviewed by the project author.

The reported metrics correspond to the project evaluation against SAM-generated pseudo-labels. The author is responsible for understanding and explaining the methods, implementation, results, and limitations.

## Interview Preparation

The following topics are relevant to the project:

- Segment Anything Model (SAM)
- Prompt-based segmentation
- HSV and Lab color spaces
- Otsu thresholding
- Morphological operations
- Connected-component analysis
- Contour selection
- GrabCut segmentation
- IoU and Dice metrics
- Data leakage and grouped data splitting
- Centroid-based tracking
- Object association
- Reflection rejection
- Segmentation and tracking failure analysis

## Conclusion

This project explores egg segmentation and video tracking using classical computer-vision methods and pretrained segmentation models.

The work includes SAM-based pseudo-label generation, classical segmentation experimentation, quantitative evaluation, and a video detection and tracking baseline.

The final results demonstrate the challenges of segmentation against imperfect reference masks and the limitations of classical tracking in scenes involving reflections, occlusion, and missed detections.

The repository preserves the implementation, experiment history, outputs, and final report for review and reproducibility.

---

**Project:** NOSH Egg Segmentation & Video Tracking  
**Author:** Dhrupad Gupta
