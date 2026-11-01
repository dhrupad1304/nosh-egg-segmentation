# NOSH Egg Segmentation and Tracking — Report

**Candidate:** [Your name]  
**Assignment date:** [Date]  
**Repository:** [Public GitHub URL]  
**Tracked video:** [Public Google Drive URL]

## Abstract
Summarize the dataset, four tasks, approach, and actual measured outcomes. Do not report results until experiments have been run.

## 1. Problem statement
Describe the need to segment eggs in images and track separate eggs through video frames.

## 2. Dataset and split
- Number of images:
- Image dimensions:
- Grouping strategy:
- Dev/eval counts:
- Potential duplicate leakage and limitations:

## 3. Task 1 — SAM pseudo-label generation
- SAM variant and checkpoint:
- Prompting method:
- Mask selection heuristic:
- Manual review/corrections:
- Examples and limitations:

## 4. Task 2 — Classical segmentation
Explain color spaces, thresholding, morphology, contour filtering, and hole filling.
Include an experiment table with every successful and failed trial.

| Trial | Change | Motivation | Parameters | Dev IoU | Dev Dice | Keep/discard | Notes |
|---|---|---|---|---:|---:|---|---|

## 5. Task 3 — Evaluation
Report mean and per-image IoU, Dice, precision, recall, boundary F-score.
Include IoU histogram, worst-case examples, error types, and improvement ideas.

## 6. Task 4 — Video tracking
Explain per-frame segmentation, connected components, centroid matching, Hungarian assignment, persistent IDs, and failure cases. Include the public Drive link.

## 7. Limitations and future work
Discuss SAM pseudo-label noise, scene variability, touching eggs, shadows, occlusion, and tracking identity switches.

## 8. AI tools disclosure
Describe exactly which AI tools were used and what code/documentation they assisted with. Confirm that you reviewed and tested the implementation.

## 9. Conclusion
Summarize only verified work and measured results.
