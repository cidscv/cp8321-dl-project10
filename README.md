# CP8321 - Deep Learning Assignment #10

## Topic

Reliable Prostate Cancer Grading: Patch-Level Gleason Pattern Classification with Uncertainty Quantification using SICAPv2

## Objective

Develop a deep learning model that not only classifies prostate cancer image patches into
Gleason patterns (3, 4, 5) but also provides uncertainty estimates with each prediction. This helps
identify when the model is unsure, improving clinical trust and enabling selective prediction
(e.g., defer to human pathologist in ambiguous cases).
The project will explore Bayesian deep learning or Monte Carlo dropout, and assess how
incorporating uncertainty improves decision-making in prostate histopathology.

## Dataset

[SICAPv2 (Prostate Cancer Histopathology)](https://data.mendeley.com/datasets/9xxm58dvs3/1)

## Task

Train a CNN (e.g., ResNet18, EfficientNet, etc.) to classify patches into 4 categories: benign,
Gleason 3, 4, 5. Evaluate with standard metrics: accuracy, confusion matrix, AUC per class.
Use one or more of:

- Monte Carlo dropout at test time (Gal & Ghahramani)
- Deep Ensembles: Train multiple models and compare predictions
- Bayesian CNNs using variational inference (if advanced students)

For each prediction, output:

- Class probabilities
- Confidence score (e.g., entropy, standard deviation of outputs)

Correlate model uncertainty with prediction correctness. Show confusion matrices for
high-certainty vs low-certainty cases. Visualize uncertainty heatmaps for sample patches.

### Bonus

Implement selective prediction: only accept model predictions above a confidence threshold.

## Our Methodology

**TODO**

## Tools and Libraries Used

1. [TensorFlow Python Library](https://www.tensorflow.org/)
2. [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb)

## Papers Referenced

1. [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep
   Learning, Yarin Gal & Zoubin Ghahramani, 2016](https://arxiv.org/abs/1506.02142)
2. [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles,
   Lakshminarayanan et al., 2017](https://arxiv.org/abs/1612.01474)
3. [On Calibration of Modern Neural Networks,
   Chuan Guo et al., 2017](https://arxiv.org/abs/1706.04599)
4. [Prostate Cancer Diagnosis using Deep Learning with Uncertainty Estimation,
   Bulten et al., 2020 (Nature Medicine)](https://pubmed.ncbi.nlm.nih.gov/31926805/)
5. [Uncertainty-Aware Deep Learning Models for Prostate Cancer Grading,
   Zhou et al., 2021](https://papers.miccai.org/miccai-2024/811-Paper2652.html)
6. [Efficient Self-Supervised Grading of Prostate Cancer Pathology, Bhattacharyya, Pal Das & Mitra, 2025](https://arxiv.org/abs/2501.15520)

## Groups Members

- Hermes Hui (hchui@torontomu.ca)
- Owen Reid (owen.reid@torontomu.ca)
- Mike Reynolds (mike.reynolds@torontomu.ca)
