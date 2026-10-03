# Malaria Detection with Explainable AI (XAI)

Experiment notebooks for the paper:
**"Towards Transparent Diagnostics: Investigating Architectural Trade-offs and Explainability in Malaria Detection"** by Suman Kunwar and Avishek Dangol.

## About

This repository contains Jupyter notebooks (conducted on Kaggle) for benchmarking deep learning architectures for malaria parasite detection from blood smear images, with a focus on the trade-off between predictive accuracy, model efficiency, and explainability.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `malaria-detection-xai.ipynb` | Malaria detection with explainability analysis (Grad-CAM, SHAP, LIME) |
| `malaria-quantification-benchmarking.ipynb` | Benchmarking multiple architectures |
| `malaria-quantification-xai.ipynb` | ResNet18_Enhanced version with XAI |

## Key Findings

- **MobileNetV2**: 96.35% accuracy, 8.49 MB model size, 1.35 ms inference — best efficiency/accuracy trade-off
- **Proposed Enhanced ResNet18**: 97.67% accuracy, 0.9756 AUC, 13.17 ms inference — highest accuracy
- Pruning the proposed model by 30% improved accuracy to 97.84% and reduced inference to 12.92 ms
- **Grad-CAM** gives holistic activation maps; **LIME** emphasizes fine-grained boundary/contour info; **SHAP** provides pixel-level feature attributions

## Dataset

**NIH Malaria Dataset (BBBC041)**
- 27,558 cell images from Giemsa-stained thin blood smears
- Equally split between infected (parasitized) and uninfected
- 150 infected and 50 healthy patients
- Patient-grouped 80/20 train/validation split (no leakage, seed = 42)
- Augmentation: flipping, rotation, color jitter

## Models Benchmarked

| Model | Accuracy (%) | Precision | Recall | F1-Score | AUC | Parameters (M) | Size (MB) | Inference (ms) |
|-------|-------------|-----------|--------|----------|-----|----------------|-----------|----------------|
| ResNet18 | 96.92 | 0.9492 | 0.9835 | 0.9661 | 0.9932 | 11.18 | 42.64 | 1.42 |
| MobileNetV2 | 96.85 | 0.9507 | 0.9803 | 0.9653 | 0.9900 | 2.23 | 8.49 | 1.35 |
| EfficientNet-B2 | 97.00 | 0.9525 | 0.9817 | 0.9669 | 0.9911 | 7.70 | 29.39 | 2.13 |
| ResNet101 | 96.75 | 0.9512 | 0.9773 | 0.9641 | 0.9902 | 42.50 | 162.14 | 5.79 |
| VGG19 | 96.69 | 0.9393 | 0.9898 | 0.9639 | 0.9933 | 139.58 | 532.45 | 6.65 |
| **Proposed (Enhanced ResNet18)** | **97.67** | **0.9795** | **0.9717** | **0.9756** | **0.9963** | **11.18** | **42.64** | **13.17** |
| Proposed (Pruned 30%) | 97.84 | — | — | — | — | — | — | 12.92 |

*Note: Inference time for the proposed model is listed as 13.42 ms in the results text but 13.17 ms in the pruning figure; the table uses 13.17 ms as reported in the pruning comparison.*

## Proposed Architecture

Enhances standard ResNet18 by:
- Using ImageNet-pretrained weights as initialization
- Replacing the original classifier with a custom head: Dropout (0.5) + Linear (512, 2) for binary classification
- Making all convolution blocks (layer 1 to layer 4) fully trainable
- Applying activation during the inference stage
- Using Test Time Augmentation (TTA) to improve performance

## Explainability Methods

- **Grad-CAM** — gradient-weighted class activation mapping; highlights regions with strong influence on the prediction
- **LIME** — local surrogate explanations; emphasizes fine-grained boundary/contour regions
- **SHAP** — Shapley value-based feature attribution; provides pixel-level quantitative contributions

### XAI Findings

- **Correctly classified samples**: SHAP, LIME, and Grad-CAM attributions were well aligned and mutually reinforced
- **Uninfected sample**: SHAP showed diffuse speckled patterns; LIME highlighted coherent superpixels; Grad-CAM emphasized subtle color regions
- **Parasitized sample**: SHAP produced dense spatial clusters over diagnostic regions, matching LIME's outline and Grad-CAM's highlighted region
- **Misclassified sample**: Methods diverged (SHAP in upper-right, LIME across disparate regions, Grad-CAM in lower portion), indicating lack of coherent internal reasoning

## Benchmarking Against State-of-the-Art

| Study | Year | Model | Accuracy (%) |
|-------|------|-------|-------------|
| Hou et al. | 2026 | MalariaNet | 95.6 |
| Laghari et al. | 2025 | ResNet-101 | 89.0 |
| **Our study** | **2026** | **Regularized ResNet18** | **97.67** |

## Usage

These notebooks were run on Kaggle. To reproduce:

1. Download the notebooks from this repo
2. Upload to Kaggle (or run locally with Jupyter)
3. Attach the NIH Malaria Dataset
4. Run all cells

Open the notebook `malaria-detection-xai.ipynb` in Jupyter or Kaggle and run all cells sequentially.

## Citation

```bibtex
@article{kunwar2026transparent,
  title={Towards Transparent Diagnostics: Investigating Architectural Trade-offs and Explainability in Malaria Detection},
  author={Kunwar, Suman and Dangol, Avishek},
  journal={arXiv preprint arXiv:2609.31682},
  year={2026}
}
```


## Acknowledgements
- NIH Malaria Dataset for providing the benchmark data
- The open-source community for PyTorch, Grad-CAM, LIME, and SHAP implementations
