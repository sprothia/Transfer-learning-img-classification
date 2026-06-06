# Transfer Learning Challenge

100-class image classification under severe data scarcity (~10 training images per class). We fine-tune an ImageNet-pretrained **ConvNeXt-Tiny** to classify the test images into 100 categories.

---

## Overview

Training a model from scratch is not viable with only ~1,000 labeled images, so we use transfer learning: we load a ConvNeXt-Tiny backbone pretrained on ImageNet, replace its final classification layer to output 100 classes, and fully fine-tune it on our data. Heavy data augmentation and regularization are used to combat overfitting on the small dataset.

Everything — data loading, training, and inference — lives in a single notebook that runs top to bottom.

## Results

- **Validation accuracy:** ~0.78 on a held-out set (~1 image per class)
  
---

## Weights

**Trained model weights:** [Google Drive link](<your-google-drive-link>) — download `best_model.pth` if you want to skip training and run inference directly.

---

## Setup

### Requirements

- Python 3.8+
- PyTorch and TorchVision
- NumPy, pandas, Pillow

```bash
pip install torch torchvision numpy pandas pillow
```

A CUDA-capable GPU is recommended for training. The notebook automatically uses GPU if available and falls back to CPU otherwise. The notebook was developed and run on Kaggle, where the competition data is mounted automatically.

### Dataset

The data is from the
[Kaggle competition page](https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project)
and is structured as:

```
ucsc-cse-144-spring-2026-final-project/
├── train/
│   ├── 0/
│   │   ├── 0.jpg
│   │   └── ...
│   ├── 1/
│   └── ... (100 class folders)
└── test/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

The notebook sets `DATA_ROOT` to the Kaggle competition path. If running locally, update `DATA_ROOT` at the top of the notebook to point to your data directory.

---

## How to Run

The entire pipeline is in `final_project.ipynb`. Run the cells in order from top to bottom.

### Training

Running the notebook through the training cell will:
1. Load the training images, mapping each class folder to its integer label (folder `"7"` → label `7`).
2. Hold out a small validation set (~1 image per class) to monitor generalization.
3. Load ConvNeXt-Tiny with ImageNet-pretrained weights and replace the final layer with a 100-class head.
4. Fine-tune the full network for 50 epochs, saving `best_model.pth` whenever validation accuracy improves.

Per-epoch loss and validation accuracy are printed during training, along with the best validation accuracy at the end.

**To skip training:** download `best_model.pth` from the Google Drive link above, place it alongside the notebook, and run only the inference cell.

### Inference

The final cell:
1. Loads `best_model.pth` (the best checkpoint saved during training).
2. Reads every image in `test/`, sorted by numeric ID.
3. Predicts a class label for each image.
4. Writes `submission.csv` with `ID` and `Label` columns, ready to upload to Kaggle.

---

## Configuration

Key hyperparameters (set in the notebook):

| Hyperparameter | Value |
|----------------|-------|
| Backbone | ConvNeXt-Tiny (ImageNet pretrained) |
| Input size | 224 × 224 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.05 |
| LR schedule | Cosine annealing |
| Epochs | 50 |
| Loss | Cross-entropy (label smoothing = 0.1) |
| Augmentation | RandomResizedCrop, horizontal flip, TrivialAugmentWide, RandomErasing |

---

## Reproducibility

All random sources are seeded for reproducibility:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

The validation split also uses a fixed-seed generator. With the same seed, code, and data, results reproduce with similar means and standard deviations. Because the dataset is small, expect minor run-to-run variance in validation accuracy.

---

## Notes

- **Label ordering:** class folders are mapped directly by integer (`int(folder_name)`), so folder `"7"` always corresponds to label `7`. This is required for correct evaluation.
- **Image format:** all images are converted to RGB and resized to 224×224, since the pretrained model expects 3-channel ImageNet-normalized input.
