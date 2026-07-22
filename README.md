# Face Detection with Age & Gender Prediction

A deep learning pipeline that detects faces in images, video, and real-time webcam
streams, then predicts the **age group** and **gender** of each detected person.

> Deep Learning 2022 — Final Project.

![Training vs. Validation Loss](Figures/EpochsLoss.png)

## Overview

The system is a two-stage pipeline:

1. **Face detection** — a YOLOv5 model locates faces and returns bounding boxes.
2. **Age & gender classification** — each detected face is cropped and passed to a
   CNN classifier that predicts an age group and the gender.

The final notebook ties both stages together and runs on still images, video files,
and live webcam input.

## Dataset

- **Face detection:** trained on the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/)
  dataset (converted to YOLO format) and evaluated against COCO 2017 face annotations.
- **Age & gender classification:** trained on the
  [UTKFace](https://susanqq.github.io/UTKFace/) dataset, whose filenames encode the
  age and gender of each subject.

Age is discretised into **9 groups** of ten years each (`00-10`, `11-20`, …, `81-90`).

## Model & Approach

**Face detection**
- YOLOv5 (`yolov5s`), fine-tuned on WIDER FACE at 1280 px input resolution.
- Exported to ONNX for inference.

**Age & gender classification**

Two backbones were trained and compared, each with a final layer producing
`9 age classes + 2 gender classes`:

- **ResNet-18** (ImageNet-pretrained) — used as the faster classifier in the final pipeline.
- **InceptionResNetV1** (`facenet-pytorch`, VGGFace2-pretrained).

Training details:
- Loss: **MSE** on the age head + **BCE** on the gender head (summed).
- Optimizer: **Adam** (`lr=0.001`, `amsgrad`, `weight_decay=1e-6`) with
  `ReduceLROnPlateau` scheduling.
- 25 epochs, batch size 64, 80/10/10 train/validation/test split.

## Results

Evaluated on the UTKFace test split:

| Model             | Age (mean group error) | Gender accuracy |
|-------------------|:----------------------:|:---------------:|
| ResNet-18         | ~0.60                  | ~93%            |
| InceptionResNetV1 | ~0.60                  | ~91%            |

*Age error is the mean absolute difference between the predicted and true age-group
index (each group spans 10 years).*

Example detections:

![Result 1](DATA/Results/Image1.png)
![Result 3](DATA/Results/Image3.png)

## Repository Contents

| Notebook | Description |
|----------|-------------|
| `Face Detection Model Creation.ipynb` | Prepares the WIDER FACE data and trains/configures the YOLOv5 detector. |
| `Face Extraction.ipynb` | Visualises the detector output, returning cropped face images. |
| `Age & Gender Prediction Model Creation.ipynb` | Trains and evaluates the ResNet-18 and InceptionResNetV1 classifiers. |
| `Face Detection: Age & Gender Detection.ipynb` | Final pipeline combining detection and classification for images, video, and webcam. |

Other directories:
- `Models/` — trained model weights (YOLOv5, ResNet-18, InceptionResNetV1).
- `DATA/` — sample test images (`WIDER_TEST`) and rendered results (`Results`).
- `Figures/` — training/validation loss plots.

## Running the Project

The notebooks were developed in **Google Colab** with GPU acceleration and support a
local fallback (they use the current working directory when Colab is unavailable).

1. Clone the repository.
2. Open the notebook of interest in Jupyter or Colab.
3. To run the full pipeline, open **`Face Detection: Age & Gender Detection.ipynb`**;
   it loads the pretrained weights from `Models/` and runs inference on the sample
   images in `DATA/WIDER_TEST/`.
4. To retrain, use the two *Model Creation* notebooks (a GPU is strongly recommended).

## Requirements

- Python 3
- PyTorch & TorchVision
- [YOLOv5](https://github.com/ultralytics/yolov5) (loaded via `torch.hub`)
- `facenet-pytorch`
- OpenCV, Pillow, NumPy, Matplotlib
- `albumentations`, `pybboxes` (data preparation)

## Authors

- Joan Gracia
- Pere Fraga
- Josep Ricci

## License

Released under the **GNU General Public License v3.0**. See [LICENSE](LICENSE).
