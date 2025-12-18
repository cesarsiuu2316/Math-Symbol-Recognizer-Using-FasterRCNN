# Math Symbol Recognizer Using Faster R-CNN

This project implements a **Faster R-CNN** object detection model (using a **ResNet50-FPN** backbone) to recognize handwritten mathematical symbols. It is specifically engineered to handle the significant domain gap between **digital ink datasets** (CROHME) and **real-world whiteboard images**.

### Key Features & Methodology

*   **Model Architecture**: The project utilizes a Faster R-CNN model pretrained on ImageNet. During training, the **entire backbone** (all 5 layers of ResNet50) is fine-tuned to adapt to the specific features of handwritten strokes, while the region proposal and classification heads are trained from scratch.
*   **Domain Adaptation via Augmentation**: To bridge the gap between the clean, thin strokes of digital ink and the noisy, thick strokes of whiteboard markers, a heavy augmentation pipeline with thresholding is employed. This includes:
    *   **Morphological Dilation**: To simulate the thickness of whiteboard markers.
    *   **Noise Injection**: To mimic sensor noise and whiteboard imperfections.
    *   **Geometric Transformations**: Random shearing, rotation, and scaling to account for handwriting variability.
*   **Statistical Anchor Calibration**: The system features a custom strategy that calibrates the model's **anchor sizes and aspect ratios** based on the statistical distribution of the target domain (whiteboard), ensuring the Region Proposal Network (RPN) is optimized for the actual scale of the symbols it will encounter.

## 📂 Project Structure

```text
Math-Symbol-Recognizer-Using-FasterRCNN/
├── configs/                    # Configuration files for CPU and CUDA training
│   ├── config_cpu.json
│   └── config_cuda.json
├── data/                       # Generated annotations and class mappings
│   ├── calibrated_whiteboard_bboxes.json  # Pre-calculated whiteboard stats
│   ├── class_mapping.json      # Map between symbol names and IDs
│   └── train_annotations.json  # Parsed training data
├── output/                     # Training outputs
│   ├── logs/                   # TensorBoard logs and JSON reports
│   └── models/                 # Saved model checkpoints (.pth)
├── TC11_CROHME23/              # Dataset folder (CROHME)
├── check_bboxes.py             # Utility to visualize ground truth bounding boxes
├── eda_calibration.py          # Calculates scaling factors between domains
├── eda_crohme_whiteboard.py    # Helper logic for EDA and calibration
├── inference.py                # Run inference on new images
├── math_symbols_dataset.py     # PyTorch Dataset implementation
├── model.py                    # Faster R-CNN model definition
├── parser.py                   # Parses raw CROHME .lg files into JSON
├── train.py                    # Main training loop
├── train_utils.py              # Training helpers (saving, logging)
├── utils.py                    # General utilities
├── requirements.txt            # CPU dependencies
└── requirements_cuda.txt       # GPU/CUDA dependencies
```

## 🚀 Installation

1.  **Clone the repository** and navigate to the project folder.

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Environment**:
    *   **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    *   **Linux/Mac**:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies**:
    *   For **GPU/CUDA** (Recommended):
        ```bash
        pip install -r requirements_cuda.txt
        ```
    *   For **CPU** only:
        ```bash
        pip install -r requirements.txt
        ```

---

## 🛠️ Usage Guide

Follow these steps to prepare data, train the model, and run inference.

### 1. Data Preparation
First, parse the raw CROHME dataset (located in `TC11_CROHME23`) into a format the model can understand. This script generates `data/train_annotations.json` and `data/class_mapping.json`.

```bash
python parser.py configs/config_cuda.json
```

### 2. Domain Calibration
This step calculates the optimal anchor sizes and scaling factors to adapt the model from digital ink to whiteboard images.

*   **Note:** The project comes with a pre-calculated `data/calibrated_whiteboard_bboxes.json`. This allows you to skip the manual bounding box selection process.

Run the calibration script to generate the final model configuration:

```bash
python eda_calibration.py configs/config_cuda.json
```

### 3. Training
Start the training process. The script will use the configuration file to set hyperparameters, paths, and model settings.

```bash
python train.py configs/config_cuda.json
```
*   Checkpoints are saved to `output/models/`.
*   Logs are saved to `output/logs/`.

### 4. Inference
To test the model on a new image (e.g., a photo of a whiteboard):

```bash
python inference.py configs/config_cuda.json
```
1.  A file dialog will open. Select your image.
2.  The script will run detection and display the result.
3.  Press any key to close the window.
4.  Results are saved in the `results/` folder.

## ⚙️ Configuration
The `configs/` folder contains JSON files to control the pipeline.
*   **`paths`**: Directories for data, logs, and models.
*   **`model_params`**: Anchor sizes, aspect ratios, and model architecture settings.
*   **`training_params`**: Learning rate, batch size, epochs, and scheduler settings.
*   **`transform_params`**: Image resizing and augmentation settings.
