# Veritas Model (Face Type Classifier)

This project uses **EfficientNet-B0** to classify faces into 4 categories:

- **3D** – Rendered, CGI, movies/series  
- **2D** – Anime, cartoons  
- **Real** – Photographs of real people  
- **AI-generated** – Deepfakes, diffusion, and other synthetic faces  

The model is trained to distinguish between these types for research, filtering, or personal projects.

---

## Dataset Flow

The workflow for dataset handling is divided into several steps:

### 1. Download dataset

Use `download_dataset.py` to download the datasets from Kaggle. The datasets are saved to the `kaggle_cache` folder.

```bash
python download_dataset.py
```plaintext

### 2. Convert to raw format

Use `dataset_to_raw.py` to move a subset of images from `kaggle_cache` into `data/raw/ai` and `data/raw/real` folders.

```bash
python dataset_to_raw.py
```plaintext

### 3. Split into train, validation, and test

Use `raw_to_trainvaltest.py` to divide the raw data into:

- `data/train/`
- `data/val/`
- `data/test/`

```bash
python raw_to_trainvaltest.py
```bash

### 4. Train the model

Train EfficientNet on the dataset using `train.py`. This will save the trained model to the `models` folder.

```bash
python train.py
```bash

### 5. Test the model

Check the accuracy of the trained model on the test set using `test.py`.

```bash
python test.py
```

### 6. Predict on new images

Use `predict.py` to classify images from a specific folder, e.g., `predict/predict_dataset`.

```bash
python predict.py
```

---

## Folder Structure

```plaintext
veritas-model/
│
├─ kaggle_cache/        # Downloaded datasets
├─ data/
│   ├─ raw/             # Selected raw images
│   │   ├─ ai/
│   │   └─ real/
│   ├─ train/
│   ├─ val/
│   └─ test/
├─ models/              # Saved trained models
├─ predict/             # Images for prediction
├─ download_dataset.py
├─ dataset_to_raw.py
├─ raw_to_trainvaltest.py
├─ train.py
├─ test.py
└─ predict.py
```

---

## Notes

- Make sure your GPU is available for faster training:

    ```python
    import torch
    print(torch.cuda.is_available())
    ```

- Adjust the number of epochs, batch size, and dataset size in the scripts depending on your system memory and dataset.

**Recommended workflow:**  
Download → Raw → Train/Val/Test → Train → Test → Predict

**Useful commands (Like execute it everytime you open the terminal so the thing can work):**  
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\effnet-env\Scripts\Activate.ps1
