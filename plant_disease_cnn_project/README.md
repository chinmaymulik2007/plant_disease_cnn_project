# Plant Disease Classification using CNN

A college mini-project that classifies plant leaf images using **PyTorch + ResNet18 transfer learning** and serves predictions through a **Streamlit** web app.

## 1. Dataset

Use the PlantVillage dataset:
- GitHub: https://github.com/spMohanty/PlantVillage-Dataset
- Hugging Face dataset information: https://huggingface.co/datasets/mohanty/PlantVillage

Download/extract the color images and place the class folders under `data/PlantVillage/`, for example:

```text
data/PlantVillage/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   └── ...
├── Apple___healthy/
├── Tomato___Early_blight/
└── Tomato___healthy/
```

The split script below selects the first 8 classes alphabetically by default for a quick demonstration. To use all classes, pass `--max-classes 0`. To use only a smaller number of images per class, use `--max-images-per-class 300`.

**Important:** the script makes a random image-level split. For research-grade results, use a leaf-group-aware split (the dataset's official Hugging Face version documents grouped train/test splits) to reduce the risk of similar images appearing in both training and test sets.

## 2. Environment setup (Windows PowerShell)

Activate your existing Python 3.12 virtual environment, if you have one:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PyTorch with CUDA already works in this environment, keep it installed. Install a matching `torchvision` build for your installed PyTorch version, then install the remaining packages:

```powershell
python -m pip install streamlit pillow numpy matplotlib scikit-learn tqdm
python -m pip install torchvision
```

If you are setting up a fresh environment, follow the official PyTorch selector for the correct CPU/CUDA command: https://pytorch.org/get-started/locally/

Check GPU:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

## 3. Split the dataset

From the project root:

```powershell
python split_dataset.py --source data/PlantVillage --output data/split --max-classes 8
```

Optional: limit each class to 300 images for a faster first run:

```powershell
python split_dataset.py --source data/PlantVillage --output data/split --max-classes 8 --max-images-per-class 300
```

## 4. Train the model

```powershell
python train.py --data data/split --epochs 5 --batch-size 32
```

If CUDA memory runs out, reduce the batch size:

```powershell
python train.py --data data/split --epochs 5 --batch-size 16
```

The first run may download ImageNet pretrained ResNet18 weights. Outputs are saved in `artifacts/`:
- `plant_disease_model.pth`
- `history.json`
- `loss_curve.png`
- `accuracy_curve.png`
- `confusion_matrix.png`
- `test_metrics.json`

## 5. Run the web app

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser and upload a leaf image.

## 6. Predict from the terminal

```powershell
python predict.py "path\to\leaf.jpg"
```

## 7. Model architecture

```text
Input leaf image (RGB, 224 x 224)
              |
      ResNet18 feature extractor
      (pretrained on ImageNet)
              |
      Global average pooling
              |
      Fully connected layer
      (number of selected classes)
              |
        Softmax probabilities
              |
      Predicted plant/disease
```

The project uses transfer learning: the pretrained ResNet18 is adapted to the selected plant disease classes. The model is trained using cross-entropy loss and AdamW.

## 8. Evaluation

Report the actual held-out test results produced by your run:
- Test accuracy
- Per-class precision and recall
- Macro F1-score
- Confusion matrix
- Training and validation loss/accuracy curves

Do not write an assumed accuracy in your report. Results depend on the selected classes, split, random seed, training time and dataset quality.

## 9. Limitations

PlantVillage images often have controlled backgrounds. Real-world field photos may contain shadows, multiple leaves, clutter and different lighting. A model's confidence is not a guarantee of correctness. This project is for education and should not be used as the sole basis for agricultural treatment decisions.

## 10. Suggested mini-project report structure

1. Abstract
2. Problem statement and objectives
3. Dataset description and preprocessing
4. CNN and transfer-learning architecture
5. Training methodology
6. Results: accuracy, precision, recall, F1 and confusion matrix
7. Streamlit application screenshots
8. Limitations and future scope
9. Conclusion
10. References

### Dataset reference

Mohanty, S. P., Hughes, D. P., and Salathé, M. (2016). *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science, 7, 1419. https://doi.org/10.3389/fpls.2016.01419
