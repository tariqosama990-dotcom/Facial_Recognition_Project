# Facial Recognition System

The desktop application uses Haar Cascade for detection, pretrained FaceNet for embeddings, and cosine distance for recognition. It keeps enrollment data locally.

## Windows setup

Use Python 3.11. Open PowerShell in the extracted project directory.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Enter a name and select **Save the new person** with exactly one face visible. Registration captures a fresh frame. The application stores samples in `dataset/` and `face_embeddings.pkl`. First use of FaceNet may require downloading its weights.

## LFW evaluation

Create `eval_lfw/` with the first command if it is missing. Run the second command after the dataset is ready.

```powershell
.\.venv\Scripts\python.exe download_lfw.py
.\.venv\Scripts\python.exe -u evaluate.py --data eval_lfw --runs 5 | Tee-Object -FilePath evaluation_results.txt
```

The default threshold is 0.40. Evaluation tunes a separate threshold using validation images for each split and reports performance on test images. The script generates `roc.png`. Exact duplicate images are excluded before evaluation.

Keep local enrollment photos and embedding files out of shared source archives.
