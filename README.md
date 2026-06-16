# Deforestation classification web app

This project is a small FastAPI web app for classifying satellite tiles as **deforested** or **aforested**. You upload a tile in the browser and the app returns:

- the predicted class (`DEFORESTED` / `AFORESTED`)
- a simple green-pixel **heuristic analysis** (green coverage %, pixel counts, thresholds)

Under the hood there are **two predictors**:

- a fast, explainable **green-pixel heuristic** (the default)
- a scikit-learn **MLP classifier** trained on flattened satellite tiles

You can switch between them with an environment variable (see below).

---

## 1. Environment setup

Run all commands from the `CascadeProjects/windsurf-project` folder.

### 1.1 Create virtual environment

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

### 1.2 Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

The project only depends on CPU libraries (scikit-learn, NumPy, Pillow, FastAPI, Uvicorn). No GPU setup is required.

---

## 2. Predictors and how they work

### 2.1 Green-pixel heuristic (default)

Implemented in `infer.py` as `GreenPixelPredictor`.

High-level logic:

1. Resize the image to a working resolution and convert to RGB.
2. For every pixel, compute how much greener it is than red/blue.
3. Mark pixels as "green" when green exceeds max(red, blue) by a small delta.
4. Compute the **green coverage ratio** = `green_pixels / total_pixels`.
5. If coverage ≥ `coverage_threshold` (currently ~45%), classify as **aforested**, else **deforested**.

The result object that reaches the template looks roughly like:

```python
{
    "class": "aforested" | "deforested",
    "index": 0 or 1,
    "confidence": green_ratio,          # 0–1, used as a soft confidence
    "top_scores": [...],               # simple 2-class scores
    "metrics": None,
    "heuristic": {
        "green_ratio": float,
        "green_pixels": int,
        "total_pixels": int,
        "avg_green_channel": float,
        "std_green_channel": float,
        "greenness_delta_mean": float,
        "green_delta_threshold": float,
        "coverage_threshold": float,
    },
}
```

These fields are rendered on the result page in a soft-edged table.

### 2.2 MLP model predictor

Implemented in `infer.py` as `Predictor`.

- Uses `dataset.py` and `train.py` to build a scikit-learn MLPClassifier.
- Images are resized, normalized, flattened and scaled with `StandardScaler`.
- The bundle is stored as a Joblib file containing:

  - `model`: the trained MLPClassifier
  - `classes`: label encoder classes (e.g. `['deforested', 'non-deforested']`)
  - `scaler`: fitted StandardScaler
  - `image_size`: image resolution used during training
  - `metrics`: optional training metrics (accuracy, precision, etc.)

At prediction time, `Predictor.predict`:

1. Preprocesses the image with the same pipeline as training.
2. Applies the saved scaler.
3. Runs `model.predict_proba` to get probabilities.
4. Returns the top class and a small top-k list.
5. Optionally writes **trace files** (`.npz`) when `PREDICT_TRACE_DIR` is set.

Class names in the API are normalized so the UI always shows `deforested` and `aforested`.

---

## 3. Training the MLP model (`train.py`)

The heuristic predictor does not require training. You only need this section if you want to use or refresh the MLP model.

### 3.1 Basic training run

From the project root:

```powershell
python train.py
```

This trains an MLP on the default dataset paths and writes a bundle (by default `model.joblib` in the project root).

### 3.2 Custom dataset path / output model

```powershell
python train.py \
  --data ".\data\deforestation dataset" \
  --epochs 20 \
  --batch 128 \
  --out models\custom.joblib
```

You can run `python train.py --help` to see all available options.

If you save to `models\\custom.joblib` (recommended), `app.py` is already configured to look there when the MLP predictor is enabled.

---

## 4. Running the web app (`app.py`)

### 4.1 Choose predictor via environment variable

`app.py` reads `PREDICTOR_MODE` once at startup:

- `heuristic` (default): use `GreenPixelPredictor` (no model file required)
- `mlp`: use the saved scikit-learn bundle via `Predictor`

Example (PowerShell):

```powershell
# Heuristic predictor (default)
$env:PREDICTOR_MODE = "heuristic"

# Or enable the MLP model
# $env:PREDICTOR_MODE = "mlp"
```

Optional: enable tracing of feature vectors for the MLP predictor:

```powershell
$env:PREDICT_TRACE_DIR = "D:\\path\\to\\traces"
```

### 4.2 Start the Uvicorn server

From `CascadeProjects/windsurf-project`:

```powershell
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

or, as a module:

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open a browser at:

- <http://127.0.0.1:8000/>

Use the upload form on the home page to select a satellite tile (PNG/JPEG). The app saves a temporary copy, runs the chosen predictor, and renders `templates/result.html`.

---

## 5. Reading the result page

The result view lives in `templates/result.html` with styles in `static/style.css`.

Key pieces:

- **Prediction badge** – a large pill showing `DEFORESTED` or `AFORESTED`.
- **Green coverage analysis** – a card containing:
  - a two-column table of heuristic stats (green coverage, pixel counts, greenness deltas, thresholds)
  - rows separated by soft borders and spacing so it reads like a modern tabular panel

If the predictor raises an exception (bad image, missing model, etc.), the page shows a red **Prediction unavailable** banner with the error message instead of the analysis.

---

## 6. Troubleshooting

- **`Model not found at ...custom.joblib`**
  - Run `train.py` and ensure the output path matches `MODEL_PATH` in `app.py`.
  - Or switch back to the heuristic predictor with `PREDICTOR_MODE=heuristic`.

- **`Unable to import inference utilities`**
  - Run all commands from the `CascadeProjects/windsurf-project` root so that `infer.py`, `dataset.py` and friends are on `PYTHONPATH`.

- **`Prediction failed: ...` in the UI**
  - Check the message: it usually indicates a corrupted image, unsupported format, or an issue loading the model bundle.

- **Traces not being written (MLP mode)**
  - Confirm `PREDICT_TRACE_DIR` points to an existing, writable folder.
  - Make sure `PREDICTOR_MODE` is set to `mlp`; the heuristic predictor does not use feature vectors.

---

## 7. Project layout (quick reference)

- `app.py` – FastAPI app and `/predict` endpoint
- `infer.py` – `Predictor` (MLP) and `GreenPixelPredictor` (heuristic)
- `dataset.py` – image preprocessing and dataset loading for training
- `train.py` – training script for the scikit-learn MLP bundle
- `templates/index.html` – upload form
- `templates/result.html` – prediction + heuristic analysis view
- `static/style.css` – dark UI styling for the web app
- `models/` – optional folder for saved `.joblib` bundles
