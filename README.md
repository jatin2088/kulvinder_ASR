# Punjabi Word Pronunciation Corrector

This project trains a closed-vocabulary model for the 50 Punjabi words in `dataset/`.

Folder labels are parsed as:

- `N`: normal/correct pronunciation
- `D`: incorrect/dyslexia-like pronunciation

The GUI listens to one short spoken word, predicts the intended word only from this dataset's 50 words, displays the corrected Punjabi word, and plays a normal reference recording from the `N` folder.

## Train

```powershell
python train_sklearn_model.py --force-cache
```

Outputs are saved in `models/`:

- `sklearn_word_model.joblib`
- `manifest.json`
- `sklearn_training_summary.json`
- `sklearn_features_cache.npz`

## Run GUI

```powershell
python realtime_gui.py
```

Use **Record Word** for automatic speech detection. If the app says no audio was detected, use **Manual 2 Sec Record** or lower the silence threshold.

## Run Web App Locally

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The web app records audio in the browser/mobile device and sends a short WAV to the server for prediction.

## Deploy To Render

Push this folder to GitHub, then create a new Render Web Service from the repo.

Render can use `render.yaml` automatically. Manual settings:

```text
Build Command: pip install -r requirements-render.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
```

No secret key is required. Mobile microphone recording needs HTTPS; Render provides HTTPS after deployment.

Current trained model:

- Validation word accuracy: `65.5%`
- Validation `D`/`N` pronunciation quality accuracy: `94.9%`
- Broken WAV files skipped: `6`

## Notes

The GUI does not use general speech recognition. It is intentionally limited to the known dataset words, which is better for short one-word Punjabi child recordings.
