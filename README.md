# Punjabi Word Pronunciation Corrector

This project trains a closed-vocabulary model for the 50 Punjabi words in `dataset/`.

Folder labels are parsed as:

- `N`: normal/correct pronunciation
- `D`: incorrect/dyslexia-like pronunciation

The GUI listens to one short spoken word, predicts the intended word only from this dataset's 50 words, displays the corrected Punjabi word, and plays a normal reference recording from the `N` folder.

## Train

```powershell
python generate_synthetic_tts_dataset.py
python train_sklearn_model.py --force-cache --noise-augment --noise-copies 2
python train_mlp_model.py --epochs 60 --batch-size 256 --d-repeat 3 --synthetic-repeat 2
```

Outputs are saved in `models/`:

- `sklearn_word_model.joblib`
- `mlp_word_model_np.npz`
- `reference_vectors_mlp.npz`
- `synthetic_tts_features.npz`
- `manifest.json`
- `sklearn_training_summary.json`
- `mlp_training_summary.json`
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

Useful test/export URLs:

```text
/health
/results.csv
/recordings/<recording_id>
```

`/health` should report `model: mlp_numpy` by default. `/results.csv` contains every test result, including top alternatives and the saved recording ID. Use `/recordings/<recording_id>` to download the exact WAV that the phone sent.

If a prediction is wrong, choose the correct word in the feedback control after the result and click **Save Correct Label**. Those labeled phone recordings are used immediately by the server as calibration examples for future predictions.

## Real Deployment Plan

The runtime screen is free-speak after the child voice map is ready: the child presses **Start Speaking**, says any one of the 50 words, presses **Stop & Correct**, and the app displays and plays the corrected Punjabi word.

Child-specific learning is mandatory in production mode. The app will not make random generic guesses before setup:

1. Open `/calibrate` or tap **Teach Child Voice**.
2. Record at least 3 clear samples for each word using the same child and phone.
3. Return to `/` for free-speak detection.

The normal runtime screen still does not show a target word before speaking. The target words are only shown during teacher setup.

Calibration requires persistent storage on Render. The Blueprint includes a disk mounted at `/opt/render/project/src/data`; if Render rejects the disk on the free plan, upgrade the service plan or add the disk manually from the Render dashboard.

## Deploy To Render

Push this folder to GitHub, then create a new Render Web Service from the repo.

Render can use `render.yaml` automatically. Manual settings:

```text
Build Command: pip install -r requirements-render.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
```

No secret key is required. Mobile microphone recording needs HTTPS; Render provides HTTPS after deployment.

Current trained model:

- Default runtime word model: `mlp_numpy` neural closed-vocabulary recognizer
- Fallback classical model: set `WORD_MODEL_KIND=sklearn`
- Production mode blocks predictions until the child/phone voice map is complete. Set `VOICE_MAP_ONLY=0` only for developer smoke tests.
- MLP validation clean word accuracy: `71.3%`
- MLP validation noisy word accuracy: `66.1%`
- Synthetic clean TTS dataset: `750` WAVs, tested `750/750`
- Local Flask smoke test: clean `46/50`, noisy `42/50`, accepted noisy correct `41/46`
- Static reference test: clean `50/50`, noisy `45/50`
- Simulated complete voice-map test: clean `50/50`, noisy `49/50`
- Validation `D`/`N` pronunciation quality accuracy: `94.9%`
- Broken WAV files skipped: `6`

## Notes

The GUI does not use general speech recognition. It is intentionally limited to the known dataset words, which is better for short one-word Punjabi child recordings.
