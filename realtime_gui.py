import json
import queue
import threading
import time
import wave
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, StringVar, Tk, filedialog, messagebox
from tkinter import ttk

import numpy as np
import sounddevice as sd
import torch
from scipy.io import wavfile
import joblib

from model_utils import SAMPLE_RATE, load_audio, log_mel_features, normalize_audio, sklearn_feature_vector_from_logmel
from train_model import WordCorrectionNet


MODEL_PATH = Path("models") / "word_correction_model.pt"
SKLEARN_MODEL_PATH = Path("models") / "sklearn_word_model.joblib"
MANIFEST_PATH = Path("models") / "manifest.json"
LAST_RECORDING_PATH = Path("models") / "last_recording.wav"


class AudioRecorder:
    def __init__(self, device=None):
        self.device = device

    def record_auto(
        self,
        threshold=0.020,
        max_seconds=3.0,
        silence_seconds=0.65,
        prebuffer_seconds=0.30,
        chunk_seconds=0.08,
        level_callback=None,
    ):
        chunk_samples = int(SAMPLE_RATE * chunk_seconds)
        silence_chunks = max(1, int(silence_seconds / chunk_seconds))
        prebuffer_chunks = max(1, int(prebuffer_seconds / chunk_seconds))
        max_chunks = max(1, int(max_seconds / chunk_seconds))

        audio_chunks = []
        prebuffer = []
        started = False
        silent_count = 0
        start_time = time.time()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=chunk_samples,
        ) as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_samples)
                chunk = np.asarray(chunk[:, 0], dtype=np.float32)
                rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
                if level_callback:
                    level_callback(min(1.0, rms / max(threshold * 3.0, 1e-6)))

                if not started:
                    prebuffer.append(chunk.copy())
                    prebuffer = prebuffer[-prebuffer_chunks:]
                    if rms >= threshold:
                        started = True
                        audio_chunks.extend(prebuffer)
                        silent_count = 0
                    continue

                audio_chunks.append(chunk.copy())
                if rms < threshold * 0.75:
                    silent_count += 1
                else:
                    silent_count = 0
                if silent_count >= silence_chunks and time.time() - start_time > 0.45:
                    break

        if not audio_chunks:
            return np.asarray([], dtype=np.float32)
        return normalize_audio(np.concatenate(audio_chunks))

    def record_fixed(self, seconds=2.0, level_callback=None):
        total_samples = int(SAMPLE_RATE * seconds)
        chunk_samples = int(SAMPLE_RATE * 0.08)
        chunks = []
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=chunk_samples,
        ) as stream:
            remaining = total_samples
            while remaining > 0:
                take = min(chunk_samples, remaining)
                chunk, _ = stream.read(take)
                chunk = np.asarray(chunk[:, 0], dtype=np.float32)
                chunks.append(chunk)
                remaining -= take
                if level_callback:
                    rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
                    level_callback(min(1.0, rms / 0.08))
        return normalize_audio(np.concatenate(chunks))


class CorrectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Punjabi Word Pronunciation Corrector")
        self.root.geometry("760x520")
        self.root.minsize(720, 480)

        self.model = None
        self.quality_model = None
        self.model_kind = None
        self.words = []
        self.references = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ui_queue = queue.Queue()
        self.is_busy = False
        self.selected_device = StringVar()
        self.threshold = StringVar(value="0.020")
        self.result_word = StringVar(value="Train/load the model, then record a word.")
        self.status = StringVar(value="Ready")
        self.confidence = StringVar(value="")
        self.quality = StringVar(value="")

        self.build_ui()
        self.load_model()
        self.refresh_devices()
        self.root.after(100, self.process_ui_queue)

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 24, "bold"))
        style.configure("Word.TLabel", font=("Segoe UI", 42, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 13, "bold"), padding=10)

        main = ttk.Frame(self.root, padding=18)
        main.pack(fill=BOTH, expand=True)

        top = ttk.Frame(main)
        top.pack(fill="x")
        ttk.Label(top, text="Punjabi Word Pronunciation Corrector", style="Title.TLabel").pack(side=LEFT)
        ttk.Button(top, text="Load Model", command=self.load_model).pack(side=RIGHT)

        controls = ttk.LabelFrame(main, text="Microphone", padding=12)
        controls.pack(fill="x", pady=(18, 10))

        ttk.Label(controls, text="Device").grid(row=0, column=0, sticky="w")
        self.device_combo = ttk.Combobox(controls, textvariable=self.selected_device, state="readonly", width=62)
        self.device_combo.grid(row=0, column=1, sticky="we", padx=8)
        ttk.Button(controls, text="Refresh", command=self.refresh_devices).grid(row=0, column=2)

        ttk.Label(controls, text="Silence threshold").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.threshold, width=12).grid(row=1, column=1, sticky="w", padx=8, pady=(10, 0))
        controls.columnconfigure(1, weight=1)

        self.level = ttk.Progressbar(controls, orient="horizontal", mode="determinate", maximum=100)
        self.level.grid(row=2, column=0, columnspan=3, sticky="we", pady=(12, 0))

        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=12)
        ttk.Button(actions, text="Record Word", style="Big.TButton", command=self.record_auto).pack(side=LEFT, padx=(0, 8))
        ttk.Button(actions, text="Manual 2 Sec Record", style="Big.TButton", command=self.record_fixed).pack(side=LEFT, padx=8)
        ttk.Button(actions, text="Replay Correct Pronunciation", command=self.play_reference).pack(side=LEFT, padx=8)
        ttk.Button(actions, text="Open WAV Test", command=self.open_wav_test).pack(side=RIGHT)

        result = ttk.LabelFrame(main, text="Correction", padding=18)
        result.pack(fill=BOTH, expand=True, pady=(8, 0))
        ttk.Label(result, textvariable=self.result_word, style="Word.TLabel", anchor="center").pack(fill=BOTH, expand=True)
        ttk.Label(result, textvariable=self.confidence, font=("Segoe UI", 13)).pack()
        ttk.Label(result, textvariable=self.quality, font=("Segoe UI", 13)).pack(pady=(4, 0))
        ttk.Label(result, textvariable=self.status, font=("Segoe UI", 10)).pack(pady=(18, 0), anchor="w")

        self.log = ttk.Treeview(main, columns=("word", "confidence", "quality"), show="headings", height=5)
        self.log.heading("word", text="Word")
        self.log.heading("confidence", text="Confidence")
        self.log.heading("quality", text="Detected")
        self.log.column("word", width=260)
        self.log.column("confidence", width=120)
        self.log.column("quality", width=160)
        self.log.pack(fill="x", pady=(12, 0))

    def load_model(self):
        if SKLEARN_MODEL_PATH.exists():
            artifact = joblib.load(SKLEARN_MODEL_PATH)
            self.model = artifact["word_model"]
            self.quality_model = artifact.get("quality_model")
            self.words = artifact["words"]
            self.references = artifact.get("references", {})
            self.model_kind = "sklearn"
            if MANIFEST_PATH.exists():
                with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                    self.references.update(manifest.get("references", {}))
            self.status.set(f"Loaded sklearn correction model with {len(self.words)} words.")
            return

        if not MODEL_PATH.exists():
            self.status.set("Model not found. Run: python train_sklearn_model.py --force-cache")
            return
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        self.words = checkpoint["words"]
        self.references = checkpoint.get("references", {})
        self.model = WordCorrectionNet(num_words=len(self.words)).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model_kind = "torch"
        if MANIFEST_PATH.exists():
            with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
                self.references.update(manifest.get("references", {}))
        self.status.set(f"Loaded model with {len(self.words)} words.")

    def refresh_devices(self):
        try:
            devices = sd.query_devices()
            inputs = []
            for idx, info in enumerate(devices):
                if info.get("max_input_channels", 0) > 0:
                    inputs.append(f"{idx}: {info['name']}")
            self.device_combo["values"] = inputs
            if inputs and not self.selected_device.get():
                self.selected_device.set(inputs[0])
        except Exception as exc:
            self.status.set(f"Cannot list microphones: {exc}")

    def get_device_index(self):
        value = self.selected_device.get()
        if ":" not in value:
            return None
        return int(value.split(":", 1)[0])

    def set_busy(self, busy):
        self.is_busy = busy

    def record_auto(self):
        if self.is_busy:
            return
        self.start_recording(auto=True)

    def record_fixed(self):
        if self.is_busy:
            return
        self.start_recording(auto=False)

    def start_recording(self, auto=True):
        if self.model is None:
            messagebox.showwarning("Model missing", "Please train/load the model first.")
            return
        self.set_busy(True)
        self.status.set("Listening...")
        self.level["value"] = 0
        thread = threading.Thread(target=self.record_worker, args=(auto,), daemon=True)
        thread.start()

    def record_worker(self, auto):
        try:
            threshold = float(self.threshold.get())
            recorder = AudioRecorder(device=self.get_device_index())

            def level_callback(value):
                self.ui_queue.put(("level", value))

            if auto:
                audio = recorder.record_auto(threshold=threshold, level_callback=level_callback)
            else:
                audio = recorder.record_fixed(seconds=2.0, level_callback=level_callback)

            if audio.size < int(0.18 * SAMPLE_RATE):
                self.ui_queue.put(("error", "No audio detected. Try Manual 2 Sec Record or lower the threshold."))
                return

            self.save_last_recording(audio)
            prediction = self.predict(audio)
            self.ui_queue.put(("prediction", prediction))
        except Exception as exc:
            self.ui_queue.put(("error", str(exc)))

    def save_last_recording(self, audio):
        LAST_RECORDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = np.clip(audio, -1.0, 1.0)
        data = (data * 32767.0).astype(np.int16)
        wavfile.write(str(LAST_RECORDING_PATH), SAMPLE_RATE, data)

    def predict(self, audio):
        features = log_mel_features(audio)
        if self.model_kind == "sklearn":
            vec = sklearn_feature_vector_from_logmel(features).reshape(1, -1)
            word_id = int(self.model.predict(vec)[0])
            confidence = self.estimate_sklearn_confidence(self.model, vec)
            if self.quality_model is not None:
                q_id = int(self.quality_model.predict(vec)[0])
                quality_confidence = self.estimate_sklearn_confidence(self.quality_model, vec)
            else:
                q_id = 1
                quality_confidence = 0.0
            return {
                "word": self.words[word_id],
                "confidence": confidence,
                "quality": "Normal / correct" if q_id == 1 else "Incorrect / dyslexia-like",
                "quality_confidence": quality_confidence,
            }

        xb = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits_word, logits_quality = self.model(xb)
            probs = torch.softmax(logits_word, dim=1)[0]
            q_probs = torch.softmax(logits_quality, dim=1)[0]
            conf, word_id = torch.max(probs, dim=0)
            q_id = int(torch.argmax(q_probs).item())
        return {
            "word": self.words[int(word_id.item())],
            "confidence": float(conf.item()),
            "quality": "Normal / correct" if q_id == 1 else "Incorrect / dyslexia-like",
            "quality_confidence": float(q_probs[q_id].item()),
        }

    def estimate_sklearn_confidence(self, model, vec):
        if hasattr(model, "decision_function"):
            scores = model.decision_function(vec)
            scores = np.asarray(scores, dtype=np.float64)
            if scores.ndim == 1:
                scores = np.stack([-scores, scores], axis=1)
            scores = scores - np.max(scores, axis=1, keepdims=True)
            probs = np.exp(scores)
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            return float(np.max(probs[0]))
        return 1.0

    def process_ui_queue(self):
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "level":
                    self.level["value"] = int(payload * 100)
                elif kind == "prediction":
                    self.handle_prediction(payload)
                    self.set_busy(False)
                elif kind == "error":
                    self.status.set(payload)
                    self.set_busy(False)
        except queue.Empty:
            pass
        self.root.after(100, self.process_ui_queue)

    def handle_prediction(self, prediction):
        word = prediction["word"]
        self.result_word.set(word)
        self.confidence.set(f"Confidence: {prediction['confidence'] * 100:.1f}%")
        self.quality.set(
            f"Detected pronunciation: {prediction['quality']} "
            f"({prediction['quality_confidence'] * 100:.1f}%)"
        )
        self.status.set(f"Saved last recording to {LAST_RECORDING_PATH}")
        self.log.insert(
            "",
            0,
            values=(word, f"{prediction['confidence'] * 100:.1f}%", prediction["quality"]),
        )
        self.play_reference(word=word)

    def play_reference(self, word=None):
        word = word or self.result_word.get()
        path = self.references.get(word)
        if not path:
            self.status.set("No normal reference recording found for this word.")
            return
        try:
            audio = load_audio(path)
            audio = normalize_audio(audio)
            sd.play(audio, SAMPLE_RATE)
            self.status.set(f"Playing correct pronunciation for: {word}")
        except Exception as exc:
            self.status.set(f"Cannot play reference: {exc}")

    def open_wav_test(self):
        if self.model is None:
            messagebox.showwarning("Model missing", "Please train/load the model first.")
            return
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        try:
            audio = load_audio(path)
            prediction = self.predict(audio)
            self.handle_prediction(prediction)
        except Exception as exc:
            messagebox.showerror("WAV test failed", str(exc))


def main():
    root = Tk()
    app = CorrectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
