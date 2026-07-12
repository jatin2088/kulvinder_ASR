import argparse
import asyncio
import csv
import json
import re
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample

try:
    import edge_tts
    import miniaudio
except ImportError:
    vendor = Path(__file__).resolve().parent / ".tts_vendor"
    if vendor.exists():
        sys.path.insert(0, str(vendor))
        import edge_tts
        import miniaudio
    else:
        raise

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DEFAULT_VOICES = [
    "en-US-EmmaMultilingualNeural",
    "en-US-AvaMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
    "en-US-BrianMultilingualNeural",
    "en-AU-WilliamMultilingualNeural",
]

DEFAULT_VARIANTS = [
    ("normal", 1.00),
    ("slow", 1.12),
    ("fast", 0.90),
]

TTS_PROMPTS = {
    "ਊਠ": "ऊठ",
    "ਉਂਗਲ": "उंगली",
    "ਅੱਖ": "आंख",
    "ਅੰਬ": "अंब",
    "ਇੱਟ": "ईंट",
    "ਇੰਜਣ": "इंजन",
    "ਸੇਬ": "सेब",
    "ਹਾਥੀ": "हाथी",
    "ਹੱਥ": "हथ",
    "ਕੁਰਸੀ": "कुरसी",
    "ਕਾਰ": "कार",
    "ਖੰਭ": "खंभ",
    "ਖੁਰਪਾ": "खुरपा",
    "ਗਮਲਾ": "गमला",
    "ਗਾਜਰ": "गाजर",
    "ਘੋੜਾ": "घोड़ा",
    "ਚਿੜੀ": "चिड़ी",
    "ਚਮਚਾ": "चमचा",
    "ਛਤਰੀ": "छतरी",
    "ਛੱਲੀ": "छल्ली",
    "ਜੱਗ": "जग",
    "ਜੁੱਤੀ": "जूती",
    "ਝੰਡਾ": "झंडा",
    "ਝਾੜੂ": "झाड़ू",
    "ਟੱਲੀ": "टली",
    "ਠੇਲਾ": "ठेला",
    "ਠੋਡੀ": "ठोड़ी",
    "ਡੱਡੂ": "डड्डू",
    "ਡਮਰੂ": "डमरू",
    "ਢੋਲ": "ढोल",
    "ਢੱਕਣ": "ढक्कन",
    "ਸਾਬਣ": "साबुन",
    "ਤੀਰ": "तीर",
    "ਤਿਤਲੀ": "तितली",
    "ਥੈਲਾ": "थैला",
    "ਥਾਲੀ": "थाली",
    "ਦੀਵਾ": "दीवा",
    "ਦੰਦ": "दंद",
    "ਧਾਗਾ": "धागा",
    "ਧੋਬੀ": "धोबी",
    "ਪੌੜੀ": "पौड़ी",
    "ਪਤੰਗ": "पतंग",
    "ਫੁੱਲ": "फूल",
    "ਫ਼ਲ": "फल",
    "ਬੱਸ": "बस",
    "ਬਿੱਲੀ": "बिल्ली",
    "ਭਿੰਡੀ": "भिंडी",
    "ਭੇਡ": "भेड़",
    "ਮੱਛੀ": "मछली",
    "ਮਟਰ": "मटर",
}


def safe_name(text):
    cleaned = re.sub(r"[^\w\u0a00-\u0a7f-]+", "_", text, flags=re.UNICODE).strip("_")
    return cleaned or "word"


def load_words(manifest_path):
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest["words"]


def samples_from_mp3(mp3_path):
    decoded = miniaudio.decode_file(
        str(mp3_path),
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=16000,
    )
    return np.frombuffer(decoded.samples, dtype="<i2").copy()


def write_wav_samples(samples, wav_path):
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples, dtype=np.int16)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples.tobytes())


def stretch_samples(samples, factor):
    if abs(factor - 1.0) < 1e-6:
        return samples
    target = max(1, int(round(len(samples) * factor)))
    stretched = resample(samples.astype(np.float32), target)
    return np.clip(stretched, -32768, 32767).astype(np.int16)


async def synthesize_base_samples(text, voice):
    with tempfile.TemporaryDirectory() as tmp:
        mp3_path = Path(tmp) / "speech.mp3"
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(str(mp3_path))
        return samples_from_mp3(mp3_path)


async def synthesize_dataset(args):
    words = load_words(args.manifest)
    voices = args.voices or DEFAULT_VOICES
    variants = DEFAULT_VARIANTS if args.variants == "default" else [("normal", 1.00)]
    out_dir = Path(args.out)
    rows = []

    selected_words = words[: args.limit] if args.limit else words
    total = len(selected_words) * len(voices) * len(variants)
    done = 0

    for word_id, word in enumerate(selected_words):
        tts_text = TTS_PROMPTS.get(word, word)
        folder = out_dir / f"{word_id + 1} {word} N"
        for voice in voices:
            voice_tag = voice.replace("Neural", "").replace("Multilingual", "").replace("-", "_")
            try:
                print(f"synth base {word} as {tts_text} {voice}", flush=True)
                base_samples = await synthesize_base_samples(tts_text, voice)
            except Exception as exc:
                print(f"  failed base: {exc}", flush=True)
                done += len(variants)
                continue

            for variant_name, stretch in variants:
                filename = f"tts_{voice_tag}_{variant_name}.wav"
                wav_path = folder / filename
                done += 1
                if wav_path.exists() and not args.force:
                    print(f"[{done}/{total}] skip {wav_path}")
                else:
                    print(f"[{done}/{total}] write {word} {voice} {variant_name}", flush=True)
                    try:
                        write_wav_samples(stretch_samples(base_samples, stretch), wav_path)
                    except Exception as exc:
                        print(f"  failed: {exc}", flush=True)
                        continue
                rows.append(
                    {
                        "word_id": word_id,
                        "word": word,
                        "tts_text": tts_text,
                        "quality": "N",
                        "voice": voice,
                        "variant": variant_name,
                        "stretch": stretch,
                        "path": str(wav_path),
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "synthetic_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["word_id", "word", "tts_text", "quality", "voice", "variant", "stretch", "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} rows to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Punjabi TTS WAV dataset for the 50-word app.")
    parser.add_argument("--manifest", default="models/manifest.json")
    parser.add_argument("--out", default="synthetic_tts_dataset")
    parser.add_argument("--voices", nargs="*", default=None)
    parser.add_argument("--variants", choices=["default", "normal"], default="default")
    parser.add_argument("--limit", type=int, default=0, help="Generate only the first N words for testing")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(synthesize_dataset(args))


if __name__ == "__main__":
    main()
