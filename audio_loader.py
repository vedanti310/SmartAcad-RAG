# audio_loader.py

from faster_whisper import WhisperModel

# Load model once (IMPORTANT for speed)
model = WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe_audio(audio_path):
    try:
        segments, info = model.transcribe(audio_path)

        full_text = ""
        for segment in segments:
            full_text += segment.text + " "

        return full_text.strip()

    except Exception as e:
        print(f"Transcription error: {e}")
        return ""