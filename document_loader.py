import os
from audio_loader import transcribe_audio
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from PIL import Image
import pytesseract

cd

# If using Windows, uncomment and adjust path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    try:
        # ---------------- PDF ----------------
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents if documents else []

        # ---------------- DOCX ----------------
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return documents if documents else []

        # ---------------- AUDIO ----------------
        elif ext in [".wav", ".mp3", ".m4a"]:
            text = transcribe_audio(file_path)

            if text and text.strip():
                return [Document(
                    page_content=text,
                    metadata={"source": file_path, "type": "audio"}
                )]
            else:
                return []

        # ---------------- IMAGE (OCR) ----------------
        elif ext in [".png", ".jpg", ".jpeg"]:
            image = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image)

            if extracted_text and extracted_text.strip():
                return [Document(
                    page_content=extracted_text,
                    metadata={"source": file_path, "type": "image"}
                )]
            else:
                return []

        # ---------------- UNSUPPORTED ----------------
        else:
            return []

    except Exception as e:
        print(f"Error loading document: {e}")
        return []