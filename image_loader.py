from PIL import Image
import pytesseract
from langchain.schema import Document

def load_image(file_path):
    image = Image.open(file_path)
    extracted_text = pytesseract.image_to_string(image)

    if not extracted_text.strip():
        return []

    return [Document(
        page_content=extracted_text,
        metadata={"source": file_path}
    )]