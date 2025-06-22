from PIL import Image
import pytesseract

def extract_text_from_image(file):
    """
    Takes a file-like object (uploaded image) and returns extracted text.
    """
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text.strip()
