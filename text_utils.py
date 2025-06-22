def extract_text_from_txt(file):
    """
    Reads and returns content from a .txt file (uploaded as file-like object).
    """
    text = file.read().decode("utf-8", errors="ignore")
    return text.strip()
