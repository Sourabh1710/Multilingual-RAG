import re
import unicodedata


def clean_and_normalize_text(text: str) -> str:
    """
    Cleans and normalizes text extracted from documents.
    Specifically handles Devanagari/Latin Unicode normalization (NFC)
    to prevent cross-lingual matching failures, and standardizes whitespace.
    """
    if not text:
        return ""

    # 1. Unicode Normalization: Force NFC composition to standardize characters like nuktas
    normalized_text = unicodedata.normalize("NFC", text)

    # 2. Standardize whitespace: Replace tabs, non-breaking spaces (\xa0) with standard spaces
    cleaned_text = re.sub(r"[\t\xa0]", " ", normalized_text)

    # 3. Collapse multiple spaces: Replace duplicate spaces with a single space
    cleaned_text = re.sub(r" +", " ", cleaned_text)

    # 4. Collapse excessive blank lines: Avoid wasting context window space on empty newlines
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    # 5. Strip leading and trailing whitespace from the final document
    return cleaned_text.strip()
