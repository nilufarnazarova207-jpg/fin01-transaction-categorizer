import re

def clean_text(s):
    s = str(s).upper()
    s = re.sub(r'CHECK\s*CRD\s*PURCHASE', ' ', s)
    s = re.sub(r'HSA\s*CARD\s*PURCHASE', ' ', s)
    s = re.sub(r'RECURRING\s*PMT', ' ', s)
    s = re.sub(r'\bMCC\b', ' ', s)
    s = re.sub(r'[^A-Z\s]', ' ', s)                  # strip digits/punct
    s = re.sub(r'\bX{3,}\b', ' ', s)                  # masked card digit remnants
    s = re.sub(r'\s+', ' ', s).strip()
    return s
