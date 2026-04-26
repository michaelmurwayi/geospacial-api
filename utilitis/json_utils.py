import json
import re


def safe_extract_json(text: str):

    if not text:
        return {}

    # Extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}

    raw = match.group()

    # Fix common LLM issues
    raw = raw.replace("'", '"')
    raw = re.sub(r'(\w+):', r'"\1":', raw)
    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r',\s*]', ']', raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}