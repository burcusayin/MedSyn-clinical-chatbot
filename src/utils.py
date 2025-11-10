from datetime import datetime
from enum import Enum

def read_prompt_from_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def serialize_dict(data):
    if isinstance(data, list):
        return [serialize_dict(item) for item in data]
    elif isinstance(data, dict):
        return {key: serialize_dict(value) for key, value in data.items()}
    elif isinstance(data, Enum):
        return str(data.value)
    elif isinstance(data, datetime):
        return data.isoformat()
    elif data is None:
        return ''
    else:
        return data