import json, hashlib, uuid, time
from pathlib import Path

def to_json(o) -> str:
    return json.dumps(o, separators=(",",":"), ensure_ascii=False)

def from_json(s: str):
    try: return json.loads(s) if s else None
    except Exception: return None

def uid() -> str: return str(uuid.uuid4())
def now_ts() -> int: return int(time.time())

def file_hash(p: Path) -> str:
    try:
        b = p.read_bytes()
        return hashlib.sha256(b).hexdigest()[:16]
    except Exception:
        return ""
