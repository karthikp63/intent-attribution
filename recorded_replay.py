"""Replay selected published model responses without an API key.

The checked-in fixture contains model text and a digest of each exact prompt.
The digest prevents a changed prompt or nonce from silently reusing an old
response. Live provider caches remain private and ignored by git.
"""

import hashlib
import json
from pathlib import Path

import proposer as P


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "published_responses.json"


def ask(provider, model, max_tokens=24576):
    with FIXTURE.open() as f:
        records = json.load(f)

    def replay(prompt, nonce=None):
        key = Path(P._cache_path(provider, model, prompt, nonce, max_tokens)).name
        entry = records[key]
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        if entry["prompt_sha256"] != digest:
            raise ValueError("recorded response prompt digest mismatch")
        return entry["text"], entry["truncated"]

    return replay
