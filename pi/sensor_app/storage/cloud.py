"""Cloud sink interface. StubCloudSink proves the push contract locally without
any real Supabase/cloud credentials."""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Union


class CloudSink(ABC):
    @abstractmethod
    def push(self, rows: Iterable[dict]) -> None:
        raise NotImplementedError


class StubCloudSink(CloudSink):
    """Appends pushed rows as JSON lines to a local file."""

    def __init__(self, jsonl_path: Union[str, Path]):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def push(self, rows: Iterable[dict]) -> None:
        with open(self.jsonl_path, "a") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
