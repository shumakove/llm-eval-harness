import hashlib, json, subprocess
from pathlib import Path

def _git_revision() -> dict:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return {"commit": sha, "dirty": dirty}

def _digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]

def write_snapshot(path: Path, manifest: dict, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def read_snapshot(path: Path) -> tuple[dict, list[dict]]:
    """Читает JSONL-снапшот: первая строка — манифест, остальные — записи."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"пустой снапшот: {path}")
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


class SnapshotWriter:
    """Пишет JSONL-снапшот по мере появления записей.

    В отличие от write_snapshot не держит записи в памяти и сбрасывает буфер
    после каждой строки: прогон, упавший на 60-м кейсе, сохраняет первые 59.
    Для дорогих прогонов это разница между потерей минуты и потерей всех
    оплаченных вызовов модели.
    """

    def __init__(self, path: Path, manifest: dict) -> None:
        self._path = path
        self._manifest = manifest
        self._file = None

    def __enter__(self) -> "SnapshotWriter":
        self._file = self._path.open("w", encoding="utf-8")
        self._write(self._manifest)
        return self

    def __exit__(self, *exc_info) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def append(self, record: dict) -> None:
        self._write(record)

    def _write(self, payload: dict) -> None:
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._file.flush()
