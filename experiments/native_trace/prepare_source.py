"""Extract FAISS 1.15.1 and apply the two retained observational patches."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "faiss-v1.15.1.zip"
SOURCE = ROOT / "faiss-1.15.1"
ARCHIVE_SHA256 = "f46290cc461f5ae88bc52f400e2066c537746328e494f79faa1fa719815e7b66"


def apply_patch(original, patch):
    """Apply a unified diff, requiring every original context line to match."""
    import re

    original = original.decode("utf-8").splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    output = []
    cursor = 0
    index = 2
    while index < len(lines):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", lines[index])
        if match is None:
            raise ValueError("Invalid unified patch header")
        start = int(match.group(1)) - 1
        if start < cursor:
            raise ValueError("Overlapping patch hunks")
        output.extend(original[cursor:start])
        cursor = start
        index += 1
        removed = added = 0
        while index < len(lines) and not lines[index].startswith("@@ "):
            line = lines[index]
            if not line or line[0] not in " +-":
                raise ValueError("Invalid patch line")
            if line[0] in " -":
                if cursor >= len(original) or original[cursor] != line[1:]:
                    raise ValueError("Patch context differs from the tagged source")
                cursor += 1
                removed += 1
            if line[0] in " +":
                output.append(line[1:])
                added += 1
            index += 1
        if removed != int(match.group(2) or 1) or added != int(match.group(4) or 1):
            raise ValueError("Patch hunk line counts differ")
    output.extend(original[cursor:])
    return "".join(output).encode("utf-8")


def expected_sources():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("FAISS archive hash mismatch")
    expected = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or ".." in name.parts or name.parts[0] != SOURCE.name:
                raise ValueError("Archive entry escapes the source directory")
            if member.is_dir():
                continue
            relative = PurePosixPath(*name.parts[1:]).as_posix()
            expected[relative] = archive.read(member)
    official = json.loads((ROOT / "official_source_hashes.json").read_text(encoding="utf-8"))
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in expected.items()}
    if hashes != official:
        raise ValueError("Tagged archive differs from the recorded source inventory")
    for filename in ("HNSW.h", "HNSW.cpp"):
        name = "faiss/impl/" + filename
        patch = (ROOT / (filename + ".patch")).read_text(encoding="utf-8")
        expected[name] = apply_patch(expected[name], patch)
    return expected


def main():
    expected = expected_sources()
    for name, data in expected.items():
        target = SOURCE / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Existing source differs; use a fresh extraction: " + name)
    for name, data in expected.items():
        target = SOURCE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
    print("Prepared FAISS 1.15.1: 1,097 source files, two observational patches.")


if __name__ == "__main__":
    main()
