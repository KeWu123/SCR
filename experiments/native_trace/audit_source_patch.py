"""Check extracted sources against the tagged archive and retained patches."""
from pathlib import Path
import hashlib
import json

from prepare_source import ARCHIVE, ARCHIVE_SHA256, ROOT, SOURCE, expected_sources


def main():
    expected = expected_sources()
    for name, data in expected.items():
        path = SOURCE / name
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError("Extracted source differs from the retained patch: " + name)
    official = json.loads((ROOT / "official_source_hashes.json").read_text(encoding="utf-8"))
    changed = sorted(name for name, data in expected.items()
                     if hashlib.sha256(data).hexdigest() != official[name])
    assert changed == ["faiss/impl/HNSW.cpp", "faiss/impl/HNSW.h"], changed
    cpp = expected["faiss/impl/HNSW.cpp"].decode("utf-8")
    assert cpp.count("const bool scr_first_visit = vt.set(v1);") == 2
    assert cpp.count("counter += scr_first_visit ? 1 : 0;") == 2
    assert cpp.count("SCRNativeExpansion event") == 1
    report = {
        "passed": True,
        "official_files_hashed": len(official),
        "changed_files": changed,
        "official_archive_sha256": ARCHIVE_SHA256,
        "source_url": "https://github.com/facebookresearch/faiss/archive/refs/tags/v1.15.1.zip",
        "instrumented_sha256": {name: hashlib.sha256(expected[name]).hexdigest() for name in changed},
        "scope": "Exact tagged-source and patch comparison; native numerical equivalence is checked by the harness and retained traces.",
    }
    (ROOT / "source_patch_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "official_files_hashed": len(official), "changed_files": changed}))


if __name__ == "__main__":
    main()
