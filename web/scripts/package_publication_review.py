"""Create a local review ZIP, never a registered/published Sites version."""
from pathlib import Path
import hashlib
import json
import zipfile

project = Path(__file__).resolve().parents[2]
review = project / "publication-review"
audit = json.loads((review / "publication-audit.json").read_text(encoding="utf-8"))
paths = [Path("dist") / item["path"] for item in audit["files"]]
paths += [Path("README.md"), Path("publication-audit.json"), Path("hosting-manifest.template.json")]
paths += [Path("qa/production-qa.json"), Path("qa/launcher-identity.json")]
for item in audit["files"]:
    assert hashlib.sha256((review / "dist" / item["path"]).read_bytes()).hexdigest() == item["sha256"]
target = project / "signbridge-publication-review.zip"
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for relative in sorted(paths):
        resolved = (review / relative).resolve()
        assert resolved.is_relative_to(review.resolve())
        archive.write(resolved, relative.as_posix())
with zipfile.ZipFile(target) as archive:
    assert archive.testzip() is None
    assert set(archive.namelist()) == {item.as_posix() for item in paths}
    assert not any(name.lower().endswith((".png", ".jpg", ".jpeg", ".mp4", ".webm")) for name in archive.namelist())
digest = hashlib.sha256(target.read_bytes()).hexdigest()
(project / "signbridge-publication-review.zip.sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
print(json.dumps({"archive": str(target), "sha256": digest, "bytes": target.stat().st_size,
                  "entries": len(paths), "published": False, "raw_media_included": False}))
