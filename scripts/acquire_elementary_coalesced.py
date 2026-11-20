"""Complete the Elementary23 reference collection using paced, coalesced ZIP ranges."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import threading
import time
import urllib.error
import urllib.request

from acquire_elementary_library_expansion import OUTPUT, WORKBOOK, acquire, inventory, load, official, sha
from acquire_fresh_continuous import write_manifest


FINAL = OUTPUT / "coalesced-2331"
BASE = OUTPUT / "words-200"
PARTIAL = OUTPUT / "mixed-2512"
WINDOW = 8 * 1024**2
MAX_SPAN = WINDOW + 2 * 1024**2
CALL_COST = 2_000_000
TOTAL_CAP = 2_000_000_000
PRIOR_RANGE_MAX = 145_000_000
EXPECTED_NEW = 2712
RESERVE = 80
WORKERS = 8
REQUEST_INTERVAL = .70


class Pace:
    def __init__(self):
        self.lock = threading.Lock()
        self.next_request = 0.
        self.server_pause = 0.
        self.successful_requests = 0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                delay = max(self.server_pause, self.next_request) - now
                if delay <= 0:
                    self.next_request = now + REQUEST_INTERVAL
                    return
            time.sleep(delay)

    def pause(self, seconds):
        with self.lock:
            self.server_pause = max(self.server_pause, time.monotonic() + seconds)


def response_delay(headers):
    value = headers.get("Retry-After", "").strip()
    if value.isdigit():
        return float(value)
    if value:
        try:
            timestamp = parsedate_to_datetime(value)
            return max(0., (timestamp - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            pass
    reset = headers.get("X-RateLimit-Reset", "")
    return max(1., float(reset) - time.time()) if reset.isdigit() else 60.


def make_plan(files):
    primary = load(BASE / "manifest.json")
    partial = load(PARTIAL / "manifest.json")
    assert primary["complete"] and len(primary["clips"]) == 200
    assert len(partial["clips"]) == 181
    existing = [*primary["clips"], *partial["clips"]]
    existing_ids = {row["workbook_video_id"] for row in existing}
    existing_hashes = {row["original_sha256"] for row in existing}
    if len(existing_ids) != len(existing) or len(existing_hashes) != len(existing):
        raise ValueError("Previously verified cohorts contain duplicate videos")
    all_candidates, index_bytes, _ = inventory(files, WORKERS)
    primary_plan = load(BASE / "manifest.planned.json")
    partial_plan = load(PARTIAL / "manifest.planned.json")
    excluded = {row["key"] for row in primary_plan["rows"]}
    excluded.update(row["key"] for row in partial_plan["rows"] if row["source_id"] in existing_ids)
    bins = defaultdict(list)
    for row in all_candidates:
        if row["key"] not in excluded:
            bins[row["archive"], row["member"]["offset"] // WINDOW].append(row)
    blocks = []
    for (archive, _), rows in bins.items():
        rows.sort(key=lambda row: row["member"]["offset"])
        start = rows[0]["member"]["offset"]
        end = max(row["member"]["offset"] + row["range_bytes"] for row in rows)
        if end - start <= MAX_SPAN:
            blocks.append({"archive": archive, "start": start, "end": end, "rows": rows})
    blocks.sort(key=lambda b: ((b["end"] - b["start"] + CALL_COST) / len(b["rows"]), b["archive"], b["start"]))
    needed = EXPECTED_NEW - len(existing)
    chosen = []
    selected, total = 0, 0
    for block in blocks:
        chosen.append(block)
        selected += len(block["rows"])
        total += block["end"] - block["start"]
        if selected >= needed + RESERVE:
            break
    if selected < needed + RESERVE or total + PRIOR_RANGE_MAX + index_bytes > TOTAL_CAP:
        raise ValueError("The coalesced transfer plan exceeds its cap or lacks enough distinct media")
    sources = (BASE / "manifest.json", BASE / "manifest.planned.json",
               PARTIAL / "manifest.json", PARTIAL / "manifest.planned.json", WORKBOOK)
    return {
        "requested": needed, "planned_clips": selected, "planned_requests": len(chosen),
        "planned_range_bytes": total, "prior_range_max_bytes": PRIOR_RANGE_MAX,
        "total_range_cap_bytes": TOTAL_CAP, "source_hashes": {str(p.relative_to(OUTPUT.parent.parent)): sha(p) for p in sources},
        "preflight_kinds": dict(Counter(row["kind"] for block in chosen for row in block["rows"])),
        "prior_original_hashes": sorted(existing_hashes), "blocks": chosen,
    }


def download(block, spec, pace):
    start, end = block["start"], block["end"]
    length = end - start
    for attempt in range(5):
        pace.wait()
        request = urllib.request.Request(spec["links"]["self"], headers={
            "Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity",
        })
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end - 1}/{spec['size']}":
                    raise ValueError("Unexpected archive response range")
                if int(response.headers.get("X-RateLimit-Remaining", "100")) <= 8:
                    pace.pause(response_delay(response.headers) + 1.)
                payload = response.read(length + 1)
            if len(payload) != length:
                raise ValueError("Short archive range")
            with pace.lock:
                pace.successful_requests += 1
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            delay = response_delay(exc.headers) + 1.
            print(f"Zenodo rate limit, pausing new requests for {delay:.0f} s", flush=True)
            pace.pause(delay)
        except OSError:
            if attempt == 4:
                raise
            pace.pause(2. ** attempt)
    raise RuntimeError("Range attempts exhausted")


def main():
    files, source = official()
    FINAL.mkdir(parents=True, exist_ok=True)
    media = FINAL / "media"
    media.mkdir(exist_ok=True)
    frozen = FINAL / "manifest.planned.json"
    plan = make_plan(files)
    if frozen.exists() and load(frozen) != plan:
        raise ValueError("The frozen coalesced plan changed")
    if not frozen.exists():
        write_manifest(frozen, plan)
    frozen_hash = sha(frozen)
    print(json.dumps({key: value for key, value in plan.items()
                      if key not in ("blocks", "source_hashes", "prior_original_hashes")}), flush=True)
    target = FINAL / "manifest.json"
    prior = load(target) if target.exists() else None
    if prior is not None and prior["plan_sha256"] != frozen_hash:
        raise ValueError("A different plan produced the current checkpoint")
    by_id = {row["workbook_video_id"]: row for row in (prior or {}).get("clips", [])}
    seen = set(plan["prior_original_hashes"])
    clips = []
    for row in by_id.values():
        path = FINAL.parent.parent.parent / row["path"]
        if not path.is_file() or sha(path) != row["mp4_sha256"] or row["original_sha256"] in seen:
            raise ValueError(f"Invalid previous media: {row['id']}")
        clips.append(row)
        seen.add(row["original_sha256"])
    failures = []
    pace = Pace()
    transferred = (prior or {}).get("successful_range_bytes", 0)
    successful = (prior or {}).get("successful_requests", 0)
    requested = plan["requested"]
    consumed = set((prior or {}).get("completed_blocks", []))

    def task(block):
        remaining = [row for row in block["rows"] if row["source_id"] not in by_id]
        missing = any(not (media / (row["source_id"] + ".flv")).is_file() for row in remaining)
        payload = download(block, files[block["archive"]], pace) if missing else None
        results = []
        for row in remaining:
            start = row["member"]["offset"] - block["start"]
            try:
                # If every FLV is cached, block=b"" forbids any unplanned HTTP retry.
                segment = payload[start:start + row["range_bytes"]] if payload is not None else b""
                results.append((acquire(row, files, source, media, block=segment), None))
            except (OSError, ValueError, RuntimeError) as exc:
                results.append((None, {"source_id": row["source_id"], "error": str(exc)}))
            except Exception as exc:
                results.append((None, {"source_id": row["source_id"], "error": repr(exc)}))
        return results, len(payload) if payload is not None else 0

    def checkpoint():
        write_manifest(target, {
            "source": source, "plan_sha256": frozen_hash, "requested": requested,
            "complete": len(clips) == requested, "clips": clips, "failures": failures,
            "word_count": sum(row["kind"] == "word" for row in clips),
            "phrase_count": sum(row["kind"] == "phrase" for row in clips),
            "completed_blocks": sorted(consumed), "successful_range_bytes": transferred,
            "successful_requests": successful + pace.successful_requests,
        })

    pending = [(index, block) for index, block in enumerate(plan["blocks"]) if index not in consumed]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for start in range(0, len(pending), WORKERS):
            batch = pending[start:start + WORKERS]
            if transferred + sum(b["end"] - b["start"] for _, b in batch) + PRIOR_RANGE_MAX > TOTAL_CAP:
                raise ValueError("Aggregate continuation transfer cap reached")
            jobs = [pool.submit(task, block) for _, block in batch]
            for (index, _), job in zip(batch, jobs, strict=True):
                results, size = job.result()
                transferred += size
                consumed.add(index)
                for clip, error in results:
                    if error:
                        failures.append(error)
                    elif clip["original_sha256"] in seen:
                        failures.append({"source_id": clip["workbook_video_id"], "error": "Duplicate original content"})
                    elif len(clips) < requested:
                        clips.append(clip)
                        seen.add(clip["original_sha256"])
            checkpoint()
            print(json.dumps({"verified": len(clips), "requests": successful + pace.successful_requests,
                              "bytes": transferred, "failures": len(failures)}), flush=True)
            if len(clips) == requested:
                return
    raise ValueError(f"Verified {len(clips)}/{requested}; partial checkpoint retained")


if __name__ == "__main__":
    main()
