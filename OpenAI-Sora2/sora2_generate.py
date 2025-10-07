import os
import sys
import time
import json
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv
from openai import OpenAI
import mimetypes
import struct

load_dotenv()

DEFAULT_BASE_URL = "https://api.openai.com/v1/videos"
FILES_BASE_URL = "https://api.openai.com/v1/files"
# Local job log to aid rediscovery if collection listing is unavailable
LOG_FILE = "sora2_jobs.jsonl"

class VideoGenerationError(Exception):
    pass

def _get_api_key() -> str:
    # WARNING: printing secrets is insecure; keep only for local debugging
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise VideoGenerationError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
        )
    return api_key

def _build_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }
    # Optional scoping headers if configured
    org = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION")
    if org:
        headers["OpenAI-Organization"] = org
    project = os.getenv("OPENAI_PROJECT")
    if project:
        headers["OpenAI-Project"] = project
    return headers

def _append_job_log(entry: Dict[str, Any]) -> None:
    """Append a single job record as JSON to the local log file."""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # Non-fatal logging failure
        pass

def generate_video(
    prompt: str,
    *,
    model: str = "sora-2-pro",
    size: str = "1792x1024",
    seconds: str = "8",
    webhook_url: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    create_timeout_s: int = 60,
    poll_interval_s: int = 5,
    poll_timeout_s: int = 1200,
    request_timeout_s: int = 60,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
    download_path: Optional[str] = None,
    input_reference: Optional[str] = None,
    match_size_to_input: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Create a video generation job using OpenAI Sora 2 and optionally download the result.

    Parameters (exposes all documented video parameters plus operational controls):
    - prompt: Text description of the desired video (required).
    - model: Video model to use. Typical options: 'sora-2', 'sora-2-pro'.
    - size: Frame size (resolution), e.g. '1280x720', '720x1280', '1024x1792', '1792x1024'.
    - seconds: Duration in seconds. Typical allowed values: 4, 8, 12.
    - webhook_url: Optional webhook to receive completion callback from the API.
    - base_url: API base for video jobs (default: https://api.openai.com/v1/videos).
    - create_timeout_s: Timeout for the create request.
    - poll_interval_s: Interval between status polls.
    - poll_timeout_s: Max time to wait for the job to complete.
    - request_timeout_s: Timeout applied to status/download requests.
    - max_retries: Max retries for transient HTTP errors on create/poll.
    - retry_backoff_s: Exponential backoff base for retries.
    - download_path: Path to save the resulting video. If not provided, the video
      will be saved in the current directory with a timestamped name like
      'sora2_YYYYMMDD_HHMMSS.mp4'.

    Returns:
    - Dict with final job payload. If download_path is provided and the job succeeds, the file is saved locally.
    """

    # progress: begin submission
    print("Submitting video generation job...", flush=True)

    _validate_parameters(model=model, size=size, seconds=seconds)

    headers = _build_headers()

    payload: Dict[str, Any] = {
        "model": model,
        "size": size,
        "seconds": seconds,
        "prompt": prompt,
    }
    if webhook_url:
        payload["webhook_url"] = webhook_url

    # If an input file is provided, ensure its dimensions match requested size, or auto-match if requested
    file_path_for_log: Optional[str] = None
    if input_reference:
        file_path_for_log, _mime_probe = _parse_input_reference(input_reference)
        try:
            img_w, img_h = _get_image_size(file_path_for_log)
        except Exception:
            img_w, img_h = (None, None)
        try:
            req_w, req_h = _parse_size_string(size)
        except Exception:
            req_w, req_h = (None, None)
        if img_w and img_h and req_w and req_h:
            if (img_w, img_h) != (req_w, req_h):
                if match_size_to_input:
                    payload["size"] = f"{img_w}x{img_h}"
                else:
                    raise VideoGenerationError(
                        f"Input image size {img_w}x{img_h} does not match requested size {req_w}x{req_h}. "
                        "Use --match-size-to-input or adjust --size."
                    )

    # Create job (JSON by default; multipart if an input_reference file is provided)
    if input_reference:
        file_path, mime_type = _parse_input_reference(input_reference)
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except Exception as exc:  # noqa: BLE001
            raise VideoGenerationError(f"Failed to read input_reference file: {file_path}") from exc

        # Form fields mirror JSON payload
        form_fields: Dict[str, Any] = dict(payload)
        # Use multipart upload; remove explicit JSON content-type so requests sets boundary
        job = _post_with_retries_multipart(
            url=base_url,
            headers=headers,
            form_fields=form_fields,
            file_field_name="input_reference",
            filename=file_name,
            file_bytes=file_bytes,
            mime_type=mime_type,
            timeout_s=create_timeout_s,
            max_retries=max_retries,
            backoff_s=retry_backoff_s,
        )
    else:
        job: Dict[str, Any] = _post_with_retries(
            url=base_url,
            headers=headers,
            json_payload=payload,
            timeout_s=create_timeout_s,
            max_retries=max_retries,
            backoff_s=retry_backoff_s,
        )

    job_id = job.get("id") or job.get("task_id")
    if not job_id:
        raise VideoGenerationError(
            f"Create response missing id: {json.dumps(job, ensure_ascii=False)}"
        )
    print(f"Job submitted. id={job_id}", flush=True)
    # Persist a local record of the job for discoverability if listing is unavailable.
    try:
        _append_job_log(
            {
                "id": job_id,
                "created_at_local": int(time.time()),
                "prompt": prompt,
                "model": model,
                "size": size,
                "seconds": seconds,
                "base_url": base_url,
                # Log only the file name for discoverability; avoid storing file bytes
                "input_reference_name": os.path.basename(file_path_for_log) if input_reference and file_path_for_log else None,
            }
        )
    except Exception:
        # Non-fatal if logging fails
        pass
    print(
        f"Polling for completion every {poll_interval_s}s (timeout {poll_timeout_s}s)...",
        flush=True,
    )

    # Poll for completion
    status_url = f"{base_url.rstrip('/')}/{job_id}"
    started_at = time.time()
    while True:
        if time.time() - started_at > poll_timeout_s:
            raise VideoGenerationError(
                f"Polling timed out after {poll_timeout_s}s for job {job_id}"
            )

        status_payload = _get_with_retries(
            url=status_url,
            headers=headers,
            timeout_s=request_timeout_s,
            max_retries=max_retries,
            backoff_s=retry_backoff_s,
        )

        status = (status_payload.get("status") or "").lower()
        elapsed = int(time.time() - started_at)
        if status:
            print(f"[{elapsed}s] status={status}", flush=True)
        else:
            print(f"[{elapsed}s] status=(unknown)", flush=True)
        if status in {"succeeded", "success", "completed", "done"}:
            # Resolve a downloadable URL (prefer direct payload, then expanded assets,
            # then heuristic endpoints, then direct download, finally brief refresh)
            # Try to extract a usable video URL from the payload first
            video_url = _extract_video_url(status_payload)

            # If not found, try status expansion
            if not video_url:
                try:
                    expanded = _get_with_retries(
                        url=status_url,
                        headers=headers,
                        timeout_s=request_timeout_s,
                        max_retries=max_retries,
                        backoff_s=retry_backoff_s,
                        params={"expand": "assets"},
                    )
                    video_url = _extract_video_url(expanded)
                    if debug and not video_url:
                        print("No URL in expanded status (expand=assets)", flush=True)
                except Exception:
                    if debug:
                        print("expand=assets request failed", flush=True)

            # If not found, try likely asset endpoints
            if not video_url:
                if debug:
                    print("No video URL in response. Attempting to retrieve assets...", flush=True)
                video_url = _try_fetch_additional_urls(
                    base_url=base_url,
                    job_id=job_id,
                    headers=headers,
                    request_timeout_s=request_timeout_s,
                    max_retries=max_retries,
                    retry_backoff_s=retry_backoff_s,
                )

            # Try direct download endpoint as a last resort
            if not video_url:
                if debug:
                    print("Attempting direct /download endpoint...", flush=True)
                video_url = _try_direct_download_endpoint(
                    base_url=base_url,
                    job_id=job_id,
                    headers=headers,
                    request_timeout_s=request_timeout_s,
                )

            # If still not found, wait briefly and re-fetch the job a few times
            if not video_url:
                for wait_idx in range(3):
                    if debug:
                        print("Waiting for video URL to become available...", flush=True)
                    time.sleep(2)
                    refreshed = _get_with_retries(
                        url=status_url,
                        headers=headers,
                        timeout_s=request_timeout_s,
                        max_retries=max_retries,
                        backoff_s=retry_backoff_s,
                    )
                    video_url = _extract_video_url(refreshed)
                    if video_url:
                        break

            if not download_path:
                ts = time.strftime("%Y%m%d_%H%M%S")
                download_path = os.path.join(os.getcwd(), f"sora2_{ts}.mp4")

            # Prefer official SDK download if available
            try:
                print("Job succeeded. Downloading via OpenAI SDK...", flush=True)
                client = OpenAI(api_key=_get_api_key())
                try:
                    content = client.videos.download_content(job_id, variant="video")
                except Exception as e1:  # noqa: BLE001
                    if debug:
                        print(f"SDK download (variant=video) failed: {e1}", flush=True)
                    # Try without variant (some SDK versions default correctly)
                    content = client.videos.download_content(job_id)
                content.write_to_file(download_path)
                abs_path = os.path.abspath(download_path)
                dir_path = os.path.dirname(abs_path)
                file_name = os.path.basename(abs_path)
                print(f"Download complete: file={file_name}", flush=True)
                print(f"Saved to directory: {dir_path}", flush=True)
                status_payload["download_path"] = download_path
                return status_payload
            except Exception as e2:  # noqa: BLE001
                if debug:
                    print(f"SDK download failed: {e2}", flush=True)
                # Fallback to URL discovery if SDK method not available/doesn't return
                if video_url:
                    print("SDK download not available, falling back to URL download...", flush=True)
                    _download_file(video_url, download_path, headers=headers, timeout_s=request_timeout_s)
                    abs_path = os.path.abspath(download_path)
                    dir_path = os.path.dirname(abs_path)
                    file_name = os.path.basename(abs_path)
                    print(f"Download complete: file={file_name}", flush=True)
                    print(f"Saved to directory: {dir_path}", flush=True)
                    status_payload["download_path"] = download_path
                    return status_payload

            print("Completed, but no downloadable URL was found.", flush=True)
            return status_payload

        if status in {"failed", "error"}:
            raise VideoGenerationError(
                f"Video generation failed for job {job_id}: {json.dumps(status_payload, ensure_ascii=False)}"
            )

        time.sleep(poll_interval_s)


def _validate_parameters(*, model: str, size: str, seconds: str) -> None:
    allowed_models = {"sora-2", "sora-2-pro"}
    if model not in allowed_models:
        raise VideoGenerationError(
            f"Invalid model '{model}'. Allowed: {sorted(allowed_models)}"
        )

    allowed_sizes_by_model = {
        "sora-2": {"1280x720", "720x1280"},
        "sora-2-pro": {"1280x720", "720x1280", "1024x1792", "1792x1024"},
    }
    allowed_sizes = allowed_sizes_by_model.get(model, set())
    if size not in allowed_sizes:
        raise VideoGenerationError(
            f"Invalid size '{size}' for model '{model}'. Allowed: {sorted(allowed_sizes)}"
        )

    if seconds not in {"4", "8", "12"}:
        raise VideoGenerationError("seconds must be one of: '4', '8', '12'")


def _parse_size_string(size: str) -> tuple[int, int]:
    """Parse a size string like '1280x720' into integers (w, h)."""
    if not isinstance(size, str) or "x" not in size:
        raise VideoGenerationError(f"Invalid size string: {size}")
    w_str, h_str = size.lower().split("x", 1)
    w = int(w_str.strip())
    h = int(h_str.strip())
    if w <= 0 or h <= 0:
        raise VideoGenerationError(f"Invalid size values: {size}")
    return w, h


def _get_image_size(path: str) -> tuple[int, int]:
    """Read width/height from PNG or JPEG without external deps.

    Supports common PNG IHDR and JPEG SOFx markers.
    """
    with open(path, "rb") as f:
        head = f.read(32)
        # PNG: 8-byte signature + IHDR chunk (width/height big-endian at bytes 16..24)
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            # Ensure IHDR layout
            # Bytes 16..20: width, 20..24: height (big-endian)
            if len(head) < 24:
                # Read more just in case
                more = f.read(24 - len(head))
                head += more
            width = int.from_bytes(head[16:20], "big")
            height = int.from_bytes(head[20:24], "big")
            return width, height

        # JPEG: scan for SOF0/2 markers to get height/width
        f.seek(0)
        data = f.read()
        if data[:2] == b"\xff\xd8":
            idx = 2
            data_len = len(data)
            while idx + 9 < data_len:
                if data[idx] != 0xFF:
                    idx += 1
                    continue
                marker = data[idx + 1]
                idx += 2
                # markers without length
                if marker in (0xD8, 0xD9):
                    continue
                if idx + 2 > data_len:
                    break
                seg_len = data[idx] << 8 | data[idx + 1]
                if seg_len < 2 or idx + seg_len > data_len:
                    break
                # SOF0..SOF3, SOF5..SOF7, SOF9..SOF11, SOF13..SOF15
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    if idx + 7 <= data_len:
                        # seg: [len(2)][precision(1)][height(2)][width(2)]...
                        height = data[idx + 3] << 8 | data[idx + 4]
                        width = data[idx + 5] << 8 | data[idx + 6]
                        return width, height
                idx += seg_len

    raise VideoGenerationError("Unsupported or unreadable image format for input_reference")


def _post_with_retries(
    *,
    url: str,
    headers: Dict[str, str],
    json_payload: Dict[str, Any],
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
) -> Dict[str, Any]:
    # Retry on 429/5xx with exponential backoff; treat other statuses as fatal
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=json_payload, timeout=timeout_s)
            # 202 Accepted is common for async jobs
            if resp.status_code in (200, 201, 202):
                return _parse_json(resp)
            # retry on 429/5xx
            if resp.status_code in (429, 500, 502, 503, 504):
                raise VideoGenerationError(
                    f"Transient error {resp.status_code}: {resp.text[:500]}"
                )
            # non-retryable
            raise VideoGenerationError(
                f"Request failed {resp.status_code}: {resp.text[:500]}"
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep((backoff_s ** attempt) if backoff_s > 1 else backoff_s)
            else:
                break
    assert last_exc is not None
    raise last_exc


def _parse_input_reference(value: str) -> tuple[str, str]:
    """Parse curl-style input reference like '@file;type=image/jpeg'.

    Returns (file_path, mime_type). If mime is omitted, guessed via mimetypes.
    """
    raw = value.strip()
    if raw.startswith("@"):
        raw = raw[1:]
    path_part = raw
    mime_type = ""
    if ";" in raw:
        path_part, meta_part = raw.split(";", 1)
        meta_part = meta_part.strip()
        if meta_part.lower().startswith("type="):
            mime_type = meta_part.split("=", 1)[1].strip()
        elif meta_part.lower().startswith("type:"):
            mime_type = meta_part.split(":", 1)[1].strip()
    file_path = path_part.strip().strip('"')
    if not mime_type:
        guessed, _ = mimetypes.guess_type(file_path)
        mime_type = guessed or "application/octet-stream"
    return file_path, mime_type


def _post_with_retries_multipart(
    *,
    url: str,
    headers: Dict[str, str],
    form_fields: Dict[str, Any],
    file_field_name: str,
    filename: str,
    file_bytes: bytes,
    mime_type: str,
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
) -> Dict[str, Any]:
    # Remove JSON content-type so requests can set multipart boundary
    headers_mp = dict(headers)
    headers_mp.pop("Content-Type", None)
    files = {file_field_name: (filename, file_bytes, mime_type)}
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers=headers_mp,
                data=form_fields,
                files=files,
                timeout=timeout_s,
            )
            if resp.status_code in (200, 201, 202):
                return _parse_json(resp)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise VideoGenerationError(
                    f"Transient error {resp.status_code}: {resp.text[:500]}"
                )
            raise VideoGenerationError(
                f"Request failed {resp.status_code}: {resp.text[:500]}"
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep((backoff_s ** attempt) if backoff_s > 1 else backoff_s)
            else:
                break
    assert last_exc is not None
    raise last_exc


def _get_with_retries(
    *,
    url: str,
    headers: Dict[str, str],
    timeout_s: int,
    max_retries: int,
    backoff_s: float,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Same retry policy as POST: 429/5xx retried with exponential backoff
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_s, params=params)
            if resp.status_code in (200, 201):
                return _parse_json(resp)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise VideoGenerationError(
                    f"Transient error {resp.status_code}: {resp.text[:500]}"
                )
            raise VideoGenerationError(
                f"Request failed {resp.status_code}: {resp.text[:500]}"
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep((backoff_s ** attempt) if backoff_s > 1 else backoff_s)
            else:
                break
    assert last_exc is not None
    raise last_exc


def _parse_json(resp: requests.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        raise VideoGenerationError(f"Invalid JSON response: {resp.text[:500]}") from exc


def _download_file(url: str, dest_path: str, headers: Optional[Dict[str, str]], timeout_s: int) -> None:
    # Use auth headers for API-hosted URLs (e.g., file content endpoints)
    eff_headers = headers or {}
    if url.startswith(FILES_BASE_URL):
        eff_headers = _build_headers()
    with requests.get(url, headers=eff_headers, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        total_bytes = int(r.headers.get("Content-Length")) if r.headers.get("Content-Length") else None
        # ensure destination directory exists
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        downloaded = 0
        next_report = 0.1  # report every 10%
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                if total_bytes is not None:
                    downloaded += len(chunk)
                    ratio = downloaded / total_bytes if total_bytes else 0
                    if ratio >= next_report:
                        pct = int(ratio * 100)
                        print(f"Download progress: {pct}%", flush=True)
                        next_report += 0.1


def _extract_video_url(payload: Dict[str, Any]) -> Optional[str]:
    # Common locations for downloadable URLs in various API responses
    candidates = [
        payload.get("video_url"),
        payload.get("output", {}).get("video_url") if isinstance(payload.get("output"), dict) else None,
        payload.get("result", {}).get("video_url") if isinstance(payload.get("result"), dict) else None,
    ]
    # Sometimes assets array contains downloadable links
    assets = payload.get("assets")
    if isinstance(assets, list):
        for asset in assets:
            if isinstance(asset, dict):
                url = asset.get("url") or asset.get("download_url")
                if url:
                    candidates.append(url)
    # Generic recursive search for any URL-like strings
    for found in _deep_find_urls(payload):
        candidates.append(found)
    for url in candidates:
        if url:
            return url
    return None


def download_existing_job(
    *,
    job_id: str,
    base_url: str = DEFAULT_BASE_URL,
    request_timeout_s: int = 60,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
    download_path: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Attempt to resolve and download a completed video's asset for an existing job id
    without creating a new job.
    """
    headers = _build_headers()
    status_url = f"{base_url.rstrip('/')}/{job_id}"
    # Fetch latest status
    payload = _get_with_retries(
        url=status_url,
        headers=headers,
        timeout_s=request_timeout_s,
        max_retries=max_retries,
        backoff_s=retry_backoff_s,
    )

    status = (payload.get("status") or "").lower()
    if debug:
        print(f"Current status for {job_id}: {status}", flush=True)

    # Try direct extraction
    video_url = _extract_video_url(payload)

    # Try expanded status
    if not video_url:
        try:
            expanded = _get_with_retries(
                url=status_url,
                headers=headers,
                timeout_s=request_timeout_s,
                max_retries=max_retries,
                backoff_s=retry_backoff_s,
                params={"expand": "assets"},
            )
            video_url = _extract_video_url(expanded)
            if debug and not video_url:
                print("No URL in expanded status (expand=assets)", flush=True)
        except Exception:
            if debug:
                print("expand=assets request failed", flush=True)

    # Try additional endpoints
    if not video_url:
        video_url = _try_fetch_additional_urls(
            base_url=base_url,
            job_id=job_id,
            headers=headers,
            request_timeout_s=request_timeout_s,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )

    # Try direct download endpoint
    if not video_url:
        video_url = _try_direct_download_endpoint(
            base_url=base_url,
            job_id=job_id,
            headers=headers,
            request_timeout_s=request_timeout_s,
        )

    result = dict(payload)
    if not download_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        download_path = os.path.join(os.getcwd(), f"sora2_{ts}.mp4")

    # Prefer official SDK download by id; does not require a URL in payload
    try:
        print("Downloading via OpenAI SDK...", flush=True)
        client = OpenAI(api_key=_get_api_key())
        try:
            content = client.videos.download_content(job_id, variant="video")
        except Exception as e1:  # noqa: BLE001
            if debug:
                print(f"SDK download (variant=video) failed: {e1}", flush=True)
            content = client.videos.download_content(job_id)
        content.write_to_file(download_path)
        abs_path = os.path.abspath(download_path)
        dir_path = os.path.dirname(abs_path)
        file_name = os.path.basename(abs_path)
        print(f"Download complete: file={file_name}", flush=True)
        print(f"Saved to directory: {dir_path}", flush=True)
        result["download_path"] = download_path
        return result
    except Exception as e2:  # noqa: BLE001
        if debug:
            print(f"SDK download failed: {e2}", flush=True)
        # fall back to URL-based download below
        pass

    if video_url:
        print("SDK download not available, falling back to URL download...", flush=True)
        _download_file(video_url, download_path, headers=headers, timeout_s=request_timeout_s)
        abs_path = os.path.abspath(download_path)
        dir_path = os.path.dirname(abs_path)
        file_name = os.path.basename(abs_path)
        print(f"Download complete: file={file_name}", flush=True)
        print(f"Saved to directory: {dir_path}", flush=True)
        result["download_path"] = download_path
        return result

    print("Completed, but no downloadable URL was found.", flush=True)
    return result


def _try_fetch_additional_urls(
    *,
    base_url: str,
    job_id: str,
    headers: Dict[str, str],
    request_timeout_s: int,
    max_retries: int,
    retry_backoff_s: float,
) -> Optional[str]:
    # Heuristic extra endpoints that may expose asset URLs
    guesses = [
        f"{base_url.rstrip('/')}/{job_id}/assets",
        f"{base_url.rstrip('/')}/{job_id}/files",
        f"{base_url.rstrip('/')}/{job_id}/results",
    ]
    for guess in guesses:
        try:
            data = _get_with_retries(
                url=guess,
                headers=headers,
                timeout_s=request_timeout_s,
                max_retries=max_retries,
                backoff_s=retry_backoff_s,
            )
            url = _extract_video_url(data)
            if url:
                return url
            # Look for file ids and try file content endpoint
            file_id = _deep_find_first_file_id(data)
            if file_id:
                content_url = f"{FILES_BASE_URL.rstrip('/')}/{file_id}/content"
                try:
                    # Return the API content URL; _download_file will pass headers
                    return content_url
                except Exception:
                    pass
        except Exception:
            # Ignore and continue trying other guesses
            pass
    return None


def _try_direct_download_endpoint(
    *,
    base_url: str,
    job_id: str,
    headers: Dict[str, str],
    request_timeout_s: int,
) -> Optional[str]:
    # Some APIs expose a /download endpoint that redirects to the asset
    dl = f"{base_url.rstrip('/')}/{job_id}/download"
    try:
        resp = requests.get(dl, headers=headers, timeout=request_timeout_s, allow_redirects=False)
        # If we get a redirect, use Location as the URL
        if resp.status_code in (302, 303, 307, 308):
            loc = resp.headers.get("Location")
            if loc:
                return loc
        # If OK and content-type is video, we can return the endpoint itself
        if resp.status_code == 200:
            ct = resp.headers.get("Content-Type", "")
            if ct.startswith("video/"):
                return dl
    except Exception:
        pass
    return None


def _deep_find_first_file_id(obj: Any) -> Optional[str]:
    # Identify strings that resemble OpenAI file IDs (heuristic: prefix 'file_')
    if isinstance(obj, dict):
        for k, v in obj.items():
            found = _deep_find_first_file_id(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _deep_find_first_file_id(v)
            if found:
                return found
    elif isinstance(obj, str):
        if obj.startswith("file_"):
            return obj
    return None


def _deep_find_urls(obj: Any) -> list[str]:
    urls: list[str] = []
    def _walk(x: Any):
        if isinstance(x, dict):
            for k, v in x.items():
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)
        elif isinstance(x, str):
            if x.startswith("http://") or x.startswith("https://"):
                urls.append(x)
    _walk(obj)
    return urls


def list_downloadable_video_urls(
    *,
    base_url: str = DEFAULT_BASE_URL,
    limit: int = 50,
    created_after: Optional[int] = None,
    include_incomplete: bool = False,
    request_timeout_s: int = 60,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
) -> Dict[str, Any]:
    """
    Return a dictionary with job metadata and all discovered downloadable video URLs.

    - base_url: API base for videos collection (default: https://api.openai.com/v1/videos)
    - limit: max number of recent jobs to inspect
    - created_after: optional unix timestamp to filter newer jobs
    - include_incomplete: if True, include jobs without a resolved URL yet
    - request_timeout_s / max_retries / retry_backoff_s: HTTP behavior controls

    Returns a dict:
    {
      "count": int,
      "jobs": [
        {"id": str, "status": str, "urls": [str], ... original job fields ...},
        ...
      ]
    }
    """
    headers = _build_headers()

    # Attempt to list recent jobs. If the API supports collection listing at base_url.
    params: Dict[str, Any] = {"limit": limit}
    if created_after is not None:
        params["created_after"] = created_after

    try:
        listing = _get_with_retries(
            url=base_url,
            headers=headers,
            timeout_s=request_timeout_s,
            max_retries=max_retries,
            backoff_s=retry_backoff_s,
            params=params,
        )
    except Exception as exc:  # noqa: BLE001
        # If listing fails, return empty result with error info
        return {"count": 0, "jobs": [], "error": str(exc)}

    jobs = []
    # Accept either a list of jobs or a wrapper with 'data'
    raw_jobs = listing if isinstance(listing, list) else listing.get("data", [])
    for item in raw_jobs:
        if not isinstance(item, dict):
            continue
        job_id = item.get("id") or item.get("video_id")
        status = (item.get("status") or "").lower()
        urls: list[str] = []

        # Try direct fields
        direct_url = _extract_video_url(item)
        if direct_url:
            urls.append(direct_url)

        # If none, attempt job detail and common asset endpoints
        if not urls and job_id:
            detail_url = f"{base_url.rstrip('/')}/{job_id}"
            try:
                detail = _get_with_retries(
                    url=detail_url,
                    headers=headers,
                    timeout_s=request_timeout_s,
                    max_retries=max_retries,
                    backoff_s=retry_backoff_s,
                )
                detail_url_candidate = _extract_video_url(detail)
                if detail_url_candidate:
                    urls.append(detail_url_candidate)
            except Exception:
                pass

            if not urls:
                try_urls = [
                    f"{detail_url}/assets",
                    f"{detail_url}/files",
                    f"{detail_url}/results",
                ]
                for guess in try_urls:
                    try:
                        data = _get_with_retries(
                            url=guess,
                            headers=headers,
                            timeout_s=request_timeout_s,
                            max_retries=max_retries,
                            backoff_s=retry_backoff_s,
                        )
                        guessed = _extract_video_url(data)
                        if guessed:
                            urls.append(guessed)
                            break
                    except Exception:
                        continue

        if urls or include_incomplete:
            entry = dict(item)  # shallow copy
            entry["urls"] = urls
            jobs.append(entry)

    return {"count": len(jobs), "jobs": jobs}


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a video with OpenAI Sora 2 (job + poll + optional download)",
    )
    parser.add_argument("prompt", type=str, nargs="?", default=None, help="Text description of the desired video (required unless using --list-urls)")
    parser.add_argument(
        "--model",
        type=str,
        default="sora-2",
        choices=["sora-2", "sora-2-pro"],
        help="Video model",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280x720",
        help="Resolution (depends on model). Examples: 1280x720, 720x1280, 1024x1792, 1792x1024",
    )
    parser.add_argument(
        "--seconds",
        type=str,
        default="4",
        choices=["4", "8", "12"],
        help="Clip length in seconds",
    )
    parser.add_argument(
        "--webhook-url",
        dest="webhook_url",
        type=str,
        default=None,
        help="Optional webhook URL for server-side callback",
    )
    parser.add_argument(
        "--prompt-file",
        dest="prompt_file",
        type=str,
        default=None,
        help="Read prompt text from file (default: sora_prompt.txt if present)",
    )
    parser.add_argument(
        "--input-reference",
        dest="input_reference",
        type=str,
        default=None,
        help="Optional input media in curl style, e.g. @sample_720p.jpeg;type=image/jpeg",
    )
    parser.add_argument(
        "--match-size-to-input",
        dest="match_size_to_input",
        action="store_true",
        help="If provided with --input-reference, overrides --size to match the input image",
    )
    parser.add_argument(
        "--download-path",
        dest="download_path",
        type=str,
        default=None,
        help="If set, download the finished video to this file path",
    )
    parser.add_argument("--base-url", dest="base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--create-timeout", dest="create_timeout_s", type=int, default=60)
    parser.add_argument("--poll-interval", dest="poll_interval_s", type=int, default=5)
    parser.add_argument("--poll-timeout", dest="poll_timeout_s", type=int, default=1200)
    parser.add_argument("--request-timeout", dest="request_timeout_s", type=int, default=60)
    parser.add_argument("--max-retries", dest="max_retries", type=int, default=3)
    parser.add_argument("--retry-backoff", dest="retry_backoff_s", type=float, default=2.0)

    # Listing mode
    parser.add_argument("--list-urls", dest="list_urls", action="store_true", help="List downloadable video URLs instead of generating a video")
    parser.add_argument("--limit", dest="limit", type=int, default=50, help="Max number of recent jobs to inspect when listing URLs")
    parser.add_argument("--created-after", dest="created_after", type=int, default=None, help="Unix timestamp; only list jobs created after this time")
    parser.add_argument("--include-incomplete", dest="include_incomplete", action="store_true", help="Include jobs without a resolved URL yet in listing output")
    parser.add_argument("--resume", dest="resume", type=str, default=None, help="Resume by downloading an existing job id without creating a new job")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable verbose debug logs")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        # Listing mode: no prompt required
        if getattr(args, "list_urls", False):
            listing = list_downloadable_video_urls(
                base_url=args.base_url,
                limit=args.limit,
                created_after=args.created_after,
                include_incomplete=args.include_incomplete,
                request_timeout_s=args.request_timeout_s,
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff_s,
            )
            print(json.dumps(listing, indent=2, ensure_ascii=False))
            return 0

        # Resume mode: attempt to download existing job asset
        if getattr(args, "resume", None):
            result = download_existing_job(
                job_id=args.resume,
                base_url=args.base_url,
                request_timeout_s=args.request_timeout_s,
                max_retries=args.max_retries,
                retry_backoff_s=args.retry_backoff_s,
                download_path=args.download_path,
                debug=args.debug,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

        prompt_text = args.prompt
        if not prompt_text:
            # Try reading the prompt from a file, preferring --prompt-file if provided,
            # otherwise falling back to a local 'sora_prompt.txt' if it exists.
            prompt_file = getattr(args, "prompt_file", None)
            candidate_paths = []
            if prompt_file:
                candidate_paths.append(prompt_file)
            candidate_paths.append(os.path.join(os.getcwd(), "sora_prompt.txt"))
            for path in candidate_paths:
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            prompt_text = f.read().strip()
                        if prompt_text:
                            break
                    except Exception:
                        # Ignore file read errors and continue to next candidate
                        pass
        if not prompt_text:
            print("Error: prompt is required (or provide --prompt-file or sora_prompt.txt)", file=sys.stderr)
            return 2

        result = generate_video(
            prompt=prompt_text,
            model=args.model,
            size=args.size,
            seconds=args.seconds,
            webhook_url=args.webhook_url,
            base_url=args.base_url,
            create_timeout_s=args.create_timeout_s,
            poll_interval_s=args.poll_interval_s,
            poll_timeout_s=args.poll_timeout_s,
            request_timeout_s=args.request_timeout_s,
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff_s,
            download_path=args.download_path,
            input_reference=getattr(args, "input_reference", None),
            match_size_to_input=getattr(args, "match_size_to_input", False),
            debug=args.debug,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


