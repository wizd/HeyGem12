"""sync.so-style HTTP API for HeyGem12 lip-sync.

Implements the lip-sync core of sync.so's v2 API (POST /v2/generate,
GET /v2/generate/{id}, /v2/generations, /v2/assets) backed by the local
HeyGem12 pipeline. See docs/API.md for what is and is not supported.
"""
import asyncio
import hashlib
import hmac
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import media
from .config import settings
from .scheduler import Scheduler, job_inputs, job_precision, job_sync_mode
from .store import Store, new_id

log = logging.getLogger("heygem12_api")

DOC_URL = "https://github.com/wizd/HeyGem12/blob/main/docs/API.md"
MEDIA_EXT = {
    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska",
    ".m4v": "video/x-m4v", ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".aac": "audio/aac", ".ogg": "audio/ogg", ".flac": "audio/flac",
}
# model name -> precision (GFPGAN face restoration before synthesis)
MODELS = {"heygem12": False, "heygem12-hq": True}
SYNC_MODES = ("bounce", "loop", "cut_off")
STATUS = {"pending": "PENDING", "running": "PROCESSING", "completed": "COMPLETED", "failed": "FAILED"}
STATUS_REV = {v: k for k, v in STATUS.items()}
IDEM_RE = re.compile(r"^[A-Za-z0-9._~-]{1,128}$")
FILE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,255}$")
# sync.so features this server does not implement: refused, never silently ignored
UNSUPPORTED_TOP = ("segments", "outputAudio", "dubParams", "projectId", "dialogueEdit")


# ------------------------------------------------------------------ errors
class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, field: Optional[str] = None):
        self.status, self.code, self.message, self.field = status, code, message, field


def error_body(status: int, code: Optional[str], message: str, field: Optional[str] = None) -> Dict[str, Any]:
    return {"message": message, "statusCode": status, "errorCode": code, "field": field, "docsUrl": DOC_URL}


# ----------------------------------------------------------- signed URLs
def _sign(kind: str, ident: str, expires: int) -> str:
    return hmac.new(settings.url_signing_secret, f"{kind}:{ident}:{expires}".encode(),
                    hashlib.sha256).hexdigest()[:32]


def signed_url(kind: str, ident: str, base_url: Optional[str]) -> str:
    exp = int(time.time()) + settings.url_ttl_s
    base = settings.public_url or (base_url or "").rstrip("/")
    return f"{base}/v2/files/{kind}/{ident}?expires={exp}&signature={_sign(kind, ident, exp)}"


def _base_url(request: Request) -> str:
    return settings.public_url or str(request.base_url).rstrip("/")


def _iso(ts: Optional[int]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


# ------------------------------------------------------------------ views
def generation_view(row: Dict[str, Any]) -> Dict[str, Any]:
    req = json.loads(row["request"])
    done = row["status"] == "completed"
    failed = row["status"] == "failed"
    inputs = job_inputs(req)
    model = req.get("model") or ("heygem12-hq" if job_precision(req) else "heygem12")
    return {
        "id": row["id"],
        "createdAt": _iso(row["created_at"]),
        "status": STATUS[row["status"]],
        "model": model,
        "input": [inputs["video"], inputs["audio"]],
        "options": {"sync_mode": job_sync_mode(req)},
        "outputUrl": signed_url("output", row["id"], row["base_url"]) if done else None,
        "outputDuration": row["duration"] if done else None,
        "outputFileName": req.get("outputFileName"),
        "webhookUrl": row["callback_url"],
        "error": row["failure_message"] if failed else None,
        "errorCode": row["failure_code"] if failed else None,
    }


def asset_view(a: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    mime = a["mime_type"] or ""
    created = _iso(a["created_at"])
    return {
        "id": a["id"], "createdAt": created, "updatedAt": created, "name": a["filename"],
        "url": signed_url("asset", a["id"], base_url), "size": a["size_bytes"],
        "format": Path(a["path"]).suffix.lstrip("."),
        "type": "VIDEO" if mime.startswith("video/") else "AUDIO",
        "visibility": "USER",
    }


# --------------------------------------------------------------- app state
settings.ensure_dirs()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(settings.logs_dir / "server.log", encoding="utf-8")])
store = Store(str(settings.db_path))
scheduler = Scheduler(store, generation_view)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    keys = settings.api_keys()
    log.info("API keys loaded from %s (%d key(s))", settings.data_dir / "api_keys.txt", len(keys))
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(title="HeyGem12 API (sync.so v2 lipsync-compatible)", version="2.0.0", lifespan=lifespan)


@app.exception_handler(ApiError)
async def _api_error(_req: Request, exc: ApiError):
    return JSONResponse(error_body(exc.status, exc.code, exc.message, exc.field), status_code=exc.status)


@app.exception_handler(RequestValidationError)
async def _validation_error(_req: Request, exc: RequestValidationError):
    e = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(x) for x in e.get("loc", [])[1:]) or None
    return JSONResponse(error_body(400, "invalid_parameter", e.get("msg", "invalid request"), field),
                        status_code=400)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_req: Request, exc: StarletteHTTPException):
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code)
    return JSONResponse(error_body(exc.status_code, code, str(exc.detail)), status_code=exc.status_code)


OPEN_PATHS = ("/health", "/v2/files/", "/docs", "/openapi.json", "/redoc")


@app.middleware("http")
async def _auth(request: Request, call_next):
    path = request.url.path
    if not any(path == p or (p.endswith("/") and path.startswith(p)) for p in OPEN_PATHS):
        key = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if not key and auth.lower().startswith("bearer "):
            key = auth[7:].strip()
        if not key or not any(hmac.compare_digest(key, k) for k in settings.api_keys()):
            return JSONResponse(error_body(401, "unauthorized", "missing or invalid API key (send x-api-key)"),
                                status_code=401)
    return await call_next(request)


# --------------------------------------------------------------- helpers
async def _json_body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        raise ApiError(400, "invalid_json", "request body must be valid JSON")
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_parameter", "request body must be a JSON object")
    return body


def _limit(v: Optional[str], default: int) -> int:
    try:
        n = int(v) if v is not None else default
    except ValueError:
        raise ApiError(400, "invalid_parameter", "limit must be an integer", "limit")
    if not 1 <= n <= 100:
        raise ApiError(400, "invalid_parameter", "limit must be between 1 and 100", "limit")
    return n


def _csv(v: Optional[str]) -> List[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


async def _save_upload(up: UploadFile, field: str) -> Dict[str, Any]:
    """Store an uploaded file as an asset. Returns the asset row."""
    ext = Path(up.filename or "").suffix.lower()
    if ext not in MEDIA_EXT:
        raise ApiError(400, "unsupported_file_type",
                       f"unsupported file type {ext or '(none)'}; allowed: {', '.join(sorted(MEDIA_EXT))}", field)
    aid = new_id()
    dest = settings.assets_dir / f"{aid}{ext}"
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    with open(dest, "wb") as f:
        while chunk := await up.read(1 << 20):
            size += len(chunk)
            if size > limit:
                f.close()
                dest.unlink(missing_ok=True)
                raise ApiError(413, "file_too_large", f"file exceeds {settings.max_upload_mb} MB", field)
            f.write(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise ApiError(400, "invalid_parameter", "file is empty", field)
    return store.add_asset(aid, up.filename, str(dest), MEDIA_EXT[ext], size)


def _drop_asset(a: Dict[str, Any]) -> None:
    store.delete_asset(a["id"])
    Path(a["path"]).unlink(missing_ok=True)


def _validate_generate(body: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a create-generation body; returns the normalized request stored with the job."""
    model = body.get("model") or "heygem12"
    if model not in MODELS:
        raise ApiError(400, "invalid_parameter", f"model must be one of: {', '.join(MODELS)}", "model")
    for name in UNSUPPORTED_TOP:
        if body.get(name) not in (None, [], {}):
            raise ApiError(400, "unsupported_parameter", f"{name} is not supported by this server", name)

    items = body.get("input")
    if not isinstance(items, list) or not items:
        raise ApiError(400, "invalid_parameter", "input must be a list with one video and one audio item", "input")
    found: Dict[str, Dict[str, Any]] = {}
    for i, it in enumerate(items):
        f = f"input[{i}]"
        if not isinstance(it, dict):
            raise ApiError(400, "invalid_parameter", f"{f} must be an object", f)
        t = it.get("type")
        if t in ("image", "text") or it.get("provider") is not None:
            raise ApiError(400, "unsupported_parameter", f"{f}: only video and audio inputs are supported", f)
        if t not in ("video", "audio"):
            raise ApiError(400, "invalid_parameter", f"{f}.type must be 'video' or 'audio'", f"{f}.type")
        if t in found:
            raise ApiError(400, "invalid_parameter", f"input must contain exactly one {t} item", f)
        for name in ("segments_secs", "segments_frames"):
            if it.get(name) is not None:
                raise ApiError(400, "unsupported_parameter", f"{f}.{name} is not supported", f"{f}.{name}")
        url, aid = it.get("url"), it.get("assetId")
        if bool(url) == bool(aid):
            raise ApiError(400, "invalid_parameter", f"{f} needs exactly one of url or assetId", f)
        if url:
            if not isinstance(url, str) or not re.match(r"^https?://", url):
                raise ApiError(400, "invalid_parameter", f"{f}.url must be an http(s) URL", f"{f}.url")
            found[t] = {"type": t, "url": url}
        else:
            if not isinstance(aid, str) or not store.get_asset(aid):
                raise ApiError(404, "asset_not_found", f"{f}.assetId does not exist", f"{f}.assetId")
            found[t] = {"type": t, "assetId": aid}
    for t in ("video", "audio"):
        if t not in found:
            raise ApiError(400, "invalid_parameter", f"input must contain one {t} item", "input")

    opts = body.get("options") or {}
    if not isinstance(opts, dict):
        raise ApiError(400, "invalid_parameter", "options must be an object", "options")
    sync_mode = opts.get("sync_mode") or "bounce"
    if sync_mode not in SYNC_MODES:
        code = "unsupported_parameter" if sync_mode in ("silence", "remap") else "invalid_parameter"
        raise ApiError(400, code, f"options.sync_mode must be one of: {', '.join(SYNC_MODES)}", "options.sync_mode")
    for k, v in opts.items():
        if k != "sync_mode" and v is not None:
            raise ApiError(400, "unsupported_parameter", f"options.{k} is not supported by this server",
                           f"options.{k}")

    hook = body.get("webhookUrl")
    if hook is not None and (not isinstance(hook, str) or not re.match(r"^https?://", hook)):
        raise ApiError(400, "invalid_parameter", "webhookUrl must be an http(s) URL", "webhookUrl")
    fname = body.get("outputFileName")
    if fname is not None and (not isinstance(fname, str) or not FILE_NAME_RE.match(fname)):
        raise ApiError(400, "invalid_parameter", "outputFileName may only contain letters, digits, _ and -, "
                       "up to 255 characters", "outputFileName")
    return {"model": model, "precision": MODELS[model], "input": [found["video"], found["audio"]],
            "options": {"sync_mode": sync_mode}, "webhookUrl": hook or None, "outputFileName": fname}


async def _read_multipart(request: Request) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """multipart/form-data create: files under video/audio, nested fields as JSON strings.
    Returns (body, assets created from the uploaded files)."""
    try:
        form = await request.form(max_files=2, max_fields=32)
    except Exception as e:
        raise ApiError(400, "invalid_parameter", f"cannot parse multipart body: {e}")
    body: Dict[str, Any] = {}
    for k, v in form.multi_items():
        if isinstance(v, UploadFile):
            continue
        if k in ("input", "options", *UNSUPPORTED_TOP):
            try:
                body[k] = json.loads(v) if v else None
            except ValueError:
                raise ApiError(400, "invalid_parameter", f"{k} must be a JSON string", k)
        else:
            body[k] = v
    created: List[Dict[str, Any]] = []
    try:
        for k, v in form.multi_items():
            if not isinstance(v, UploadFile):
                continue
            if k not in ("video", "audio"):
                raise ApiError(400, "unsupported_parameter", f"file field {k!r} is not supported "
                               "(use video and audio)", k)
            a = await _save_upload(v, k)
            created.append(a)
            body.setdefault("input", [])
            if not isinstance(body["input"], list):
                raise ApiError(400, "invalid_parameter", "input must be a list", "input")
            body["input"].append({"type": k, "assetId": a["id"]})
    except Exception:
        for a in created:
            _drop_asset(a)
        raise
    finally:
        await form.close()
    return body, created


# ================================================================= routes
@app.get("/health")
def health():
    return {"status": "ok" if scheduler.state != "error" else "degraded",
            "worker": scheduler.state, "queue_length": store.queue_length(),
            "running": scheduler.current is not None}


# ------------------------------------------------------------ generations
@app.post("/v2/generate")
async def create_generation(request: Request):
    key = request.headers.get("idempotency-key")
    scope = "POST /v2/generate"
    if key is not None:
        if not IDEM_RE.match(key):
            raise ApiError(400, "invalid_parameter", "Idempotency-Key must be 1-128 chars of [A-Za-z0-9._~-]",
                           "Idempotency-Key")
        prev = store.idem_get(key, scope)
        if prev:
            if prev["status_code"] is None:
                raise ApiError(409, "request_in_progress", "a request with this Idempotency-Key is in progress")
            row = store.get_lipsync(json.loads(prev["response"]).get("id", ""))
            if row and row["status"] != "failed":
                return JSONResponse(generation_view(row), status_code=200,
                                    headers={"Idempotency-Replayed": "true"})
            store.idem_delete(key, scope)     # original failed or was deleted: start a new one
        if not store.idem_reserve(key, scope):
            raise ApiError(409, "request_in_progress", "a request with this Idempotency-Key is in progress")
    created: List[Dict[str, Any]] = []
    try:
        if request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
            body, created = await _read_multipart(request)
        else:
            body = await _json_body(request)
        try:
            req = _validate_generate(body)
        except ApiError:
            for a in created:
                _drop_asset(a)
            raise
        lid = store.add_lipsync(req, _base_url(request))
        scheduler.notify()
    except Exception:
        if key is not None:
            store.idem_release(key, scope)
        raise
    if key is not None:
        store.idem_store(key, scope, 201, {"id": lid})
    return JSONResponse(generation_view(store.get_lipsync(lid)), status_code=201)


def _get_row(gid: str) -> Dict[str, Any]:
    r = store.get_lipsync(gid)
    if not r:
        raise ApiError(404, "generation_not_found", "generation not found", "id")
    return r


@app.get("/v2/generate/{gid}")
async def get_generation(gid: str, wait: Optional[str] = None, timeout: Optional[float] = None,
                         include: Optional[str] = None):
    r = _get_row(gid)
    if (wait or "").lower() == "true":
        t = 5.0 if timeout is None else timeout
        if not 1 <= t <= 10:
            raise ApiError(400, "invalid_parameter", "timeout must be between 1 and 10 seconds", "timeout")
        deadline = time.monotonic() + t
        while r["status"] not in ("completed", "failed") and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            r = _get_row(gid)
    headers = {} if r["status"] in ("completed", "failed") else {"Retry-After": "2"}
    return JSONResponse(generation_view(r), headers=headers)


@app.get("/v2/generations")
def list_generations(status: Optional[str] = None, ids: Optional[str] = None,
                     limit: Optional[str] = None, cursor: Optional[str] = None):
    statuses = _csv(status)
    bad = [s for s in statuses if s not in STATUS_REV]
    if bad:
        raise ApiError(400, "invalid_parameter", f"unknown status {bad[0]!r}; use {', '.join(STATUS_REV)}",
                       "status")
    if cursor and limit is None:
        raise ApiError(400, "invalid_parameter", "cursor requires limit", "cursor")
    n = _limit(limit, 100)
    before = None
    if cursor:
        c = store.get_lipsync(cursor)
        if not c:
            raise ApiError(400, "invalid_parameter", "cursor must be the id of a generation", "cursor")
        before = c["seq"]
    rows = store.list_lipsyncs(n, before, [STATUS_REV[s] for s in statuses], _csv(ids))
    return [generation_view(r) for r in rows[:n]]


@app.get("/v2/generations/{gid}/download")
def download_generation(gid: str, request: Request):
    r = _get_row(gid)
    if r["status"] == "failed":
        raise ApiError(422, "generation_not_downloadable", "generation failed; see its error", "id")
    if r["status"] != "completed":
        raise ApiError(409, "generation_not_ready", "generation is still pending or processing", "id")
    if not r["output_path"] or not Path(r["output_path"]).exists():
        raise ApiError(404, "generation_output_unavailable", "output file is missing", "id")
    return PlainTextResponse(signed_url("output", gid, _base_url(request)))


@app.delete("/v2/generations/{gid}")
def delete_generation(gid: str):
    r = _get_row(gid)
    if r["status"] == "running":
        raise ApiError(409, "generation_in_progress", "generation is processing; delete it after it finishes")
    store.delete_lipsync(gid)
    if r["output_path"]:
        Path(r["output_path"]).unlink(missing_ok=True)
    return {"id": gid}


# ----------------------------------------------------------------- assets
@app.post("/v2/assets", status_code=201)
async def create_asset(request: Request):
    """multipart field 'file' (upload), or JSON {url, type, name} (server downloads the URL)."""
    if request.headers.get("content-type", "").lower().startswith("multipart/form-data"):
        form = await request.form(max_files=1, max_fields=8)
        try:
            up = form.get("file")
            if not isinstance(up, UploadFile):
                raise ApiError(400, "invalid_parameter", "multipart upload needs a 'file' field", "file")
            a = await _save_upload(up, "file")
        finally:
            await form.close()
        return asset_view(a, _base_url(request))
    body = await _json_body(request)
    url, typ = body.get("url"), body.get("type")
    if not isinstance(url, str) or not re.match(r"^https?://", url):
        raise ApiError(400, "invalid_parameter", "url must be an http(s) URL", "url")
    if typ not in ("VIDEO", "AUDIO"):
        raise ApiError(400, "invalid_parameter", "type must be VIDEO or AUDIO", "type")
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in MEDIA_EXT:
        ext = ".mp4" if typ == "VIDEO" else ".wav"
    aid = new_id()
    dest = settings.assets_dir / f"{aid}{ext}"
    try:
        await run_in_threadpool(media.download, url, dest)
        info = await run_in_threadpool(media.probe, str(dest))
    except media.MediaError as e:
        dest.unlink(missing_ok=True)
        raise ApiError(422, e.code, e.message, "url")
    if not info["video" if typ == "VIDEO" else "audio"]:
        dest.unlink(missing_ok=True)
        raise ApiError(422, "invalid_media", f"file has no {typ.lower()} stream", "url")
    name = body.get("name") if isinstance(body.get("name"), str) else Path(url.split("?")[0]).name
    a = store.add_asset(aid, name, str(dest), MEDIA_EXT[ext], dest.stat().st_size)
    return asset_view(a, _base_url(request))


@app.get("/v2/assets")
def list_assets(request: Request, limit: Optional[str] = None, cursor: Optional[str] = None,
                types: Optional[str] = None):
    tl = [t.upper() for t in _csv(types)]
    if any(t not in ("VIDEO", "AUDIO", "IMAGE") for t in tl):
        raise ApiError(400, "invalid_parameter", "types must be VIDEO, AUDIO or IMAGE", "types")
    prefixes = [t.lower() + "/" for t in tl]
    if tl and not [p for p in prefixes if p != "image/"]:
        return {"items": [], "nextCursor": None, "totalCount": 0}
    n = _limit(limit, 20)
    try:
        off = int(cursor) if cursor else 0
    except ValueError:
        raise ApiError(400, "invalid_parameter", "invalid cursor", "cursor")
    rows = store.list_assets(n, max(0, off), prefixes)
    base = _base_url(request)
    return {"items": [asset_view(a, base) for a in rows[:n]],
            "nextCursor": str(off + n) if len(rows) > n else None,
            "totalCount": store.count_assets(prefixes)}


@app.get("/v2/assets/{asset_id}")
def get_asset(asset_id: str, request: Request):
    a = store.get_asset(asset_id)
    if not a:
        raise ApiError(404, "asset_not_found", "asset not found", "id")
    return asset_view(a, _base_url(request))


@app.delete("/v2/assets/{asset_id}")
def delete_asset(asset_id: str):
    a = store.get_asset(asset_id)
    if not a:
        raise ApiError(404, "asset_not_found", "asset not found", "id")
    _drop_asset(a)
    return {"id": asset_id}


# ------------------------------------------------- signed file downloads
@app.get("/v2/files/{kind}/{ident}")
def download_file(kind: str, ident: str, expires: int = 0, signature: str = ""):
    if kind not in ("output", "asset") or expires < time.time() or \
            not hmac.compare_digest(signature, _sign(kind, ident, expires)):
        raise ApiError(403, "forbidden", "invalid or expired link")
    if kind == "output":
        r = store.get_lipsync(ident)
        path = r and r["output_path"]
        fname = r and json.loads(r["request"]).get("outputFileName")
        name, mime = f"{fname or ident}.mp4", "video/mp4"
    else:
        a = store.get_asset(ident)
        path = a and a["path"]
        name, mime = (a["filename"] if a else ident), (a["mime_type"] if a else None)
    if not path or not Path(path).exists():
        raise ApiError(404, "not_found", "file not found")
    return FileResponse(path, media_type=mime, filename=name)
