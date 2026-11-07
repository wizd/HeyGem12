# HeyGem12 HTTP API

A small HTTP service for lip-sync: send one video and one audio track, get back a
video whose mouth movements follow the audio. The API follows the lip-sync core
of [sync.so](https://sync.so)'s v2 API, so the request and response shapes will be
familiar if you have used it. Features that need sync.so's cloud (text-to-speech,
dubbing, segments, projects, batches) are not implemented and are rejected with
an error rather than silently ignored.

```
POST /v2/generate                      create a job (JSON or multipart)       -> 201 generation
GET  /v2/generate/{id}                 poll (optional ?wait=true long-poll)   -> generation
GET  /v2/generations                   list jobs
GET  /v2/generations/{id}/download     signed download URL (text/plain)
DELETE /v2/generations/{id}
POST /v2/assets                        upload a file, or register a URL       -> 201 asset
GET  /v2/assets, GET/DELETE /v2/assets/{id}
GET  /health                           no key needed
```

## Running

```bash
pip install -r requirements-api.txt        # plus the pipeline's own requirements
python -m heygem12_api                     # listens on 0.0.0.0:8800
```

`ffmpeg` and `ffprobe` must be on `PATH`. On first start the server creates
`data/api_keys.txt` (one key per line; add or remove lines to manage keys, no
restart needed) and the URL-signing and webhook secrets in `data/`.

| Variable | Default | Meaning |
|---|---|---|
| `HEYGEM12_API_HOST` / `_PORT` | `0.0.0.0` / `8800` | Listen address |
| `HEYGEM12_API_PUBLIC_URL` | (request host) | Base URL used in `outputUrl`s, e.g. `https://lipsync.example.com` |
| `HEYGEM12_API_DATA` | `./data` | Database, uploads, outputs, logs, secrets |
| `HEYGEM12_API_CHECKPOINTS` | `./checkpoints` | Model weights |
| `HEYGEM12_API_DEVICE` | `cuda` | Torch device (`cuda` also means ROCm builds) |
| `HEYGEM12_API_MAX_UPLOAD_MB` | `1024` | Per-file limit for uploads and URL downloads |
| `HEYGEM12_API_MAX_DURATION` | `600` | Longest output in seconds (all frames are held in RAM) |
| `HEYGEM12_API_ALLOW_PRIVATE_URLS` | `0` | Allow input/webhook URLs that resolve to private addresses |
| `HEYGEM12_API_JOB_TIMEOUT_BASE` / `_PER_FRAME` | `300` / `0.5` | Watchdog budget; the GPU worker is restarted on overrun |
| `HEYGEM12_API_URL_TTL` | `604800` | Lifetime of signed download URLs (seconds) |
| `HEYGEM12_API_WARMUP` | `1` | Run a short warm-up job when the worker starts |

## Conventions

- **Auth:** `x-api-key: <key>` (or `Authorization: Bearer <key>`).
- **IDs** are UUIDs. Timestamps are ISO 8601 UTC.
- **Errors** are JSON: `{"message", "statusCode", "errorCode", "field", "docsUrl"}`.
  Branch on `errorCode`, not on `message`.
- **Idempotency:** send `Idempotency-Key` (1–128 chars of `[A-Za-z0-9._~-]`) on
  `POST /v2/generate`. A retry with the same key returns `200` with the original
  generation and `Idempotency-Replayed: true`; if the original failed or was
  deleted, a new one is started (`201`). Keys are kept for 7 days.
- **Download URLs** (`outputUrl`, asset `url`) are signed and expire (7 days by
  default). They need no API key, so treat them as secrets.

## Create a generation — `POST /v2/generate`

JSON form; each input is given by `url` or `assetId`:

```json
{
  "model": "heygem12",
  "input": [
    {"type": "video", "url": "https://example.com/face.mp4"},
    {"type": "audio", "assetId": "3f0c…"}
  ],
  "options": {"sync_mode": "bounce"},
  "webhookUrl": "https://example.com/hook",
  "outputFileName": "my_clip"
}
```

Multipart form, to send local files directly: file fields `video` and `audio`,
other fields as plain values, `input` and `options` as JSON strings. Uploaded
files are stored as assets.

```bash
curl -H "x-api-key: $KEY" -F model=heygem12 -F video=@face.mp4 -F audio=@voice.wav \
     -F 'options={"sync_mode":"cut_off"}' $API/v2/generate
```

| Field | Values |
|---|---|
| `model` | `heygem12` (default, fastest) or `heygem12-hq` (GFPGAN face restoration first; about 1.8× slower) |
| `input` | exactly one `video` and one `audio` item |
| `options.sync_mode` | what to do when the lengths differ, see below |
| `webhookUrl` | http(s) URL notified when the job ends |
| `outputFileName` | download file name without extension (`[A-Za-z0-9_-]`, ≤255) |

`sync_mode` — the output always uses the whole audio unless `cut_off` is chosen:

| Mode | Audio longer than video | Audio shorter than video |
|---|---|---|
| `bounce` (default) | video plays forward, backward, forward… (no visible jump) | video is trimmed |
| `loop` | video restarts from the beginning | video is trimmed |
| `cut_off` | audio is cut at the end of the video | video is trimmed |

Refused with `400 unsupported_parameter`: `image` and `text` (TTS) inputs,
`segments_secs`/`segments_frames`, top-level `segments`, `outputAudio`,
`dubParams`, `projectId`, `dialogueEdit`, sync modes `silence` and `remap`, and
any other `options` key (e.g. `temperature`, `active_speaker_detection`).

Response (`201`), also returned by `GET /v2/generate/{id}` and sent to webhooks:

```json
{
  "id": "1b6e…", "createdAt": "2026-11-07T18:40:02Z",
  "status": "COMPLETED",
  "model": "heygem12",
  "input": [{"type": "video", "assetId": "…"}, {"type": "audio", "assetId": "…"}],
  "options": {"sync_mode": "bounce"},
  "outputUrl": "http://host:8800/v2/files/output/1b6e…?expires=…&signature=…",
  "outputDuration": 6.0, "outputFileName": "my_clip",
  "webhookUrl": null, "error": null, "errorCode": null
}
```

`status` is `PENDING` → `PROCESSING` → `COMPLETED` or `FAILED`. `outputUrl` and
`outputDuration` are set only when completed; `error`/`errorCode` only when failed.

## Poll and download

- `GET /v2/generate/{id}` — current state. Non-final responses carry `Retry-After: 2`.
  `?wait=true&timeout=N` (1–10 s, default 5) holds the request until the job
  finishes or the timeout passes.
- `GET /v2/generations?status=COMPLETED,FAILED&ids=a,b&limit=20&cursor=<id>` —
  newest first, a JSON array. `cursor` is the id of the last item of the previous
  page and requires `limit`.
- `GET /v2/generations/{id}/download` — a fresh signed URL as `text/plain`.
  `409 generation_not_ready` while pending/processing, `422 generation_not_downloadable` if failed.
- `DELETE /v2/generations/{id}` — removes the job and its output (`409` while processing).

## Assets

- `POST /v2/assets` multipart with a `file` field → asset. Or JSON
  `{"url": "https://…", "type": "VIDEO" | "AUDIO", "name": "…"}`: the server
  downloads and checks the file now, so later jobs don't depend on the URL.
- `GET /v2/assets?limit=20&cursor=…&types=VIDEO,AUDIO` → `{items, nextCursor, totalCount}`.
- `GET /v2/assets/{id}`, `DELETE /v2/assets/{id}`.

Asset object: `{id, createdAt, updatedAt, name, url, size, format, type, visibility}`.
Accepted files: mp4, mov, webm, mkv, m4v, mp3, wav, m4a, aac, ogg, flac.

## Webhooks

When a job with `webhookUrl` ends, the server POSTs the generation object (as
above) to it, with:

- `Sync-Signature: t=<unix seconds>,v1=<hex HMAC-SHA256>`, computed over
  `"<t>.<raw body>"` with the secret in `data/webhook.secret`;
- `Sync-Delivery-Id`, unchanged across retries (use it to de-duplicate).

Delivery is retried up to 4 times (5 s, 15 s, 45 s, 135 s). Verify like this:

```python
t, v1 = (p.split("=", 1)[1] for p in header.split(","))
ok = hmac.compare_digest(v1, hmac.new(secret, f"{t}.".encode() + raw_body, hashlib.sha256).hexdigest())
```

## Failure codes (`errorCode` on a failed generation)

`download_failed`, `invalid_url`, `file_too_large`, `invalid_media`,
`invalid_video`, `invalid_audio`, `asset_not_found`, `duration_too_long`,
`timeout` (GPU worker was restarted), `worker_crashed`, `processing_failed`,
`internal_error`.

## Differences from sync.so

- Models are `heygem12` / `heygem12-hq`, not sync.so's model names.
- No TTS, dubbing, segments, image input, batches, projects, cost estimates,
  `include=progress`, or `GET /v2/errors`.
- Assets are uploaded directly (`POST /v2/assets` multipart) instead of through
  presigned upload URLs; `POST /v2/assets/upload` does not exist.
- Without an `Idempotency-Key`, identical requests are not rejected as duplicates.
- Webhook secret is a local file, not `GET /v2/organizations/webhook/secret`.
- The spec was taken from sync.so's official Python SDK (`syncsdk` 0.4.0); it has
  not been tested with sync.so's own client libraries.

## Operational notes

- One job runs at a time; others wait in a FIFO queue that survives restarts
  (jobs left running are re-queued).
- Input and webhook URLs must resolve to public addresses unless
  `HEYGEM12_API_ALLOW_PRIVATE_URLS=1`; redirects are re-checked.
- The server speaks plain HTTP. For internet use, put it behind an HTTPS
  reverse proxy and set `HEYGEM12_API_PUBLIC_URL`.
- Jobs created by the 1.x (HeyGen-style `/v3`) API remain readable through
  `/v2/generate/{id}` and `/v2/generations`.
