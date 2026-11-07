"""Job scheduler: one GPU worker process, FIFO queue, watchdog, webhooks."""
import hashlib
import hmac
import json
import logging
import multiprocessing as mp
import shutil
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import media
from .config import settings
from .store import Store, now
from .worker import worker_main

log = logging.getLogger("heygem12_api.scheduler")


class JobFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def job_inputs(req: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{'video': spec, 'audio': spec}; spec has 'url' or 'assetId'. Also reads the
    1.x request format ({video: {type: url|asset_id, ...}, audio: ...})."""
    if "input" in req:
        return {it["type"]: it for it in req["input"]}
    out = {}
    for t in ("video", "audio"):
        old = req[t]
        out[t] = {"type": t, "url": old["url"]} if old["type"] == "url" else {"type": t, "assetId": old["asset_id"]}
    return out


def job_sync_mode(req: Dict[str, Any]) -> str:
    if "input" in req:
        return (req.get("options") or {}).get("sync_mode") or "bounce"
    return "bounce" if req.get("enable_dynamic_duration", True) else "cut_off"


def job_precision(req: Dict[str, Any]) -> bool:
    return bool(req.get("precision")) or req.get("mode") == "precision"


class Scheduler:
    def __init__(self, store: Store, view: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.store = store
        self.view = view            # callable(row) -> public generation object (webhook payload)
        self.proc: Optional[mp.Process] = None
        self.conn = None
        self.state = "starting"     # starting | ready | busy | restarting | error
        self.current: Optional[str] = None
        self.worker_error: Optional[str] = None
        self.warmup_s: Optional[float] = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        n = self.store.requeue_running()
        if n:
            log.warning("requeued %d job(s) left running by a previous run", n)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._kill_worker()

    def notify(self) -> None:
        self._wake.set()

    def _start_worker(self) -> None:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        logp = str(settings.logs_dir / "worker.log")
        self.proc = ctx.Process(target=worker_main, name="heygem12-gpu-worker", daemon=True,
                                args=(child, settings.checkpoints, settings.device, logp, settings.warmup))
        self.proc.start()
        self.conn = parent
        log.info("GPU worker starting (pid %s)", self.proc.pid)
        if not parent.poll(600):
            self._kill_worker()
            raise RuntimeError("GPU worker did not become ready within 600 s")
        msg = parent.recv()
        if not msg.get("ready"):
            self._kill_worker()
            raise RuntimeError(f"GPU worker failed to start: {msg.get('error')}")
        self.warmup_s = msg.get("warmup_s")
        log.info("GPU worker ready in %.1fs", self.warmup_s or 0)

    def _kill_worker(self) -> None:
        p, self.proc = self.proc, None
        if p is None:
            return
        if p.is_alive():
            # /T also kills anything the worker spawned (ffmpeg)
            subprocess.run(["taskkill", "/f", "/t", "/pid", str(p.pid)], capture_output=True)
            p.join(10)
            if p.is_alive():
                p.kill()
                p.join(5)
        if p.is_alive():
            log.error("GPU worker pid %s could not be terminated (GPU hang?)", p.pid)

    def _ensure_worker(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            return
        self.state = "restarting" if self.proc is not None else "starting"
        self._kill_worker()
        self._start_worker()

    # ------------------------------------------------------------------ loop
    def _loop(self) -> None:
        backoff = 5
        while not self._stop.is_set():
            try:
                self._ensure_worker()
                self.state, self.worker_error, backoff = "ready", None, 5
            except Exception as e:
                self.state, self.worker_error = "error", str(e)[-2000:]
                log.error("worker start failed: %s", e)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 300)
                continue
            job = self.store.claim_next()
            if job is None:
                self._wake.wait(5)
                self._wake.clear()
                continue
            self.state, self.current = "busy", job["id"]
            try:
                self._process(job)
            except Exception as e:  # safety net: never let the loop die
                log.exception("unexpected error in job %s", job["id"])
                self._finish(job, failure=("internal_error", str(e)[:500]))
            finally:
                self.current = None
                self.state = "ready" if self.proc is not None and self.proc.is_alive() else "restarting"

    # ------------------------------------------------------------------- job
    def _resolve_input(self, spec: Dict[str, Any], workdir: Path, name: str) -> str:
        if spec.get("assetId"):
            a = self.store.get_asset(spec["assetId"])
            if not a or not Path(a["path"]).exists():
                raise JobFailure("asset_not_found", f"{name} asset {spec['assetId']} not found")
            return a["path"]
        suffix = Path(spec["url"].split("?")[0]).suffix[:8] or ".bin"
        return str(media.download(spec["url"], workdir / f"{name}_src{suffix}"))

    def _process(self, job: Dict[str, Any]) -> None:
        lid = job["id"]
        req = json.loads(job["request"])
        inputs = job_inputs(req)
        mode = job_sync_mode(req)
        workdir = settings.work_dir / lid
        workdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            try:
                vsrc = self._resolve_input(inputs["video"], workdir, "video")
                asrc = self._resolve_input(inputs["audio"], workdir, "audio")
                vinfo = media.probe(vsrc)
                if not vinfo["video"]:
                    raise JobFailure("invalid_video", "video input has no video stream")
                vdur = vinfo["video"]["duration"] or vinfo["duration"]
                wav = workdir / "audio.wav"
                # cut_off: output ends with the shorter input; otherwise it follows the audio
                adur = media.prepare_audio(asrc, wav, max_duration=vdur if mode == "cut_off" else None)
                out_dur = min(adur, vdur) if mode == "cut_off" else adur
                if out_dur > settings.max_duration_s:
                    raise JobFailure("duration_too_long",
                                     f"output would be {out_dur:.1f}s; limit is {settings.max_duration_s:.0f}s")
                vprep = workdir / "video.mp4"
                pinfo = media.prepare_video(vsrc, vprep, out_dur, workdir,
                                            extend="loop" if mode == "loop" else "bounce")
            except media.MediaError as e:
                raise JobFailure(e.code, e.message)
            t_prep = time.perf_counter() - t0

            frames = pinfo["video"]["duration"] * pinfo["video"]["fps"]
            budget = settings.job_timeout_base_s + settings.job_timeout_per_frame_s * frames
            out_tmp = workdir / "out.mp4"
            self.conn.send({"id": lid, "video": str(vprep), "audio": str(wav), "output": str(out_tmp),
                            "precision": job_precision(req)})
            if not self.conn.poll(budget):
                log.error("job %s exceeded %.0fs budget; killing GPU worker", lid, budget)
                self._kill_worker()
                raise JobFailure("timeout", f"processing exceeded {budget:.0f}s; GPU worker was restarted")
            try:
                res = self.conn.recv()
            except EOFError:
                self._kill_worker()
                raise JobFailure("worker_crashed", "GPU worker exited during processing")
            if not res.get("ok"):
                raise JobFailure("processing_failed", res.get("error") or "unknown error")
            if not out_tmp.exists():
                raise JobFailure("processing_failed", "pipeline produced no output file")
            final = settings.outputs_dir / f"{lid}.mp4"
            shutil.move(str(out_tmp), final)
            dur = media.probe(str(final))["duration"]
            log.info("job %s done (%s): prep %.1fs, gpu %.1fs, total %.1fs, %.1fs video",
                     lid, mode, t_prep, res.get("wall_s", 0), time.perf_counter() - t0, dur)
            self._finish(job, output=str(final), duration=dur)
        except JobFailure as e:
            log.warning("job %s failed: %s %s", lid, e.code, e.message)
            self._finish(job, failure=(e.code, e.message))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _finish(self, job, output: str = None, duration: float = None, failure=None) -> None:
        lid = job["id"]
        if self.store.get_lipsync(lid) is None:      # deleted meanwhile
            if output:
                Path(output).unlink(missing_ok=True)
            return
        if failure:
            self.store.update_lipsync(lid, status="failed", completed_at=now(),
                                      failure_code=failure[0], failure_message=failure[1])
        else:
            self.store.update_lipsync(lid, status="completed", completed_at=now(),
                                      output_path=output, duration=duration)
        job = self.store.get_lipsync(lid)
        if job.get("callback_url"):
            body = json.dumps(self.view(job)).encode()
            threading.Thread(target=self._post, args=(job["callback_url"], body), daemon=True).start()

    # -------------------------------------------------------------- webhooks
    def _post(self, url: str, body: bytes) -> None:
        """POST a webhook signed like sync.so: Sync-Signature: t=<unix>,v1=<hex HMAC-SHA256
        of '<t>.<raw body>'>, plus a Sync-Delivery-Id that stays the same across retries."""
        delivery = str(uuid.uuid4())
        for attempt in range(4):
            try:
                media._check_public_host(url)
                t = str(int(time.time()))
                sig = hmac.new(settings.webhook_secret, t.encode() + b"." + body, hashlib.sha256).hexdigest()
                req = urllib.request.Request(url, data=body, method="POST", headers={
                    "Content-Type": "application/json", "User-Agent": "heygem12-api/2.0",
                    "Sync-Signature": f"t={t},v1={sig}", "Sync-Delivery-Id": delivery})
                with urllib.request.urlopen(req, timeout=15) as r:
                    if 200 <= r.status < 300:
                        return
            except Exception as e:
                log.warning("webhook %s attempt %d failed: %s", url, attempt + 1, e)
            time.sleep(5 * (3 ** attempt))
        log.error("webhook %s gave up", url)
