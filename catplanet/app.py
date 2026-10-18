"""
app.py

The Cat Planet control panel: a small web page, served only on this Mac, for
choosing settings and running batch.py or pipeline.py without typing flags.

    python app.py              # opens http://localhost:8765 in your browser
    python app.py --port 9000 --no-browser

It uses only Python's standard library. Behind the scenes it runs the same
batch.py / pipeline.py commands you could type yourself (the page shows the
exact command), so retries, resume and every safety check work the same way.

Pieces
------
    app_options.py     every option's label, dropdown choices and ⓘ note
    ui/index.html      the page itself
    this file          the local web server, the run manager and the music
                       server manager

Saved profiles live in config/ui_profiles.json (not in git). Each run's full
log is kept in data/app_logs/.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import app_options, local_settings, paths, progress

UI_FILE = Path(__file__).resolve().parent / "ui" / "index.html"
PROFILES_FILE = paths.CONFIG_DIR / "ui_profiles.json"
LOG_DIR = paths.DATA_DIR / "app_logs"
MUSIC_LOG = LOG_DIR / "music_server.log"
OLLAMA_URL = "http://127.0.0.1:11434/api/tags"
DEFAULT_MUSIC_SERVER = "http://127.0.0.1:8001"
MAX_LOG_LINES = 6000

VERSION_LINE = re.compile(r"^\[(\d+)/(\d+)\] (.+?)  ->  version (\d+)\s*$")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _relative(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(paths.REPO))
    except ValueError:
        return str(p)


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


# =========================================================
# Profiles
# =========================================================
class Profiles:
    def __init__(self, path: Path = PROFILES_FILE):
        self.path = path
        self.lock = threading.Lock()

    def load(self) -> dict:
        with self.lock:
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text())
                    if data.get("profiles"):
                        return data
                except (json.JSONDecodeError, OSError):
                    pass
            data = {"profiles": app_options.starter_profiles(), "last_used": "My usual"}
            self._save(data)
            return data

    def _save(self, data: dict) -> None:
        progress.write_text(self.path, json.dumps(data, indent=2) + "\n")

    def save(self, name: str, values: dict) -> dict:
        name = name.strip()
        if not name:
            raise ValueError("Give the profile a name.")
        keep = {k: v for k, v in values.items() if k != "video"}
        data = self.load()
        with self.lock:
            data["profiles"][name] = keep
            data["last_used"] = name
            self._save(data)
        return data

    def delete(self, name: str) -> dict:
        data = self.load()
        with self.lock:
            if name in data["profiles"] and len(data["profiles"]) > 1:
                del data["profiles"][name]
                if data.get("last_used") == name:
                    data["last_used"] = next(iter(data["profiles"]))
                self._save(data)
        return data

    def use(self, name: str) -> None:
        data = self.load()
        with self.lock:
            if name in data["profiles"]:
                data["last_used"] = name
                self._save(data)


# =========================================================
# Music server (ACE-Step) manager
# =========================================================
class MusicServer:
    """Starts and stops "MUSIC GEN/start_api_server.command" in the background."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.started_at: float | None = None
        self.lock = threading.Lock()

    def status(self, url: str) -> dict:
        healthy = _http_ok(url.rstrip("/") + "/health")
        ours = self.proc is not None and self.proc.poll() is None
        if healthy:
            state = "running"
        elif ours:
            state = "starting"
        elif self.proc is not None and self.proc.poll() is not None:
            state = "crashed"
        else:
            state = "off"
        return {
            "state": state,
            "managed": ours,
            "launcher_found": paths.MUSIC_SERVER_LAUNCHER.exists(),
            "seconds": round(time.time() - self.started_at) if ours and self.started_at else None,
            "log": _relative(MUSIC_LOG),
        }

    def start(self, url: str) -> dict:
        with self.lock:
            if _http_ok(url.rstrip("/") + "/health"):
                return {"ok": True, "message": "The music server is already running."}
            if self.proc is not None and self.proc.poll() is None:
                return {"ok": True, "message": "The music server is already starting."}
            if not paths.MUSIC_SERVER_LAUNCHER.exists():
                return {"ok": False, "message": f"Can't find {_relative(paths.MUSIC_SERVER_LAUNCHER)}."}
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            log = open(MUSIC_LOG, "ab")
            log.write(f"\n===== started by the control panel {_now()} =====\n".encode())
            log.flush()
            self.proc = subprocess.Popen(
                ["/bin/bash", str(paths.MUSIC_SERVER_LAUNCHER)],
                cwd=paths.MUSIC_GEN_DIR, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            self.started_at = time.time()
            return {"ok": True, "message": "Starting the music server. The first "
                    "start loads about 5 GB, so give it a minute or two."}

    def stop(self) -> dict:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                return {"ok": False, "message": "The panel didn't start the music "
                        "server, so it won't stop it. Close its own window instead."}
            _stop_group(self.proc, grace=15)
            return {"ok": True, "message": "Stopped the music server."}


def _stop_group(proc: subprocess.Popen, grace: float) -> None:
    """Ctrl-C a process group, then insist if it doesn't stop in time."""
    for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


# =========================================================
# Runs (batch.py / pipeline.py)
# =========================================================
class Run:
    def __init__(self, mode: str, values: dict, argv: list[str], dry_run: bool):
        self.mode = mode
        self.values = values
        self.argv = argv
        self.dry_run = dry_run
        self.started = time.time()
        self.started_iso = _now()
        self.ended: float | None = None
        self.exit_code: int | None = None
        self.lines: list[str] = []
        self.dropped = 0
        self.proc: subprocess.Popen | None = None
        self.stop_requested = 0
        self.clips: list[dict] = []        # rows for the progress screen
        self.log_path = LOG_DIR / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{mode}.log")
        self.lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class Runner:
    def __init__(self):
        self.run: Run | None = None
        self.lock = threading.Lock()

    # ----------------------------------------------------------- start --
    def start(self, mode: str, values: dict, dry_run: bool) -> Run:
        with self.lock:
            if self.run is not None and self.run.running:
                raise RuntimeError("A run is already going. Stop it or wait for it to finish.")
            clean = app_options.normalize(values)
            argv = app_options.build_command(mode, values, dry_run=dry_run)
            run = Run(mode, clean, argv, dry_run)
            run.clips = self._planned_clips(mode, clean)
            if mode == "batch" and not run.clips:
                raise RuntimeError("The inbox is empty. Drop some videos into data/inbox/ first.")
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            script = paths.REPO / argv[0]
            run.proc = subprocess.Popen(
                [sys.executable, str(script)] + argv[1:],
                cwd=paths.REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True,
            )
            self.run = run
            threading.Thread(target=self._pump, args=(run,), daemon=True).start()
            return run

    def _planned_clips(self, mode: str, values: dict) -> list[dict]:
        if mode == "single":
            video = paths.resolve(values["video"])
            folder = app_options.single_output_dir(video)
            return [{"name": video.name, "version": None, "folder": folder,
                     "stem": video.stem}]
        from . import batch
        inbox = paths.resolve(values["inbox"]) if values.get("inbox") else paths.INBOX_DIR
        vids = batch.find_videos(inbox)
        if values.get("limit"):
            vids = vids[: values["limit"]]
        return [{"name": v.name, "version": None, "folder": None, "stem": None} for v in vids]

    def _outputs_dir(self, run: Run) -> Path:
        v = run.values.get("outputs")
        return paths.resolve(v) if v else paths.OUTPUTS_DIR

    def _pump(self, run: Run) -> None:
        with open(run.log_path, "w") as log:
            log.write("$ python " + " ".join(run.argv) + "\n\n")
            assert run.proc and run.proc.stdout
            for line in run.proc.stdout:
                line = line.rstrip("\n")
                log.write(line + "\n")
                log.flush()
                with run.lock:
                    run.lines.append(line)
                    if len(run.lines) > MAX_LOG_LINES:
                        cut = len(run.lines) - MAX_LOG_LINES
                        del run.lines[:cut]
                        run.dropped += cut
                    m = VERSION_LINE.match(line)
                    if m and run.mode == "batch":
                        i, version = int(m.group(1)) - 1, int(m.group(4))
                        if 0 <= i < len(run.clips):
                            row = run.clips[i]
                            row["version"] = version
                            row["stem"] = str(version)
                            row["folder"] = self._outputs_dir(run) / str(version)
            run.proc.wait()
            run.exit_code = run.proc.returncode
            run.ended = time.time()
            log.write(f"\n[exit code {run.exit_code}]\n")

    # ------------------------------------------------------------ stop --
    def stop(self) -> str:
        run = self.run
        if run is None or not run.running:
            return "Nothing is running."
        run.stop_requested += 1
        sig = signal.SIGINT if run.stop_requested == 1 else signal.SIGTERM
        try:
            os.killpg(run.proc.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGINT:
            return "Stopping after the current step is saved…"
        return "Forcing it to stop."

    def shutdown(self) -> None:
        run = self.run
        if run is not None and run.running:
            _stop_group(run.proc, grace=20)

    # -------------------------------------------------------- reporting --
    def snapshot(self, since: int) -> dict:
        run = self.run
        if run is None:
            return {"state": "idle"}
        with run.lock:
            start = max(0, since - run.dropped)
            lines = run.lines[start:]
            nxt = run.dropped + len(run.lines)
            clips = [self._clip_view(run, c) for c in run.clips]
        if run.running:
            state = "stopping" if run.stop_requested else "running"
        else:
            state = "finished"
        return {
            "state": state, "mode": run.mode, "dry_run": run.dry_run,
            "command": "python " + " ".join(run.argv),
            "argv": ["python"] + [_quote(a) for a in run.argv],
            "started": run.started_iso,
            "elapsed": round((run.ended or time.time()) - run.started),
            "exit_code": run.exit_code, "stopped": bool(run.stop_requested),
            "lines": lines, "next": nxt, "clips": clips,
            "log_file": _relative(run.log_path),
        }

    def _clip_view(self, run: Run, c: dict) -> dict:
        view = {"name": c["name"], "version": c["version"], "state": "waiting",
                "step": None, "why": None, "stages": []}
        want = [("describe", "describe"), ("script", "script")]
        if run.values.get("score"):
            want.append(("music_plan", "music plan"))
        if run.values.get("narrate"):
            want.append(("voice", "voiceover"))
        if run.values.get("score"):
            want.append(("music", "music"))
        folder, stem = c.get("folder"), c.get("stem")
        status = None
        if folder and stem:
            p = Path(folder) / f"{stem}.status.json"
            try:
                data = json.loads(p.read_text())
                last = data["runs"][-1]
                # Only trust a progress entry written during this run.
                if last.get("started_at", "") >= _utc_iso(run.started - 2):
                    status = last
            except (OSError, json.JSONDecodeError, KeyError, IndexError):
                status = None
        prog = (status or {}).get("progress", {})
        step = (status or {}).get("step", "")
        for key, label in want:
            how = prog.get(key)
            s = {"key": key, "label": label, "state": "todo"}
            if key == "music":
                takes = run.values.get("music_takes") or 1
                done = how["done"] if isinstance(how, dict) else 0
                s["detail"] = f"{done}/{takes}"
                if isinstance(how, dict) and how.get("done", 0) >= how.get("of", takes):
                    s["state"] = "reused" if how.get("reused") else "done"
                elif step.startswith("stage 4"):
                    s["state"] = "active"
            elif how in ("done", "reused"):
                s["state"] = how
            elif step and _step_is(step, key):
                s["state"] = "active"
            view["stages"].append(s)
        if status:
            result = status.get("result")
            view["step"] = step
            if result == "ok":
                view["state"] = "done"
            elif result == "failed":
                view["state"] = "failed"
                view["why"] = progress.describe_stop(Path(folder) / f"{stem}.status.json")
            else:
                view["state"] = "running" if run.running else "stopped"
                if not run.running:
                    view["why"] = f"stopped during {step}"
        elif run.running and c.get("version") is not None:
            view["state"] = "running"
        return view

    def results(self) -> dict:
        run = self.run
        if run is None:
            return {"clips": []}
        out = []
        for c in run.clips:
            view = self._clip_view(run, c)
            folder, stem = c.get("folder"), c.get("stem")
            item = {"name": c["name"], "version": c["version"], "state": view["state"],
                    "why": view["why"], "folder": None, "script": None,
                    "voiceover": None, "music": [], "servable": False}
            if folder and stem and Path(folder).is_dir():
                folder = Path(folder)
                item["folder"] = _relative(folder)
                item["servable"] = _servable(folder)
                sp = folder / f"{stem}.script.txt"
                if sp.exists():
                    item["script"] = sp.read_text()[:4000]
                vo = folder / f"{stem}.voiceover.wav"
                if vo.exists():
                    item["voiceover"] = _relative(vo)
                takes = sorted(folder.glob(f"{stem}.music*.wav"),
                               key=lambda p: [int(t) if t.isdigit() else t
                                              for t in p.name.split(".")])
                item["music"] = [_relative(p) for p in takes]
            out.append(item)
        return {"clips": out, "mode": run.mode, "dry_run": run.dry_run,
                "finished": not run.running, "exit_code": run.exit_code,
                "elapsed": round((run.ended or time.time()) - run.started),
                "stopped": bool(run.stop_requested)}


def _utc_iso(ts: float) -> str:
    from datetime import timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _step_is(step: str, key: str) -> bool:
    return {
        "describe": step.startswith("stage 1 "),
        "script": step.startswith("stage 2 "),
        "music_plan": step.startswith("stage 2b"),
        "voice": step.startswith("stage 3"),
    }.get(key, False)


def _servable(p: Path) -> bool:
    try:
        Path(p).resolve().relative_to(paths.DATA_DIR.resolve())
        return True
    except ValueError:
        return False


# =========================================================
# Ready check
# =========================================================
def ready_check(music: MusicServer, query: dict) -> dict:
    music_url = (query.get("music_server") or [DEFAULT_MUSIC_SERVER])[0] or DEFAULT_MUSIC_SERVER
    ref = (query.get("reference_audio") or [""])[0]
    video = (query.get("video") or [""])[0]

    from . import batch
    inbox = [p for p in batch.find_videos(paths.INBOX_DIR)]
    queued = []
    for v in inbox:
        note = None
        if v.stem.isdigit():
            note = progress.describe_stop(paths.OUTPUTS_DIR / v.stem / f"{v.stem}.status.json")
        queued.append({"name": v.name, "note": note})

    return {
        "ollama": _http_ok(OLLAMA_URL),
        "music": music.status(music_url),
        "inbox": queued,
        "reference_audio": {"path": ref, "exists": bool(ref) and paths.resolve(ref).exists()},
        "video": {"path": video, "exists": bool(video) and paths.resolve(video).is_file(),
                  "saves_to": _relative(app_options.single_output_dir(video)) + "/" if video else None},
        "saved_voice": local_settings.get("reference_audio"),
        "mac": sys.platform == "darwin",
    }


# =========================================================
# HTTP
# =========================================================
class Handler(BaseHTTPRequestHandler):
    server_version = "CatPlanet/1"
    app: "App"

    def log_message(self, fmt, *args):   # keep the terminal quiet
        pass

    # ---- helpers ----
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _json(self, data, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        return json.loads(self.rfile.read(n) or b"{}")

    # ---- GET ----
    def do_GET(self):
        if not self._host_ok():
            return self._error("Forbidden", 403)
        url = urlparse(self.path)
        q = parse_qs(url.query)
        app = self.app
        try:
            if url.path in ("/", "/index.html"):
                body = UI_FILE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/schema":
                self._json(app_options.schema())
            elif url.path == "/api/profiles":
                self._json(app.profiles.load())
            elif url.path == "/api/status":
                self._json(ready_check(app.music, q))
            elif url.path == "/api/run":
                since = int((q.get("since") or ["0"])[0])
                self._json(app.runner.snapshot(since))
            elif url.path == "/api/results":
                self._json(app.runner.results())
            elif url.path.startswith("/files/"):
                self._serve_file(unquote(url.path[len("/files/"):]))
            else:
                self._error("Not found", 404)
        except BrokenPipeError:
            pass
        except Exception as e:  # report, don't crash the server
            self._error(f"{type(e).__name__}: {e}", 500)

    def _serve_file(self, rel: str) -> None:
        p = (paths.REPO / rel).resolve()
        if not _servable(p) or not p.is_file():
            return self._error("Not found", 404)
        size = p.stat().st_size
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            if a:
                start = int(a)
                end = int(b) if b else size - 1
            else:   # suffix range: last N bytes
                start = max(0, size - int(b))
            end = min(end, size - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    # ---- POST ----
    def do_POST(self):
        # The custom header blocks other websites from posting here: browsers
        # won't send it cross-origin without a preflight this server refuses.
        if not self._host_ok() or self.headers.get("X-CatPlanet") != "1":
            return self._error("Forbidden", 403)
        url = urlparse(self.path)
        app = self.app
        try:
            body = self._body()
            if url.path == "/api/command":
                try:
                    argv = app_options.build_command(body["mode"], body.get("values", {}),
                                                     dry_run=bool(body.get("dry_run")))
                    self._json({"command": "python " + " ".join(_quote(a) for a in argv),
                                "argv": ["python"] + [_quote(a) for a in argv]})
                except ValueError as e:
                    self._json({"command": None, "error": str(e)})
            elif url.path == "/api/run":
                try:
                    app.runner.start(body["mode"], body.get("values", {}),
                                     bool(body.get("dry_run")))
                    if body.get("profile"):
                        app.profiles.use(body["profile"])
                    self._json({"ok": True})
                except (RuntimeError, ValueError) as e:
                    self._error(str(e), 409)
            elif url.path == "/api/stop":
                self._json({"message": app.runner.stop()})
            elif url.path == "/api/profiles":
                action = body.get("action")
                if action == "save":
                    self._json(app.profiles.save(body.get("name", ""), body.get("values", {})))
                elif action == "delete":
                    self._json(app.profiles.delete(body.get("name", "")))
                else:
                    self._error("Unknown profile action")
            elif url.path == "/api/music-server":
                url_ = body.get("url") or DEFAULT_MUSIC_SERVER
                if body.get("action") == "start":
                    self._json(app.music.start(url_))
                else:
                    self._json(app.music.stop())
            elif url.path == "/api/reveal":
                p = (paths.REPO / body.get("path", "")).resolve()
                if not str(p).startswith(str(paths.REPO.resolve())) or not p.exists():
                    return self._error("Not found", 404)
                if sys.platform == "darwin":
                    # A file is shown selected in its folder; a folder is opened.
                    subprocess.Popen(["open", "-R", str(p)] if p.is_file() else ["open", str(p)])
                    self._json({"ok": True})
                else:
                    self._json({"ok": False, "message": f"Open {_relative(p)} yourself."})
            else:
                self._error("Not found", 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            self._error(f"{type(e).__name__}: {e}", 500)


def _quote(a: str) -> str:
    return a if re.fullmatch(r"[\w@%+=:,./-]+", a) else "'" + a.replace("'", "'\\''") + "'"


class App:
    def __init__(self):
        self.profiles = Profiles()
        self.music = MusicServer()
        self.runner = Runner()

    def shutdown(self) -> None:
        print("\nShutting down: stopping any run (it can resume later)…", flush=True)
        self.runner.shutdown()
        if self.music.proc is not None and self.music.proc.poll() is None:
            print("Stopping the music server the panel started…", flush=True)
            _stop_group(self.music.proc, grace=15)


def main() -> int:
    ap = argparse.ArgumentParser(prog="app.py", description="The Cat Planet control panel.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="Don't open a browser tab.")
    args = ap.parse_args()

    problems = app_options.check_against_parser() + app_options.check_info()
    for p in problems:
        print(f"[warn] {p}", file=sys.stderr)

    app = App()
    Handler.app = app
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as e:
        print(f"error: can't use port {args.port} ({e}). Is the panel already open? "
              f"Try --port {args.port + 1}.", file=sys.stderr)
        return 1
    server.daemon_threads = True
    url = f"http://localhost:{args.port}"
    print(f"Cat Planet control panel: {url}\nPress Ctrl-C here to quit.", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    # Closing the Terminal window (SIGHUP) or a plain kill (SIGTERM) shuts
    # down the same careful way as Ctrl-C: the run is stopped so it can
    # resume later, and a music server the panel started is stopped too.
    def _graceful(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _graceful)
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
