"""
neuroforge.dashboard.server
===========================

A zero-dependency live training dashboard served from the Python standard
library's ``http.server``. The training loop calls ``dashboard.log(...)``;
the browser polls ``/api/state`` and renders loss curves, learning rate,
throughput, generated samples, an interactive text-generation playground and
per-head attention heatmaps.

Usage
-----
    dash = Dashboard(port=8787)
    dash.set_config(model="GPT", params="1.2M")
    dash.generate_fn = lambda prompt, n, temp: ...   # optional
    dash.attention_fn = lambda prompt: {...}        # optional
    for step in range(steps):
        ...
        dash.log(step, loss=float(loss.data), lr=lr)
    dash.finish()
"""

from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class Dashboard:
    def __init__(self, port: int = 8787, host: str = "127.0.0.1", open_browser: bool = False,
                 title: str = "NeuroForge"):
        self.port, self.host, self.title = port, host, title
        self.lock = threading.Lock()
        self.metrics: List[Dict] = []
        self.samples: List[Dict] = []
        self.config: Dict = {"title": title}
        self.status = "running"
        self.started_at = time.time()
        self.generate_fn: Optional[Callable[[str, int, float], str]] = None
        self.attention_fn: Optional[Callable[[str], Dict]] = None
        self.graph: Optional[Dict] = None
        self._server = ThreadingHTTPServer((host, port), self._handler_class())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[dashboard] live at {self.url}")
        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def set_config(self, **kwargs) -> None:
        with self.lock:
            self.config.update(kwargs)

    def log(self, step: int, **metrics) -> None:
        row = {"step": int(step), "time": time.time() - self.started_at}
        row.update({k: (float(v) if v is not None else None) for k, v in metrics.items()})
        with self.lock:
            self.metrics.append(row)

    def add_sample(self, step: int, text: str, label: str = "sample") -> None:
        with self.lock:
            self.samples.append({"step": int(step), "text": text, "label": label})
            self.samples = self.samples[-50:]

    def finish(self, keep_alive: bool = True) -> None:
        with self.lock:
            self.status = "finished"
        if keep_alive:
            print(f"[dashboard] training finished. Dashboard still live at {self.url} - press Ctrl+C to exit.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        self.close()

    def close(self) -> None:
        self._server.shutdown()

    # ------------------------------------------------------------------ #
    def _state(self) -> Dict:
        with self.lock:
            return {
                "config": self.config,
                "metrics": self.metrics,
                "samples": self.samples,
                "status": self.status,
                "elapsed": time.time() - self.started_at,
                "has_generate": self.generate_fn is not None,
                "has_attention": self.attention_fn is not None,
                "has_graph": self.graph is not None,
            }

    def _handler_class(self):
        dash = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence request logging
                pass

            def _send(self, code: int, body: bytes, ctype: str = "application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, obj, code: int = 200):
                self._send(code, json.dumps(obj).encode("utf-8"))

            def do_GET(self):
                parsed = urlparse(self.path)
                qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
                path = parsed.path
                try:
                    if path in ("/", "/index.html"):
                        with open(os.path.join(_STATIC, "index.html"), "rb") as f:
                            self._send(200, f.read(), "text/html; charset=utf-8")
                    elif path == "/api/state":
                        self._json(dash._state())
                    elif path == "/api/generate":
                        if dash.generate_fn is None:
                            return self._json({"error": "generation not available"}, 404)
                        prompt = qs.get("prompt", "")
                        n = int(qs.get("tokens", 100))
                        temp = float(qs.get("temperature", 0.8))
                        t0 = time.time()
                        text = dash.generate_fn(prompt, n, temp)
                        self._json({"text": text, "seconds": time.time() - t0})
                    elif path == "/api/attention":
                        if dash.attention_fn is None:
                            return self._json({"error": "attention not available"}, 404)
                        self._json(dash.attention_fn(qs.get("prompt", "")))
                    elif path == "/api/graph":
                        self._json(dash.graph or {"nodes": [], "edges": []})
                    else:
                        self._json({"error": "not found"}, 404)
                except Exception as e:  # surface errors to the browser instead of killing training
                    self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        return Handler
