from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from time import perf_counter


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.scoring import build_analysis_response, resolve_mask_ratios, score_text


class handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("content-length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8") or "{}")

            text = str(payload.get("text", "")).strip()
            if len(text) < 40 or len(text) > 8000:
                self._send_json(
                    422,
                    {"detail": "Text must be between 40 and 8000 characters."},
                )
                return
            if len(text.split()) < 20:
                self._send_json(
                    422,
                    {"detail": "Please provide at least 20 words for a meaningful signal."},
                )
                return

            try:
                mask_ratios = resolve_mask_ratios(
                    str(payload.get("mode", "balanced")),
                    payload.get("mask_ratios"),
                )
            except ValueError as exc:
                self._send_json(422, {"detail": str(exc)})
                return

            started = perf_counter()
            provider_result = score_text(text, mask_ratios)
            response = build_analysis_response(provider_result, perf_counter() - started)
            self._send_json(200, response)
        except Exception as exc:
            self._send_json(502, {"detail": str(exc)})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status: int, body: dict) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_cors_headers(self) -> None:
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")
