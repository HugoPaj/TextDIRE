from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.scoring import provider_name


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = json.dumps({"status": "ok", "provider": provider_name()}).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
