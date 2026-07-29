#!/usr/bin/env python3
"""A deliberately awkward HTTP server, for testing the agent's transport.

Every route here is a thing a real server does that the agent must survive: a redirect that would leak the
bearer token, a body that never ends, a response that arrives too slowly, an HTML error page from a proxy.
Prints the port it bound to on stdout so the test does not have to guess a free one.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass                                        # quiet: the test owns stdout

    def _send(self, status, body=b"", ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _echo(self):
        """Reflect what the agent sent, so the test can assert on headers and body."""
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else b""
        payload = {
            "method": self.command,
            "path": self.path,
            "authorization": self.headers.get("Authorization", ""),
            "user_agent": self.headers.get("User-Agent", ""),
            "content_type": self.headers.get("Content-Type", ""),
            "node_header": self.headers.get("X-Brae-Node", ""),
            "body": body.decode("utf-8", "replace"),
        }
        self._send(200, json.dumps(payload).encode())

    def _route(self):
        path = self.path.split("?")[0]
        if path in ("/echo", "/"):
            return self._echo()
        if path == "/status/500":
            return self._send(500, b'{"error":{"code":"internal_error","message":"boom"}}')
        if path == "/status/401":
            return self._send(401, b'{"error":{"code":"unauthorized","message":"no"}}')
        if path == "/proxy-html":
            return self._send(502, b"<html><body>502 Bad Gateway</body></html>", "text/html")
        if path == "/redirect":
            # If the agent followed this, it would hand the bearer token to another host.
            self.send_response(302)
            self.send_header("Location", "/leaked")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/leaked":
            return self._send(200, b'{"leaked":true}')
        if path == "/huge":
            # Far larger than the agent's cap; it must stop reading rather than swallow it.
            chunk = b"x" * 65536
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(chunk) * 64))     # 4 MiB
            self.end_headers()
            try:
                for _ in range(64):
                    self.wfile.write(chunk)
            except Exception:
                pass
            return
        if path == "/slow":
            time.sleep(10)
            return self._send(200, b'{"slow":true}')
        return self._send(404, b'{"error":{"code":"not_found","message":"no such route"}}')

    def do_GET(self):
        self._route()

    def do_POST(self):
        self._route()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_address[1], flush=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
