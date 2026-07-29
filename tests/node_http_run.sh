#!/usr/bin/env bash
# Start the awkward test server, run the transport test against it, stop the server.
# The server binds port 0 and prints what it got, so parallel ctest runs cannot collide.
set -u
BIN="${1:?test_node_http binary}"
SERVER="${2:?node_http_server.py}"

command -v python3 > /dev/null || { echo "SKIP: python3 not available"; exit 125; }

port_file="$(mktemp)"
python3 "$SERVER" > "$port_file" 2>/dev/null &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null; rm -f "$port_file"' EXIT

for _ in $(seq 1 50); do
    PORT="$(head -1 "$port_file" 2>/dev/null)"
    [ -n "${PORT:-}" ] && break
    sleep 0.1
done
[ -n "${PORT:-}" ] || { echo "FAIL: the test server never reported a port"; exit 1; }

"$BIN" "http://127.0.0.1:$PORT"
