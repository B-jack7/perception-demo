#!/bin/zsh
set -e

cd -- "$(dirname -- "$0")"

demo_port=8765
demo_url="http://127.0.0.1:${demo_port}/"
demo_log="/tmp/perception-demo-server.log"

python3 -m http.server "$demo_port" >"$demo_log" 2>&1 &
demo_server_pid=$!

cleanup() {
  kill "$demo_server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 1
open "$demo_url"

echo "Active Multi-View RGB-D Demo"
echo "Open: $demo_url"
echo "Press Control-C to stop the local server."

wait "$demo_server_pid"

