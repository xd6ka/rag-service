"""Probe: is the measured latency the service or the HTTP client on this host?"""
import json
import os
import subprocess
import sys
import time

import httpx

PORT = 8766
env = {**os.environ, "RAG_BACKEND": "stub"}
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "service.api:app", "--host", "127.0.0.1",
     "--port", str(PORT), "--log-level", "warning"],
    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
time.sleep(6)
body = json.dumps({"question": "curl probe"})
try:
    for index in range(3):
        out = subprocess.run(
            ["curl.exe", "-s", "-o", "NUL", "-w", "%{time_total}", "-X", "POST",
             f"http://127.0.0.1:{PORT}/ask", "-H", "Content-Type: application/json",
             "-d", body],
            capture_output=True, text=True,
        )
        print("curl_s", out.stdout.strip(), out.stderr.strip()[:60])

    with httpx.Client(trust_env=False, timeout=60) as client:
        for index in range(3):
            started = time.perf_counter()
            client.get(f"http://127.0.0.1:{PORT}/health")
            print("httpx pooled health ms", round((time.perf_counter() - started) * 1000, 1))
finally:
    proc.terminate()
