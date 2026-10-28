"""Fault injection in disposable test containers only."""
from __future__ import annotations

import json
import subprocess
import uuid

from ner_test_support import ROOT, docker, wait_ready


def output(*args):
    return docker(*args, capture_output=True, text=True).stdout.strip()


COUNTER_SERVER = """
from http.server import BaseHTTPRequestHandler, HTTPServer
count = 0
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global count
        count += 1
        self.rfile.read(int(self.headers.get('Content-Length', '0')))
        self.send_response(503)
        self.end_headers()
        self.wfile.write(b'busy')
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str(count).encode())
    def log_message(self, *args): pass
HTTPServer(('0.0.0.0', 8000), Handler).serve_forever()
"""


def check_no_replay():
    prefix = "ru-ner-no-replay-" + uuid.uuid4().hex[:8]
    nginx, first, second = [prefix + suffix for suffix in ("-nginx", "-a", "-b")]
    names = [nginx, first, second]
    try:
        output("network", "create", "--internal", prefix)
        for name in (first, second):
            output("run", "-d", "--name", name, "--network", prefix, "--network-alias", "ner-triton",
                   "python:3.12-slim", "python", "-c", COUNTER_SERVER)
        output("run", "-d", "--name", nginx, "--network", prefix,
               "-v", f"{ROOT}/nginx/conf.d/triton.conf:/etc/nginx/conf.d/default.conf:ro", "nginx:1.27-alpine")
        output("exec", nginx, "nginx", "-t")
        get = "import urllib.request; print(urllib.request.urlopen('{url}', timeout=3).read().decode())"

        def ready():
            try:
                return all(output("exec", name, "python", "-c", get.format(url="http://127.0.0.1:8000/count")) == "0"
                           for name in (first, second)) and output(
                               "exec", nginx, "wget", "-qO-", "-T", "3",
                               "http://127.0.0.1:8000/v2/models/tiny2/versions/1/ready") == "0"
            except subprocess.CalledProcessError:
                return False

        wait_ready(ready, 30)
        ip = json.loads(output("inspect", nginx))[0]["NetworkSettings"]["Networks"][prefix]["IPAddress"]
        post = (
            "import urllib.request,urllib.error; "
            f"r=urllib.request.Request('http://{ip}:8000/v2/models/tiny2/versions/1/infer',data=b'test');\n"
            "try:\n urllib.request.urlopen(r,timeout=10)\n"
            "except urllib.error.HTTPError as e:\n print(e.code)"
        )
        assert output("exec", first, "python", "-c", post) == "503"
        counts = [int(output("exec", name, "python", "-c", get.format(url="http://127.0.0.1:8000/count")))
                  for name in (first, second)]
        assert sorted(counts) == [0, 1], "Nginx replayed one POST to another peer"
        return {"check": "single_post_no_replay", "status": "passed", "peer_post_counts": counts}
    finally:
        subprocess.run(["docker", "rm", "-f", *names], capture_output=True, timeout=30)
        removed = subprocess.run(["docker", "network", "rm", prefix], capture_output=True, timeout=30)
        if removed.returncode and subprocess.run(
            ["docker", "network", "inspect", prefix], capture_output=True, timeout=30
        ).returncode == 0:
            raise RuntimeError("No-replay test network cleanup failed")


def check_replica_recovery(gate, ctx, token):
    from deployment import DeploymentError
    from deployment_topology_gate import request, assert_request

    replicas = [c for c in ctx.containers() if c["Config"]["Labels"].get("com.docker.compose.service") == "ner-triton"]
    assert len(replicas) == 2
    first = replicas[0]
    patch = (
        "import json; from pathlib import Path; p=Path('/models/tiny2/export-manifest.json'); "
        "p.with_suffix('.test-backup').write_bytes(p.read_bytes()); d=json.loads(p.read_text()); "
        "d['onnx_sha256']='b'*64; p.write_text(json.dumps(d))"
    )
    output("exec", first["Id"], "python3", "-c", patch)
    try:
        try:
            ctx.health()
        except DeploymentError:
            pass
        else:
            raise AssertionError("One mismatched replica was hidden by the healthy peer")
    finally:
        output("exec", first["Id"], "python3", "-c",
               "from pathlib import Path; p=Path('/models/tiny2/export-manifest.json'); "
               "p.write_bytes(p.with_suffix('.test-backup').read_bytes()); p.with_suffix('.test-backup').unlink()")
    ctx.health(20)
    output("stop", first["Id"])
    try:
        ctx.health()
    except DeploymentError:
        pass
    else:
        raise AssertionError("Stopped replica was accepted as a complete healthy stack")
    url = gate.url(ctx, "nginx", 80)
    attempts = []

    def remaining_capacity():
        status, body = request(url, "/v1/chat/completions", {"model": "mock-chat", "messages": [
            {"role": "user", "content": f"Телефон клиента +79031234569. Проверка {len(attempts)}."}]}, token)
        attempts.append(status)
        if status == 200:
            assert "+79031234569" in body
            return True
        assert status == 503, "Unexpected failure while the remaining same-model peer recovers"
        assert "+79031234569" not in body
        return False

    # These are separate client attempts, not automatic replay of one POST.
    wait_ready(remaining_capacity, 45)
    network, old_info = next(iter(first["NetworkSettings"]["Networks"].items()))
    old_ip = old_info["IPAddress"]
    output("rm", first["Id"])
    holder = gate.project + "-address-holder"
    try:
        output("run", "-d", "--name", holder, "--network", network, "--ip", old_ip,
               "python:3.12-slim", "sleep", "240")
        ctx.start()
        current = [c for c in ctx.containers() if c["Config"]["Labels"].get("com.docker.compose.service") == "ner-triton"]
        replacement = next(c for c in current if c["Id"] not in {r["Id"] for r in replicas})
        assert replacement["NetworkSettings"]["Networks"][network]["IPAddress"] != old_ip
        assert_request(gate.url(ctx, "nginx", 80), "mask", "responses", True, token, "+79031234569")
    finally:
        subprocess.run(["docker", "rm", "-f", holder], capture_output=True, timeout=30)
    gate.result["checks"].append({"check": "triton_replica_recovery", "status": "passed",
                                  "wrong_artifact_rejected": True, "stopped_replica_rejected": True,
                                  "client_attempt_statuses": attempts, "replacement_new_ip": True})
