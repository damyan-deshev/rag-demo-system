"""Provision the pinned upstream CPU deployment and a scoped teaching account."""

import getpass
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parent
STATE = ROOT / ".local-stack"
SOURCE = STATE / "ragflow"
VERSION = "db1cd5929a7885a3074547791f5ecc4d2f891739"
RAG = "http://127.0.0.1:9380"
OLLAMA = "http://127.0.0.1:11435/v1"
PROVIDER = "OpenAI-API-Compatible"


def write_private(path, value):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        f.write(value)
    Path(f.name).replace(path)


def keyfile():
    return (
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "rag-demo-system/keys.json"
    )


def saved_key(service, url):
    p = keyfile()
    return (
        json.loads(p.read_text()).get(service + ":" + url.rstrip("/"), "")
        if p.exists()
        else ""
    )


def remember(service, url, value):
    p = keyfile()
    keys = json.loads(p.read_text()) if p.exists() else {}
    keys[service + ":" + url.rstrip("/")] = value.removeprefix("Bearer ")
    write_private(p, json.dumps(keys))


def checkout():
    STATE.mkdir(mode=0o700, exist_ok=True)
    if not SOURCE.exists():
        with tempfile.TemporaryDirectory(dir=STATE) as tmp:
            clone = Path(tmp) / "ragflow"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    "v1.0.0",
                    "https://github.com/infiniflow/ragflow.git",
                    str(clone),
                ],
                check=True,
            )
            clone.rename(SOURCE)
    rev = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if rev != VERSION:
        raise ValueError("Unexpected upstream revision in .local-stack/ragflow.")
    print(f"RAGFlow v1.0.0 source · {SOURCE}", flush=True)


def prepare():
    checkout()
    deployment = STATE / "docker"
    if not deployment.exists():
        with tempfile.TemporaryDirectory(dir=STATE) as tmp:
            dest = Path(tmp) / "docker"
            shutil.copytree(SOURCE / "docker", dest)
            env = (dest / ".env").read_text().splitlines()
            names = {
                "MYSQL_PASSWORD",
                "MINIO_PASSWORD",
                "REDIS_PASSWORD",
                "ELASTIC_PASSWORD",
                "ADMIN_DEFAULT_PASSWORD",
            }
            env = [
                (
                    line.split("=", 1)[0] + "=" + secrets.token_hex(24)
                    if line.partition("=")[0] in names
                    else line
                )
                for line in env
            ]
            write_private(dest / ".env", "\n".join(env) + "\n")
            dest.rename(deployment)
    return [
        "docker",
        "compose",
        "-p",
        "rag-demo-system",
        "--profile",
        "cpu",
        "--profile",
        "metadata-mysql",
        "--profile",
        "elasticsearch",
        "--env-file",
        str(deployment / ".env"),
        "-f",
        str(deployment / "docker-compose.yml"),
        "-f",
        str(ROOT / "stack/local.yml"),
    ]


def request(client, method, path, token="", **kwargs):
    headers = (
        {"Authorization": "Bearer " + token.removeprefix("Bearer ")} if token else {}
    )
    r = client.request(method, RAG + "/api/v1" + path, headers=headers, **kwargs)
    r.raise_for_status()
    body = r.json()
    if body.get("code", 0) not in (0, 200):
        raise ValueError(body.get("message", "RAGFlow request failed"))
    return body.get("data"), r


def account(client):
    path = STATE / "account.json"
    if not path.exists():
        write_private(
            path,
            json.dumps(
                {
                    "email": "student-" + secrets.token_hex(5) + "@demo.local",
                    "password": secrets.token_urlsafe(32),
                }
            ),
        )
    login = json.loads(path.read_text())
    previous = saved_key("ragflow_url", RAG)
    if previous:
        try:
            request(client, "GET", "/datasets", previous)
            return previous
        except (ValueError, httpx.HTTPStatusError):
            pass
    try:
        _, r = request(client, "POST", "/auth/login", json=login)
    except (ValueError, httpx.HTTPStatusError):
        _, r = request(
            client, "POST", "/users", json={**login, "nickname": "Demo Student"}
        )
    token = r.headers["Authorization"]
    key, _ = request(client, "POST", "/system/tokens", token, json={})
    remember("ragflow_url", RAG, key["token"])
    return key["token"]


def register(client, token, base, api_key, embed, chat):
    import hashlib

    ident = (
        "demo-"
        + hashlib.sha256(json.dumps([base, embed, chat, api_key]).encode()).hexdigest()[
            :12
        ]
    )
    providers, _ = request(client, "GET", "/providers", token)
    if not any(p.get("provider_name", p.get("name")) == PROVIDER for p in providers):
        request(client, "PUT", "/providers", token, json={"provider_name": PROVIDER})
    path = "/providers/" + PROVIDER + "/instances"
    rows, _ = request(client, "GET", path, token)
    if isinstance(rows, dict):
        rows = rows.get("instances", [])
    if not any(r["instance_name"] == ident for r in rows):
        request(
            client,
            "POST",
            path,
            token,
            json={
                "instance_name": ident,
                "api_key": api_key or "local",
                "base_url": base,
                "model_info": [
                    {
                        "model_name": embed,
                        "model_type": ["embedding"],
                        "max_tokens": 2048,
                    },
                    {"model_name": chat, "model_type": ["chat"], "max_tokens": 8192},
                ],
            },
        )
    models, _ = request(client, "GET", "/models", token)
    if isinstance(models, dict):
        models = models["models"]
    for kind, name in (("embedding", embed), ("chat", chat)):
        model = next(
            (m for m in models if m["name"] == name and m["instance_name"] == ident),
            None,
        )
        if not model:
            raise ValueError(f"Model registration was not visible for {name}")
        request(
            client,
            "PATCH",
            "/models/default",
            token,
            json={"model_id": model["model_id"], "model_type": kind},
        )
    return f"{embed}@{ident}@{PROVIDER}"


def monitoring(target):
    folder = ROOT / "monitoring"
    cfg = STATE / "prometheus.yml"
    base = (
        (folder / "prometheus.yml")
        .read_text()
        .replace("host.docker.internal:9380", "ragflow-cpu:9380")
        .replace("[host.docker.internal:1234]", json.dumps([target]))
    )
    if not cfg.exists() or cfg.read_text() != base:
        write_private(cfg, base)
    cfg.chmod(0o644)  # Prometheus runs as an unprivileged container user.
    extra = STATE / "monitor.json"
    write_private(
        extra,
        json.dumps(
            {
                "services": {
                    "probe": {"networks": ["default", "lab"]},
                    "prometheus": {
                        "labels": {"rag-demo.probe-target": target},
                        "volumes": [f"{cfg}:/etc/prometheus/prometheus.yml:ro"],
                    },
                },
                "networks": {
                    "lab": {"external": True, "name": "rag-demo-system_ragflow"}
                },
            }
        ),
    )
    return [
        "docker",
        "compose",
        "-p",
        "rag-demo-monitor",
        "-f",
        str(folder / "compose.yaml"),
        "-f",
        str(extra),
    ]


def model_choices(client):
    url = (
        input("Gateway URL visible from this terminal (including /v1) › ")
        .strip()
        .rstrip("/")
    )
    key = os.environ.get("LLM_API_KEY") or getpass.getpass(
        "Gateway API key (Enter for none) › "
    )
    headers = {"Authorization": "Bearer " + key} if key else {}
    response = client.get(url + "/models", headers=headers)
    response.raise_for_status()
    names = [m["id"] for m in response.json()["data"]]
    for i, name in enumerate(names, 1):
        print(i, name)

    def select(label):
        value = input(label + " (number or exact model ID) › ").strip()
        if value.isdecimal() and 1 <= int(value) <= len(names):
            return names[int(value) - 1]
        if not value:
            raise ValueError("A model ID is required.")
        return value

    embed, chat = select("Embedding model"), select("Chat model")
    probe = client.post(
        url + "/embeddings",
        headers=headers,
        json={"model": embed, "input": "RAG demo connection test"},
    )
    probe.raise_for_status()
    if not probe.json().get("data", [{}])[0].get("embedding"):
        raise ValueError("The gateway returned no embedding for the selected model.")
    parts = urlsplit(url)
    internal = url
    if parts.hostname in ("localhost", "127.0.0.1", "::1"):
        internal = parts._replace(
            netloc="host.docker.internal" + (f":{parts.port}" if parts.port else "")
        ).geturl()
    inside = (
        input(f"Gateway URL visible from RAGFlow container [{internal}] › ").strip()
        or internal
    )
    return url, inside, key, embed, chat


def start(api, gateway=False):
    subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], check=True)
    cmd = prepare()
    if gateway:
        url, inside, key, embed, chat = model_choices(api.client)
    else:
        url, inside, key, embed, chat = (
            OLLAMA,
            "http://ollama:11434/v1",
            "",
            "nomic-embed-text",
            "qwen2.5:1.5b",
        )
        cmd += ["--profile", "models"]
    subprocess.run(cmd + ["up", "-d"], check=True)
    if not gateway:
        for name in (embed, chat):
            ollama = cmd + ["exec", "-T", "ollama", "ollama"]
            if subprocess.run(
                ollama + ["show", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode:
                subprocess.run(ollama + ["pull", name], check=True)
    print("Waiting for the RAGFlow API …", flush=True)
    deadline = time.monotonic() + 300
    while True:
        try:
            ping = api.client.get(RAG + "/api/v1/system/ping", timeout=3)
            ping.raise_for_status()
            if ping.text.strip() != "pong":
                raise ValueError("Unexpected ping response")
            break
        except (httpx.HTTPError, ValueError):
            if time.monotonic() >= deadline:
                raise ValueError(
                    "RAGFlow startup timed out. Inspect ./demo stack-logs and rerun ./demo up."
                )
            time.sleep(2)
    token = account(api.client)
    embedding = register(api.client, token, inside, key, embed, chat)
    remember("llm_url", url, key)
    parts = urlsplit(inside)
    target = f"{parts.hostname}:{parts.port or (443 if parts.scheme=='https' else 80)}"
    subprocess.run(monitoring(target) + ["up", "-d"], check=True)
    api.cfg.update(
        ragflow_url=RAG, llm_url=url, embedding_model=embedding, llm_model=chat
    )
    print(
        "Local stack ready · RAGFlow http://127.0.0.1:9388 · Grafana http://127.0.0.1:3300",
        flush=True,
    )
