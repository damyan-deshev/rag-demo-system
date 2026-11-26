"""Interactive terminal client for the RAGFlow v1.0 Go API."""

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.prompt import Prompt

import local_stack
import subprocess
import getpass
import sessions
from epub import fix_opf
from iterative import add_pass_option, fused_candidates, review_line, review_pass

ui = Console(markup=False, highlight=False)
CONFIG = (
    Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    / "rag-demo-system/settings.json"
)


def choose(items, label):
    if not items:
        ui.print("No entries.")
        return None
    for i, (name, _) in enumerate(items, 1):
        ui.print(f"{i:3}. {name}")
    answer = input(f"{label} (number, Enter to go back) › ").strip()
    if answer.isdecimal() and 1 <= int(answer) <= len(items):
        return items[int(answer) - 1][1]
    return None


def browse():
    folder = Path.cwd()
    while True:
        ui.print(str(folder))
        entries = [p for p in folder.iterdir() if not p.name.startswith(".")]
        files = [folder.parent] + sorted(
            entries, key=lambda p: (not p.is_dir(), p.name.casefold())
        )
        for i, p in enumerate(files, 1):
            ui.print(
                f"{i:3}. " + ("../" if i == 1 else p.name + ("/" if p.is_dir() else ""))
            )
        value = input("Number or document path (Enter to cancel) › ").strip()
        if not value:
            return None
        if value.isdecimal() and 1 <= int(value) <= len(files):
            p = files[int(value) - 1]
        else:
            p = folder / Path(value).expanduser()
        if p.is_file():
            return p
        if p.is_dir():
            folder = p.resolve()
        else:
            ui.print("File not found.")


def settings():
    saved = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    defaults = dict(
        ragflow_url="http://localhost:9380",
        llm_url="http://localhost:1234/v1",
        embedding_model="",
        llm_model="",
    )
    return {k: os.environ.get(k.upper(), saved.get(k, v)) for k, v in defaults.items()}


def save(cfg):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=CONFIG.parent, delete=False) as f:
        json.dump(cfg, f, indent=2)
    Path(f.name).replace(CONFIG)


class Backend:
    def __init__(self, cfg, client):
        self.cfg = cfg
        self.client = client

    def request(self, method, path, *, llm=False, **kw):
        base, key = (
            ("llm_url", "LLM_API_KEY") if llm else ("ragflow_url", "RAGFLOW_API_KEY")
        )
        token = (
            os.environ.get(key) or local_stack.saved_key(base, self.cfg[base])
        ).removeprefix("Bearer ")
        headers = {"Authorization": "Bearer " + token} if token else {}
        r = self.client.request(
            method, self.cfg[base].rstrip("/") + path, headers=headers, **kw
        )
        r.raise_for_status()
        body = r.json()
        if not llm and body.get("code", 0) not in (0, 200):
            raise ValueError(str(body.get("message", body.get("code"))))
        return body.get("data", body) if not llm else body

    async def complete(self, messages, run_id, max_tokens, flow=None):
        model = self.cfg["llm_model"]
        if not model:
            raise ValueError("Select an LLM with the wizard or set LLM_MODEL.")
        payload = dict(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0
        )
        if "qwen" in model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            body = self.request("POST", "/chat/completions", llm=True, json=payload)
        except httpx.HTTPStatusError as error:
            if (
                error.response.status_code not in (400, 422)
                or "chat_template_kwargs" not in payload
            ):
                raise
            del payload["chat_template_kwargs"]
            body = self.request("POST", "/chat/completions", llm=True, json=payload)
        return body["choices"][0]["message"]["content"], body

    def listing(self, path, key):
        result = []
        for page in range(1, 10001):
            body = self.request("GET", path, params={"page": page, "page_size": 100})
            rows = body if isinstance(body, list) else body[key]
            result.extend(rows)
            if len(rows) < 100:
                return result
        raise ValueError("Pagination limit reached.")


def setup(api):
    mode = input(
        "Setup · [1] full local CPU stack (default) · [2] local RAGFlow + existing model gateway · [3] existing services · Enter for 1 › "
    ).strip()
    if mode in ("", "1", "2"):
        local_stack.start(api, gateway=mode == "2")
        save(api.cfg)
    elif mode == "3":
        wizard(api, prompt_keys=True)
    else:
        raise ValueError("Choose 1, 2 or 3.")


def wizard(api, prompt_keys=False):
    cfg = api.cfg.copy()
    probe = Backend(cfg, api.client)
    for llm in (False, True):
        endpoint, model = (
            ("llm_url", "llm_model") if llm else ("ragflow_url", "embedding_model")
        )
        cfg[endpoint] = Prompt.ask(endpoint, default=cfg[endpoint])
        env_key = "LLM_API_KEY" if llm else "RAGFLOW_API_KEY"
        if prompt_keys and not os.environ.get(env_key):
            value = getpass.getpass(f"{env_key} (Enter to use saved key or no key) › ")
            if value:
                local_stack.remember(endpoint, cfg[endpoint], value)
        names = []
        try:
            if llm:
                names = [
                    m["id"] for m in probe.request("GET", "/models", llm=True)["data"]
                ]
            else:
                rows = probe.request("GET", "/api/v1/models")
                if isinstance(rows, dict):
                    rows = rows.get("models", [])
                for m in rows:
                    if "embedding" in (m.get("model_type") or []):
                        names.append(
                            "@".join(
                                [m["name"], m["instance_name"], m["provider_name"]]
                            )
                        )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            ui.print(f"Discovery unavailable · {e}")
        pick = choose([(n, n) for n in names], model) if names else None
        cfg[model] = pick or Prompt.ask(model + " ID", default=cfg[model])
    save(cfg)
    api.cfg.update(cfg)


def upload(api, base, p):
    with p.open("rb") as f:
        uploaded = api.request("POST", base + "/documents", files={"file": (p.name, f)})
    rows = uploaded if isinstance(uploaded, list) else uploaded.get("docs", [uploaded])
    ident = rows[0]["id"]
    ui.print(f"Uploaded {p.name} · {ident}")
    return ident


def index(api, base, ident):
    api.request("POST", base + "/chunks", json={"document_ids": [ident]})
    previous = None
    while True:
        body = api.request("GET", base + "/documents", params={"id": ident})
        rows = body if isinstance(body, list) else body["docs"]
        doc = next((d for d in rows if d["id"] == ident), None)
        if doc is None:
            raise ValueError(f"Document {ident} was not found in this collection.")
        phase = str(doc.get("ingestion_status", doc.get("run", ""))).upper()
        progress = max(0, min(1, doc.get("progress") or 0))
        count = doc.get("chunk_count", 0)
        message = (doc.get("latest_ingestion_event") or {}).get("message") or doc.get(
            "progress_msg", ""
        )
        line = f"{phase}  {'█' * round(progress*20)}{'░' * (20-round(progress*20))} {progress:.0%} · {count} chunks · {message}"
        if line != previous:
            ui.print(line)
            previous = line
        if phase in {
            "DONE",
            "COMPLETED",
            "3",
            "FAIL",
            "FAILED",
            "ERROR",
            "STOPPED",
            "CANCEL",
            "4",
            "2",
            "-1",
        }:
            return doc
        time.sleep(1)


def trace(label, color, body):
    ui.print(label, style=f"bold {color}")
    ui.print(body)


def retrieve(api, ds, query, label):
    trace(label, "cyan", query)
    chunks = api.request(
        "POST",
        "/api/v1/retrieval",
        json={
            "dataset_ids": [ds],
            "question": query,
            "page_size": 8,
            "include_knowledge_compilation": False,
        },
    )["chunks"]
    ui.print(f"RAGFlow → harness · {len(chunks)} откъса", style="bold yellow")
    for rank, chunk in enumerate(chunks, 1):
        score = chunk.get("similarity")
        suffix = f" · similarity {score:.3f}" if isinstance(score, (float, int)) else ""
        ui.print(f"  {rank}. {chunk['id']}{suffix}", style="yellow")
        ui.print(chunk["content"])
    return chunks


def source_blocks(chunks):
    return [
        f"[S{n}] {c.get('document_name') or c.get('docnm_kw') or c.get('document_keyword') or ''} · {c['id']}\n{c['content']}"
        for n, c in enumerate(chunks, 1)
    ]


async def ask(api, ds, question, limit, history, raw, context_chunks=4, instruction=""):
    baseline = retrieve(api, ds, question, "Въпрос → RAGFlow · дословна заявка")
    batches = [baseline]
    if raw:
        blocks = source_blocks(baseline[:context_chunks])
        trace("Избран контекст", "yellow", "\n\n".join(blocks) or "0 откъса")
        return

    rewrite, _ = await api.complete(
        [
            {
                "role": "system",
                "content": "You write search queries. Copy the latest question exactly when it is already standalone. Otherwise resolve references from the dialogue. Never supply a guessed answer. Example: When did Maple open? → When did Maple open? Return only the question.",
            },
            *history,
            {"role": "user", "content": question},
        ],
        "",
        128,
    )
    query = (rewrite or "").strip() or question
    trace("LLM → harness · заявка след пренаписване", "green", query)
    tried = []
    for i in range(1, limit + 1):
        tried.append(query)
        if i == 1 and query == question:
            ui.print("Заявката съвпада дословно. Използваме вече върнатите откъси.")
            chunks = baseline
        else:
            chunks = retrieve(api, ds, query, f"Harness → RAGFlow · заявка {i}/{limit}")
            batches.append(chunks)
        picked = fused_candidates(batches, context_chunks)
        blocks = source_blocks(picked)
        context = "\n\n".join(blocks)
        if limit == 1:
            break
        review = await review_pass(
            api,
            {
                "question": question,
                "run_id": "terminal",
                "options": {"flow": "chat"},
                "conversation": history,
                "retrieval_candidates": chunks,
                "context": picked,
                "evidence_text": context,
            },
            tried,
            limit,
        )
        trace("LLM → harness · оценка на откъсите", "green", review_line(review))
        if review["action"] != "retry":
            break
        query = review["next_query"]

    messages = []
    if instruction:
        trace("Harness → LLM · твоята инструкция", "cyan", instruction)
    messages.extend(history)
    trace(
        f"Harness → LLM · контекст ({len(blocks)} откъса)",
        "cyan",
        context or "0 откъса",
    )
    trace("Потребител → LLM · въпрос", "cyan", question)
    payload = {"question": question, "retrieved_excerpts": blocks}
    content = json.dumps(payload, ensure_ascii=False)
    if instruction:
        content += "\n\n" + instruction
    messages.append({"role": "user", "content": content})
    answer, _ = await api.complete(messages, "", 600)
    trace("LLM → потребител · отговор", "green", answer)
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    del history[:-6]
    return {
        "question": question,
        "queries": [question] + [q for q in tried if q != question],
        "sources": blocks,
        "instruction": instruction,
        "answer": answer,
    }


def chat(api, ds, limit, context_chunks):
    history, transcript = [], []
    instruction = ""
    ui.print(
        "Свободен чат. За инструкция към модела въведи /prompt и текста. /prompt без текст я изчиства."
    )
    while q := input("Question (/prompt, /sources, Enter to finish) › ").strip():
        if q == "/prompt" or q.startswith("/prompt "):
            instruction = q[len("/prompt") :].strip()
            trace(
                "Инструкция за следващите въпроси",
                "cyan",
                instruction or "Свободен чат",
            )
            continue
        if q == "/sources":
            ui.print(
                ("\n\n".join(transcript[-1]["sources"]) or "0 откъса")
                if transcript
                else "Ask a question first."
            )
            continue
        turn = asyncio.run(
            ask(api, ds, q, limit, history, False, context_chunks, instruction)
        )
        if turn:
            transcript.append(turn)
    if (
        transcript
        and input("Save this conversation with its excerpts? [y/N] ").lower() == "y"
    ):
        sessions.save(
            (api.cfg["ragflow_url"].rstrip("/"), ds),
            transcript[0]["question"],
            transcript,
        )


def collection(api, ds, limit, context_chunks):
    base = "/api/v1/datasets/" + ds["id"]
    scope = (api.cfg["ragflow_url"].rstrip("/"), ds["id"])
    ui.print(f"Collection ID · {ds['id']}")
    while True:
        action = (
            input(
                f"\n{ds['name']} · [i]ngest [d]ocuments [s]earch [c]hat [t]ranscripts [x]delete collection [Enter] back › "
            )
            .strip()
            .lower()
        )
        if not action:
            return
        if action == "i":
            path = browse()
            if path:
                doc = index(api, base, upload(api, base, path))
                error = (doc.get("latest_ingestion_event") or {}).get(
                    "message"
                ) or doc.get("progress_msg", "")
                if (
                    path.suffix.lower() == ".epub"
                    and "charsetreader is nil" in error.lower()
                    and "parse opf" in error.lower()
                ):
                    ui.print(
                        "The EPUB reader stopped at the XML catalogue before extracting the chapters."
                    )
                    if (
                        input(
                            "Repair OPF metadata in a separate copy and retry? [y/N] "
                        ).lower()
                        == "y"
                    ):
                        with tempfile.TemporaryDirectory() as folder:
                            fixed = Path(folder) / (path.stem + "-utf8.epub")
                            member, encoding = fix_opf(path, fixed, error)
                            ui.print(f"{member} · {encoding} → UTF-8")
                            index(api, base, upload(api, base, fixed))
        elif action == "d":
            docs = api.listing(base + "/documents", "docs")
            doc = choose(
                [
                    (
                        f"{d['name']} · {d.get('ingestion_status','')} · {d.get('chunk_count',0)} chunks",
                        d,
                    )
                    for d in docs
                ],
                "Document",
            )
            if doc:
                doc_action = input("[v]iew chunks · [d]elete document · Enter back › ")
                if (
                    doc_action == "d"
                    and input(f"Delete {doc['name']} and its chunks? Type DELETE › ")
                    == "DELETE"
                ):
                    api.request(
                        "DELETE", base + "/documents", json={"ids": [doc["id"]]}
                    )
                    ui.print("Document deleted.")
                elif doc_action == "v":
                    path = base + "/documents/" + doc["id"] + "/chunks"
                    chunks = api.listing(path, "chunks")
                    chunk = choose([(c["content"][:75], c) for c in chunks], "Chunk")
                    if chunk:
                        ui.print(chunk["content"])
                        if input("Delete this chunk? Type DELETE › ") == "DELETE":
                            api.request(
                                "DELETE", path, json={"chunk_ids": [chunk["id"]]}
                            )
        elif (
            action == "x"
            and input(
                f"Delete collection {ds['name']} and its documents? Type DELETE › "
            )
            == "DELETE"
        ):
            api.request("DELETE", "/api/v1/datasets", json={"ids": [ds["id"]]})
            sessions.delete(scope)
            return
        elif action == "t":
            ident = choose(sessions.list_saved(scope), "Saved conversation")
            if ident:
                ui.print_json(data=sessions.read(scope, ident))
                if input("Delete this conversation? Type DELETE › ") == "DELETE":
                    sessions.delete(scope, ident)
        elif action == "c":
            chat(api, ds["id"], limit, context_chunks)
        elif action == "s":
            history = []
            while q := input("Question (Enter to return) › ").strip():
                asyncio.run(
                    ask(api, ds["id"], q, limit, history, action == "s", context_chunks)
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-wizard", action="store_true")
    add_pass_option(parser)
    parser.add_argument("--dataset", help="Collection ID for staged commands")
    parser.add_argument("--context-chunks", type=int, choices=range(1, 9), default=4)
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=160,
        help="Target tokens per chunk for new collections",
    )
    stages = parser.add_subparsers(dest="stage")
    stages.add_parser("upload").add_argument("file", type=Path)
    stages.add_parser("index").add_argument("document")
    stages.add_parser("search").add_argument("question")
    stages.add_parser("chat")
    stages.add_parser("up")
    stages.add_parser("stack-logs")
    stages.add_parser("stop")
    args = parser.parse_args()
    if args.stage in ("upload", "index", "chat", "search") and not args.dataset:
        parser.error("--dataset ID is required for a staged command")
    if args.chunk_tokens < 1:
        parser.error("--chunk-tokens must be positive")
    with httpx.Client(timeout=180) as client:
        api = Backend(settings(), client)
        if args.stage == "up":
            local_stack.start(api)
            save(api.cfg)
            return
        if args.stage in ("stack-logs", "stop"):
            if not (local_stack.STATE / "docker").exists():
                raise ValueError("Start a local stack first with ./demo up.")
            cmd = local_stack.prepare() + ["--profile", "models"]
            if args.stage == "stop":
                subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-p",
                        "rag-demo-monitor",
                        "-f",
                        str(local_stack.ROOT / "monitoring/compose.yaml"),
                        "stop",
                    ],
                    check=True,
                )
                subprocess.run(cmd + ["stop"], check=True)
            else:
                subprocess.run(
                    cmd + ["logs", "--tail", "80", "ragflow-cpu"], check=True
                )
            return
        if (
            not args.skip_wizard
            and not CONFIG.exists()
            and input("Configure endpoints and models? [Y/n] ").lower() != "n"
        ):
            setup(api)
        if args.stage:
            base = "/api/v1/datasets/" + args.dataset
            if args.stage == "upload":
                upload(api, base, args.file.expanduser())
            elif args.stage == "index":
                doc = index(api, base, args.document)
                if str(doc.get("ingestion_status", doc.get("run"))).upper() not in {
                    "DONE",
                    "COMPLETED",
                    "3",
                }:
                    raise SystemExit(1)
            elif args.stage == "search":
                asyncio.run(
                    ask(
                        api,
                        args.dataset,
                        args.question,
                        1,
                        [],
                        True,
                        args.context_chunks,
                    )
                )
            else:
                chat(api, args.dataset, args.max_passes, args.context_chunks)
            return
        while True:
            cmd = input(
                "\n[n]ew collection [o]pen collection [w]izard [q]uit › "
            ).strip()
            try:
                if cmd == "q":
                    return
                if cmd == "w":
                    setup(api)
                if cmd == "n":
                    name = input("Collection name › ").strip()
                    if name:
                        data = {"name": name, "parse_type": 1, "parser_id": "general"}
                        if api.cfg["embedding_model"]:
                            data["embedding_model"] = api.cfg["embedding_model"]
                        ds = api.request("POST", "/api/v1/datasets", json=data)
                        config = ds.get("parser_config", {})
                        for key, value in config.items():
                            if "Chunker:" in key:
                                value["chunk_token_size"] = args.chunk_tokens
                        if config:
                            api.request(
                                "PUT",
                                "/api/v1/datasets/" + ds["id"],
                                json={"parser_config": config},
                            )
                        collection(api, ds, args.max_passes, args.context_chunks)
                if cmd == "o":
                    rows = api.listing("/api/v1/datasets", "datasets")
                    ds = choose([(r["name"], r) for r in rows], "Collection")
                    if ds:
                        collection(api, ds, args.max_passes, args.context_chunks)
            except (httpx.HTTPError, ValueError, OSError, KeyError) as e:
                ui.print(f"Operation failed · {e}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, httpx.HTTPError, subprocess.CalledProcessError) as e:
        raise SystemExit(str(e))
    except (EOFError, KeyboardInterrupt):
        print()
