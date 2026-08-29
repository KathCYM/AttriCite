import json
import os
import shutil
import uuid
from flask import Flask, request, Response, jsonify, render_template, stream_with_context
from src.run_main import run
from argparse import Namespace
import io
import sys
from rich.console import Console
import time
from queue import Queue
import threading

app = Flask(__name__)
APP_ROOT = os.path.abspath(os.getcwd())
LAST_RUN_PAYLOAD = None
LAST_RUN_RESULT_PATH = None
RUN_PAYLOADS = {}
RUN_RESULT_PATHS = {}
RUN_REQUESTED_PATHS = {}
APP_VERSION = "web-ui-run-scope-v1"


def resolve_result_path(path: str) -> str:
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(APP_ROOT, path))


def resolve_requested_result_path(path: str) -> str:
    requested = resolve_result_path(path)
    if not requested.startswith(APP_ROOT):
        raise ValueError("Result path must stay within the app workspace.")
    return requested


def build_run_result_path(run_id: str, requested_path: str) -> str:
    requested_abs_path = resolve_requested_result_path(requested_path)
    run_dir = os.path.join(APP_ROOT, ".web_runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, os.path.basename(requested_abs_path))


def seed_run_result_path(run_result_path: str, requested_result_path: str) -> None:
    os.makedirs(os.path.dirname(run_result_path), exist_ok=True)
    if os.path.exists(requested_result_path):
        shutil.copyfile(requested_result_path, run_result_path)


def load_payload_from_path(result_path: str) -> dict:
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])
    latest = results[-1] if results else {}
    return {
        "results": results,
        "paper_buffer": latest.get("papers", []),
        "selected": latest.get("selected"),
        "result_path": result_path,
    }


def get_payload_for_run(run_id: str | None):
    if run_id:
        payload = RUN_PAYLOADS.get(run_id)
        if payload is not None:
            return payload
        result_path = RUN_RESULT_PATHS.get(run_id)
        if result_path and os.path.exists(result_path):
            payload = load_payload_from_path(result_path)
            payload["run_id"] = run_id
            payload["requested_result_path"] = RUN_REQUESTED_PATHS.get(run_id)
            return payload
        requested_result_path = RUN_REQUESTED_PATHS.get(run_id)
        if requested_result_path and os.path.exists(requested_result_path):
            payload = load_payload_from_path(requested_result_path)
            payload["run_id"] = run_id
            payload["requested_result_path"] = requested_result_path
            return payload
        return None

    if LAST_RUN_PAYLOAD is not None:
        return LAST_RUN_PAYLOAD
    if LAST_RUN_RESULT_PATH and os.path.exists(LAST_RUN_RESULT_PATH):
        return load_payload_from_path(LAST_RUN_RESULT_PATH)
    return None

class StreamingStdout:
    def __init__(self, q: Queue):
        self.q = q
        self.buffer = ""

    def write(self, data):
        self.buffer += data
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.q.put(line + "\n")  # keep the newline

    def flush(self):
        pass

def stream_output(args, run_id: str, requested_result_path: str):
    q = Queue()

    old_stdout = sys.stdout
    sys.stdout = StreamingStdout(q)

    buffer = io.StringIO()
    console = Console() 

    def worker():
        global LAST_RUN_PAYLOAD, LAST_RUN_RESULT_PATH
        try:
            result, paper_buffer = run(args, console)
            q.put("\n=== FINAL RESULT ===\n")
            latest_result = result[-1] if result else {}
            q.put(
                "Processed "
                + f"{len(result)} result(s); latest id={latest_result.get('id', 'N/A')}, "
                + f"status={latest_result.get('status', 'unknown')}\n"
            )
            payload = {
                "run_id": run_id,
                "results": result,
                "paper_buffer": latest_result.get("papers", []),
                "selected": latest_result.get("selected"),
                "result_path": args.result_path,
                "requested_result_path": requested_result_path,
                "export_error": None,
            }
            RUN_PAYLOADS[run_id] = payload
            RUN_RESULT_PATHS[run_id] = args.result_path
            RUN_REQUESTED_PATHS[run_id] = requested_result_path
            LAST_RUN_RESULT_PATH = args.result_path
            LAST_RUN_PAYLOAD = payload
            if requested_result_path != args.result_path:
                try:
                    os.makedirs(os.path.dirname(requested_result_path), exist_ok=True)
                    shutil.copyfile(args.result_path, requested_result_path)
                except Exception as export_error:
                    payload["export_error"] = str(export_error)
            if payload["export_error"]:
                q.put(f"[export warning] {payload['export_error']}\n")
            final_payload = json.dumps(payload, ensure_ascii=False)
            q.put(f"__FINAL_PAYLOAD__={final_payload}\n")
        except Exception as e:
            q.put(f"\n[stream error] {e}\n")
        finally:
            sys.stdout = old_stdout
            q.put(None) 

    threading.Thread(target=worker).start()

    while True:
        line = q.get()
        if line is None:
            break
        yield line

@app.route("/run_stream", methods=["POST"])
def run_app():
    data = request.get_json()
    run_id = data.get("run_id") or str(uuid.uuid4())
    try:
        requested_result_path = resolve_requested_result_path(data.get("result_path", "results.json"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    scoped_result_path = build_run_result_path(run_id, data.get("result_path", "results.json"))
    seed_run_result_path(scoped_result_path, requested_result_path)
    RUN_RESULT_PATHS[run_id] = scoped_result_path
    RUN_REQUESTED_PATHS[run_id] = requested_result_path
    args = Namespace(
        dataset=data.get("dataset"),
        result_path=scoped_result_path,
        model_name=data["model_name"],
        local_model=data.get("local_model", False),
        use_vllm=data.get("use_vllm", False),
        vllm_base_url=data.get("vllm_base_url"),
        id=data.get("id", "manual"),
        source_paper_title=data.get("source_paper_title"),
        target_paper_title=data.get("target_paper_title"),
        excerpt=data.get("excerpt"),
        year=data.get("year", 2025),
        skip_citations=data.get("skip_citations", ""),
        additional_context=data.get("additional_context"),
        interactive_context=False,
        temperature=data.get("temperature", 0.95),
    )

    return Response(stream_with_context(stream_output(args, run_id, requested_result_path)),
                    mimetype="text/plain; charset=utf-8")

@app.route("/test")
def test_stream():
    def generate():
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer  # redirect print to buffer

        try:
            # Example prints
            for i in range(5):
                print(f"Line {i}")
                time.sleep(1)

        finally:
            sys.stdout = old_stdout  # restore stdout

        buffer.seek(0)
        for line in buffer:
            yield line

        yield "\n=== FINAL RESULT ===\n"

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/result_json", methods=["GET"])
def result_json():
    run_id = request.args.get("run_id")
    if run_id:
        abs_path = RUN_RESULT_PATHS.get(run_id)
        if abs_path and not os.path.exists(abs_path):
            requested_result_path = RUN_REQUESTED_PATHS.get(run_id)
            if requested_result_path and os.path.exists(requested_result_path):
                abs_path = requested_result_path
        if not abs_path:
            return jsonify({"error": f"Unknown run_id: {run_id}"}), 404
    else:
        requested_path = request.args.get("path", "results.json")
        abs_path = resolve_result_path(requested_path)
        if not abs_path.startswith(APP_ROOT):
            return jsonify({"error": "Path must stay within the app workspace."}), 400
    if not os.path.exists(abs_path):
        missing_ref = run_id if run_id else request.args.get("path", "results.json")
        return jsonify({"error": f"Result file not found: {missing_ref}"}), 404

    with open(abs_path, "r", encoding="utf-8") as f:
        return Response(f.read(), mimetype="application/json")


@app.route("/latest_result", methods=["GET"])
def latest_result():
    run_id = request.args.get("run_id")
    payload = get_payload_for_run(run_id)
    if payload is None:
        message = f"No run has completed yet for run_id {run_id}." if run_id else "No run has completed yet."
        return jsonify({"error": message}), 404
    return jsonify(payload)


@app.route("/debug_state", methods=["GET"])
def debug_state():
    run_id = request.args.get("run_id")
    run_result_path = RUN_RESULT_PATHS.get(run_id) if run_id else LAST_RUN_RESULT_PATH
    return jsonify(
        {
            "app_version": APP_VERSION,
            "app_root": APP_ROOT,
            "cwd": os.path.abspath(os.getcwd()),
            "run_id": run_id,
            "known_run_ids": list(RUN_RESULT_PATHS.keys())[-10:],
            "last_run_result_path": LAST_RUN_RESULT_PATH,
            "last_run_result_exists": bool(LAST_RUN_RESULT_PATH and os.path.exists(LAST_RUN_RESULT_PATH)),
            "has_last_run_payload": LAST_RUN_PAYLOAD is not None,
            "run_result_path": run_result_path,
            "run_result_exists": bool(run_result_path and os.path.exists(run_result_path)),
            "requested_result_path": RUN_REQUESTED_PATHS.get(run_id) if run_id else None,
            "has_run_payload": run_id in RUN_PAYLOADS if run_id else False,
        }
    )

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # serve a HTML page

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
