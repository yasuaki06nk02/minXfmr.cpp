#!/usr/bin/env python3
"""Production-ready transpose search helper.

Improvements over the original script:
- structured logging (console + run log)
- robust scoring heuristic (filters debug noise)
- retries for transient failures
- optional verification runs for the best candidate
- safer defaults and explicit opt-in for high load
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import signal
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

CURRENT_PROC: Optional[subprocess.Popen] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search best transpose combination with production guardrails."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="build\\src\\Release\\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    )
    parser.add_argument(
        "cli",
        nargs="?",
        default="build\\src\\Release\\minxfmr_cli.exe",
    )
    parser.add_argument("prompt", nargs="?", default="hello")
    parser.add_argument("--logdir", default="scripts/transpose_search_logs")
    parser.add_argument("--start-mask", type=int, default=0)
    parser.add_argument(
        "--max-masks",
        type=int,
        default=4,
        help="How many masks to run from start-mask. Default 4 for safe operation.",
    )
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--cooldown-sec", type=float, default=3.0)
    parser.add_argument(
        "--max-gen-tokens",
        type=int,
        default=8,
        help="Generation token cap per trial for fast sweeps.",
    )
    parser.add_argument(
        "--priority",
        choices=["idle", "below_normal", "normal"],
        default="below_normal",
        help="Process priority on Windows.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="If >0, export OMP/MKL/OPENBLAS/NUMEXPR thread limits.",
    )
    parser.add_argument(
        "--allow-high-load",
        action="store_true",
        help="Allow running more than 4 masks in one command.",
    )
    parser.add_argument(
        "--test-all-masks",
        action="store_true",
        help="Run all masks 0..15 (overrides safety clamp).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--chat-mode",
        action="store_true",
        help="Use chat mode stdin flow instead of one-shot --prompt mode.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Number of attempts per mask on transient failure (timeouts).",
    )
    parser.add_argument(
        "--emit-json",
        action="store_true",
        help="Use CLI JSON output + model vocab for precise scoring (preferred).",
    )
    parser.add_argument(
        "--verify-best",
        action="store_true",
        help="After search, re-run the best candidate with a larger token budget to verify quality.",
    )
    parser.add_argument(
        "--verify-gen-tokens",
        type=int,
        default=24,
        help="Generation token cap to use for verification runs.",
    )
    parser.add_argument(
        "--verify-runs",
        type=int,
        default=2,
        help="How many verification runs to perform for the best mask.",
    )
    parser.add_argument(
        "--summary-file",
        default="summary.json",
        help="Path to write run summary JSON (inside --logdir by default).",
    )
    return parser.parse_args()


def windows_creationflags(priority: str) -> int:
    if os.name != "nt":
        return 0
    mapping = {
        "idle": getattr(subprocess, "IDLE_PRIORITY_CLASS", 0),
        "below_normal": getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
        "normal": getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0),
    }
    return mapping.get(priority, 0)


def _handle_sigint(signum, frame) -> None:
    global CURRENT_PROC
    logging.warning("Interrupted (SIGINT). Cleaning up child process if any.")
    if CURRENT_PROC is not None:
        try:
            CURRENT_PROC.kill()
        except Exception:
            pass
    sys.exit(130)


def configure_logging(logdir: Path) -> None:
    logdir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-5s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logdir / "run.log", encoding="utf-8"),
        ],
    )


def strip_ansi(text: str) -> str:
    ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_re.sub("", text)


def score_output(out_text: str) -> int:
    """Return an integer score for model output. Higher is better.

    Heuristic: composite of UTF-8 validity rate and vocabulary-like hit rate.
    - valid_utf8_rate: fraction of chars that were not replaced during decoding
    - vocab_like_rate: fraction of tokens that look like real words (ASCII letters)
    Filters obvious debug lines and ANSI noise.
    """
    if not out_text:
        return 0
    text = strip_ansi(out_text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # filter noisy debug lines
    lines = [ln for ln in lines if not re.search(r"\b(DEBUG|INFO|TRACE|WARNING|ERROR)\b", ln, re.I)]
    # prefer assistant lines when available
    assist_lines = [ln for ln in lines if "assistant" in ln.lower()]
    if not assist_lines:
        assist_lines = lines

    sample = "\n".join(assist_lines[:12])

    # UTF-8 validity: count replacement characters (U+FFFD) introduced by decode errors
    total_chars = len(sample)
    if total_chars == 0:
        return 0
    repl = sample.count('\ufffd')
    valid_utf8_rate = max(0.0, 1.0 - (repl / total_chars))

    # Vocabulary-like hit rate: fraction of tokens that are ASCII-letter words (approximation)
    tokens = re.findall(r"\S+", sample)
    if not tokens:
        vocab_rate = 0.0
    else:
        word_like = 0
        for t in tokens:
            # strip surrounding punctuation
            tt = t.strip(string.punctuation)
            # consider token word-like if it contains 2+ ASCII letters
            if re.match(r"^[A-Za-z]{2,}$", tt):
                word_like += 1
        vocab_rate = float(word_like) / float(len(tokens))

    # combine rates (weights favor UTF-8 validity)
    combined = (0.6 * valid_utf8_rate) + (0.4 * vocab_rate)
    # scale to integer for ranking
    score = int(combined * 1000)
    return score


def run_proc(cmd: list, env: Dict[str, str], cflags: int, timeout: int, chat_mode: bool, prompt: str) -> Tuple[str, str, bool, int]:
    """Run command and return (stdout, stderr, timed_out, returncode)."""
    global CURRENT_PROC
    out_text = ""
    err_text = ""
    timed_out = False
    try:
        CURRENT_PROC = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if chat_mode else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            creationflags=cflags,
        )
        if chat_mode:
            inp = prompt + "\nexit\n"
            out_text, err_text = CURRENT_PROC.communicate(input=inp, timeout=timeout)
        else:
            out_text, err_text = CURRENT_PROC.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            CURRENT_PROC.kill()
            out_text, err_text = CURRENT_PROC.communicate(timeout=5)
        except Exception:
            out_text = out_text or ""
            err_text = (err_text or "") + "\n[TIMEOUT] process exceeded timeout and was killed.\n"
    except Exception as exc:
        err_text = (err_text or "") + f"\n[ERROR] {exc}\n"
    finally:
        rc = CURRENT_PROC.returncode if CURRENT_PROC is not None else -1
        CURRENT_PROC = None
    return out_text or "", err_text or "", timed_out, rc


def main() -> int:
    signal.signal(signal.SIGINT, _handle_sigint)
    args = parse_args()

    model = Path(args.model)
    cli = Path(args.cli)
    logdir = Path(args.logdir)
    configure_logging(logdir)

    if not cli.exists():
        logging.error("CLI not found: %s", cli)
        return 2
    if not model.exists():
        logging.error("Model not found: %s", model)
        return 2
    if args.start_mask < 0 or args.start_mask > 15:
        logging.error("start-mask must be in [0, 15]")
        return 2
    if args.max_masks < 1:
        logging.error("max-masks must be >= 1")
        return 2

    safe_max = args.max_masks
    if args.test_all_masks:
        logging.info("[override] --test-all-masks set: running all masks 0..15 starting at %d", args.start_mask)
        safe_max = 16 - args.start_mask
    else:
        if safe_max > 4 and not args.allow_high_load:
            logging.warning("[safety] max-masks > 4 requires --allow-high-load. Clamping to 4.")
            safe_max = 4
        safe_max = min(safe_max, 16 - args.start_mask)

    env = os.environ.copy()
    if args.cpu_threads > 0:
        th = str(args.cpu_threads)
        env["OMP_NUM_THREADS"] = th
        env["MKL_NUM_THREADS"] = th
        env["OPENBLAS_NUM_THREADS"] = th
        env["NUMEXPR_NUM_THREADS"] = th
        env["MINXFMR_CPU_THREADS"] = th

    flags = ["wq", "wk", "wv", "wo"]
    best: Dict[str, Optional[object]] = {"score": -1, "mask": None, "log": None, "cmd": None}
    cflags = windows_creationflags(args.priority)

    logging.info(
        "start_mask=%d max_masks=%d timeout=%ds cooldown=%.1fs priority=%s",
        args.start_mask,
        safe_max,
        args.timeout_sec,
        args.cooldown_sec,
        args.priority,
    )
    if args.cpu_threads > 0:
        logging.info("cpu thread limits set to %d", args.cpu_threads)

    summary = {"runs": [], "best": None}

    # If requested, attempt to load model vocab via CLI --emit-vocab
    vocab = None
    if args.emit_json:
        try:
            logging.info("Fetching model vocab via CLI --emit-vocab")
            # ensure JSON emission not set
            env_nojson = env.copy()
            if "MINXFMR_EMIT_JSON" in env_nojson: del env_nojson["MINXFMR_EMIT_JSON"]
            out_text, err_text, timed_out, rc = run_proc([str(cli), str(model), "--emit-vocab"], env_nojson, cflags, args.timeout_sec, False, args.prompt)
            if timed_out or rc != 0:
                logging.warning("Failed to fetch vocab: rc=%s timed_out=%s stderr=%s", rc, timed_out, err_text)
            else:
                try:
                    import json as _json

                    vocab = _json.loads(out_text)
                    if not isinstance(vocab, list):
                        logging.warning("vocab JSON not a list, falling back")
                        vocab = None
                    else:
                        logging.info("Loaded vocab size=%d", len(vocab))
                except Exception as e:
                    logging.warning("Failed to parse vocab JSON: %s", e)
                    vocab = None
        except Exception as e:
            logging.warning("Exception reading vocab: %s", e)

    for idx, mask in enumerate(range(args.start_mask, args.start_mask + safe_max)):
        if idx > 0 and args.cooldown_sec > 0:
            time.sleep(args.cooldown_sec)

        
            
            if args.dry_run:
                logging.info("dry-run: would write %s", logfile)
                run_attempts.append({"attempt": attempt, "dry_run": True})
                break

            # if using JSON scoring, request CLI to emit JSON
            run_env = env.copy()
            use_json = args.emit_json and vocab is not None
            if use_json:
                run_env["MINXFMR_EMIT_JSON"] = "1"

            out_text, err_text, timed_out, returncode = run_proc(
                cmd, run_env, cflags, args.timeout_sec, args.chat_mode, args.prompt
            )

            # write per-attempt log (stdout+stderr)
            full = "[STDOUT]\n" + (out_text or "") + "\n[STDERR]\n" + (err_text or "")
            logfile.write_text(full, encoding="utf-8")

            # compute score: prefer JSON-based precise scoring when available
            score = 0
            if use_json and out_text:
                try:
                    import json as _json, base64 as _base64

                    j = _json.loads(out_text)
                    token_ids = j.get("token_ids", [])
                    text_b64 = j.get("text_b64", "")
                    # decode base64 to bytes then to str (utf-8)
                    decoded = ""
                    if text_b64:
                        try:
                            b = _base64.b64decode(text_b64)
                            decoded = b.decode("utf-8", errors="replace")
                        except Exception:
                            decoded = ""
                    # valid utf8 rate
                    total_chars = len(decoded)
                    if total_chars == 0:
                        valid_utf8_rate = 0.0
                    else:
                        repl = decoded.count('\ufffd')
                        valid_utf8_rate = max(0.0, 1.0 - (repl / total_chars))

                    # vocab hit rate using model vocab
                    if token_ids and vocab:
                        hits = 0
                        for tid in token_ids:
                            if not isinstance(tid, int):
                                continue
                            if tid < 0 or tid >= len(vocab):
                                continue
                            tok = vocab[tid]
                            # skip explicit byte fallbacks / unknowns
                            if not tok or tok.startswith("<0x") or tok == "<unk>" or ("<" in tok and ">" in tok and len(tok) <= 8):
                                continue
                            # consider word-like token if contains an ASCII letter
                            if any(('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in tok):
                                hits += 1
                        vocab_rate = float(hits) / float(len(token_ids)) if token_ids else 0.0
                    else:
                        vocab_rate = 0.0

                    combined = (0.6 * valid_utf8_rate) + (0.4 * vocab_rate)
                    score = int(combined * 1000)
                except Exception as e:
                    logging.warning("Failed JSON scoring: %s", e)
                    score = score_output(out_text)
            else:
                score = score_output(out_text)

            logging.info("mask=%d attempt=%d score=%d timed_out=%s rc=%s", mask, attempt, score, timed_out, returncode)

            run_attempts.append(
                {
                    "attempt": attempt,
                    "log": str(logfile),
                    "score": score,
                    "timed_out": bool(timed_out),
                    "returncode": int(returncode),
                }
            )

            # on timeout we may retry
            if timed_out and attempt < args.retries:
                backoff = min(30, 2 ** attempt)
                logging.warning("mask=%d timed out, retrying after %ds (attempt %d/%d)", mask, backoff, attempt + 1, args.retries)
                time.sleep(backoff)
                continue

            # otherwise stop attempts for this mask
            break

        # merge attempt results into summary
        best_run = max((r for r in run_attempts if r.get("score") is not None), key=lambda x: x.get("score", 0), default=None)
        summary_entry = {"mask": mask, "attempts": run_attempts, "best_attempt": best_run}
        summary["runs"].append(summary_entry)

        if best_run and best_run.get("score", 0) > best["score"]:
            best.update(score=best_run.get("score"), mask=mask, log=best_run.get("log"), cmd=shlex.join(base_cmd))

    summary["best"] = best
    # write summary JSON
    summary_path = (logdir / args.summary_file).resolve()
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Wrote summary to %s", summary_path)

    # optional verification of best candidate
    if args.verify_best and best.get("mask") is not None:
        mask = int(best["mask"])
        logging.info("Verifying best mask=%d with %d runs at %d tokens", mask, args.verify_runs, args.verify_gen_tokens)
        verify_cmd = [
            str(cli),
            str(model),
            "--temp",
            "0",
            "--top_k",
            "1",
            "--max-gen-tokens",
            str(args.verify_gen_tokens),
        ]
        if args.chat_mode:
            verify_cmd.append("--chat")
        else:
            verify_cmd.extend(["--prompt", args.prompt])
        for i, name in enumerate(flags):
            if (mask >> i) & 1:
                verify_cmd.append(f"--transpose-{name}")
            else:
                verify_cmd.append(f"--no-transpose-{name}")

        verify_results = []
        # Use JSON emission for verification when possible
        use_json_verify = args.emit_json and vocab is not None
        for i in range(1, args.verify_runs + 1):
            stamp = int(time.time() * 1000)
            logfile = logdir / f"verify_{mask}_{stamp}_run{i}.log"
            logging.info("verify run %d -> %s", i, shlex.join(verify_cmd))

            run_env = env.copy()
            if use_json_verify:
                run_env["MINXFMR_EMIT_JSON"] = "1"

            out_text, err_text, timed_out, returncode = run_proc(
                list(verify_cmd), run_env, cflags, max(args.timeout_sec, 120), args.chat_mode, args.prompt
            )
            full = "[STDOUT]\n" + (out_text or "") + "\n[STDERR]\n" + (err_text or "")
            logfile.write_text(full, encoding="utf-8")

            # score verification runs using JSON when available, fallback otherwise
            score = 0
            if use_json_verify and out_text:
                try:
                    import json as _json, base64 as _base64

                    j = _json.loads(out_text)
                    token_ids = j.get("token_ids", [])
                    text_b64 = j.get("text_b64", "")
                    decoded = ""
                    if text_b64:
                        try:
                            b = _base64.b64decode(text_b64)
                            decoded = b.decode("utf-8", errors="replace")
                        except Exception:
                            decoded = ""

                    total_chars = len(decoded)
                    if total_chars == 0:
                        valid_utf8_rate = 0.0
                    else:
                        repl = decoded.count('\ufffd')
                        valid_utf8_rate = max(0.0, 1.0 - (repl / total_chars))

                    if token_ids and vocab:
                        hits = 0
                        for tid in token_ids:
                            if not isinstance(tid, int):
                                continue
                            if tid < 0 or tid >= len(vocab):
                                continue
                            tok = vocab[tid]
                            if not tok or tok.startswith("<0x") or tok == "<unk>" or ("<" in tok and ">" in tok and len(tok) <= 8):
                                continue
                            if any(('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in tok):
                                hits += 1
                        vocab_rate = float(hits) / float(len(token_ids)) if token_ids else 0.0
                    else:
                        vocab_rate = 0.0

                    combined = (0.6 * valid_utf8_rate) + (0.4 * vocab_rate)
                    score = int(combined * 1000)
                except Exception as e:
                    logging.warning("Failed JSON scoring verify: %s", e)
                    score = score_output(out_text)
            else:
                score = score_output(out_text)

            verify_results.append({"run": i, "log": str(logfile), "score": score, "timed_out": bool(timed_out)})

        summary["verification"] = {"mask": mask, "results": verify_results}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logging.info("Verification results appended to %s", summary_path)

    logging.info("Best candidate: %s", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
