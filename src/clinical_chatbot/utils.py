import re
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import Tuple, Optional, List

import pandas as pd
import chainlit as cl


# -------- User helpers --------

def get_current_user():
    """
    Return cl.User across Chainlit versions (function vs module).
    """
    # Newer API: cl.user() callable
    try:
        maybe = getattr(cl, "user", None)
        if callable(maybe):
            u = maybe()
            if u:
                return u
    except Exception:
        pass

    # Fallback: context
    try:
        from chainlit.context import get_context
        ctx = get_context()
        if getattr(ctx, "session", None) and ctx.session.user:
            return ctx.session.user
    except Exception:
        pass

    # Last resort: whatever we stashed
    try:
        tmp = cl.user_session.get("user")
        if tmp:
            if hasattr(tmp, "identifier"):
                return tmp  # it's a cl.User-like object
            if isinstance(tmp, dict):
                return cl.User(
                    identifier=tmp.get("name", "unknown"),
                    metadata={"role": tmp.get("role", "user")}
                )
    except Exception:
        pass

    return None


def session_id_generator() -> str:
    """
    Build a session id using the username + timestamp.
    """
    u = get_current_user()
    username = (u.identifier if u else "user").replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{username}_{ts}"


# -------- Filename / logging helpers --------

def _sanitize_for_path(s: str) -> str:
    """Keep simple cross-platform-safe characters."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def get_session_meta_for_filename() -> Tuple[str, str]:
    """
    Return (safe_username, safe_session_id) for filenames.
    Accepts cl.user_session['user'] as either a dict or a cl.User.
    """
    sess_id = cl.user_session.get("session_id")
    if not sess_id:
        sess_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    username = None
    user_obj = cl.user_session.get("user")

    # handle dict or cl.User
    if isinstance(user_obj, dict):
        username = user_obj.get("name") or user_obj.get("identifier") or user_obj.get("username")
    elif user_obj is not None:
        username = getattr(user_obj, "identifier", None) or getattr(user_obj, "name", None)

    # fallback to current user
    if not username:
        u = get_current_user()
        if u:
            username = getattr(u, "identifier", None) or getattr(u, "name", None)

    if not username:
        username = "unknown"

    return _sanitize_for_path(username), _sanitize_for_path(sess_id)


def log_line(text: str) -> None:
    """Append a single line to the current session log (if available)."""
    path = cl.user_session.get("log_path")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")
    except Exception:
        pass


# -------- Timing helpers --------

def q_start() -> None:
    """Mark the start time for the current question."""
    cl.user_session.set("question_t0", perf_counter())


def q_stop_and_record(qid) -> float:
    """
    Stop timer; store elapsed for this question; return seconds.
    Stores into:
      - 'elapsed_list' (ordered list)
      - 'elapsed_by_qid' (dict)
    """
    t0 = cl.user_session.get("question_t0")
    if t0 is None:
        t0 = perf_counter()
    elapsed = max(0.0, perf_counter() - t0)

    lst = cl.user_session.get("elapsed_list") or []
    byq = cl.user_session.get("elapsed_by_qid") or {}
    lst.append(elapsed)
    byq[str(qid)] = elapsed
    cl.user_session.set("elapsed_list", lst)
    cl.user_session.set("elapsed_by_qid", byq)
    return elapsed


# -------- CSV / finalize helpers --------

async def save_csv(
    df: pd.DataFrame,
    answers: List[str],
    *,
    out_dir: Path,
    model_name: str,
    final: bool = True,
    suffix: Optional[str] = None,
) -> None:
    """
    Write CSV with `real_phy_answer` + `time_spent_seconds`, named with user & session.
    """
    out = df.copy()
    out["real_phy_answer"] = answers

    elapsed = cl.user_session.get("elapsed_list") or []
    elapsed = (elapsed + [None] * len(out))[:len(out)]
    out["time_spent_seconds"] = elapsed

    out_dir.mkdir(parents=True, exist_ok=True)

    safe_user, safe_sess = get_session_meta_for_filename()
    base = f"out_interactive_ass_{model_name}__{safe_user}__{safe_sess}"
    if not final:
        base += f"__{suffix or 'partial'}"
    out_csv = out_dir / f"{base}.csv"

    out.to_csv(out_csv, index=False)
    log_line(f"Saved CSV -> {out_csv}")


async def finalize_session(
    df: Optional[pd.DataFrame],
    answers: List[str],
    *,
    partial: bool,
    out_dir: Path,
    model_name: str,
) -> None:
    """
    On finish/stop/end: write totals and optional partial CSV.
    """
    t0 = cl.user_session.get("session_t0")
    if t0 is not None:
        total = perf_counter() - t0
        cl.user_session.set("session_total_sec", total)
        log_line(f"Total session time: {total:.2f} sec")

    if partial and df is not None and answers:
        # Save only answered rows so far
        await save_csv(
            df.iloc[:len(answers)],
            answers,
            out_dir=out_dir,
            model_name=model_name,
            final=False,
            suffix="partial",
        )
