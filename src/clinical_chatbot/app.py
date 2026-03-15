import os
import inspect
from pathlib import Path
from datetime import datetime
from time import perf_counter
import asyncio
import traceback
from textwrap import dedent
import json
from urllib.parse import quote

import pandas as pd

import chainlit as cl
import langroid as lr
import langroid.language_models as lm
from langroid.agent.callbacks.chainlit import add_instructions, ChainlitAgentCallbacks

import src.clinical_chatbot.health  # noqa: F401
from src.physician_agent import PhysicianModel
from src.assistant_agent import AssistantModel
from src.utils import read_prompt_from_file, serialize_dict  # noqa: F401

from src.clinical_chatbot.db import get_conn
from src.clinical_chatbot.auth import verify_password, ensure_admin

from src.clinical_chatbot.utils import (
    get_current_user,
    session_id_generator,
    get_session_meta_for_filename as _get_session_meta_for_filename,
    log_line as _log_line,
    q_start as _q_start,
    q_stop_and_record as _q_stop_and_record,
    save_csv as _save_csv,
    finalize_session as _finalize_session,
    log_dialogue_turn as _log_dialogue_turn,
)


import logging
from logging.handlers import RotatingFileHandler

# =========================
# Env / config
# =========================
LOG_DIR        = Path(os.getenv("LOG_DIR"))
DATASET_FOLDER = Path(os.getenv("DATASET_FOLDER"))
PROMPTS_DIR    = Path(os.getenv("PROMPT"))
OUT_DIR        = Path(os.getenv("OUT_DIR"))

INPUT_DATA_FILE = os.getenv("INPUT_DATA_FILE")
SAMPLE_SIZE     = int(os.getenv("SAMPLE_SIZE", "0") or "0")

PHY_MODEL       = os.getenv("PHY_MODEL")
ASS_MODEL       = os.getenv("ASS_MODEL")
PHY_MODEL_NAME  = os.getenv("PHY_MODEL_NAME")
ASS_MODEL_NAME  = os.getenv("ASS_MODEL_NAME")

# Which backend to use for the assistant LLM:
#   - "ollama"    (default): use the Ollama sidecar
#   - "vllm"      : use a vLLM OpenAI-compatible server at VLLM_API_BASE
#   - "openrouter": use commercial models via OpenRouter
LLM_BACKEND   = os.getenv("LLM_BACKEND", "ollama").strip().lower()

# For vLLM you must set this to something like "http://vllm:8000/v1"
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "").strip().rstrip("/")

# Ollama sidecar (used when LLM_BACKEND == "ollama")
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://ollama:11434")  # default for Docker
OLLAMA_API_BASE = f"{OLLAMA_HOST.rstrip('/')}/v1"

RANDOM_SEED     = 42

# Experiment mode
#   - "interactive" (default): chief complaint only + assistant LLM
#   - "baseline"             : full note, no assistant (user submits final diagnosis directly)
EXPERIMENT_MODE = os.getenv("EXPERIMENT_MODE", "interactive").strip().lower()
IS_BASELINE = EXPERIMENT_MODE == "baseline"

# Prompt files (configurable via environment)
PHY_SYSTEM_PROMPT_FILE       = os.getenv("PHY_SYSTEM_PROMPT_FILE", "real_phy_system_prompt.txt")
PHY_PROMPT_TEMPLATE_FILE     = os.getenv("PHY_PROMPT_TEMPLATE_FILE", "real_phy_user_prompt.txt")
ASS_SYSTEM_PROMPT_FILE       = os.getenv("ASS_SYSTEM_PROMPT_FILE", "real_ass_system_prompt.txt")
ASS_PROMPT_TEMPLATE_FILE     = os.getenv("ASS_PROMPT_TEMPLATE_FILE", "real_ass_user_prompt.txt")

phy_system_prompt   = PROMPTS_DIR / PHY_SYSTEM_PROMPT_FILE
phy_prompt_template = PROMPTS_DIR / PHY_PROMPT_TEMPLATE_FILE
ass_system_prompt   = PROMPTS_DIR / ASS_SYSTEM_PROMPT_FILE
ass_prompt_template = PROMPTS_DIR / ASS_PROMPT_TEMPLATE_FILE

model_responses: list[str] = []

# =========================
# Agent / prompt builders
# =========================
def create_assistant_agent(ass_prompt: str) -> lr.ChatAgent:
    """
    Build the Langroid ChatAgent for the assistant, using the backend selected by
    the LLM_BACKEND env var.

    Supported backends:
      - "ollama"    (default): use the Ollama sidecar at OLLAMA_API_BASE
      - "vllm"      : use a vLLM OpenAI-compatible server at VLLM_API_BASE
      - "openrouter": use commercial models via OpenRouter (OPENROUTER_API_KEY)
    """
    backend = (LLM_BACKEND or "ollama").lower()

    if backend == "ollama":
        # Local open-source models via the Ollama sidecar
        cfg = lm.OpenAIGPTConfig(
            timeout=180,
            chat_context_length=1_040_000,
            chat_model=f"ollama/{ASS_MODEL}",
            api_base=OLLAMA_API_BASE,
            seed=RANDOM_SEED,
        )

    elif backend == "vllm":
        # vLLM exposes an OpenAI-compatible /v1/chat/completions endpoint.
        # Example VLLM_API_BASE: "http://vllm:8000/v1"
        if not VLLM_API_BASE:
            raise RuntimeError(
                "LLM_BACKEND is set to 'vllm' but VLLM_API_BASE is empty. "
                "Set VLLM_API_BASE to your vLLM OpenAI-compatible base URL "
                "(e.g. http://vllm:8000/v1)."
            )
        cfg = lm.OpenAIGPTConfig(
            timeout=180,
            chat_context_length=1_040_000,
            chat_model=ASS_MODEL,      # vLLM model name, e.g. "Mistral-7B-Instruct-v0.2"
            api_base=VLLM_API_BASE,
            seed=RANDOM_SEED,
        )

    elif backend == "openrouter":
        # Commercial models via OpenRouter (OpenAI-compatible).
        # Langroid will read OPENROUTER_API_KEY and route requests to
        # https://openrouter.ai/api/v1 under the hood when chat_model starts
        # with "openrouter/".
        #
        # ASS_MODEL should be a full OpenRouter model id such as
        #   "openai/gpt-4o" or "google/gemini-2.5-flash"
        cfg = lm.OpenAIGPTConfig(
            timeout=180,
            chat_context_length=1_040_000,
            chat_model=f"openrouter/{ASS_MODEL}",
            seed=RANDOM_SEED,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_BACKEND={backend!r}. "
            "Use 'ollama', 'vllm', or 'openrouter'."
        )

    return lr.ChatAgent(lr.ChatAgentConfig(system_message=ass_prompt, llm=cfg))

def create_prompts(row) -> tuple[str, str]:
    chief_complaint = str(row.get("chief_complaint", ""))
    patient_history = str(row.get("history", ""))
    physical_exam   = str(row.get("physical_exam", ""))
    exam_results    = str(row.get("results", ""))

    phy_clinical_note = "CHIEF COMPLAINT: " + chief_complaint
    ass_clinical_note = (
        "----------------------------------------------------------\n\nCHIEF COMPLAINT: " + chief_complaint
        + "\n\n----------------------------------------------------------\n\nPATIENT HISTORY: \n\n" + patient_history
        + "\n\n----------------------------------------------------------\n\nPHYSICAL EXAM: \n\n" + physical_exam
        + "\n\n----------------------------------------------------------\n\nPERTINENT RESULTS: \n\n" + exam_results
    )

    read_phy_sys  = read_prompt_from_file(phy_system_prompt)
    read_phy_tmpl = read_prompt_from_file(phy_prompt_template)
    read_ass_sys  = read_prompt_from_file(ass_system_prompt)
    read_ass_tmpl = read_prompt_from_file(ass_prompt_template)

    phy_model = PhysicianModel(
        PHY_MODEL_NAME,
        system_prompt=read_phy_sys,
        user_prompt_template=read_phy_tmpl,
    )
    ass_model = AssistantModel(
        ASS_MODEL_NAME,
        system_prompt=read_ass_sys,
        user_prompt_template=read_ass_tmpl,
    )

    phy_prompt = phy_model.generate_prompt()
    ass_prompt = ass_model.generate_prompt(ass_clinical_note)
    return ass_prompt, ass_clinical_note, phy_prompt, phy_clinical_note

def format_clinical_note_for_baseline(full_note: str) -> str:
    """
    Format the full clinical note (given as a single string) for human reading
    in baseline mode.

    We wrap the entire note in a fenced code block to prevent Markdown from
    interpreting any characters (e.g. '=', '_') as headings or formatting.
    """
    text = (full_note or "").strip()
    # Avoid accidentally closing the code block if ``` appears in the note
    text = text.replace("```", "'''")

    return f"### Clinical note\n\n```text\n{text}\n```"


# =========================
# Auth
# =========================
ensure_admin()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT password_hash, role FROM users WHERE username=?",
        (username,),
    ).fetchone()
    conn.close()
    if row and verify_password(password, row[0]):
        return cl.User(
            identifier=username,
            metadata={"role": row[1], "provider": "password"},
        )
    return None

# =========================
# Streaming/task stopper
# =========================
async def _stop_streaming_and_optionally_close(close_agent: bool = False):
    task: asyncio.Task | None = cl.user_session.get("last_assistant_task")
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    cl.user_session.set("last_assistant_task", None)

    if close_agent:
        agent = cl.user_session.get("ass_agent")
        if agent:
            for m in ("reset", "close", "aclose"):
                fn = getattr(agent, m, None)
                if callable(fn):
                    try:
                        res = fn()
                        if inspect.iscoroutine(res):
                            await res
                    except Exception:
                        pass
        cl.user_session.set("ass_agent", None)

# =========================
# Client redirect trigger
# =========================
async def _force_client_redirect_to_thanks(
    username: str,
    sess_id: str,
    total_sec: float,
    n_cases_done: int,
) -> None:
    """
    Notify the frontend that the session is finished and redirect the user
    to the thank-you/survey page.

    We do two things:
      1. Send a visible fallback message with a direct link to the summary page.
      2. Send the [[AUTO_LOGOUT::...]] marker used by public/login.js for
         automatic redirect + logout, for environments where that JS hook
         is active.
    """
    safe_username = username or "unknown"
    safe_sess_id = sess_id or ""
    safe_total = float(total_sec or 0.0)
    safe_n_cases = int(n_cases_done or 0)

    # Payload understood by public/thank-you.html (via decodeFromHash()).
    summary = {
        "username": safe_username,
        "sessionId": safe_sess_id,
        "totalTimeSec": safe_total,
        "nCases": safe_n_cases,
        "endedAt": datetime.now().isoformat(),
    }
    try:
        payload = quote(json.dumps(summary))
        thank_you_url = f"/public/thank-you.html#{payload}"
    except Exception:
        # Very defensive: fall back to a simple URL without payload
        thank_you_url = "/public/thank-you.html"

    # 1) Visible fallback link for users (works even if JS marker watcher fails)
    await cl.Message(
        content=(
            "You will now be redirected to the session summary and survey. Please complete the post-session questionnaire. \n\n"
            f"If nothing happens automatically, please open this link:\n{thank_you_url}"
        )
    ).send()

    # 2) Marker for automatic redirect via public/login.js (if active)
    await cl.Message(
        content=(
            f"[[AUTO_LOGOUT::{safe_username}::{safe_sess_id}::"
            f"{safe_total:.2f}::{safe_n_cases}]]"
        )
    ).send()

    # Give the client a moment to receive/process messages
    await asyncio.sleep(0.05)


def _log_conversation_snapshot(prefix: str = "") -> None:
    """
    Log the current agent.message_history (user + LLM messages) to the session log.
    """
    agent = cl.user_session.get("ass_agent")
    if not agent:
        return

    try:
        mh = getattr(agent, "message_history", None)
        if not (mh and hasattr(mh, "messages")):
            return

        if prefix:
            _log_line(
                f"{prefix} Conversation snapshot ({datetime.now().isoformat()}):"
            )

        for m in mh.messages:
            raw_role = getattr(m, "role", None) or getattr(m, "sender", "assistant")
            text = getattr(m, "content", "")

            if raw_role in ("assistant", "system"):
                who = "LLM"
            elif raw_role in ("user", "human"):
                who = "USER"
            else:
                who = raw_role or "UNKNOWN"

            ts = datetime.now().isoformat()
            _log_line(f"[{ts}] {who}: {text}")
    except Exception:
        # Never break the app because logging failed
        pass

async def _agent_respond_and_log(
    agent: lr.ChatAgent,
    user_text: str,
    idx: int,
    qid,
) -> None:
    """
    Wrap agent.llm_response_async to both stream to the UI (via
    ChainlitAgentCallbacks) and log the final LLM response text.
    """
    try:
        resp = await agent.llm_response_async(user_text)
        # Try to extract text from the response object
        text = None
        for attr in ("content", "text", "message", "msg"):
            if hasattr(resp, attr):
                text = getattr(resp, attr)
                break
        if text is None:
            text = str(resp) if resp is not None else "<empty response>"

        _log_line(f"LLM MESSAGE Q{idx + 1} (ID: {qid}): {text}")
        
        # Structured dialogue log: llm turn
        _log_dialogue_turn(
            sender="llm",
            message=text,
            idx=idx,
            qid=qid,
        )

    except asyncio.CancelledError:
        # Streaming cancelled (stop button etc.) - just propagate
        raise
    except Exception as e:
        _log_line(f"ERROR logging LLM response for Q{idx + 1} (ID: {qid}): {e!r}")

async def _start_case(idx: int) -> None:
    """Initialize and start the given case index (0-based)."""
    df: pd.DataFrame = cl.user_session.get("questions_df")
    if df is None or len(df) == 0:
        await cl.Message(content="No cases loaded.").send()
        return

    total = len(df)
    if idx < 0 or idx >= total:
        await cl.Message(content="No more cases available.").send()
        return

    cl.user_session.set("current_question_index", idx)

    row = df.iloc[idx]
    ass_prompt, ass_clinical_note, phy_prompt, phy_clinical_note = create_prompts(row)

    # In baseline mode we do NOT create an assistant; users only see the note
    # and then submit their final diagnosis directly.
    if IS_BASELINE:
        agent = None
        cl.user_session.set("ass_agent", None)
    else:
        agent = create_assistant_agent(ass_prompt)
        cl.user_session.set("ass_agent", agent)
        ChainlitAgentCallbacks(agent)

    qid = row.get("note_id", idx + 1)
    _q_start()
    _log_line(f"Q{idx + 1} (ID: {qid}) started at {datetime.now().isoformat()}")
    
    # Structured dialogue marker: case start
    _log_dialogue_turn(
        sender="system",
        message=f"CASE_START: note_id={qid}",
        idx=idx,
        qid=qid,
    )

    # Mark that a case has started => we now accept user input
    cl.user_session.set("case_started", True)

    # Decide which clinical note to show to the clinician
    # - Baseline   -> full clinical note (assessment view)
    # - Interactive-> chief complaint only (physician view)
    note_to_show = ass_clinical_note if IS_BASELINE else phy_clinical_note

    if IS_BASELINE:
        # Nice, structured view of the full note for clinicians
        note_to_show = format_clinical_note_for_baseline(note_to_show)
        msg_content = (
            f"Case {idx+1} is ready. Please see the full clinical note below "
            "and then type your final discharge diagnosis in the chat.\n\n"
            + note_to_show
        )
    else:
        # Interactive mode: only chief complaint, plain text
        msg_content = (
            f"Case {idx+1} is ready. Please see the clinical note for the patient "
            "below. You can now start the conversation with the assistant.\n\n"
            + dedent(note_to_show)
        )

    try:
        await cl.Message(content=msg_content).send()
    except Exception:
        await cl.Message(
            content=f"Started case {idx + 1}. Please continue the conversation."
        ).send()


# =========================
# Lifecycle
# =========================
@cl.on_chat_start
async def on_chat_start():
    input_file = DATASET_FOLDER / INPUT_DATA_FILE
    df = pd.read_csv(input_file).reset_index(drop=True)
    if SAMPLE_SIZE:
        df = df.iloc[:SAMPLE_SIZE].copy()

    needed = ["note_id", "chief_complaint", "history", "physical_exam", "results"]
    for col in needed:
        if col not in df.columns:
            df[col] = ""

    user = get_current_user()
    session_id = session_id_generator()
    cl.user_session.set("user", user)
    cl.user_session.set("session_id", session_id)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_user, safe_sess = _get_session_meta_for_filename()
    date_str = datetime.now().strftime("%Y%m%d")
    log_path = LOG_DIR / f"chat__{date_str}__{safe_user}__{safe_sess}.log"

    # Configure logging for this session
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    cl.user_session.set("log_path", str(log_path))
    cl.user_session.set("questions_df", df)
    cl.user_session.set("current_question_index", 0)
    cl.user_session.set("answers", {})
    cl.user_session.set("total_questions", len(df))
    cl.user_session.set("elapsed_list", [])
    cl.user_session.set("elapsed_by_qid", {})
    cl.user_session.set("question_t0", None)
    cl.user_session.set("session_t0", perf_counter())
    cl.user_session.set("last_assistant_task", None)
    cl.user_session.set("case_started", False)  # gate input until Start Case 1
    cl.user_session.set("session_completed", False)


    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Session started: {datetime.now().isoformat()}\n")
        f.write(f"Session ID: {session_id}\n")
        f.write(f"Experiment mode: {EXPERIMENT_MODE}\n")
        f.write(f"Total questions in session: {len(df)}\n\n")

    # Build and show instructions from the first case, but DO NOT start the agent yet
    first_row = df.iloc[0]
    ass_prompt, ass_clinical_note, phy_prompt, phy_clinical_note = create_prompts(first_row)

    await add_instructions(
        title="Welcome to MedSyn!",
        content=dedent(phy_prompt),
    )

    # Mode-specific clarifications for the user
    if IS_BASELINE:
        await cl.Message(
            content=dedent(
                """
                **Study condition: Baseline (no AI support)**
                """
            )
        ).send()
    else:
        await cl.Message(
            content=dedent(
                """
                **Study condition: Interactive (with AI assistant)**
                """
            )
        ).send()

    cl.user_session.set("instructions_shown", True)


    # Hard gate: user must explicitly start or exit
    res = await cl.AskActionMessage(
        content=(
            "Please read the instructions carefully.\n\n"
            "When you are ready, you can click on Start to proceed with first patient case. "
            "If you prefer doing the task later, you can click on Exit to end your session."
        ),
        actions=[
            cl.Action(
                name="start_case",
                payload={"value": "start", "case_index": 0},
                label="Start Case 1",
            ),
            cl.Action(
                name="exit_study",
                payload={"value": "exit"},
                label="Exit without starting",
            ),
        ],
        timeout=3600,  # allow up to 1 hour to answer
        raise_on_timeout=False,
    ).send()

    _log_line(f"Start screen AskAction response: {res}")

    if not res:
        _log_line("User did not answer AskAction at start (no response).")
        await cl.Message(
            content="Session ended before starting any case."
        ).send()
        return

    payload = res.get("payload") or {}
    value = payload.get("value")

    if value == "exit":
        _log_line("User chose to exit at start, no cases started.")
        
        await cl.Message(
            content="You chose to exit."
        ).send()
        
        cl.user_session.set("case_started", False)
        cl.user_session.set("session_completed", True)
        # No cases started: mark session as finalized (0 completed)
        total_sec = 0.0
        user_obj = cl.user_session.get("user") or get_current_user()
        if isinstance(user_obj, dict):
            username = (
                user_obj.get("name")
                or user_obj.get("identifier")
                or "unknown"
            )
        else:
            username = (
                getattr(user_obj, "identifier", None)
                or getattr(user_obj, "name", None)
                or "unknown"
            )
        n_cases_done = 0

        # Baseline: do NOT redirect to the thank-you/survey page
        if not IS_BASELINE:
            await _force_client_redirect_to_thanks(
                username, session_id, total_sec, n_cases_done
            )
        return

    # Default: start case 1
    case_index = int(payload.get("case_index", 0))
    await _start_case(case_index)

@cl.on_message
async def on_message(message: cl.Message):
    try:
        # If session has been marked as completed, block further input
        if cl.user_session.get("session_completed", False):
            await cl.Message(
                content=(
                    "This session has already ended. "
                    "Please refresh the page to start a new session."
                )
            ).send()
            return

        # Gate: ignore user input until a case has started
        if not cl.user_session.get("case_started", False):
            await cl.Message(
                content=(
                    "Please click **Start Case 1** to begin the session. "
                    "Your input has not been recorded."
                )
            ).send()
            return

        df: pd.DataFrame = cl.user_session.get("questions_df")
        agent: lr.ChatAgent = cl.user_session.get("ass_agent")
        idx = cl.user_session.get("current_question_index", 0)
        total = cl.user_session.get("total_questions", len(df) if df is not None else 0)
        answers: dict = cl.user_session.get("answers") or {}
        global model_responses

        if df is None or len(df) == 0:
            await cl.Message(content="No cases loaded.").send()
            return

        # Defensive: ensure index is within bounds of the dataframe
        if idx < 0 or idx >= len(df):
            _log_line(
                f"Received message for out-of-range case index {idx} "
                f"(total cases: {len(df)}). Treating as session already completed."
            )
            await cl.Message(
                content=(
                    "You have already completed all cases in this session. "
                    "If you wish to start a new session, please refresh the page."
                )
            ).send()
            cl.user_session.set("session_completed", True)
            cl.user_session.set("case_started", False)
            return
        
        if agent:
            ChainlitAgentCallbacks(agent)

        row = df.iloc[idx]
        qid = row.get("note_id", idx + 1)

        content = message.content.strip()
        upper = content.upper()

        # Log every incoming user message
        _log_line(f"USER MESSAGE Q{idx + 1} (ID: {qid}): {content}")
        
        # Structured dialogue log: user turn
        _log_dialogue_turn(
            sender="user",
            message=content,
            idx=idx,
            qid=qid,
        )

        # In baseline mode, any non-EXIT message is treated as the final diagnosis.
        # We internally rewrite it into a FINAL ANSWER command so that the existing
        # flow for recording answers and ending the session can be reused.
        if IS_BASELINE and not upper.startswith("EXIT"):
            content = f"FINAL ANSWER: {content}"
            upper = content.upper()

        # EXIT -> partial finalize + marker
        if upper.startswith("EXIT"):
            t0 = cl.user_session.get("question_t0")
            if t0 is not None:
                sofar = perf_counter() - t0
                _log_line(
                    f"EXIT received. Elapsed so far on Q{idx + 1} (ID: {qid}): {sofar:.2f} sec"
                )

            # Log what has been discussed so far in this session
            _log_conversation_snapshot(
                prefix=f"On EXIT from Q{idx + 1} (ID: {qid})"
            )
            
            # Structured dialogue marker: case aborted by user exit
            _log_dialogue_turn(
                sender="system",
                message="CASE_ABORT: user_exit",
                idx=idx,
                qid=qid,
            )

            await _stop_streaming_and_optionally_close(close_agent=True)
            await _finalize_session(
                df,
                list(answers.values()),
                partial=True,
                out_dir=OUT_DIR,
                model_name=ASS_MODEL_NAME,
            )

            total_sec = cl.user_session.get("session_total_sec") or 0.0
            sess_id = cl.user_session.get("session_id") or ""
            user_obj = cl.user_session.get("user") or get_current_user()
            if isinstance(user_obj, dict):
                username = (
                    user_obj.get("name")
                    or user_obj.get("identifier")
                    or "unknown"
                )
            else:
                username = (
                    getattr(user_obj, "identifier", None)
                    or getattr(user_obj, "name", None)
                    or "unknown"
                )
            n_cases_done = len(answers)
            
            cl.user_session.set("session_completed", True)
            cl.user_session.set("case_started", False)

            await cl.Message(content="Session ended. Thank you.").send()
            
            if not IS_BASELINE:
                await _force_client_redirect_to_thanks(
                    username, sess_id, total_sec, n_cases_done
                )
                
            return

        # FINAL ANSWER -> record, move to next or end
        if upper.startswith("FINAL ANSWER:") or upper.startswith("FINAL DIAGNOSIS:"):
            final_answer = content.split(":", 1)[1].strip()
            answers[qid] = final_answer
            cl.user_session.set("answers", answers)
            model_responses.append(final_answer)

            elapsed = _q_stop_and_record(qid)
            _log_line(
                f"FINAL ANSWER Q{idx + 1} (ID: {qid}): {final_answer}"
            )
            _log_line(
                f"Time for Q{idx + 1} (ID: {qid}): {elapsed:.2f} sec"
            )
            
            # Structured dialogue marker: case end with final answer
            _log_dialogue_turn(
                sender="system",
                message=f"CASE_END: final_answer={final_answer}",
                idx=idx,
                qid=qid,
            )

            # Acknowledge the answer and stop current task
            await cl.Message(
                content=f"Thank you for your answer to Question {idx + 1}."
            ).send()
            await _stop_streaming_and_optionally_close(close_agent=False)

            # Log the conversation up to this point
            _log_conversation_snapshot(
                prefix=f"After FINAL ANSWER for Q{idx + 1} (ID: {qid})"
            )

            idx += 1
            cl.user_session.set("current_question_index", idx)

            if idx < total:
                # There are more cases: start next without re-showing full instructions
                await cl.Message(
                    content=(
                        f"✅ Answer recorded for Question {idx}. "
                        "Loading the next case..."
                    )
                ).send()
                await _start_case(idx)
                return  # don't let the LLM answer the FINAL ANSWER message
            else:
                # Last case completed
                sess_id = cl.user_session.get("session_id") or ""
                user_obj = cl.user_session.get("user") or get_current_user()
                if isinstance(user_obj, dict):
                    username = (
                        user_obj.get("name")
                        or user_obj.get("identifier")
                        or "unknown"
                    )
                else:
                    username = (
                        getattr(user_obj, "identifier", None)
                        or getattr(user_obj, "name", None)
                        or "unknown"
                    )
                n_cases_done = len(answers)

                # BASELINE: finalize immediately, no "End session" button, no redirect
                if IS_BASELINE:
                    await _finalize_session(
                        df,
                        list(answers.values()),
                        partial=False,
                        out_dir=OUT_DIR,
                        model_name=ASS_MODEL_NAME,
                    )
                    await _save_csv(
                        df,
                        list(answers.values()),
                        out_dir=OUT_DIR,
                        model_name=ASS_MODEL_NAME,
                        final=True,
                    )
                    await _stop_streaming_and_optionally_close(close_agent=True)

                    cl.user_session.set("session_completed", True)
                    cl.user_session.set("case_started", False)

                    total_sec = cl.user_session.get("session_total_sec") or 0.0
                    _log_line(
                        f"Baseline session completed for user={username}, "
                        f"session={sess_id}, total_sec={total_sec:.2f}, "
                        f"n_cases_done={n_cases_done}"
                    )

                    await cl.Message(
                        content="✅ All cases completed. Thank you! You can now log out from this session."
                    ).send()
                    # Important: NO redirect in baseline mode
                    return

                # INTERACTIVE: keep existing behaviour with End Session button + redirect
                res = await cl.AskActionMessage(
                    content=(
                        "You have completed all cases.\n\n"
                        "Please click the button below to finalize your session."
                    ),
                    actions=[
                        cl.Action(
                            name="end_session",
                            payload={"value": "end"},
                            label="End session",
                        ),
                    ],
                    timeout=3600,
                    raise_on_timeout=False,
                ).send()

                _log_line(f"End-of-session AskAction response: {res}")

                if not res:
                    # User closed the tab or did not answer; keep session open
                    return

                payload = res.get("payload") or {}
                value = payload.get("value")

                # 1) Finalize -> this sets session_total_sec in user_session
                await _finalize_session(
                    df,
                    list(answers.values()),
                    partial=False,
                    out_dir=OUT_DIR,
                    model_name=ASS_MODEL_NAME,
                )
                await _save_csv(
                    df,
                    list(answers.values()),
                    out_dir=OUT_DIR,
                    model_name=ASS_MODEL_NAME,
                    final=True,
                )
                await _stop_streaming_and_optionally_close(close_agent=True)

                cl.user_session.set("session_completed", True)
                cl.user_session.set("case_started", False)

                # 2) NOW read the total session time
                total_sec = cl.user_session.get("session_total_sec") or 0.0

                await cl.Message(
                    content="✅ All cases completed. Thank you!"
                ).send()
                await _force_client_redirect_to_thanks(
                    username, sess_id, total_sec, n_cases_done
                )

                return  # important: don't fall through


        # Normal discussion with assistant
        if not agent:
            await cl.Message(
                content="Assistant not available at the moment."
            ).send()
            return

        # Run LLM, stream to UI (via ChainlitAgentCallbacks) and log response text
        task = asyncio.create_task(
            _agent_respond_and_log(agent, message.content, idx, qid)
        )
        cl.user_session.set("last_assistant_task", task)
        try:
            await task
        finally:
            cl.user_session.set("last_assistant_task", task)

    except Exception:
        tb = traceback.format_exc()
        _log_line(f"ERROR in on_message:\n{tb}")
        raise

@cl.on_stop
async def _on_stop():
    # If the session is already completed, just log and ignore
    if cl.user_session.get("session_completed", False):
        _log_line("on_stop called but session already completed; skipping partial finalize.")
        return
    
    # Log whatever conversation we have up to this point
    _log_conversation_snapshot(prefix="on_stop")

    await _stop_streaming_and_optionally_close(close_agent=True)
    df: pd.DataFrame = cl.user_session.get("questions_df")
    answers_dict = cl.user_session.get("answers") or {}
    await _finalize_session(
        df,
        list(answers_dict.values()),
        partial=True,
        out_dir=OUT_DIR,
        model_name=ASS_MODEL_NAME,
    )

@cl.on_chat_end
async def _on_chat_end():
    if cl.user_session.get("session_completed", False):
        _log_line("on_chat_end called but session already completed; skipping partial finalize.")
        return
    
    # Log whatever conversation we have up to this point
    _log_conversation_snapshot(prefix="on_chat_end")

    await _stop_streaming_and_optionally_close(close_agent=True)
    df: pd.DataFrame = cl.user_session.get("questions_df")
    answers_dict = cl.user_session.get("answers") or {}
    await _finalize_session(
        df,
        list(answers_dict.values()),
        partial=True,
        out_dir=OUT_DIR,
        model_name=ASS_MODEL_NAME,
    )
