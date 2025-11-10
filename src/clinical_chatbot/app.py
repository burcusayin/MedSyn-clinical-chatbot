# src/clinical_chatbot/app.py
import os
import inspect
from pathlib import Path
from datetime import datetime
from time import perf_counter
import asyncio
from textwrap import dedent

import pandas as pd
import chainlit as cl
import langroid as lr
import langroid.language_models as lm
from langroid.agent.callbacks.chainlit import add_instructions, ChainlitAgentCallbacks

from src.physician_agent import PhysicianModel
from src.assistant_agent import AssistantModel
from src.utils import read_prompt_from_file, serialize_dict

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
)

# =========================
# Env / config
# =========================
LOG_DIR        = Path(os.getenv("LOG_DIR"))
DATASET_FOLDER = Path(os.getenv("DATASET_FOLDER"))
PROMPTS_DIR    = Path(os.getenv("PROMPT"))
OUT_DIR        = Path(os.getenv("OUT_DIR"))

INPUT_DATA_FILE = os.getenv("INPUT_DATA_FILE")
SAMPLE_SIZE     = int(os.getenv("SAMPLE_SIZE"))

PHY_MODEL       = os.getenv("PHY_MODEL")
ASS_MODEL       = os.getenv("ASS_MODEL")
PHY_MODEL_NAME  = os.getenv("PHY_MODEL_NAME")
ASS_MODEL_NAME  = os.getenv("ASS_MODEL_NAME")

RANDOM_SEED     = 42

# Prompt files (your original design)
phy_system_prompt   = PROMPTS_DIR / "real_phy_system_prompt.txt"
phy_prompt_template = PROMPTS_DIR / "real_phy_user_prompt.txt"
ass_system_prompt   = PROMPTS_DIR / "ass_system_prompt.txt"
ass_prompt_template = PROMPTS_DIR / "ass_user_prompt.txt"

model_responses: list[str] = []

# =========================
# Agent / prompt builders
# =========================
def create_assistant_agent(ass_prompt: str) -> lr.ChatAgent:
    cfg = lm.OpenAIGPTConfig(
        timeout=180,
        chat_context_length=1_040_000,
        chat_model=f"ollama/{ASS_MODEL}",
        seed=RANDOM_SEED,
    )
    return lr.ChatAgent(lr.ChatAgentConfig(system_message=ass_prompt, llm=cfg))

def create_prompts(row) -> tuple[str, str]:
    chief_complaint = str(row.get("chief_complaint", ""))
    patient_history = str(row.get("history", ""))
    physical_exam   = str(row.get("physical_exam", ""))
    exam_results    = str(row.get("results", ""))

    phy_clinical_note = "Chief complaint: " + chief_complaint
    ass_clinical_note = (
        "Chief complaint: " + chief_complaint
        + "\nPatient history: " + patient_history
        + "\nPhysical exam: " + physical_exam
        + "\nPertinent results: " + exam_results
    )

    read_phy_sys  = read_prompt_from_file(phy_system_prompt)
    read_phy_tmpl = read_prompt_from_file(phy_prompt_template)
    read_ass_sys  = read_prompt_from_file(ass_system_prompt)
    read_ass_tmpl = read_prompt_from_file(ass_prompt_template)

    phy_model = PhysicianModel(PHY_MODEL_NAME, system_prompt=read_phy_sys, user_prompt_template=read_phy_tmpl)
    ass_model = AssistantModel(ASS_MODEL_NAME, system_prompt=read_ass_sys, user_prompt_template=read_ass_tmpl)

    phy_prompt = phy_model.generate_prompt(phy_clinical_note)
    ass_prompt = ass_model.generate_prompt(ass_clinical_note)
    return ass_prompt, phy_prompt

# =========================
# Auth
# =========================
ensure_admin()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    conn = get_conn()
    row = conn.execute("SELECT password_hash, role FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    if row and verify_password(password, row[0]):
        return cl.User(identifier=username, metadata={"role": row[1], "provider": "password"})
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
# Client redirect trigger (marker-only; JS handles logout)
# =========================
async def _force_client_redirect_to_thanks(username: str, sess_id: str, total_sec: float, n_cases_done: int):
    await cl.Message(
        content=f"[[AUTO_LOGOUT::{username or 'unknown'}::{sess_id or ''}::{(total_sec or 0.0):.2f}::{int(n_cases_done or 0)}]]"
    ).send()
    await asyncio.sleep(0.05)  # let it flush to client

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
    cl.user_session.set("user", {"name": user.identifier if user else "guest",
                                 "role": (user.metadata or {}).get("role", "user") if user else "user"})
    cl.user_session.set("session_id", session_id)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_user, safe_sess = _get_session_meta_for_filename()
    log_path = LOG_DIR / f"chat__{safe_user}__{safe_sess}.log"

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

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Session started: {datetime.now().isoformat()}\n")
        f.write(f"Total questions in session: {len(df)}\n\n")

    row = df.iloc[0]
    ass_prompt, phy_prompt = create_prompts(row)
    agent = create_assistant_agent(ass_prompt)
    cl.user_session.set("ass_agent", agent)
    ChainlitAgentCallbacks(agent)

    await add_instructions(title="Welcome to Clinical Chatbot!", content=dedent(phy_prompt))

    qid = row.get("note_id", 1)
    _q_start()
    _log_line(f"Q1 (ID: {qid}) started at {datetime.now().isoformat()}")

@cl.on_message
async def on_message(message: cl.Message):
    df: pd.DataFrame = cl.user_session.get("questions_df")
    agent: lr.ChatAgent = cl.user_session.get("ass_agent")
    idx = cl.user_session.get("current_question_index", 0)
    total = cl.user_session.get("total_questions", len(df) if df is not None else 0)
    answers: dict = cl.user_session.get("answers") or {}
    global model_responses

    if df is None or len(df) == 0:
        await cl.Message(content="No cases loaded.").send()
        return

    if agent:
        ChainlitAgentCallbacks(agent)

    row = df.iloc[idx]
    qid = row.get("note_id", idx + 1)

    content = message.content.strip()
    upper = content.upper()

    # EXIT -> partial finalize + marker
    if upper.startswith("EXIT"):
        t0 = cl.user_session.get("question_t0")
        if t0 is not None:
            sofar = perf_counter() - t0
            _log_line(f"EXIT received. Elapsed so far on Q{idx + 1} (ID: {qid}): {sofar:.2f} sec")

        await _stop_streaming_and_optionally_close(close_agent=True)
        await _finalize_session(df, list(answers.values()), partial=True, out_dir=OUT_DIR, model_name=ASS_MODEL_NAME)

        total_sec = cl.user_session.get("session_total_sec") or 0.0
        sess_id   = cl.user_session.get("session_id") or ""
        user_obj  = cl.user_session.get("user") or get_current_user()
        if isinstance(user_obj, dict):
            username = user_obj.get("name") or user_obj.get("identifier") or "unknown"
        else:
            username = getattr(user_obj, "identifier", None) or getattr(user_obj, "name", None) or "unknown"
        n_cases_done = len(answers)

        await cl.Message(content="Session ended. Thank you.").send()
        await _force_client_redirect_to_thanks(username, sess_id, total_sec, n_cases_done)
        return

    # FINAL ANSWER -> record, move to next
    if upper.startswith("FINAL ANSWER:"):
        final_answer = content.split(":", 1)[1].strip()
        answers[qid] = final_answer
        cl.user_session.set("answers", answers)
        model_responses.append(final_answer)

        elapsed = _q_stop_and_record(qid)
        _log_line(f"FINAL ANSWER Q{idx + 1} (ID: {qid}): {final_answer}")
        _log_line(f"Time for Q{idx + 1} (ID: {qid}): {elapsed:.2f} sec")
       
        # Acknowledge the answer and stop current task
        await cl.Message(content=f"Thank you for your answer to Question {idx + 1}.").send()
        await _stop_streaming_and_optionally_close(close_agent=False)

        # optional: log conversation
        try:
            mh = getattr(agent, "message_history", None)
            if mh and hasattr(mh, "messages"):
                convo = []
                for m in mh.messages:
                    role = getattr(m, "role", None) or getattr(m, "sender", "assistant")
                    text = getattr(m, "content", "")
                    convo.append(f"{role}: {text}")
                _log_line("\n".join(convo) + "\n")
        except Exception:
            pass

        idx += 1
        cl.user_session.set("current_question_index", idx)

        if idx < total:
            next_row = df.iloc[idx]
            ass_prompt, phy_prompt = create_prompts(next_row)
            next_agent = create_assistant_agent(ass_prompt)
            cl.user_session.set("ass_agent", next_agent)
            ChainlitAgentCallbacks(next_agent)

            _q_start()
            next_qid = next_row.get("note_id", idx + 1)
            _log_line(f"Q{idx + 1} (ID: {next_qid}) started at {datetime.now().isoformat()}")

            await add_instructions(title=f"Case {idx + 1}", content=dedent(phy_prompt))
        else:
            await _finalize_session(df, list(answers.values()), partial=False, out_dir=OUT_DIR, model_name=ASS_MODEL_NAME)
            await _save_csv(df, list(answers.values()), out_dir=OUT_DIR, model_name=ASS_MODEL_NAME, final=True)
            await _stop_streaming_and_optionally_close(close_agent=True)

            total_sec = cl.user_session.get("session_total_sec") or 0.0
            sess_id   = cl.user_session.get("session_id") or ""
            user_obj  = cl.user_session.get("user") or get_current_user()
            if isinstance(user_obj, dict):
                username = user_obj.get("name") or user_obj.get("identifier") or "unknown"
            else:
                username = getattr(user_obj, "identifier", None) or getattr(user_obj, "name", None) or "unknown"
            n_cases_done = len(answers)

            await cl.Message(content="✅ All cases completed. Thank you!").send()
            await _force_client_redirect_to_thanks(username, sess_id, total_sec, n_cases_done)
        return  # important: don't fall through

    # Normal discussion with assistant
    if not agent:
        await cl.Message(content="Assistant not available at the moment.").send()
        return

    task = asyncio.create_task(agent.llm_response_async(message.content))
    cl.user_session.set("last_assistant_task", task)
    try:
        await task
    finally:
        cl.user_session.set("last_assistant_task", task)

@cl.on_stop
async def _on_stop():
    await _stop_streaming_and_optionally_close(close_agent=True)
    df: pd.DataFrame = cl.user_session.get("questions_df")
    answers_dict = cl.user_session.get("answers") or {}
    await _finalize_session(df, list(answers_dict.values()), partial=True, out_dir=OUT_DIR, model_name=ASS_MODEL_NAME)

@cl.on_chat_end
async def _on_chat_end():
    await _stop_streaming_and_optionally_close(close_agent=True)
    df: pd.DataFrame = cl.user_session.get("questions_df")
    answers_dict = cl.user_session.get("answers") or {}
    await _finalize_session(df, list(answers_dict.values()), partial=True, out_dir=OUT_DIR, model_name=ASS_MODEL_NAME)
