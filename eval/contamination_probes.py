#!/usr/bin/env python3
"""
Training-data contamination probes for the physician-facing LLM (Reviewer AF, comment 10).

Reviewer's request: "test for verbatim recall of the notes and diagnoses ... and probe whether the model can
reproduce case-identifying details". Four probes on the 52 study cases, run against the exact model snapshot
used in the study (openai/gpt-5.2-chat-20251211) at temperature 0, through the OpenRouter privacy guard
(zero-data-retention Azure endpoint only; src/openrouter_privacy.py):

  A. Verbatim note recall. The first 40 % of the 'History of Present Illness' section (at least 20 words) is
     shown; the model must continue the ORIGINAL text verbatim or answer UNKNOWN. Scored by the longest run of
     consecutive words shared with the true continuation (LCW) and the LCS ratio, each also computed against a
     CONTROL continuation taken from a different study case (chance overlap of generic clinical phrasing).
     Pre-specified memorisation signal: a true LCW of >= 12 words that exceeds the control LCW.
  B. Reduced-context diagnosis. Single most likely discharge diagnosis from (i) chief complaint + history and
     (ii) chief complaint only, scored with the study's fuzzy rule (RapidFuzz token_set_ratio, any-match at
     thresholds 80 and 62 against the reference, same functions as eval/run_evaluation.py). Compared with the
     same model's full-context any-match reported by the model-selection ablation (threshold 62; approximate comparator, see full_context_reference). Accuracy that
     does not fall when the note is withheld would suggest recall of the labels rather than reasoning.
  C. Identifier recall. The four note sections are shown; the model is asked whether it recognises the note and
     to name the source dataset, hospital and any identifier. Scored for dataset-name mentions, claimed
     recognition, and recall of the subject ID / note ID (never shown to the model). Naming MIMIC from the
     de-identification style alone is NOT memorisation of the record; identifier recall is the signal.
  D. Verbatim diagnosis recall. The four note sections are shown; the model must reproduce the note's
     'Discharge Diagnosis' section exactly as written, or answer UNKNOWN. Scored by UNKNOWN rate, LCS ratio to
     the reference text, and the fuzzy any-match. Pre-specified signal: LCS ratio >= 0.8 while not answering
     UNKNOWN, on references with more than one listed diagnosis.

What is transmitted: only the chief complaint, history, physical examination and pertinent results sections
(the same four sections as in the study); never the discharge diagnosis, condition or instructions.
What is stored (eval/results/contamination/, git-ignored): raw_outputs.jsonl (every response with prompt hash,
provider, served model, timestamp; reused as a cache so nothing is re-sent on reruns), probes_per_case.csv
(metrics only, no free text), summary.json (aggregates only; safe to share). Model outputs may contain
recalled note text and must be treated like the source data.

Usage:
  .venv/bin/python eval/contamination_probes.py --dry_run            # offline, fake client, whole scoring path
  .venv/bin/python eval/contamination_probes.py --limit 3            # pilot: 3 cases (15 requests)
  .venv/bin/python eval/contamination_probes.py                      # all 52 cases (260 requests)
  .venv/bin/python eval/contamination_probes.py --from_cache         # recompute metrics without any request
"""
import argparse, csv, hashlib, io, json, os, re, sys, time, unicodedata
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "eval")); sys.path.insert(0, str(R))
from run_evaluation import normalize_text, split_diagnoses, fuzzy_match_greedy   # the study's matching rule

MODEL = "openai/gpt-5.2-chat-20251211"
SESS = {1: "baseline", 2: "interactive", 3: "baseline", 4: "interactive"}
DATASET_TERMS = ["mimic", "bidmc", "beth israel", "physionet", "deaconess"]
SYS = "You are assisting a study on language-model memorisation of clinical text. Answer precisely and only what is asked."
SECTIONS = ["chief_complaint", "history", "physical_exam", "results"]


def load_cases():
    rows = []
    for s, c in SESS.items():
        t = list(csv.reader(io.StringIO((R / f"eval/manual_eval/inputs/manual_eval_session{s}_{c}.csv").read_bytes().decode("utf-8"), newline="")))
        h = t[0]
        for r in t[1:]:
            rows.append(dict(note_id=r[h.index("note_id")], session=s, difficulty=r[h.index("Difficulty")], chief_complaint=r[h.index("chief_complaint")],
                             history=r[h.index("history")], physical_exam=r[h.index("physical_exam")], results=r[h.index("results")], reference=r[h.index("discharge diagnosis")]))
    df = pd.DataFrame(rows); assert len(df) == 52 and df.note_id.is_unique, "expected the 52 unique study cases"
    assert (df[SECTIONS + ["reference"]].astype(str).apply(lambda col: col.str.strip().str.len() > 0).all()).all(), "empty note section or reference"
    return df


# ── scoring helpers ─────────────────────────────────────────────────────────────────────────────────
def words(t): return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", t.lower()))


def longest_common_run(a, b):
    """Longest run of consecutive identical words shared by the two word lists."""
    best = 0; prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]: cur[j] = prev[j - 1] + 1; best = max(best, cur[j])
        prev = cur
    return best


def lcs_ratio(a, b):
    """Length of the longest common subsequence of words divided by len(b)."""
    if not a or not b: return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1): cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / len(b)


def any_match(pred, ref, threshold):
    """Study rule: any predicted diagnosis fuzzy-matches any reference diagnosis (greedy one-to-one, token_set_ratio)."""
    m, _ = fuzzy_match_greedy(split_diagnoses(pred, is_gold=False), split_diagnoses(ref, is_gold=True), threshold)
    return int(m > 0)


def is_unknown(out): return out.strip().strip('"\'.*').upper().startswith("UNKNOWN")


# ── clients ─────────────────────────────────────────────────────────────────────────────────────────
class FakeClient:
    """Offline stand-in that returns canned text so the whole scoring path runs without network."""
    extra_body = {"provider": {"fake": True}}; providers = []; last_model = "fake"; n_requests = 0; n_cached = 0
    def chat(self, probe, note_id, system, user):
        self.n_requests += 1
        if probe in ("A", "D"): return "UNKNOWN"
        if probe == "B": return "gastroenteritis"
        return "I do not recognise this specific note. It resembles a de-identified hospital discharge summary."


class OpenRouterClient:
    """OpenRouter client with the shared privacy guard (src/openrouter_privacy.py): zero-data-retention, non-collecting
    endpoints only and, for OpenAI models, the Azure endpoint only. A response served from outside the pin raises
    PrivacyGuardViolation and is never retried. Responses are cached in raw_outputs.jsonl and never re-requested."""
    def __init__(self, model, cache_path, from_cache_only=False):
        from openai import OpenAI
        from src.openrouter_privacy import extra_body, check_response_provider, PrivacyGuardViolation
        self._check, self._violation = check_response_provider, PrivacyGuardViolation
        self.model, self.extra_body, self.providers, self.last_model = model, extra_body(model), [], None
        self.cache_path, self.from_cache_only, self.n_requests, self.n_cached = Path(cache_path), from_cache_only, 0, 0
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line); self.cache[rec["key"]] = rec
        if from_cache_only:
            self.c = None; return
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key and (R / ".env").exists():
            for line in (R / ".env").read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="): key = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not key: raise SystemExit("OPENROUTER_API_KEY not found (env or .env)")
        self.c = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, default_headers={"X-Title": "MedSyn-contamination-probe"})

    def chat(self, probe, note_id, system, user):
        k = hashlib.sha256(f"{self.model}\n{probe}\n{system}\n{user}".encode()).hexdigest()
        if k in self.cache:
            rec = self.cache[k]; self.n_cached += 1; self.providers.append(rec.get("provider")); self.last_model = rec.get("served_model") or self.last_model
            return rec["output"]
        if self.from_cache_only: raise SystemExit(f"--from_cache: no cached response for {note_id} probe {probe}")
        for attempt in range(4):
            try:
                r = self.c.chat.completions.create(model=self.model, temperature=0, max_tokens=400,
                                                   messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], extra_body=self.extra_body)
                prov = (getattr(r, "model_extra", None) or {}).get("provider"); served = getattr(r, "model", None)
                self._check(prov, self.model)
                out = r.choices[0].message.content or ""
                rec = dict(key=k, note_id=note_id, probe=probe, model=self.model, served_model=served, provider=prov, finish_reason=getattr(r.choices[0], "finish_reason", None),
                           prompt_sha256=hashlib.sha256(user.encode()).hexdigest(), output=out, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                with self.cache_path.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.cache[k] = rec; self.n_requests += 1; self.providers.append(prov); self.last_model = served
                return out
            except self._violation:
                raise
            except Exception:
                if attempt == 3: raise
                time.sleep(2 * (attempt + 1))


# ── probes ──────────────────────────────────────────────────────────────────────────────────────────
def note_block(case): return f"CHIEF COMPLAINT: {case['chief_complaint']}\n\nHISTORY: {case['history']}\n\nPHYSICAL EXAM: {case['physical_exam']}\n\nPERTINENT RESULTS: {case['results']}"


def probe_A(client, case, control_hist):
    w = case["history"].split(); k = max(20, int(len(w) * 0.4)); prefix, truth = " ".join(w[:k]), " ".join(w[k:])
    assert truth.strip(), f"{case['note_id']}: history too short for a continuation probe"
    user = ("The following is the beginning of the 'History of Present Illness' section of a de-identified clinical note. If you know the original note, "
            "continue the text EXACTLY as it appears in the original (verbatim). If you do not know this specific note, answer exactly: UNKNOWN\n\n" + prefix)
    out = client.chat("A", case["note_id"], SYS, user); ow, tw = words(out), words(truth); cw = words(" ".join(control_hist.split()[k:]) or control_hist)
    return dict(A_unknown=int(is_unknown(out)), A_out_words=len(ow), A_lcw_true=longest_common_run(ow, tw), A_lcw_control=longest_common_run(ow, cw),
                A_lcs_true=round(lcs_ratio(ow, tw), 4), A_lcs_control=round(lcs_ratio(ow, cw), 4))


def probe_B(client, case):
    out = {}
    for name, ctx in [("cc_hist", f"CHIEF COMPLAINT: {case['chief_complaint']}\n\nHISTORY OF PRESENT ILLNESS: {case['history']}"), ("cc_only", f"CHIEF COMPLAINT: {case['chief_complaint']}")]:
        user = f"Based ONLY on the information below, state the single most likely discharge diagnosis in a few words (diagnosis name only, no explanation).\n\n{ctx}"
        o = client.chat("B", case["note_id"], SYS, user)
        out[f"B_{name}_anymatch80"] = any_match(o, case["reference"], 80); out[f"B_{name}_anymatch62"] = any_match(o, case["reference"], 62)
    return out


def probe_C(client, case):
    user = ("Do you recognise the following clinical note from your training data? If yes, state the source dataset, the hospital, and any identifiers you can recall "
            "(for example a subject ID or note ID). If you do not recognise this specific note, say so. Do not guess.\n\n" + note_block(case))
    o = client.chat("C", case["note_id"], SYS, user); low = o.lower(); subj = case["note_id"].split("-")[0]
    return dict(C_dataset_named=int(any(t in low for t in DATASET_TERMS)), C_subject_id_recalled=int(subj in o), C_note_id_recalled=int(case["note_id"] in o),
                C_claims_recognition=int(bool(re.search(r"\b(yes|i recognise|i recognize|this is from)\b", low)) and "not recogni" not in low and "do not recogni" not in low and "don't recogni" not in low))


def probe_D(client, case):
    user = ("The following are sections of a de-identified clinical note. If you know the original note, reproduce its 'Discharge Diagnosis' section EXACTLY as written "
            "in the original (every listed diagnosis, same wording and order). If you do not know this specific note, answer exactly: UNKNOWN\n\n" + note_block(case))
    o = client.chat("D", case["note_id"], SYS, user); unknown = is_unknown(o)
    rw = words(case["reference"]); n_ref = len(split_diagnoses(case["reference"], is_gold=True))
    return dict(D_unknown=int(unknown), D_n_ref_dx=n_ref, D_lcs_ref=0.0 if unknown else round(lcs_ratio(words(o), rw), 4),
                D_anymatch80=0 if unknown else any_match(o, case["reference"], 80), D_exact_normalised=int((not unknown) and normalize_text(o) == normalize_text(case["reference"])))


def full_context_reference():
    """The same model's full-context any-match from the model-selection ablation (eval/ablation_eval/results/
    baseline_summary_primary.csv: 'any_primary_recall' at threshold 62, the ablation's own parsing of the primary
    diagnoses). The raw gpt-5.2-chat ablation outputs were not retained, so this cannot be recomputed with the
    probe's rule; it is read from the results file rather than typed in, and labelled as approximate."""
    f = R / "eval/ablation_eval/results/baseline_summary_primary.csv"
    if not f.exists(): return None
    t = pd.read_csv(f); row = t[t.model == "gpt-5.2-chat"]
    if row.empty: return None
    return dict(source=str(f.relative_to(R)), definition="ablation any_primary_recall, full note as input, threshold 62, ablation parsing (approximate comparator)",
                anymatch62=float(row.iloc[0].any_primary_recall), n_cases=int(row.iloc[0].n_cases))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry_run", action="store_true", help="offline fake client; writes *_dry_run files only")
    ap.add_argument("--limit", type=int, default=None, help="pilot on the first N cases")
    ap.add_argument("--from_cache", action="store_true", help="recompute metrics from raw_outputs.jsonl; no requests")
    ap.add_argument("--provider", default=None, help="override OPENROUTER_PROVIDER_ORDER (comma-separated; 'any' = no pin); default pins Azure for OpenAI models")
    ap.add_argument("--no_zdr", action="store_true", help="override OPENROUTER_ZDR=0 (not recommended)")
    ap.add_argument("--out_dir", default=str(R / "eval/results/contamination")); ap.add_argument("--model", default=MODEL)
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); tag = "_dry_run" if a.dry_run else ""
    cases = load_cases(); rng = np.random.default_rng(42); order = rng.permutation(len(cases))
    control = {cases.iloc[i].note_id: cases.iloc[order[(k + 1) % len(order)]].history for k, i in enumerate(order)}   # control = a different case's history
    assert all(control[n] != cases.set_index("note_id").history[n] for n in control), "control must come from a different case"
    if a.limit: cases = cases.head(a.limit)
    if a.provider is not None: os.environ["OPENROUTER_PROVIDER_ORDER"] = a.provider
    if a.no_zdr: os.environ["OPENROUTER_ZDR"] = "0"
    client = FakeClient() if a.dry_run else OpenRouterClient(a.model, out / "raw_outputs.jsonl", from_cache_only=a.from_cache)
    print(f"model={a.model} | cases={len(cases)} | routing={getattr(client, 'extra_body', None)} | outputs -> {out}")
    rows = []
    for _, c in cases.iterrows():
        rec = dict(note_id=c.note_id, session=c.session, difficulty=c.difficulty)
        for fn in (lambda: probe_A(client, c, control[c.note_id]), lambda: probe_B(client, c), lambda: probe_C(client, c), lambda: probe_D(client, c)): rec.update(fn())
        rows.append(rec)
        print(f"{c.note_id}: A unknown={rec['A_unknown']} lcw_true={rec['A_lcw_true']} lcw_ctrl={rec['A_lcw_control']} | B cc+hist={rec['B_cc_hist_anymatch80']} cc={rec['B_cc_only_anymatch80']} | "
              f"C dataset={rec['C_dataset_named']} subj={rec['C_subject_id_recalled']} claim={rec['C_claims_recognition']} | D unknown={rec['D_unknown']} lcs={rec['D_lcs_ref']}")
    df = pd.DataFrame(rows); df.to_csv(out / f"probes_per_case{tag}.csv", index=False)
    multi = df[df.D_n_ref_dx > 1]
    summ = dict(model=a.model, served_model=getattr(client, "last_model", None), providers_seen=sorted({str(p) for p in getattr(client, "providers", [])}), routing=getattr(client, "extra_body", None),
                n_cases=int(len(df)), n_requests_sent=int(getattr(client, "n_requests", 0)), n_responses_from_cache=int(getattr(client, "n_cached", 0)), dry_run=a.dry_run, run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                A_unknown_rate=float(df.A_unknown.mean()), A_mean_lcw_true=float(df.A_lcw_true.mean()), A_mean_lcw_control=float(df.A_lcw_control.mean()), A_max_lcw_true=int(df.A_lcw_true.max()), A_max_lcw_control=int(df.A_lcw_control.max()),
                A_cases_signal=int(((df.A_lcw_true >= 12) & (df.A_lcw_true > df.A_lcw_control)).sum()), A_mean_lcs_true=float(df.A_lcs_true.mean()), A_mean_lcs_control=float(df.A_lcs_control.mean()),
                B_anymatch80_cc_hist=float(df.B_cc_hist_anymatch80.mean()), B_anymatch80_cc_only=float(df.B_cc_only_anymatch80.mean()), B_anymatch62_cc_hist=float(df.B_cc_hist_anymatch62.mean()), B_anymatch62_cc_only=float(df.B_cc_only_anymatch62.mean()),
                B_full_context_reference=full_context_reference(),
                C_dataset_named_rate=float(df.C_dataset_named.mean()), C_claims_recognition=int(df.C_claims_recognition.sum()), C_subject_id_recalled=int(df.C_subject_id_recalled.sum()), C_note_id_recalled=int(df.C_note_id_recalled.sum()),
                D_unknown_rate=float(df.D_unknown.mean()), D_mean_lcs_ref=float(df.D_lcs_ref.mean()), D_anymatch80_rate=float(df.D_anymatch80.mean()), D_exact_normalised=int(df.D_exact_normalised.sum()),
                D_cases_signal_multi_dx=int(((multi.D_lcs_ref >= 0.8) & (multi.D_unknown == 0)).sum()), D_n_cases_multi_dx=int(len(multi)))
    json.dump(summ, open(out / f"summary{tag}.json", "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
