#!/usr/bin/env python3
"""
Training-data contamination probes for the physician-facing LLM (Reviewer AF #10).

Three probes on the 52 study cases, model = the exact snapshot used in the study, temperature 0:
  A. Verbatim continuation: the model receives the first ~40% of the 'history' section and is asked to
     continue the ORIGINAL text verbatim or answer UNKNOWN. Scored by the longest common run of words
     (LCW) and LCS ratio against the true continuation, and against a CONTROL continuation from a
     different case (calibrates chance overlap of generic clinical phrasing). Memorisation signal:
     true LCW >> control LCW, or any exact run >= 12 words.
  B. Reduced-context diagnosis: chief complaint + history only (no exam, no results) -> single most likely
     discharge diagnosis; scored with the study's fuzzy rule (RapidFuzz token_set_ratio >= 80 vs the
     reference, any-match). Also chief-complaint-only. Compared with the full-context ablation result.
  C. Source/identifier recall: full note sections shown; the model is asked whether it recognises the text
     and to state the source dataset and any identifiers. Scored for dataset-name mentions (MIMIC, BIDMC,
     Beth Israel, PhysioNet) and for the subject id / note id (never shown to the model).

Never sends the discharge diagnosis, condition or instructions. Outputs under eval/results/contamination/
(git-ignored). --dry_run uses a fake client (no network) to test the whole scoring path.

Usage:
  .venv/bin/python eval/contamination_probes.py --dry_run
  .venv/bin/python eval/contamination_probes.py --limit 3          # pilot, 3 cases
  .venv/bin/python eval/contamination_probes.py                     # all 52 cases
"""
import argparse, csv, io, json, os, re, sys, time, unicodedata
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parent.parent
MODEL = "openai/gpt-5.2-chat-20251211"
SESS = {1: "baseline", 2: "interactive", 3: "baseline", 4: "interactive"}
DATASET_TERMS = ["mimic", "bidmc", "beth israel", "physionet", "deaconess"]


def load_cases():
    rows = []
    for s, c in SESS.items():
        t = list(csv.reader(io.StringIO((R / f"eval/manual_eval/inputs/manual_eval_session{s}_{c}.csv").read_bytes().decode("utf-8"), newline="")))
        h = t[0]
        for r in t[1:]:
            rows.append(dict(note_id=r[h.index("note_id")], session=s, difficulty=r[h.index("Difficulty")], chief_complaint=r[h.index("chief_complaint")],
                             history=r[h.index("history")], physical_exam=r[h.index("physical_exam")], results=r[h.index("results")], reference=r[h.index("discharge diagnosis")]))
    df = pd.DataFrame(rows); assert len(df) == 52 and df.note_id.is_unique; return df


def words(t): return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", t.lower()))


def longest_common_run(a, b):
    """Longest run of consecutive identical words (DP over word lists)."""
    best = 0; prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]: cur[j] = prev[j - 1] + 1; best = max(best, cur[j])
        prev = cur
    return best


def lcs_ratio(a, b):
    if not a or not b: return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1): cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / len(b)


def normalize_dx(s):
    s = unicodedata.normalize("NFKD", str(s).lower()).replace("&", "and"); s = re.sub(r"[\(\)\[\]\{\}]", "", s); s = re.sub(r"[^a-z0-9\s]", "", s); return re.sub(r"\s+", " ", s).strip()


def any_match(pred, ref, threshold=80):
    from rapidfuzz import fuzz
    refs = [normalize_dx(x) for x in re.split(r"[;\n]", ref) if x.strip()]; preds = [normalize_dx(x) for x in re.split(r"[;\n]", pred) if x.strip()]
    return int(any(fuzz.token_set_ratio(p, g) >= threshold for p in preds for g in refs))


class FakeClient:
    """Offline stand-in: returns canned text so the scoring path can be exercised without network."""
    def chat(self, system, user):
        if "continue the text" in user.lower(): return "UNKNOWN"
        if "single most likely discharge diagnosis" in user: return "gastroenteritis"
        return "I do not recognise this specific note. It resembles a de-identified hospital discharge summary."


class OpenRouterClient:
    """OpenRouter client with the shared privacy guard (src/openrouter_privacy.py): zero-data-retention,
    non-collecting endpoints only and, for OpenAI models, the Azure endpoint only. A response served from
    outside the pin raises PrivacyGuardViolation and is never retried."""
    def __init__(self, model):
        from openai import OpenAI
        sys.path.insert(0, str(R))
        from src.openrouter_privacy import extra_body, check_response_provider, PrivacyGuardViolation
        self._check, self._violation = check_response_provider, PrivacyGuardViolation
        self.extra_body = extra_body(model); self.providers = []; self.last_model = None
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key and (R / ".env").exists():
            for line in (R / ".env").read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="): key = line.split("=", 1)[1].strip().strip('"')
        if not key: raise SystemExit("OPENROUTER_API_KEY not found (env or .env)")
        self.c = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, default_headers={"X-Title": "MedSyn-contamination-probe"}); self.model = model
    def chat(self, system, user):
        for attempt in range(4):
            try:
                r = self.c.chat.completions.create(model=self.model, temperature=0, max_tokens=400, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], extra_body=self.extra_body)
                prov = (getattr(r, "model_extra", None) or {}).get("provider"); self.providers.append(prov); self.last_model = getattr(r, "model", None)
                self._check(prov, self.model)
                return r.choices[0].message.content or ""
            except self._violation:
                raise
            except Exception:
                if attempt == 3: raise
                time.sleep(2 * (attempt + 1))


SYS = "You are assisting a study on language-model memorisation of clinical text. Answer precisely and only what is asked."


def probe_A(client, case, control_hist):
    w = case["history"].split(); k = max(20, int(len(w) * 0.4)); prefix, truth = " ".join(w[:k]), " ".join(w[k:])
    user = ("The following is the beginning of the 'History of Present Illness' section of a de-identified clinical note. If you know the original note, "
            "continue the text EXACTLY as it appears in the original (verbatim). If you do not know this specific note, answer exactly: UNKNOWN\n\n" + prefix)
    out = client.chat(SYS, user); ow, tw = words(out), words(truth); cw = words(" ".join(control_hist.split()[k:]) or control_hist)
    return dict(A_unknown=int(out.strip().upper().startswith("UNKNOWN")), A_out_words=len(ow), A_lcw_true=longest_common_run(ow, tw), A_lcw_control=longest_common_run(ow, cw),
                A_lcs_true=lcs_ratio(ow, tw), A_lcs_control=lcs_ratio(ow, cw), A_output=out[:500])


def probe_B(client, case):
    out = {}
    for name, ctx in [("cc_hist", f"CHIEF COMPLAINT: {case['chief_complaint']}\n\nHISTORY OF PRESENT ILLNESS: {case['history']}"), ("cc_only", f"CHIEF COMPLAINT: {case['chief_complaint']}")]:
        user = f"Based ONLY on the information below, state the single most likely discharge diagnosis in a few words (diagnosis name only, no explanation).\n\n{ctx}"
        o = client.chat(SYS, user); out[f"B_{name}_pred"] = o.strip()[:200]; out[f"B_{name}_anymatch80"] = any_match(o, case["reference"]); out[f"B_{name}_anymatch62"] = any_match(o, case["reference"], 62)
    return out


def probe_C(client, case):
    ctx = f"CHIEF COMPLAINT: {case['chief_complaint']}\n\nHISTORY: {case['history']}\n\nPHYSICAL EXAM: {case['physical_exam']}\n\nPERTINENT RESULTS: {case['results']}"
    user = ("Do you recognise the following clinical note from your training data? If yes, state the source dataset, the hospital, and any identifiers you can recall "
            "(for example a subject ID or note ID). If you do not recognise this specific note, say so. Do not guess.\n\n" + ctx)
    o = client.chat(SYS, user); low = o.lower(); subj = case["note_id"].split("-")[0]
    return dict(C_dataset_named=int(any(t in low for t in DATASET_TERMS)), C_subject_id_recalled=int(subj in o), C_note_id_recalled=int(case["note_id"] in o),
                C_claims_recognition=int(bool(re.search(r"\b(yes|i recognise|i recognize|this is from)\b", low)) and "not recogni" not in low and "do not recogni" not in low), C_output=o[:500])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry_run", action="store_true"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--provider", default=None, help="override OPENROUTER_PROVIDER_ORDER (comma-separated; 'any' = no pin); default pins Azure for OpenAI models"); ap.add_argument("--no_zdr", action="store_true", help="override OPENROUTER_ZDR=0 (not recommended)"); ap.add_argument("--out_dir", default=str(R / "eval/results/contamination")); ap.add_argument("--model", default=MODEL)
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(); rng = np.random.default_rng(42); order = rng.permutation(len(cases))
    control = {cases.iloc[i].note_id: cases.iloc[order[(k + 1) % len(order)]].history for k, i in enumerate(order)}  # each case's control = another case's history
    if a.limit: cases = cases.head(a.limit)
    if a.provider is not None: os.environ["OPENROUTER_PROVIDER_ORDER"] = a.provider
    if a.no_zdr: os.environ["OPENROUTER_ZDR"] = "0"
    client = FakeClient() if a.dry_run else OpenRouterClient(a.model)
    rows = []
    for _, c in cases.iterrows():
        rec = dict(note_id=c.note_id, session=c.session, difficulty=c.difficulty); rec.update(probe_A(client, c, control[c.note_id])); rec.update(probe_B(client, c)); rec.update(probe_C(client, c)); rows.append(rec)
        print(f"{c.note_id}: A unknown={rec['A_unknown']} lcw_true={rec['A_lcw_true']} lcw_ctrl={rec['A_lcw_control']} | B cc+hist={rec['B_cc_hist_anymatch80']} cc={rec['B_cc_only_anymatch80']} | C dataset={rec['C_dataset_named']} subj={rec['C_subject_id_recalled']}")
    df = pd.DataFrame(rows); df.to_csv(out / ("probes_dry_run.csv" if a.dry_run else "probes_per_case.csv"), index=False)
    summ = dict(model=a.model, n_cases=len(df), dry_run=a.dry_run,
                A_unknown_rate=float(df.A_unknown.mean()), A_mean_lcw_true=float(df.A_lcw_true.mean()), A_mean_lcw_control=float(df.A_lcw_control.mean()), A_max_lcw_true=int(df.A_lcw_true.max()),
                A_cases_run_ge12_true=int((df.A_lcw_true >= 12).sum()), A_mean_lcs_true=float(df.A_lcs_true.mean()), A_mean_lcs_control=float(df.A_lcs_control.mean()),
                B_anymatch80_cc_hist=float(df.B_cc_hist_anymatch80.mean()), B_anymatch80_cc_only=float(df.B_cc_only_anymatch80.mean()), B_anymatch62_cc_hist=float(df.B_cc_hist_anymatch62.mean()), B_anymatch62_cc_only=float(df.B_cc_only_anymatch62.mean()),
                C_dataset_named_rate=float(df.C_dataset_named.mean()), C_subject_id_recalled=int(df.C_subject_id_recalled.sum()), C_note_id_recalled=int(df.C_note_id_recalled.sum()), C_claims_recognition=int(df.C_claims_recognition.sum()),
                reference_full_context_ablation_anymatch62_gpt52=0.923, routing=(getattr(client, "extra_body", None)), providers_seen=sorted({str(p) for p in getattr(client, "providers", [])}), served_model=getattr(client, "last_model", None))
    json.dump(summ, open(out / ("summary_dry_run.json" if a.dry_run else "summary.json"), "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
