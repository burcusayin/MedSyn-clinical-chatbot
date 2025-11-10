#!/usr/bin/env bash
# Launch Chainlit for clinical_chatbot

set -Eeuo pipefail

# ---------- Config (override via env vars when calling the script) ----------
MAIN_DIR="${MAIN_DIR:-/home/gunel/medSyn/}"
APP_REL="${APP_REL:-src/clinical_chatbot/app.py}"      # path to your Chainlit entry file
ENV_FILE="${ENV_FILE:-src/clinical_chatbot/.env}"      # .env with app settings
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WATCH="${WATCH:-true}"                                  # empty to disable --watch
CHAINLIT_ARGS="${CHAINLIT_ARGS:-}"                      # extra args passed to chainlit
# ----------------------------------------------------------------------------

log()  { printf "\033[1;34m[INFO]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*"; exit 1; }

# Go to project root
[[ -d "$MAIN_DIR" ]] || die "MAIN_DIR not found: $MAIN_DIR"
cd "$MAIN_DIR"
log "PWD=$(pwd)"

# Ensure packages exist for absolute imports like `from src....`
[[ -f src/__init__.py ]] || { touch src/__init__.py; log "Created src/__init__.py"; }
[[ -d src/clinical_chatbot ]] || die "Missing folder: src/clinical_chatbot"
[[ -f src/clinical_chatbot/__init__.py ]] || { touch src/clinical_chatbot/__init__.py; log "Created src/clinical_chatbot/__init__.py"; }

# Export project root to PYTHONPATH so `src.*` imports work
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
log "PYTHONPATH includes project root."

# 5) Load .env if present
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  log "Loaded env vars from $ENV_FILE"
else
  warn "Env file not found at $ENV_FILE (continuing without it)"
fi

# Ensure essential env defaults (keep minimal & safe)
#    Add more defaults here only if your app expects them.
if [[ -z "${LOG_DIR:-}" ]]; then
  export LOG_DIR="$MAIN_DIR/logs"
  log "LOG_DIR not set; defaulting to $LOG_DIR"
fi
mkdir -p "$LOG_DIR"

# Pre-flight import check to catch import errors early
python - <<'PY' || die "Python import of src.clinical_chatbot.app failed. Check errors above."
import sys, importlib
print("Python:", sys.version.split()[0])
m = importlib.import_module("src.clinical_chatbot.app")
print("Imported:", m.__file__)
PY

# Ensure Chainlit is installed
command -v chainlit >/dev/null 2>&1 || die "chainlit not found. Install with: python -m pip install chainlit"

# Kill any existing Chainlit running this app (best-effort)
pkill -f "chainlit run .*$(printf "%s" "$APP_REL" | sed 's/[.[\*^$()+?{}|]/\\&/g')" >/dev/null 2>&1 || true

# Start Chainlit
log "Starting Chainlit → $APP_REL (host=$HOST port=$PORT watch=${WATCH:-false})"
if [[ -n "$WATCH" ]]; then
  exec chainlit run "$APP_REL" 
else
  exec chainlit run "$APP_REL" 
fi
