try:
    # Chainlit >= 1.0
    from chainlit.server import app
except Exception:
    # fallback for older chainlit builds
    from chainlit.server import app  # keep same import; most versions expose FastAPI app here

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
