# Deploying the HSN Checker

Vercel is not supported: it only runs short serverless functions, while Streamlit needs a
long-running server (WebSocket) and the Tesseract OCR system package.

## Option A — Streamlit Community Cloud (free, easiest)
1. Create a GitHub repository and upload: `app.py`, `hsn_master.csv`, `requirements.txt`,
   `packages.txt`, `.streamlit/config.toml`, `.gitignore`.
   (Do NOT upload customer invoices — `.gitignore` already excludes PDFs/images/Excel.)
2. Go to https://share.streamlit.io → sign in with GitHub → **Create app**.
3. Pick the repository, branch `main`, main file `app.py` → **Deploy**.
   `packages.txt` installs Tesseract automatically. You get a URL like
   `https://<name>.streamlit.app`.
4. To keep it private: App settings → Sharing → "Only specific people can view".

Limits: ~1 GB RAM, app sleeps after inactivity (wakes in ~30 s), scanned PDFs are slower
than on your PC.

## Option B — Render (Docker, always-on paid plan or free with sleep)
1. Push the same files plus `Dockerfile` and `.dockerignore` to GitHub.
2. https://render.com → New → **Web Service** → connect the repo.
3. Runtime: **Docker**. Instance: at least 1 GB RAM (Starter) for OCR.
4. Deploy. Render sets `$PORT`; the Dockerfile already uses it.

Railway works the same way (New Project → Deploy from GitHub repo; it detects the Dockerfile).

## Updating the HSN database
Replace `hsn_master.csv` in the repository and push — the app redeploys automatically.
