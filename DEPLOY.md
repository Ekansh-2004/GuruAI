# Deploying GuruAI to Hugging Face Spaces

This guide gets a public, HTTPS demo of GuruAI running on a **free** Hugging Face
Space with a pre-seeded demo account, so anyone with the link can try it without
registering or uploading anything.

Why HF Spaces: the app depends on PyTorch + a sentence-transformers embedding
model, whose memory footprint OOM-crashes 512 MB free tiers (Render/Railway
free). HF free CPU Spaces give **16 GB RAM**, which is comfortable.

---

## What's already wired up

The repo is deployment-ready:

| Concern | How it's handled |
|---|---|
| Port | `server.py` reads `$PORT` (Dockerfile sets `7860` for HF) |
| Prod server | `reload` is off unless `RELOAD=1` |
| Secret key | `JWT_SECRET_KEY` from env; warns loudly if unset |
| HTTPS cookies | `COOKIE_SECURE=1` makes the auth cookie HTTPS-only |
| Demo account | `SEED_DEMO=1` populates a demo user on startup |
| Container | `Dockerfile` runs as UID 1000 and pre-bakes the embedding model |
| Database | `DATABASE_URL` (Postgres + pgvector) holds all data, including embedded document chunks — see "Persistence" below |

---

## One prerequisite: Space metadata in README

HF Spaces reads its config from YAML frontmatter at the **top of the Space's
`README.md`**. Add this block as the very first lines (above the existing title):

```yaml
---
title: GuruAI
emoji: 🎓
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
```

`app_port: 7860` must match the `EXPOSE`/`PORT` in the Dockerfile. (This block
renders as a tidy info card on HF; on GitHub it shows as a small YAML header.)

---

## Steps

### 1. Create the Space
- huggingface.co → your profile → **New Space**.
- **SDK: Docker** → **Blank**. Free CPU basic hardware is enough.

### 2. Push the code
Add the Space as a git remote and push (the Space is just a git repo):

```bash
git remote add space https://huggingface.co/spaces/<your-username>/GuruAI
git push space main
```

The first build takes several minutes (installing PyTorch + baking the embedding
model into the image). Watch the **Logs** tab.

### 3. Set secrets and env vars
In the Space: **Settings → Variables and secrets**.

**Secrets** (encrypted — for keys):
| Name | Value |
|---|---|
| `GROQ_API_KEY` | your Groq key |
| `GOOGLE_API_KEY` | your Gemini key |
| `JWT_SECRET_KEY` | a long random string (e.g. `openssl rand -hex 32`) |
| `DATABASE_URL` | your Neon connection string — see "Persistence" below |

**Variables** (plain — for flags):
| Name | Value |
|---|---|
| `COOKIE_SECURE` | `1` |
| `SEED_DEMO` | `1` |

Saving secrets rebuilds/restarts the Space.

### 4. Try it
Open the Space's direct URL — `https://<your-username>-guruai.hf.space` — and log
in with:

```
username: demo
password: demo1234
```

The demo account comes pre-loaded with subjects, a knowledge profile with
weak/average/strong topics, a spaced-repetition schedule, and an Operating
Systems document you can chat with right away.

> Put the **direct `*.hf.space` URL** on your resume, not the embedded
> `huggingface.co/spaces/...` page — the direct URL is first-party, so login
> cookies behave normally.

---

## Persistence (Postgres + pgvector)

All data — users, sessions, messages, mastery scores, and embedded document
chunks — lives in a single Postgres database with the `pgvector` extension,
reached via `DATABASE_URL`. This replaced the earlier local-SQLite +
local-FAISS setup specifically because HF free Spaces have an **ephemeral
disk**: anything written to the container's filesystem is wiped on every
rebuild or sleep/wake cycle, which meant real visitors' accounts and uploads
used to vanish on restart. A hosted Postgres database survives that.

**Neon** (neon.tech) is recommended for the free tier: unlike some
alternatives, it auto-suspends when idle and auto-wakes on the next query,
with no manual "unpause" step.

1. Create a free Neon project and database.
2. In Neon's SQL editor (or via `psql`), run once: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Copy the connection string Neon gives you and set it as the `DATABASE_URL`
   secret on the Space (table above). The app's `init_db()` creates every
   table automatically on next startup — no manual schema step needed beyond
   the extension.
4. Local development against this same setup: see "Running the container
   locally" below, or run Postgres directly via
   `docker run -d --name guruai-postgres -e POSTGRES_USER=guruai -e POSTGRES_PASSWORD=guruai_dev_pw -e POSTGRES_DB=guruai_dev -p 5432:5432 pgvector/pgvector:pg16`
   and point `DATABASE_URL` at `postgresql://guruai:guruai_dev_pw@localhost:5432/guruai_dev`.

## Notes & gotchas

- **Free Spaces sleep after inactivity.** First hit after sleep cold-starts in
  ~30 s. Keep a short GIF in the README as a fallback for when it's asleep.
- **Cost/abuse:** every chat calls Groq + Gemini on your keys. The demo account
  keeps casual usage cheap, but the endpoint is public — watch your usage, and
  rotate `GROQ_API_KEY`/`GOOGLE_API_KEY` if you ever suspect abuse.
- **Changing the demo password:** set `DEMO_USERNAME` / `DEMO_PASSWORD` as Space
  variables; `scripts/seed_demo.py` reads them.

## Running the container locally (optional sanity check)

```bash
docker build -t guruai .
docker run -p 7860:7860 \
  -e GROQ_API_KEY=... -e GOOGLE_API_KEY=... -e JWT_SECRET_KEY=... \
  -e DATABASE_URL=postgresql://guruai:guruai_dev_pw@host.docker.internal:5432/guruai_dev \
  -e SEED_DEMO=1 \
  guruai
# → http://localhost:7860
# (assumes the local Postgres+pgvector container from "Persistence" above is
# already running; host.docker.internal lets this container reach it)
```
