# Web UI — the desk in a browser (cloud / anywhere access)

The web UI is the same desk as the desktop app, served from a browser. It reuses
the entire engine: every action is the same `app.runner` flow, run on the server
and **streamed live to the browser** over Server-Sent Events. Run it on an
always-on host and you get one cloud brain, accessible from any device — no
desktop, no local/cloud divergence.

> Status: **Phase 1.** The Desk (screens, deep-dives, the trade/monitor actions,
> live output, order tickets, watchlist), the Performance and Learning views, and
> a Settings form are wired. Scheduling stays as documented in
> [`AUTOMATION.md`](AUTOMATION.md) (the host's cron / the GitHub Actions workflow).

---

## Run it locally

```bash
pip install -e ".[web,live-data,llm]"
uvicorn web.server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

State (config, ledger, learning, watchlist) lives under `~/.swing_system/` exactly
as the desktop app, so the two share data on the same machine. Set `SWING_HOME`
to relocate it. Credentials can come from the Settings form **or** the environment
(`ANTHROPIC_API_KEY`, `ALPACA_KEY_ID`, `ALPACA_SECRET`, `EDGAR_USER_AGENT`, …).

---

## Deploy free + always-on (Oracle Cloud Always-Free)

Oracle Cloud's **Always-Free** tier gives a small VM that never expires ($0) — the
one mainstream truly-free *always-on* option (Render/Railway free tiers sleep,
which would stop the scheduler). Outline:

1. Create an Always-Free **Ampere/VM.Standard.E2.1.Micro** instance (Ubuntu).
   Open port 8000 (or put it behind a free reverse proxy / Cloudflare Tunnel for
   HTTPS).
2. On the VM:
   ```bash
   git clone <your private repo> && cd swing-system
   python3.11 -m venv .venv && . .venv/bin/activate
   pip install -e ".[web,live-data,llm]"
   export ANTHROPIC_API_KEY=... ALPACA_KEY_ID=... ALPACA_SECRET=... EDGAR_USER_AGENT=...
   uvicorn web.server:app --host 0.0.0.0 --port 8000
   ```
3. Keep it running with **systemd** (survives reboots):
   ```ini
   # /etc/systemd/system/swing-web.service
   [Unit]
   Description=Swing System web
   After=network.target
   [Service]
   WorkingDirectory=/home/ubuntu/swing-system
   EnvironmentFile=/home/ubuntu/swing.env
   ExecStart=/home/ubuntu/swing-system/.venv/bin/uvicorn web.server:app --host 0.0.0.0 --port 8000
   Restart=always
   [Install]
   WantedBy=multi-user.target
   ```
   `sudo systemctl enable --now swing-web`
4. Schedule the desk on the same VM with cron (using the [headless
   commands](AUTOMATION.md#headless-commands)) — e.g. `--review`, `--watch`,
   `--screen`, so the scheduler and the UI share one state store.

Because the VM is always on, this is the single coherent brain: the schedule and
your interactive sessions read/write the same `~/.swing_system/`, and you reach
the UI from any browser or phone.

**Lock it down.** Set `SWING_WEB_PASSWORD=<your password>` in the environment and
the app requires a login (a sign-in page, an HMAC session cookie, all `/api/`
routes gated; restarting the server logs everyone out). With no password set the
app is open — fine for `localhost`, not for a public IP. For real exposure also
put it behind HTTPS (a Cloudflare Tunnel is free) and/or a firewall rule limiting
the port to your IP.

The **Settings → Scheduled automation** section lets you choose the preset and
universes, edit custom ET times, toggle unattended exit management, and copy a
ready-made **crontab** (already converted to UTC) to paste on the host.

---

## Architecture

- `web/server.py` — FastAPI. `POST /api/run` starts a flow on a worker thread;
  `GET /api/stream/{id}` streams its `emit()` output as SSE; `POST /api/stop`
  cancels cooperatively. `GET/POST /api/config` (secrets masked), `/api/orders` +
  `POST /api/order`, `/api/performance`, `/api/learning`, `/api/watchlist`. One run
  at a time, like the desktop.
- `web/static/` — a dependency-free SPA (`index.html` / `style.css` / `app.js`)
  matching the desktop's dark theme: four tabs, a live colour-coded console with a
  Stop button, the order-ticket table, the equity curve, and the Settings form.

No new engine code — the web layer is pure plumbing over `app.runner`, so the
desktop and the browser always behave identically.
