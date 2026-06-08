"""Reddit fetch via the official OAuth API (anonymous .json is now 403-blocked).

Requires free Reddit app credentials (https://www.reddit.com/prefs/apps -> create
app). A "script" app uses the password grant (client id+secret + your reddit
username+password); a "web app" can use client-credentials (id+secret only). We
try password grant when a username is supplied, else client-credentials.

Read-only: we only GET hot/new listings from finance subreddits.
"""

from __future__ import annotations

import re

import requests

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API_BASE = "https://oauth.reddit.com"
DEFAULT_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket"]


def get_token(client_id: str, client_secret: str, user_agent: str,
              username: str | None = None, password: str | None = None) -> str:
    if username and password:
        data = {"grant_type": "password", "username": username, "password": password}
    else:
        data = {"grant_type": "client_credentials"}
    r = requests.post(TOKEN_URL, auth=(client_id, client_secret), data=data,
                      headers={"User-Agent": user_agent}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Reddit auth failed ({r.status_code}): {r.text[:160]}")
    tok = r.json().get("access_token")
    if not tok:
        raise RuntimeError(f"Reddit auth returned no token: {r.json()}")
    return tok


def fetch_posts(token: str, user_agent: str, subreddits: list[str] | None = None,
                limit: int = 100, sort: str = "hot") -> list[dict]:
    """Recent posts from the given subreddits: title, selftext, score, comments."""
    subs = subreddits or DEFAULT_SUBREDDITS
    headers = {"Authorization": f"bearer {token}", "User-Agent": user_agent}
    out = []
    for sub in subs:
        url = f"{API_BASE}/r/{sub}/{sort}?limit={min(limit, 100)}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            continue
        for child in r.json().get("data", {}).get("children", []):
            d = child.get("data", {})
            out.append({
                "subreddit": sub,
                "title": d.get("title", ""),
                "text": d.get("selftext", "") or "",
                "score": int(d.get("score", 0)),
                "comments": int(d.get("num_comments", 0)),
                "created": d.get("created_utc"),
            })
    return out


# Tickers that are also common English words -> only count when written as $TICKER
_AMBIGUOUS = {"A", "ALL", "ON", "IT", "GE", "MO", "KO", "BA", "MS", "SO", "GO",
              "PG", "C", "T", "F", "D", "U", "K", "L"}


def extract_mentions(posts: list[dict], tickers: list[str]) -> dict[str, dict]:
    """Map ticker -> {count, score, comments, posts[]} from post title+text.

    A plain uppercase token counts for unambiguous tickers; ambiguous ones
    (common English words) count only when written with a leading '$'.
    """
    tset = set(tickers)
    out: dict[str, dict] = {t: {"count": 0, "score": 0, "comments": 0, "posts": []}
                            for t in tickers}
    for p in posts:
        blob = f"{p['title']} {p['text']}"
        dollar = set(re.findall(r"\$([A-Z]{1,5})\b", blob))
        plain = set(re.findall(r"\b([A-Z]{1,5})\b", blob))
        hits = {t for t in tset if t in dollar
                or (t in plain and t not in _AMBIGUOUS)}
        for t in hits:
            e = out[t]
            e["count"] += 1
            e["score"] += p["score"]
            e["comments"] += p["comments"]
            if len(e["posts"]) < 8:
                e["posts"].append({"title": p["title"][:200], "subreddit": p["subreddit"],
                                   "score": p["score"]})
    return {t: e for t, e in out.items() if e["count"] > 0}
