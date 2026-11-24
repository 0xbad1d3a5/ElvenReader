"""Users: who is reading. Each user has a token of their own (their link), and the token says who a request is from,
so each device stays with its user (there is no switching on a device). Each user keeps their own bookmarks
(bookmark.py); everyone reads every book, and changes only the books they added (permissions.py).

The owner is the user whose token is the server's own (config.json `token`, embedded in pages opened on the PC);
only the owner manages users (the Users page: add, rename, a new link, delete). data/profiles.json (users were
"profiles" before they had tokens: the same ids, so bookmarks carry over).
"""
from __future__ import annotations

import json
import re
import secrets
import threading
import time

from aiohttp import web

_ID = re.compile(r"p-[0-9a-f]{8}")
_lock = threading.Lock()
USER_KEY = "user"  # request[USER_KEY]: the user the request's token belongs to


def _path(app: web.Application):
    from .app import CONFIG_KEY

    return app[CONFIG_KEY].data_path / "profiles.json"


def _load(app: web.Application) -> list[dict]:
    try:
        return json.loads(_path(app).read_text(encoding="utf-8")).get("profiles", [])
    except (OSError, ValueError):
        return []


def _save(app: web.Application, users: list[dict]) -> None:
    p = _path(app)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"profiles": users}, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def new_token() -> str:
    return secrets.token_urlsafe(24)


def all_users(app: web.Application) -> list[dict]:
    """The users, the owner first. The owner's token is always the server's own; a user from before tokens gets one
    (the first of them is the owner)."""
    from .app import CONFIG_KEY

    owner_token = app[CONFIG_KEY].token
    with _lock:
        users = _load(app)
        changed = False
        if not users:
            users, changed = [{"id": "p-" + secrets.token_hex(4), "name": "Me", "created": time.time()}], True
        owner = next((u for u in users if u.get("owner")), None) or next(
            (u for u in users if u.get("token") == owner_token), None) or users[0]
        for u in users:
            want_owner = u is owner
            if bool(u.get("owner")) != want_owner:
                u["owner"], changed = want_owner, True
            if want_owner and u.get("token") != owner_token:
                u["token"], changed = owner_token, True
            if not u.get("token"):
                u["token"], changed = new_token(), True
        users.sort(key=lambda u: not u.get("owner"))
        if changed:
            _save(app, users)
        return users


def for_token(app: web.Application, token: str) -> dict | None:
    """The user a token belongs to."""
    if not token:
        return None
    for u in all_users(app):
        if secrets.compare_digest(u["token"].encode(), token.encode()):
            return u
    return None


def current(request: web.Request) -> str | None:
    """The id of the user the request is from (its token)."""
    user = request.get(USER_KEY)
    return user["id"] if user else None


def public(user: dict) -> dict:
    return {"id": user["id"], "name": user["name"], "owner": bool(user.get("owner"))}


def _name(body) -> str | None:
    name = " ".join(str((body or {}).get("name") or "").split())
    return name[:40] or None


def _owner_only(request: web.Request) -> web.Response | None:
    user = request.get(USER_KEY)
    if not user or not user.get("owner"):
        return web.json_response({"error": "only the server's owner manages users"}, status=403)
    return None


def _managed(request: web.Request, user: dict) -> dict:
    """A user as the Users page shows them: with their links (one per address the server answers on) and QR codes."""
    from .app import CONFIG_KEY, _qr_svg, server_urls, viewer_url

    cfg = request.app[CONFIG_KEY]
    links = [{"url": viewer_url(base, user["token"]), "base": base} for base in server_urls(cfg.port, cfg.extra_urls)]
    for link in links:
        link["qr"] = _qr_svg(link["url"])
    return {**public(user), "token": user["token"], "created": user.get("created"), "links": links,
            "you": user["id"] == current(request)}


async def me(request: web.Request) -> web.Response:
    """GET /api/me: who this device's token belongs to."""
    return web.json_response(public(request[USER_KEY]))


async def list_users(request: web.Request) -> web.Response:
    if (denied := _owner_only(request)) is not None:
        return denied
    return web.json_response({"users": [_managed(request, u) for u in all_users(request.app)]})


async def add_user(request: web.Request) -> web.Response:
    if (denied := _owner_only(request)) is not None:
        return denied
    name = _name(await request.json())
    if not name:
        return web.json_response({"error": "a user needs a name"}, status=400)
    users = all_users(request.app)
    with _lock:
        if any(u["name"].lower() == name.lower() for u in users):
            return web.json_response({"error": f"there is already a user called {name}"}, status=409)
        user = {"id": "p-" + secrets.token_hex(4), "name": name, "created": time.time(), "token": new_token()}
        _save(request.app, users + [user])
    return web.json_response(_managed(request, user), status=201)


async def change_user(request: web.Request) -> web.Response:
    """{"name": ...} renames a user; {"new_link": true} gives them a new token (their old link stops working). The
    owner's token is the server's own (config.json): it isn't changed here."""
    if (denied := _owner_only(request)) is not None:
        return denied
    uid, body = request.match_info["user"], await request.json()
    users = all_users(request.app)
    with _lock:
        user = next((u for u in users if u["id"] == uid), None)
        if user is None:
            return web.json_response({"error": "no such user"}, status=404)
        if "name" in body:
            name = _name(body)
            if not name:
                return web.json_response({"error": "a user needs a name"}, status=400)
            if any(o["name"].lower() == name.lower() and o["id"] != uid for o in users):
                return web.json_response({"error": f"there is already a user called {name}"}, status=409)
            user["name"] = name
        if body.get("new_link"):
            if user.get("owner"):
                return web.json_response({"error": "the owner's link is the server's token (config.json)"}, status=400)
            user["token"] = new_token()
        _save(request.app, users)
    return web.json_response(_managed(request, user))


async def delete_user(request: web.Request) -> web.Response:
    """Removes a user (their link stops working) and everything kept for them (their bookmarks)."""
    from . import bookmark

    if (denied := _owner_only(request)) is not None:
        return denied
    uid = request.match_info["user"]
    users = all_users(request.app)
    with _lock:
        user = next((u for u in users if u["id"] == uid), None)
        if user is None:
            return web.json_response({"error": "no such user"}, status=404)
        if user.get("owner"):
            return web.json_response({"error": "the owner can't be deleted"}, status=400)
        _save(request.app, [u for u in users if u["id"] != uid])
    bookmark.forget_profile(request.app, uid)
    return web.json_response({"users": [_managed(request, u) for u in all_users(request.app)]})


def setup(app: web.Application) -> None:
    app.router.add_get("/api/me", me)
    app.router.add_get("/api/users", list_users)
    app.router.add_post("/api/users", add_user)
    app.router.add_patch("/api/users/{user}", change_user)
    app.router.add_delete("/api/users/{user}", delete_user)
