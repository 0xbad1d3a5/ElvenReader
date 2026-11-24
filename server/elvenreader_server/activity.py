"""What the site is being used for right now, which decides what runs on the GPU (llm_mode "adaptive").

Three states:
* reading: someone reads or listens (a heartbeat from open pages while someone interacts or audio plays,
  and every speech request). The speech model is loaded; book analysis runs next to it, more slowly;
  automatic voice creation waits.
* designing: someone designed a voice within `design_pause_s`. The speech model and the voice designer are
  loaded; analysis stops at once and waits until `design_pause_s` after the last design.
* idle: nobody has used the site for `idle_after_s`. Analysis gets the whole GPU and new books' voices are
  created; when someone comes back, the LLM is stopped and the speech model loads again.

And on top of reading, `crowded`: two or more people are listening to speech that has to be made (a reader whose
page asked for a passage not made yet within READER_WINDOW_S). The speech models need all the GPU's time then, so
analysis stops at once and waits until there has been at most one such reader for CROWD_HOLD_S (moving the LLM
costs tens of seconds, so it doesn't come and go with every pause).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from aiohttp import web

logger = logging.getLogger("elvenreader")

READING, DESIGNING, IDLE = "reading", "designing", "idle"
READER_WINDOW_S = 60.0  # a reader counts while their page asked for new speech within this
CROWD_HOLD_S = 180.0  # analysis comes back this long after the last moment with two such readers


class Activity:
    def __init__(self, idle_after_s: float = 300.0, design_pause_s: float = 600.0):
        self.idle_after_s = idle_after_s
        self.design_pause_s = design_pause_s
        self.last_use = float("-inf")  # nobody yet
        self.last_design = float("-inf")
        # Callbacks when someone starts using the site again (stop a whole-GPU LLM, load the speech model)
        # and when someone starts designing voices (stop the LLM to make room for the designer).
        self.on_use: list[Callable[[], Awaitable[None]]] = []
        self.on_design: list[Callable[[], Awaitable[None]]] = []
        # Called when a second reader starts needing new speech (stop the LLM: the speech models need the GPU).
        self.on_crowd: list[Callable[[], Awaitable[None]]] = []
        self._making: dict[str, float] = {}  # reader -> when their page last asked for speech not made yet
        self.last_crowded = float("-inf")
        self._tasks: set[asyncio.Task] = set()

    @property
    def designing(self) -> bool:
        return time.monotonic() - self.last_design < self.design_pause_s

    @property
    def in_use(self) -> bool:
        return self.designing or time.monotonic() - self.last_use < self.idle_after_s

    def readers(self) -> int:
        """How many people are listening to speech that has to be made now."""
        cutoff = time.monotonic() - READER_WINDOW_S
        for r in [r for r, t in self._making.items() if t < cutoff]:
            del self._making[r]
        return len(self._making)

    @property
    def crowded(self) -> bool:
        """Two or more readers need new speech (or did within CROWD_HOLD_S): analysis stays off the GPU."""
        if self.readers() >= 2:
            self.last_crowded = time.monotonic()
        return time.monotonic() - self.last_crowded < CROWD_HOLD_S

    def making(self, reader: str) -> None:
        """A reader's page asked for a passage that has to be made (not in the cache or the book file)."""
        was = self.readers() >= 2
        self._making[reader] = time.monotonic()
        self.touch()
        if not was and self.readers() >= 2:
            self.last_crowded = time.monotonic()
            logger.info("%d people are listening to new speech: analysis leaves the GPU", self.readers())
            self._run(self.on_crowd)

    @property
    def state(self) -> str:
        return DESIGNING if self.designing else READING if self.in_use else IDLE

    def _run(self, callbacks) -> None:
        for cb in callbacks:
            task = asyncio.ensure_future(cb())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    def touch(self) -> None:
        """Someone reads, listens or otherwise uses the site."""
        was_in_use = self.in_use
        self.last_use = time.monotonic()
        if not was_in_use:
            logger.info("the site is in use: whole-GPU background work stops")
            self._run(self.on_use)

    def touch_design(self) -> None:
        """Someone designs a voice: analysis stops until design_pause_s after the last one."""
        was_designing = self.designing
        self.touch()
        self.last_design = time.monotonic()
        if not was_designing:
            logger.info("voices are being designed: analysis pauses")
            self._run(self.on_design)

    async def wait_for(self, ok: Callable[[], bool], on_wait: Callable[[], Awaitable[None]] | None = None) -> None:
        """Return once `ok()` holds (at once if it does); `on_wait` runs once if it has to wait."""
        if ok():
            return
        if on_wait is not None:
            await on_wait()
        while not ok():
            await asyncio.sleep(1.0)

    def json(self) -> dict:
        now = time.monotonic()
        return {"state": self.state, "in_use": self.in_use, "designing": self.designing,
                "readers": self.readers(), "crowded": self.crowded,
                "idle_in_s": round(max(0.0, self.idle_after_s - (now - self.last_use),
                                       self.design_pause_s - (now - self.last_design))),
                "design_pause_left_s": round(max(0.0, self.design_pause_s - (now - self.last_design))),
                "idle_after_s": self.idle_after_s, "design_pause_s": self.design_pause_s}


ACTIVITY_KEY = web.AppKey("activity", Activity)


async def activity_route(request: web.Request) -> web.Response:
    """POST /api/activity: an open page says someone is using it (interacting with it, or audio is playing)."""
    request.app[ACTIVITY_KEY].touch()
    return web.json_response(request.app[ACTIVITY_KEY].json())


def touch(app: web.Application, design: bool = False) -> None:
    if ACTIVITY_KEY in app:
        (app[ACTIVITY_KEY].touch_design if design else app[ACTIVITY_KEY].touch)()
