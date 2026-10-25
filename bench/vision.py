#!/usr/bin/env python3
"""Image input through the MiniMax-M3.1 server (stdlib only): a smoke check and a bench.

Every test image is drawn at run time by this file (a PNG writer over zlib, a 5x7 bitmap font): no image file is
committed, no image leaves the run, and the same run draws the same bytes. Images go as OpenAI ``image_url`` parts
with a ``data:image/png;base64,...`` URL, or as Anthropic ``/v1/messages`` ``image`` blocks with a base64 source.
The server needs ``--vision`` (``VISION=1``, scripts/config.sh); an engine without the branch mm3/vision refuses images.

  vision.py smoke [--out R/vision-smoke.json]
      describe            a red circle on white (OpenAI route): the reply names red and a circle
      describe-anthropic  a blue square on white (/v1/messages): the reply names blue and a square
      tool                a green square and a ``record_color(color)`` tool (tool_choice "required" with
                          ENGINE_API_TOOLS=1, else "auto"): one call, JSON arguments, color green
      refuse-bytes        one PNG over ``--max-image-mb`` (default 10; the engine's limit is 10 MB an image)
      refuse-count        ``--max-images`` + 1 images (default $VISION_MAX_IMAGES, else 4)
      refuse-video        a ``video_url`` part
      refuse-gif          a GIF image (JPEG, PNG and WebP are served; other formats are refused)
      refuse-url          a public https URL while the server runs without ``--vision-urls`` (skipped with
                          ``--vision-urls`` / VISION_URLS=1: the server would fetch it)
      alive               after the refusals: /health 200 and a text reply (no rank went down)
      Each refusal must be a clean HTTP 4xx (not 401 / 403) with an error message: a 5xx, a 200, a dropped
      connection or a dead server afterwards fails it.
  vision.py bench [--reps 3] [--sizes small,square,wide] [--suites timing,vqa] [--out R/vision-bench.json]
      timing  per size (small 224x224, square 672x672 = the most pixels, wide 2016x224 = the same pixels at 9:1),
              ``--reps`` streamed requests: time to the first token, total time, and the image's prompt tokens
              (usage.prompt_tokens less the same prompt without the image, sent right before it); a MiniMax
              image is at most 576 tokens (672 x 672 pixels, 28 x 28 pixels a token)
      vqa     10 synthetic images with exact answers (counts of shapes, colors, a shape, a digit, a two-digit
              number, a word): the accuracy, each answer and what was read from it

Thinking is off on every request (``chat_template_kwargs.enable_thinking`` false; the old engine may still think, and
only ``content`` is scored), greedy, a per-run nonce in each prompt so a kept prompt state cannot answer it. Raw
MiniMax markup in ``content`` (``]<]minimax[>[``, ``<invoke name=``, ``<tool_call>``, think tags) fails a smoke case.

Defaults come from HOST / PORT / SERVED_NAME / API_KEY_FILE / ENGINE_API_TOOLS / VISION_MAX_IMAGES / VISION_URLS
(scripts/config.sh exports them); the key goes in a header (``Authorization: Bearer``; ``x-api-key`` too on
/v1/messages) and is masked in everything printed or saved. Exit 0: every smoke case passed / no bench request failed
(and the VQA accuracy is at least ``--min-accuracy``); 1: a failure; 2: the server could not be reached or refused the
key.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# engine_api beside this file (python3 -I leaves the script's directory off sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine_api as ea

LEAK = re.compile(r"\]<\]minimax\[>\[|<invoke name=|</?tool_call>|</?mm:think>")
MASK = "<redacted>"
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
# MiniMax's image geometry: 28 x 28 pixels a token, 672 x 672 pixels at most, so 576 tokens an image at most
PATCH = 28
MAX_PIXELS = 672 * 672
MAX_IMAGE_TOKENS = (672 // PATCH) ** 2
MARKER_SLACK = 16  # the template's image markers around an image's rows

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 30, 30)
GREEN = (30, 160, 60)
BLUE = (30, 70, 220)
YELLOW = (240, 200, 20)


# -- drawing: a PNG writer and a canvas --------------------------------------------------------------------------------


def png(width: int, height: int, rgb: bytes | bytearray, level: int = 6) -> bytes:
    """An 8-bit RGB PNG of ``rgb`` (rows top to bottom, 3 bytes a pixel), filter 0 on every row."""

    if len(rgb) != width * height * 3:
        raise ValueError(f"{len(rgb)} bytes for {width}x{height} RGB")
    stride = width * 3
    raw = b"".join(
        b"\x00" + bytes(rgb[y * stride : (y + 1) * stride]) for y in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, level))
        + chunk(b"IEND", b"")
    )


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG's IHDR."""

    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


# 5 x 7 glyphs for the digits and the letters the VQA set writes
GLYPHS = {
    "0": (" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "),
    "1": ("  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    "2": (" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"),
    "3": ("#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "),
    "4": ("   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "),
    "5": ("#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "),
    "6": ("  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "),
    "7": ("#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "),
    "8": (" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "),
    "9": (" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "),
    "A": (" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "C": (" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "),
    "T": ("#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "),
}


class Canvas:
    """An RGB canvas, white unless told otherwise; shapes are filled row spans (fast in pure Python)."""

    def __init__(self, width: int, height: int, background=WHITE) -> None:
        self.w, self.h = width, height
        self.px = bytearray(bytes(background) * (width * height))

    def span(self, y: int, x0: int, x1: int, color) -> None:
        if 0 <= y < self.h:
            x0, x1 = max(0, x0), min(self.w, x1)
            if x1 > x0:
                off = (y * self.w + x0) * 3
                self.px[off : off + (x1 - x0) * 3] = bytes(color) * (x1 - x0)

    def rect(self, x0: int, y0: int, x1: int, y1: int, color) -> Canvas:
        for y in range(y0, y1):
            self.span(y, x0, x1, color)
        return self

    def square(self, cx: int, cy: int, side: int, color) -> Canvas:
        return self.rect(cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2, color)

    def circle(self, cx: int, cy: int, r: int, color) -> Canvas:
        for y in range(cy - r, cy + r + 1):
            dx = int(math.sqrt(max(r * r - (y - cy) ** 2, 0)))
            self.span(y, cx - dx, cx + dx + 1, color)
        return self

    def triangle(self, cx: int, cy: int, size: int, color) -> Canvas:
        """An upright triangle ``size`` wide and high, centred on (cx, cy)."""

        top = cy - size // 2
        for i in range(size):
            half = i // 2
            self.span(top + i, cx - half, cx + half + 1, color)
        return self

    def text(self, s: str, cx: int, cy: int, scale: int, color=BLACK) -> Canvas:
        """``s`` in the 5x7 font, ``scale`` pixels a dot, centred on (cx, cy)."""

        width = (len(s) * 6 - 1) * scale
        x0, y0 = cx - width // 2, cy - 7 * scale // 2
        for n, ch in enumerate(s):
            for row, bits in enumerate(GLYPHS[ch]):
                for col, bit in enumerate(bits):
                    if bit == "#":
                        x = x0 + (n * 6 + col) * scale
                        self.rect(x, y0 + row * scale, x + scale, y0 + (row + 1) * scale, color)
        return self

    def png(self) -> bytes:
        return png(self.w, self.h, self.px)


def data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def oversize_png(min_bytes: int, seed: int = 7) -> bytes:
    """A valid PNG of at least ``min_bytes``: seeded noise, stored uncompressed (zlib level 0)."""

    w = 2048
    h = min_bytes // (w * 3) + 2
    return png(w, h, random.Random(seed).randbytes(w * h * 3), level=0)


# The smallest GIF89a (1 x 1, white): a format the server does not decode
GIF_1PX = (
    b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00"
    b",\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
)


def estimate_tokens(width: int, height: int) -> int:
    """An image's rows as MiniMax's processor sizes it: at most 672 x 672 pixels (aspect kept), 28 x 28 a token."""

    if width * height > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (width * height))
        w, h = max(PATCH, math.floor(width * s / PATCH) * PATCH), max(PATCH, math.floor(height * s / PATCH) * PATCH)
    else:
        w, h = max(PATCH, round(width / PATCH) * PATCH), max(PATCH, round(height / PATCH) * PATCH)
    return min(MAX_IMAGE_TOKENS, (w // PATCH) * (h // PATCH))


# -- the images ------------------------------------------------------------------------------------------------------


def smoke_circle() -> bytes:
    return Canvas(448, 448).circle(224, 224, 150, RED).png()


def smoke_square() -> bytes:
    return Canvas(448, 448).square(224, 224, 260, BLUE).png()


def smoke_green() -> bytes:
    return Canvas(448, 448).square(224, 224, 300, GREEN).png()


def scene(width: int, height: int) -> bytes:
    """Three shapes spread over the canvas (the timing images): a red circle, a blue square, a green triangle."""

    c, u = Canvas(width, height), min(width, height)
    c.circle(width // 6, height // 2, int(u * 0.3), RED)
    c.square(width // 2, height // 2, int(u * 0.55), BLUE)
    c.triangle(width * 5 // 6, height // 2, int(u * 0.6), GREEN)
    return c.png()


SIZES: dict[str, tuple[int, int]] = {"small": (224, 224), "square": (672, 672), "wide": (2016, 224)}


@dataclass(frozen=True)
class Item:
    name: str
    draw: Callable[[], bytes]
    question: str
    kind: str  # number | color | shape | word
    expected: str


def _count(shape: str, color, centres: list[tuple[int, int]], size: int) -> Callable[[], bytes]:
    def draw() -> bytes:
        c = Canvas(448, 448)
        for x, y in centres:
            getattr(c, shape)(x, y, size // 2 if shape == "circle" else size, color)
        return c.png()

    return draw


def _mixed() -> bytes:
    c = Canvas(448, 448)
    c.circle(110, 110, 55, RED).square(338, 110, 110, BLUE).triangle(110, 338, 120, GREEN).circle(338, 338, 55, YELLOW)
    return c.png()


NUMBER_ONLY = " Answer with only the number."
VQA: tuple[Item, ...] = (
    Item("count-circles", _count("circle", RED, [(110, 130), (330, 120), (220, 330)], 100),
         "How many circles are in this image?" + NUMBER_ONLY, "number", "3"),
    Item("count-squares", _count("square", BLUE, [(75, 75), (285, 85), (95, 285), (335, 315), (200, 200)], 70),
         "How many squares are in this image?" + NUMBER_ONLY, "number", "5"),
    Item("count-triangles", _count("triangle", GREEN, [(120, 224), (330, 224)], 140),
         "How many triangles are in this image?" + NUMBER_ONLY, "number", "2"),
    Item("count-shapes", _mixed, "How many shapes are in this image?" + NUMBER_ONLY, "number", "4"),
    Item("color-circle", lambda: Canvas(448, 448).circle(224, 224, 160, YELLOW).png(),
         "What color is the circle? Answer with one word.", "color", "yellow"),
    Item("color-square", lambda: Canvas(448, 448).square(224, 224, 280, GREEN).png(),
         "What color is the square? Answer with one word.", "color", "green"),
    Item("shape", lambda: Canvas(448, 448).triangle(224, 224, 300, RED).png(),
         "What shape is drawn in this image? Answer with one word.", "shape", "triangle"),
    Item("digit", lambda: Canvas(448, 448).text("7", 224, 224, 40).png(),
         "What digit is written in this image? Answer with only the digit.", "number", "7"),
    Item("number", lambda: Canvas(448, 448).text("38", 224, 224, 28).png(),
         "What number is written in this image?" + NUMBER_ONLY, "number", "38"),
    Item("word", lambda: Canvas(448, 448).text("CAT", 224, 224, 20).png(),
         "What word is written in this image? Answer with only the word.", "word", "cat"),
)


# -- reading the answers ---------------------------------------------------------------------------------------------

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
COLORS = ("red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "grey")
SHAPES = {
    "circle": ("circle", "circles", "circular", "round", "disc", "disk", "dot", "dots", "ball", "sphere", "oval"),
    "square": ("square", "squares", "rectangle", "rectangles", "box", "cube"),
    "triangle": ("triangle", "triangles", "triangular", "pyramid"),
}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", (text or "").lower())


def first_number(text: str) -> int | None:
    """The first number in ``text``, digits or a word (zero to twelve)."""

    for w in words(text):
        if w.isdigit():
            return int(w)
        if w in NUMBER_WORDS:
            return NUMBER_WORDS[w]
    return None


def first_color(text: str) -> str | None:
    for w in words(text):
        if w in COLORS:
            return "gray" if w == "grey" else w
    return None


def first_shape(text: str) -> str | None:
    for w in words(text):
        for shape, names in SHAPES.items():
            if w in names:
                return shape
    return None


def read_answer(kind: str, text: str) -> str | None:
    """What a reply says for a question of ``kind``: the first number, color, shape or word in it."""

    if kind == "number":
        n = first_number(text)
        return None if n is None else str(n)
    if kind == "color":
        return first_color(text)
    if kind == "shape":
        return first_shape(text)
    ws = [w for w in words(text) if not w.isdigit()]
    return ws[0] if ws else None


def vqa_correct(item: Item, text: str) -> tuple[bool, str | None]:
    """(right, what was read). A word counts when it appears as a whole word anywhere in the reply."""

    got = read_answer(item.kind, text)
    if item.kind == "word" and item.expected in words(text):
        return True, item.expected
    return got == item.expected, got


def mentions(text: str, color: str, shape: str) -> list[str]:
    """Problems when ``text`` does not name ``color`` and ``shape`` (synonyms of the shape count)."""

    ws, out = set(words(text)), []
    if color not in ws:
        out.append(f"the reply does not name {color}")
    if not ws & set(SHAPES[shape]):
        out.append(f"the reply does not name a {shape}")
    return out


def degenerate(text: str) -> str | None:
    """Why ``text`` looks like a broken engine's output, or None."""

    if not text.strip():
        return "empty reply"
    if "�" in text:
        return "replacement character (U+FFFD) in the reply"
    if re.search(r"(.)\1{23,}", text):
        return "one character repeated 24+ times"
    ws = re.findall(r"\S+", text)
    if len(ws) >= 12 and len(set(ws)) / len(ws) < 0.3:
        return f"repetitive reply ({len(set(ws)) / len(ws):.2f} distinct words)"
    return None


# -- requests --------------------------------------------------------------------------------------------------------


def image_part(data: bytes, mime: str = "image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(data, mime)}}


def openai_body(model: str, prompt: str, images: list[dict], max_tokens: int = 256, **extra) -> dict:
    """A chat request: the image parts, then the prompt; thinking off, greedy."""

    content = [*images, {"type": "text", "text": prompt}]
    return dict(OFF, model=model, messages=[{"role": "user", "content": content}],
                temperature=0, max_tokens=max_tokens, **extra)


def anthropic_body(model: str, prompt: str, data: bytes, max_tokens: int = 256) -> dict:
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                          "data": base64.b64encode(data).decode()}}
    return {"model": model, "max_tokens": max_tokens, "temperature": 0, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [block, {"type": "text", "text": prompt}]}]}


RECORD_COLOR = {
    "type": "function",
    "function": {
        "name": "record_color",
        "description": "Record the color of the shape the user shows",
        "parameters": {
            "type": "object",
            "properties": {"color": {"type": "string", "description": "the color, one lowercase word"}},
            "required": ["color"],
        },
    },
}


class Unreachable(Exception):
    """No server, or it refused the key."""


@dataclass
class Reply:
    status: int  # 0: no HTTP status (the connection dropped)
    body: Any  # parsed JSON, else None
    text: str  # the raw body (masked), or why there is none
    seconds: float = 0.0


def default_base() -> str:
    """http://HOST:PORT from scripts/config.sh's settings (a wildcard HOST is reached on 127.0.0.1)."""

    host = os.environ.get("HOST") or "127.0.0.1"
    host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{host}:{os.environ.get('PORT') or 8000}"


def read_key(path: str | None) -> str:
    """The first key of an API key file (one key, or "label: key", a line); "" when there is none to read.

    ``path`` None: API_KEY_FILE from the environment, used when it exists (required when API_KEY=1)."""

    strict = path is not None or os.environ.get("API_KEY") == "1"
    path = path if path is not None else os.environ.get("API_KEY_FILE", "")
    if not path or (not strict and not os.path.exists(path)):
        return ""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as exc:
        raise SystemExit(f"cannot read the API key file {path}: {exc.strerror}") from None
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split(":", 1)[1].strip() if ":" in line else line
    raise SystemExit(f"{path}: no key in it")


class Api:
    """JSON over HTTP with the key as a header; ``mask`` hides the key in any text that is printed or saved."""

    def __init__(self, base: str, key: str = "") -> None:
        self.base, self.key = base.rstrip("/").removesuffix("/v1"), key

    def headers(self, path: str) -> dict:
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
            if path == "/v1/messages":
                h["x-api-key"] = self.key
        if path == "/v1/messages":
            h["anthropic-version"] = "2023-06-01"
        return h

    def request(self, path: str, body: dict | None, timeout: float) -> Reply:
        """One request; an HTTP error is a Reply with its status, not an exception."""

        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data, self.headers(path))
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (ConnectionResetError, BrokenPipeError, TimeoutError)):
                return Reply(0, None, f"no HTTP status: {exc.reason!r}", time.perf_counter() - t0)
            raise Unreachable(str(exc.reason)) from None
        except (ConnectionResetError, BrokenPipeError, TimeoutError) as exc:
            return Reply(0, None, f"no HTTP status: {type(exc).__name__}: {exc}", time.perf_counter() - t0)
        if status in (401, 403):
            raise Unreachable(f"HTTP {status}: the server refused the key")
        text = self.mask(raw.decode(errors="replace"))  # a reply never holds the key
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        return Reply(status, parsed, text, time.perf_counter() - t0)

    def stream(self, body: dict, timeout: float) -> dict:
        """A streamed chat request: first-token and total seconds, content, reasoning, usage."""

        body = dict(body, stream=True, stream_options={"include_usage": True})
        req = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     self.headers("/v1/chat/completions"))
        out: dict[str, Any] = {"ttft_s": None, "content": "", "reasoning": "", "usage": None, "error": None}
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                for raw in r:
                    line = self.mask(raw.decode(errors="replace")).strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    if chunk.get("error"):
                        out["error"] = "stream error: " + str(chunk["error"])[:300]
                        break
                    if chunk.get("usage"):
                        out["usage"] = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        d = ch.get("delta") or {}
                        piece, thought = d.get("content") or "", d.get("reasoning_content") or ""
                        if (piece or thought or d.get("tool_calls")) and out["ttft_s"] is None:
                            out["ttft_s"] = round(time.perf_counter() - t0, 3)
                        out["content"] += piece
                        out["reasoning"] += thought
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise Unreachable(f"HTTP {exc.code}: the server refused the key") from None
            out["error"] = f"HTTP {exc.code}: " + self.mask(exc.read().decode(errors="replace"))[:300]
        except urllib.error.URLError as exc:
            if not isinstance(exc.reason, (ConnectionResetError, BrokenPipeError, TimeoutError)):
                raise Unreachable(str(exc.reason)) from None
            out["error"] = f"no HTTP status: {exc.reason!r}"
        except (ConnectionResetError, BrokenPipeError, TimeoutError) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        out["total_s"] = round(time.perf_counter() - t0, 3)
        return out

    def mask(self, text: str) -> str:
        return text.replace(self.key, MASK) if self.key else text


def message(reply: Reply) -> dict:
    """The OpenAI reply's message, or {}."""

    body = reply.body if isinstance(reply.body, dict) else {}
    return ((body.get("choices") or [{}])[0]).get("message") or {}


def error_message(body: Any) -> str:
    """The message of an OpenAI ({"error": {"message"}} or {"error": "..."}) or Anthropic error body."""

    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or "")
    return str(err or body.get("message") or "")


def refusal_problems(reply: Reply) -> list[str]:
    """[] when ``reply`` is a clean refusal: HTTP 4xx (not 401 / 403) with an error message."""

    if reply.status == 0:
        return [f"no clean refusal ({reply.text[:200]})"]
    if not 400 <= reply.status < 500:
        return [f"HTTP {reply.status}, not a 4xx refusal: {reply.text[:200]!r}"]
    if not error_message(reply.body).strip():
        return [f"HTTP {reply.status} without an error message: {reply.text[:200]!r}"]
    return []


def answer_problems(msg: dict) -> list[str]:
    out = []
    content = msg.get("content") or ""
    if LEAK.search(content):
        out.append(f"raw MiniMax markup in content: {content[:120]!r}")
    if LEAK.search(msg.get("reasoning_content") or ""):
        out.append("a think tag left inside reasoning_content")
    return out


# -- smoke -----------------------------------------------------------------------------------------------------------


def case_describe(api: Api, a, nonce: str) -> tuple[list[str], dict]:
    prompt = f"[{nonce}] Describe this image in one short sentence: what shape is it, and what color?"
    r = api.request("/v1/chat/completions", openai_body(a.model, prompt, [image_part(smoke_circle())]), a.timeout)
    if r.status != 200:
        return [f"HTTP {r.status}: {r.text[:300]}"], {}
    msg = message(r)
    content = msg.get("content") or ""
    problems = answer_problems(msg) + mentions(content, "red", "circle")
    if why := degenerate(content):
        problems.append(why)
    return problems, {"content": content[:300], "usage": (r.body or {}).get("usage")}


def case_anthropic(api: Api, a, nonce: str) -> tuple[list[str], dict]:
    prompt = f"[{nonce}] Describe this image in one short sentence: what shape is it, and what color?"
    r = api.request("/v1/messages", anthropic_body(a.model, prompt, smoke_square()), a.timeout)
    if r.status != 200:
        return [f"HTTP {r.status}: {r.text[:300]}"], {}
    blocks = (r.body or {}).get("content") or []
    text = "".join(b.get("text") or "" for b in blocks if isinstance(b, dict) and b.get("type") == "text")
    thinking = "".join(b.get("thinking") or "" for b in blocks if isinstance(b, dict) and b.get("type") == "thinking")
    problems = []
    if LEAK.search(text) or LEAK.search(thinking):
        problems.append(f"raw MiniMax markup in the reply: {text[:120]!r}")
    problems += mentions(text, "blue", "square")
    if why := degenerate(text):
        problems.append(why)
    return problems, {"content": text[:300], "usage": (r.body or {}).get("usage")}


def case_tool(api: Api, a, nonce: str) -> tuple[list[str], dict]:
    prompt = (f"[{nonce}] Look at the image and call the record_color tool with the color of the shape "
              "(one lowercase word). Do not answer in text.")
    choice = "required" if a.api_tools else "auto"
    body = openai_body(a.model, prompt, [image_part(smoke_green())], max_tokens=512, tools=[RECORD_COLOR],
                       tool_choice=choice)
    r = api.request("/v1/chat/completions", body, a.timeout)
    if r.status != 200:
        return [f"HTTP {r.status} (tool_choice {choice}): {r.text[:300]}"], {}
    msg = message(r)
    problems = answer_problems(msg)
    calls = msg.get("tool_calls") or []
    detail = {"tool_choice": choice, "content": (msg.get("content") or "")[:300], "tool_calls": calls[:4]}
    if not calls:
        return problems + [f"no tool_calls: {(msg.get('content') or '')[:120]!r}"], detail
    fn = calls[0].get("function") or {}
    if fn.get("name") != "record_color":
        problems.append(f"called {fn.get('name')!r}, not record_color")
    try:
        args = json.loads(fn.get("arguments") or "")
        if not isinstance(args, dict):
            raise TypeError("not an object")
    except (ValueError, TypeError) as exc:
        return problems + [f"arguments not a JSON object ({exc}): {str(fn.get('arguments'))[:120]!r}"], detail
    if e := ea.validate(args, RECORD_COLOR["function"]["parameters"]):
        problems.append(f"arguments break the schema: {e}")
    color = args.get("color")
    if not isinstance(color, str) or first_color(color) != "green":
        problems.append(f"color {color!r}, not green")
    return problems, detail


def refusal_case(body_of: Callable[[Any], tuple[str, dict]]) -> Callable:
    def run(api: Api, a, nonce: str) -> tuple[list[str], dict]:
        path, body = body_of(a)
        r = api.request(path, body, a.timeout)
        return refusal_problems(r), {"status": r.status, "error": (error_message(r.body) or r.text)[:300],
                                     "seconds": round(r.seconds, 2)}

    return run


def _too_big(a) -> tuple[str, dict]:
    data = oversize_png(int(a.max_image_mb * 2**20) + 2**20)
    return "/v1/chat/completions", openai_body(a.model, "Describe this image.", [image_part(data)], max_tokens=16)


def _too_many(a) -> tuple[str, dict]:
    colors = [RED, GREEN, BLUE, YELLOW, BLACK]
    parts = [image_part(Canvas(56, 56, colors[i % len(colors)]).png()) for i in range(a.max_images + 1)]
    return "/v1/chat/completions", openai_body(a.model, "How many images are there?", parts, max_tokens=16)


def _video(a) -> tuple[str, dict]:
    part = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28y"}}
    return "/v1/chat/completions", openai_body(a.model, "Describe this video.", [part], max_tokens=16)


def _gif(a) -> tuple[str, dict]:
    return "/v1/chat/completions", openai_body(a.model, "Describe this image.", [image_part(GIF_1PX, "image/gif")],
                                               max_tokens=16)


def _url(a) -> tuple[str, dict]:
    part = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    return "/v1/chat/completions", openai_body(a.model, "Describe this image.", [part], max_tokens=16)


def case_alive(api: Api, a, nonce: str) -> tuple[list[str], dict]:
    h = api.request("/health", None, 30)
    problems = [] if h.status == 200 else [f"/health HTTP {h.status}: {h.text[:200]}"]
    r = api.request("/v1/chat/completions", dict(OFF, model=a.model, temperature=0, max_tokens=32, messages=[
        {"role": "user", "content": f"[{nonce}] What is the capital of France? One word."}]), a.timeout)
    content = message(r).get("content") or ""
    if r.status != 200 or not re.search(r"\bParis\b", content):
        problems.append(f"no text reply after the refusals (HTTP {r.status}): {(content or r.text)[:200]!r}")
    return problems, {"health": h.status, "content": content[:120]}


SMOKE: tuple[tuple[str, Callable], ...] = (
    ("describe", case_describe),
    ("describe-anthropic", case_anthropic),
    ("tool", case_tool),
    ("refuse-bytes", refusal_case(_too_big)),
    ("refuse-count", refusal_case(_too_many)),
    ("refuse-video", refusal_case(_video)),
    ("refuse-gif", refusal_case(_gif)),
    ("refuse-url", refusal_case(_url)),
    ("alive", case_alive),
)


def smoke(api: Api, a) -> dict:
    nonce = f"vision {os.getpid()}-{int(time.time())}"
    cases = []
    for name, run in SMOKE:
        if name == "refuse-url" and a.vision_urls:
            cases.append({"case": name, "ok": None, "problems": [], "detail": {"skipped": "VISION_URLS=1"}})
            print(f"SKIP {name:20s} (the server fetches https URLs: VISION_URLS=1)", flush=True)
            continue
        t0 = time.perf_counter()
        try:
            problems, detail = run(api, a, nonce)
        except Unreachable as exc:
            if not cases:
                raise  # no server at all, or the key refused: exit 2
            problems, detail = [f"the server stopped answering: {exc}"], {}
        except Exception as exc:  # noqa: BLE001 - a failed request fails the case
            problems, detail = [f"request failed: {type(exc).__name__}: {exc}"[:300]], {}
        res = {"case": name, "ok": not problems, "problems": problems, "seconds": round(time.perf_counter() - t0, 2),
               "detail": detail}
        cases.append(res)
        line = f"{'OK  ' if res['ok'] else 'FAIL'} {name:20s} {res['seconds']}s"
        print(api.mask(line + (" | " + "; ".join(problems) if problems else "")), flush=True)
    ran = [c for c in cases if c["ok"] is not None]
    return {"ok": all(c["ok"] for c in ran), "passed": sum(1 for c in ran if c["ok"]), "total": len(ran),
            "cases": cases}


# -- bench -----------------------------------------------------------------------------------------------------------


def timing(api: Api, a, nonce: str) -> dict:
    out = {}
    for size in a.sizes.split(","):
        if size not in SIZES:
            raise SystemExit(f"--sizes: {size!r} is not one of {', '.join(SIZES)}")
        w, h = SIZES[size]
        data = scene(w, h)
        runs = []
        for rep in range(a.reps):
            prompt = f"[{nonce} {size} {rep}] Describe this image in one sentence."
            base = api.request("/v1/chat/completions", openai_body(a.model, prompt, [], max_tokens=1), a.timeout)
            base_tokens = ((base.body or {}).get("usage") or {}).get("prompt_tokens") if base.status == 200 else None
            run = api.stream(openai_body(a.model, prompt, [image_part(data)], max_tokens=a.max_tokens), a.timeout)
            usage = run.pop("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            image_tokens = prompt_tokens - base_tokens if prompt_tokens and base_tokens else None
            problems = [run["error"]] if run["error"] else []
            if not problems and image_tokens is None:
                problems.append("no usage.prompt_tokens to compare (text-only HTTP %s)" % base.status)
            elif image_tokens is not None and not 0 < image_tokens <= MAX_IMAGE_TOKENS + MARKER_SLACK:
                problems.append(f"{image_tokens} image tokens (an image is 1 to {MAX_IMAGE_TOKENS} + markers)")
            if not problems and not run["content"].strip():
                problems.append("empty reply")
            runs.append({"rep": rep, "ttft_s": run["ttft_s"], "total_s": run["total_s"], "prompt_tokens": prompt_tokens,
                         "text_prompt_tokens": base_tokens, "image_tokens": image_tokens,
                         "completion_tokens": usage.get("completion_tokens"), "content": run["content"][:160],
                         "problems": problems})
            print(api.mask(f"{'OK  ' if not problems else 'FAIL'} timing {size:6s} [{rep}] ttft {run['ttft_s']}s "
                           f"total {run['total_s']}s image tokens {image_tokens}"
                           + (" | " + "; ".join(problems) if problems else "")), flush=True)
        good = [r for r in runs if not r["problems"]]

        def median(key: str, rows=good):
            vals = [r[key] for r in rows if r[key] is not None]
            return round(statistics.median(vals), 3) if vals else None

        out[size] = {"width": w, "height": h, "png_bytes": len(data), "estimated_tokens": estimate_tokens(w, h),
                     "image_tokens": median("image_tokens"), "ttft_s_median": median("ttft_s"),
                     "total_s_median": median("total_s"), "errors": len(runs) - len(good), "runs": runs}
    return out


def vqa(api: Api, a, nonce: str) -> dict:
    items = []
    for n, item in enumerate(VQA):
        prompt = f"[{nonce} q{n}] {item.question}"
        r = api.request("/v1/chat/completions", openai_body(a.model, prompt, [image_part(item.draw())],
                                                            max_tokens=a.max_tokens), a.timeout)
        content = message(r).get("content") or ""
        error = None if r.status == 200 else f"HTTP {r.status}: {r.text[:200]}"
        ok, got = (False, None) if error else vqa_correct(item, content)
        items.append({"name": item.name, "kind": item.kind, "expected": item.expected, "read": got, "ok": ok,
                      "content": content[:200], "error": error, "seconds": round(r.seconds, 2)})
        print(api.mask(f"{'OK  ' if ok else 'MISS'} vqa {item.name:16s} expected {item.expected!r} read {got!r}"
                       + (f" | {error}" if error else "")), flush=True)
    correct = sum(1 for i in items if i["ok"])
    return {"correct": correct, "total": len(items), "accuracy": round(correct / len(items), 3),
            "errors": sum(1 for i in items if i["error"]), "items": items}


def bench(api: Api, a) -> dict:
    nonce = f"vision {os.getpid()}-{int(time.time())}"
    suites = a.suites.split(",")
    for s in suites:
        if s not in ("timing", "vqa"):
            raise SystemExit(f"--suites: {s!r} is not timing or vqa")
    report: dict[str, Any] = {}
    if "timing" in suites:
        report["timing"] = timing(api, a, nonce)
    if "vqa" in suites:
        report["vqa"] = vqa(api, a, nonce)
    errors = sum(v["errors"] for v in report.get("timing", {}).values()) + report.get("vqa", {}).get("errors", 0)
    report["errors"] = errors
    acc = report.get("vqa", {}).get("accuracy")
    report["ok"] = errors == 0 and (acc is None or acc >= a.min_accuracy)
    return report


# -- main ------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("smoke", "bench"))
    ap.add_argument("--base", default=default_base())
    ap.add_argument("--model", default=os.environ.get("SERVED_NAME") or "MiniMax-M3.1-Flash-EXL3")
    ap.add_argument("--api-key-file", default=None, help="the server's key file (default: $API_KEY_FILE if present)")
    ap.add_argument("--timeout", type=float, default=600, help="seconds a request (the first may compile kernels)")
    ap.add_argument("--out", default="", help="the JSON report (e.g. results/vision/<date>-smoke.json)")
    ap.add_argument("--label", default="", help="a name for the run, saved in the report")
    ap.add_argument("--max-images", type=int, default=int(os.environ.get("VISION_MAX_IMAGES") or 4),
                    help="the server's --vision-max-images (default $VISION_MAX_IMAGES, else 4)")
    ap.add_argument("--max-image-mb", type=float, default=10.0,
                    help="the server's bytes limit an image, MB (the engine's: 10); refuse-bytes sends 1 MiB more")
    ap.add_argument("--vision-urls", choices=("0", "1"), default=os.environ.get("VISION_URLS") or "0",
                    help="1: the server runs with --vision-urls (refuse-url is skipped); default $VISION_URLS, else 0")
    ap.add_argument("--reps", type=int, default=3, help="bench: requests a size")
    ap.add_argument("--sizes", default="small,square,wide", help=f"bench: of {', '.join(SIZES)}")
    ap.add_argument("--suites", default="timing,vqa", help="bench: timing, vqa")
    ap.add_argument("--max-tokens", type=int, default=256, help="bench: a reply's tokens at most")
    ap.add_argument("--min-accuracy", type=float, default=0.0, help="bench: fail under this VQA accuracy (0 to 1)")
    ea.add_flag(ap)
    a = ap.parse_args(argv)
    a.api_tools = ea.enabled(a.engine_api_tools)
    a.vision_urls = a.vision_urls == "1"
    if a.max_images < 1 or a.reps < 1:
        ap.error("--max-images and --reps are positive")
    api = Api(a.base, read_key(a.api_key_file))
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        report = smoke(api, a) if a.command == "smoke" else bench(api, a)
    except Unreachable as exc:
        print(api.mask(f"[vision] cannot query {api.base}: {exc}"), file=sys.stderr)
        return 2
    config = {k: v for k, v in vars(a).items() if k != "api_key_file"}
    report = {"label": a.label, "started": started, "config": config, **report}
    if a.command == "smoke":
        print(f"\n[vision] smoke: {report['passed']}/{report['total']} passed", flush=True)
    else:
        parts = [f"{s} ttft {v['ttft_s_median']}s total {v['total_s_median']}s tokens {v['image_tokens']}"
                 for s, v in report.get("timing", {}).items()]
        if "vqa" in report:
            q = report["vqa"]
            parts.append(f"VQA {q['correct']}/{q['total']} ({q['accuracy']:.0%})")
        print(f"\n[vision] bench: {'; '.join(parts)}; errors {report['errors']}", flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(api.mask(json.dumps(report, indent=2)))
        print(f"wrote {a.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
