"""bench/vision.py: its drawn images (valid PNGs, deterministic, the geometry MiniMax's processor gives them), its answer
readers and refusal check, and the smoke and bench runs against the stand-in server: clean runs pass, a wrong answer,
a 5xx or a served video fail, the tool case follows ENGINE_API_TOOLS, and the key never appears in output or files."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import fake_minimax as fm
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import vision as vz  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_png(data: bytes) -> tuple[int, int, bytes]:
    """(width, height, RGB bytes) of an 8-bit RGB PNG whose rows all use filter 0 (what vz.png writes)."""

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos : pos + 8])
        body = data[pos + 8 : pos + 8 + n]
        assert struct.unpack(">I", data[pos + 8 + n : pos + 12 + n])[0] == zlib.crc32(kind + body) & 0xFFFFFFFF
        if kind == b"IHDR":
            w, h, depth, color = struct.unpack(">IIBB", body[:10])
            assert (depth, color) == (8, 2)
        elif kind == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    stride = w * 3 + 1
    assert len(raw) == stride * h and all(raw[y * stride] == 0 for y in range(h))
    return w, h, b"".join(raw[y * stride + 1 : (y + 1) * stride] for y in range(h))


def pixel(rgb: bytes, w: int, x: int, y: int) -> tuple[int, int, int]:
    off = (y * w + x) * 3
    return tuple(rgb[off : off + 3])


# -- images ----------------------------------------------------------------------------------------------------------


def test_png_round_trip_and_shapes():
    w, h, rgb = decode_png(vz.smoke_circle())
    assert (w, h) == (448, 448) == vz.png_size(vz.smoke_circle())
    assert pixel(rgb, w, 224, 224) == vz.RED
    assert pixel(rgb, w, 5, 5) == vz.WHITE
    w, h, rgb = decode_png(vz.Canvas(100, 60).triangle(50, 30, 40, vz.GREEN).png())
    assert pixel(rgb, w, 50, 45) == vz.GREEN and pixel(rgb, w, 30, 12) == vz.WHITE


def test_the_font_draws_the_digit():
    w, h, rgb = decode_png(vz.Canvas(60, 80).text("7", 30, 40, 8).png())
    # the 7's top bar, and the blank under its right end
    assert pixel(rgb, w, 30, 40 - 28 + 2) == vz.BLACK
    assert pixel(rgb, w, 10, 60) == vz.WHITE


def test_images_are_deterministic_small_and_distinct():
    drawn = [item.draw() for item in vz.VQA] + [vz.smoke_circle(), vz.smoke_square(), vz.smoke_green()]
    again = [item.draw() for item in vz.VQA] + [vz.smoke_circle(), vz.smoke_square(), vz.smoke_green()]
    assert drawn == again
    assert len({sha(d) for d in drawn}) == len(drawn)
    assert max(len(d) for d in drawn) < 20_000
    for size, (w, h) in vz.SIZES.items():
        assert vz.png_size(vz.scene(w, h)) == (w, h), size


def test_estimated_tokens_follow_minimaxs_geometry():
    assert vz.estimate_tokens(224, 224) == 64
    assert vz.estimate_tokens(672, 672) == 576 == vz.MAX_IMAGE_TOKENS
    assert vz.estimate_tokens(2016, 224) == 576
    assert vz.estimate_tokens(1344, 1344) == 576
    assert vz.estimate_tokens(56, 56) == 4


def test_the_over_large_image_is_a_valid_png_over_the_limit():
    data = vz.oversize_png(3 * 2**20)
    assert len(data) > 3 * 2**20
    assert vz.png_size(data)[0] == 2048


# -- answers and refusals --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,text,read",
    [
        ("number", "There are three circles.", "3"),
        ("number", "38.", "38"),
        ("number", "**5**", "5"),
        ("number", "I cannot tell.", None),
        ("color", "Yellow.", "yellow"),
        ("color", "A grey square", "gray"),
        ("shape", "It is a triangular shape.", "triangle"),
        ("shape", "a round red disc", "circle"),
        ("word", "Cat", "cat"),
    ],
)
def test_answer_readers(kind, text, read):
    assert vz.read_answer(kind, text) == read


def test_vqa_scoring_is_exact():
    by = {i.name: i for i in vz.VQA}
    assert vz.vqa_correct(by["count-circles"], "3") == (True, "3")
    assert vz.vqa_correct(by["count-circles"], "There are 4 circles") == (False, "4")
    assert vz.vqa_correct(by["word"], 'The word is "CAT".') == (True, "cat")
    assert vz.vqa_correct(by["number"], "83") == (False, "83")
    assert vz.mentions("A red circle.", "red", "circle") == []
    assert len(vz.mentions("A blue box.", "red", "circle")) == 2


@pytest.mark.parametrize(
    "reply,clean",
    [
        (vz.Reply(400, {"error": {"message": "too many images"}}, ""), True),
        (vz.Reply(413, {"error": "request body too large"}, ""), True),
        (vz.Reply(400, {"type": "error", "error": {"type": "x", "message": "no video"}}, ""), True),
        (vz.Reply(400, {"error": {}}, "{}"), False),
        (vz.Reply(500, {"error": {"message": "internal"}}, ""), False),
        (vz.Reply(200, {"choices": []}, ""), False),
        (vz.Reply(0, None, "the connection dropped"), False),
    ],
)
def test_a_refusal_is_a_4xx_with_a_message(reply, clean):
    assert (vz.refusal_problems(reply) == []) is clean


def test_request_builders():
    body = vz.openai_body("m", "Q?", [vz.image_part(b"\x89PNG")], max_tokens=8)
    content = body["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[-1] == {"type": "text", "text": "Q?"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0
    ab = vz.anthropic_body("m", "Q?", b"abc")
    block = ab["messages"][0]["content"][0]
    assert block["source"] == {"type": "base64", "media_type": "image/png", "data": "YWJj"}


# -- against the stand-in server -------------------------------------------------------------------------------------


def seen(**extra: str) -> dict:
    """The stand-in's answers: right ones for every drawn image, ``extra`` (item name -> reply) instead where given."""

    replies = {
        "count-circles": "3", "count-squares": "There are 5 squares.", "count-triangles": "two",
        "count-shapes": "4", "color-circle": "Yellow.", "color-square": "Green", "shape": "A triangle.",
        "digit": "7", "number": "38", "word": "CAT",
    }
    replies.update(extra)
    answers = {sha(i.draw()): replies[i.name] for i in vz.VQA}
    answers[sha(vz.smoke_circle())] = "A red circle on a white background."
    answers[sha(vz.smoke_square())] = "A blue square on white."
    answers[sha(vz.scene(*vz.SIZES["small"]))] = "A red circle, a blue square and a green triangle."
    answers[sha(vz.scene(*vz.SIZES["square"]))] = "Three shapes."
    answers[sha(vz.scene(*vz.SIZES["wide"]))] = "Three shapes in a row."
    return answers


GREEN_CALL = {"record_color": {"color": "green"}}


def cases(out: Path) -> dict:
    return {c["case"]: c for c in json.loads(out.read_text())["cases"]}


def test_smoke_passes_and_masks_the_key(fake, run_py, key, key_file, tmp_path):
    f = fake(key=key, echo=True, vision={"answers": seen()}, args=GREEN_CALL)
    out = tmp_path / "smoke.json"
    r = run_py("bench/vision.py", "smoke", "--base", f.url, "--api-key-file", key_file, "--out", out)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "smoke: 9/9 passed" in r.stdout
    got = cases(out)
    assert all(c["ok"] for c in got.values())
    assert got["refuse-count"]["detail"]["status"] == 400
    assert "at most 4 images" in got["refuse-count"]["detail"]["error"]
    assert key not in r.stdout + r.stderr + out.read_text()
    paths = [q["path"] for q in f.requests]
    assert "/v1/messages" in paths and "/health" in paths
    anthropic = next(q for q in f.requests if q["path"] == "/v1/messages")
    assert {k.lower(): v for k, v in anthropic["headers"].items()}.get("x-api-key") == key
    tool = next(b for b in f.bodies() if b.get("tools"))
    assert tool["tool_choice"] == "auto"
    sizes = [len(json.dumps(b)) for b in f.bodies()]
    assert max(sizes) > 10_000_000, "refuse-bytes sends an image over the 10 MB limit"


def test_smoke_fails_a_wrong_description_and_a_server_without_vision(fake, run_py, tmp_path):
    answers = seen()
    answers[sha(vz.smoke_circle())] = "A blue square."
    f = fake(vision={"answers": answers}, args=GREEN_CALL)
    out = tmp_path / "smoke.json"
    r = run_py("bench/vision.py", "smoke", "--base", f.url, "--out", out)
    assert r.returncode == 1
    assert cases(out)["describe"]["problems"] == ["the reply does not name red", "the reply does not name a circle"]
    f2 = fake(args=GREEN_CALL)  # no --vision: every image part is refused
    r = run_py("bench/vision.py", "smoke", "--base", f2.url, "--out", out)
    assert r.returncode == 1
    got = cases(out)
    assert not got["describe"]["ok"] and "needs the server's --vision" in got["describe"]["problems"][0]
    assert got["refuse-count"]["ok"] and got["alive"]["ok"]


@pytest.mark.parametrize("bad,case", [("oversize-500", "refuse-bytes"), ("video-200", "refuse-video"),
                                      ("count-200", "refuse-count")])
def test_smoke_fails_an_unclean_refusal(fake, run_py, tmp_path, bad, case):
    f = fake(vision={"answers": seen(), "misbehave": {bad}}, args=GREEN_CALL)
    out = tmp_path / "smoke.json"
    r = run_py("bench/vision.py", "smoke", "--base", f.url, "--out", out)
    assert r.returncode == 1
    got = cases(out)
    assert [n for n, c in got.items() if c["ok"] is False] == [case]
    assert "not a 4xx refusal" in got[case]["problems"][0]


def test_smoke_tool_case_follows_engine_api_tools(fake, run_py, tmp_path):
    out = tmp_path / "smoke.json"
    old = fake(vision={"answers": seen()}, args=GREEN_CALL)  # the family as imported refuses "required"
    r = run_py("bench/vision.py", "smoke", "--base", old.url, "--out", out, env={"ENGINE_API_TOOLS": "1"})
    assert r.returncode == 1 and not cases(out)["tool"]["ok"]
    new = fake(vision={"answers": seen()}, args=GREEN_CALL, api=fm.NEW_ENGINE)
    r = run_py("bench/vision.py", "smoke", "--base", new.url, "--out", out, env={"ENGINE_API_TOOLS": "1"})
    assert r.returncode == 0, r.stdout
    assert next(b for b in new.bodies() if b.get("tools"))["tool_choice"] == "required"
    wrong = fake(vision={"answers": seen()}, args={"record_color": {"color": "red"}})
    r = run_py("bench/vision.py", "smoke", "--base", wrong.url, "--out", out)
    assert r.returncode == 1 and cases(out)["tool"]["problems"] == ["color 'red', not green"]


def test_smoke_reads_the_limits_from_the_environment(fake, run_py, tmp_path):
    f = fake(vision={"answers": seen(), "max_images": 2, "urls": True}, args=GREEN_CALL)
    out = tmp_path / "smoke.json"
    r = run_py("bench/vision.py", "smoke", "--base", f.url, "--out", out,
               env={"VISION_MAX_IMAGES": "2", "VISION_URLS": "1"})
    assert r.returncode == 0, r.stdout
    got = cases(out)
    assert got["refuse-url"]["ok"] is None and "8/8 passed" in r.stdout
    counted = [b for b in f.bodies() if "How many images" in json.dumps(b)]
    assert len(counted[0]["messages"][0]["content"]) == 3 + 1  # three images and the prompt
    assert not any("https://" in json.dumps(b) for b in f.bodies())


def test_bench_reports_timing_tokens_and_vqa(fake, run_py, key, key_file, tmp_path):
    f = fake(key=key, vision={"answers": seen(**{"count-squares": "6"})})
    out = tmp_path / "bench.json"
    r = run_py("bench/vision.py", "bench", "--base", f.url, "--api-key-file", key_file, "--reps", "2",
               "--label", "t1", "--out", out)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads(out.read_text())
    assert rep["label"] == "t1" and rep["errors"] == 0
    timing = rep["timing"]
    assert [timing[s]["image_tokens"] for s in ("small", "square", "wide")] == [66, 578, 578]
    assert [timing[s]["estimated_tokens"] for s in ("small", "square", "wide")] == [64, 576, 576]
    assert all(timing[s]["ttft_s_median"] is not None and len(timing[s]["runs"]) == 2 for s in timing)
    q = rep["vqa"]
    assert (q["correct"], q["total"]) == (9, 10)
    assert [i["name"] for i in q["items"] if not i["ok"]] == ["count-squares"]
    assert "VQA 9/10" in r.stdout and key not in r.stdout + out.read_text()
    r = run_py("bench/vision.py", "bench", "--base", f.url, "--api-key-file", key_file, "--suites", "vqa",
               "--min-accuracy", "1", "--out", out)
    assert r.returncode == 1 and "timing" not in json.loads(out.read_text())


def test_bench_counts_refused_images_as_errors(fake, run_py, tmp_path):
    f = fake()  # no --vision
    out = tmp_path / "bench.json"
    r = run_py("bench/vision.py", "bench", "--base", f.url, "--reps", "1", "--sizes", "small", "--out", out)
    assert r.returncode == 1
    rep = json.loads(out.read_text())
    assert rep["timing"]["small"]["errors"] == 1 and rep["vqa"]["errors"] == 10


def test_no_server_is_exit_2(run_py):
    r = run_py("bench/vision.py", "smoke", "--base", "http://127.0.0.1:9", "--timeout", "5")
    assert r.returncode == 2 and "cannot query" in r.stderr
