# bench: clients for a running MiniMax-M3.1-Flash server

Standard-library Python (3.11+, runnable with `python3 -I`) and one bash script, run on the head node against the
server's OpenAI API. None of them has run against the real server yet: every number they print is a
measurement of the run that printed it, and nothing here claims a result.

| Harness | Measures | Run |
| --- | --- | --- |
| `toolcall.py` | MiniMax tool-call round trips: single, parallel (2-4 calls), nested objects, arrays, booleans, integers, floats, markup-like text, earlier calls and results in the history, live follow-up turns; `edit_replace_all`, `long_reasoning_tool` and `followup_answer` answer up to 2 preparatory calls (read, todowrite, glob) with canned results first and report `direct` for passes without one. Counts corruption (unknown tool, bad JSON, markup in a value), typing slips (`"3"` for 3, `"true"` for true) and raw-markup leaks in `content` | `python3 bench/toolcall.py --reps 2 --out R/toolcall.json [--mode off\|adaptive\|low\|high] [--temperature -1] [--stream]` |
| `tooleval/chains.py` | 6 multi-step chains with simulated tools (a result feeds the next call); 2 points a scenario, history sent back with `tool_calls` and `reasoning_content`; the simulated files keep what `write_file` writes | `python3 bench/tooleval/chains.py --mode off --out R/chains.json` |
| `tooleval/run.sh` | tool-eval-bench and spark-bench TrueScore at pinned commits (`setup` installs them under `~/.cache/tooleval`), plus `own` (chains.py) and `toolcall` | `bench/tooleval/run.sh setup`, then `bench/tooleval/run.sh teb\|sb\|both\|chains\|own\|toolcall off\|adaptive\|low\|high [label]` (`PY_TEMP`, `PY_REPS` for own / toolcall) |
| `structured.py` | JSON-schema and `json_object` replies (6 schemas, thinking off / on, single = serial = batched), tool-call shapes (`required`, named, `strict`, `parallel_tool_calls`, `none`, streamed `auto` and `required`) and thinking controls (off, `thinking_budget`, `reasoning_effort`). Engine features follow `ENGINE_API_TOOLS` (below); another refused feature is UNSUPPORTED (`--strict` fails both) | `python3 bench/structured.py --out R/structured.json [--suites schemas,tools,thinking]` |
| `quality.py` | MMLU-200 (`data/mmlu200.jsonl`) at 0 or N shots, a passkey needle at given sizes, `compare` of two runs | `python3 bench/quality.py mmlu --out R/mmlu.json`, `... needle --sizes 32768,100000 --out R/needle.json` |
| `stress.py` | one long prefill beside `--streams` - 1 decoding streams (default `$PARALLEL`, prod 1: the long prefill alone; decode tok/s while the long one prefills), and admission latency | `python3 bench/stress.py stress --out R/stress.json`, `... admit --tokens 512` |
| `vision.py` | image input (`VISION=1`, off by default; an engine with `mm3/vision`). `smoke`: a drawn red circle described over the OpenAI route, a blue square over `/v1/messages`, a green square with a `record_color` tool call (JSON arguments, `green`), then clean 4xx refusals of an image over 10 MB, one image over `VISION_MAX_IMAGES`, a video part, a GIF and an https URL (skipped with `VISION_URLS=1`), and the server still answering. `bench`: time to the first token, total time and the image's prompt tokens (against the same prompt without it) at 224x224, 672x672 and 2016x224, and 10 visual questions with exact answers (shape counts, colors, a shape, a digit, a number, a word) | `python3 bench/vision.py smoke --out R/vision-smoke.json`, `python3 bench/vision.py bench --reps 3 --out R/vision-bench.json [--label L]` |
| `drafts.py` | drafts off against the dspark draft (docs/DRAFTS.md): C1, thinking off, 256 tokens, four fixed classes (`data/drafts-prompts.json`: code, prose, structured, agent with a tool turn and its follow-up), greedy and keyed-sampled reps. Per request client TTFT and decode tok/s, `tensorfold.tokens_per_second`, `speculative.*`, `tensorfold.token_sha256`; with rank 0's log (`TF_MM3_DSPARK_STATS=1`) each request's `dspark acceptance stream:` line: acceptance by position (conditional, cumulative) and tau per class. Both nodes' device state and temperatures before and after each prompt. `compare`: speedup per class and token identity (a greedy mismatch fails); `log`: a saved rank 0 log's totals. `DRY_RUN=1` prints the requests | `python3 bench/drafts.py run --label drafts-off\|drafts-on --server-log-cmd 'docker logs -t minimax-m3-flash-tf' --out R/x.json`, `... compare --off R/off.json --on R/on.json` |
| `draft_positions.py` | drafts on against drafts off at C1 by draft position (docs/DFLASH2.md): 16 greedy requests in four classes (code, prose, structured, agent turns); per request tok/s, tau, acceptance, positions tried / kept, token_sha256; per class a_j = kept_j / tried_j, survival, tau, mean tok/s, the speed ratio and token equality against the plain run (a difference exits 1); `run --draft-off` sends `"draft": false`. `scripts/measure-drafts.sh` runs it in the window (docs/DFLASH2.md) | `python3 bench/draft_positions.py run --label dflash2 --out R/dflash2.jsonl`, `... report R/plain.jsonl R/dflash2.jsonl` |
| `soak.py` | mixed traffic for N minutes, 1 to `--workers` (4) concurrent (beyond `PARALLEL` they queue), cancels and disconnects; errors, leaks, latency per kind, slot drain, 17*23 at the end. `--kinds session` adds growing tool sessions | `python3 bench/soak.py --minutes 30 --out R/soak.json` |

`scripts/canary.py` is the post-start check (`scripts/serve.sh` runs it): a greedy chat, a thinking reply
with `reasoning_content`, a `get_weather` call parsed into `tool_calls`, a JSON reply; with `ENGINE_API_TOOLS=1` also a
`required` call and a thinking-off reply. It prints one `[canary] ok: ...` line or exits non-zero.

MiniMax specifics the harnesses follow (from the engine patch and the pack's template):

- Thinking: `chat_template_kwargs: {"enable_thinking": bool}`, effort in `reasoning_effort` (both in the kwargs and
  top level); reasoning comes back in `message.reasoning_content`. The pack's `chat_template.jinja` itself is adaptive
  (it reads no `thinking_mode`), so a "thinking off" request can still think: `quality.py` counts such replies.
- Raw markup in `content` (`]<]minimax[>[`, `<invoke name=`, `<tool_call>`, `<mm:think>`, `</mm:think>`) is a parser
  or template failure, counted by every harness and failing the canary.
- `tool_choice` "required" or a named function is refused by the family as imported, so every harness asks with
  `tool_choice` "auto" and a prompt that names the tool; the cases that send "required" follow `ENGINE_API_TOOLS`
  (below). A `response_format` grammar on the first token is refused too (StructuresUnsupported): `structured.py`
  follows the switch, `soak.py` and the canary (switch at 0) fall back to a JSON-only prompt.

## Engine features (`ENGINE_API_TOOLS`)

Required and named `tool_choice`, `response_format`, thinking off, `thinking_budget` and the five `reasoning_effort`
levels arrive with the engine branch `mm3/api-tools`. `ENGINE_API_TOOLS` (environment, exported by `scripts/config.sh`;
`--engine-api-tools [0|1]` on `structured.py`, `toolcall.py` and the canary) says whether the engine has them:

- `0` (default): a refusal of one of these, or the old engine's known behaviour (a thinking-off reply that still
  thinks, high / max run as xhigh), is `SKIP (engine lacks mm3/api-tools)` and fails nothing; the canary does not
  probe them.
- `1`: the same is a failure, and the canary adds a `required` probe and a thinking-off probe.

Either way a feature the engine accepts must behave, or its case fails. `bench/engine_api.py` holds the switch, the
engine's refusal texts (HTTP status and message, from the family patch and the branch) and the JSON-schema check.

| Case | Where | Passes when |
| --- | --- | --- |
| `tool_choice` "required" (thinking off / on) | `structured.py` tools `required/*`; `toolcall.py --tool-choice required`; canary `required` | >= 1 call, each an offered tool, arguments JSON valid against its schema, finish `tool_calls`, no `]<]minimax[>[` / `<invoke` / `<tool_call>` / think tag in content |
| named function, also with `parallel_tool_calls` false | tools `named/*`, `named-single/*` | only that function; exactly one call with parallel false |
| `response_format` json_schema and json_object (thinking off / on) | schemas `<schema>/*`, `json_object/*` | finish `stop`, content parses and validates, no markup in content, reasoning only in `reasoning_content` |
| thinking off (plain, with tools, with `response_format`) | thinking `off/plain`, every `*/chat` case of tools and schemas; canary `nothink` | no `reasoning_content`, an answer, no `</mm:think>` in content |
| `thinking_budget` 64 and 256 | thinking `budget/*` | reasoning tokens (`usage.completion_tokens_details.reasoning_tokens`, else 4 characters a token) <= budget + slack (`budget/8`, at least 16; half the budget + 32 when estimated); finish `stop` with an answer after the close |
| `reasoning_effort` low, medium, high, xhigh, max, and `minimal` | thinking `effort/*` | HTTP 200; `tensorfold.reasoning_effort`, when reported, is the level asked for (`minimal`: low) |
| streamed `required` call | tools `stream-required/chat` | the delta fragments concatenate to the arguments of the same request answered whole |

The engine's narrowing for required calls (untyped values as text, objects with their named properties only, string
patterns / formats / lengths dropped, a tool without parameters refused) does not touch the harnesses' tools: each has
typed, named parameters.

## Settings and keys

Defaults come from the environment `scripts/config.sh` exports: `HOST` / `PORT` (base URL, a wildcard host is
reached on 127.0.0.1; else `http://127.0.0.1:8000`), `SERVED_NAME` (else `MiniMax-M3.1-Flash-EXL3`) and
`API_KEY_FILE`. Each harness takes `--base`, `--model`, `--api-key-file` to override them; `run.sh` reads
`BASE_URL` and `MODEL`.

The key never goes on a command line, into a saved file or onto the terminal:

- The Python harnesses read the first key of the file (one key, or `label: key`, a line; used when the file exists,
  required when `API_KEY=1`), send it as `Authorization: Bearer`, and mask it in every reply as it arrives and in
  everything they print or save.
- `run.sh` reads it into its own shell: curl gets it as a header on a file descriptor, tool-eval-bench as
  `TOOL_EVAL_API_KEY` and spark-bench as `SPARK_BENCH_API_KEY` in their environment. Saved command lines go through
  `scripts/redact.py argv`, logs through `scripts/redact.py text`, and `scripts/redact.py scrub` masks the run
  directory (the benches' own reports and request dumps) when the run ends. `API_KEY` must be 0 or 1 here; a key
  placed in it is refused, not echoed.

## Images (`vision.py`)

Every image is drawn at run time by `vision.py` itself (a stdlib PNG writer and a 5x7 bitmap font), so no image file is
committed and the same run draws the same bytes. Requests go with thinking off, greedy, and a nonce in the prompt; only
`content` is scored. A refusal passes when it is an HTTP 4xx other than 401 / 403 with an error message; a 5xx, a 200, a
dropped connection or a server that stops answering afterwards fails it. `--max-images` (default `$VISION_MAX_IMAGES`, 4),
`--max-image-mb` (10, the engine's limit) and `--vision-urls` (`$VISION_URLS`) must match the server's settings. The tool
case sends `tool_choice` "required" with `ENGINE_API_TOOLS=1`, else "auto". Image tokens are `usage.prompt_tokens` less the
same prompt's without the image (sent just before it); MiniMax gives an image at most 576 tokens (672 x 672 pixels, 28 x 28
pixels a token) plus its markers, and more fails the run. The VQA accuracy is reported, not a gate, unless
`--min-accuracy` is given.

## Tests

`python3 -m pytest -q tests/bench` runs every harness against `tests/bench/fake_minimax.py`, a stand-in server on
127.0.0.1 that answers as the MiniMax server does and can be told to leak raw markup, type every value as a string,
refuse `response_format`, or echo the request's key into its replies (to prove the masking). With `vision` it takes
images as `--vision` does (limits, refusals, MiniMax's token geometry, the Anthropic route) and can be told to answer an
over-large image with a 500 or serve a video part (`tests/bench/test_vision.py`). Its `api` switch makes it
refuse each engine feature as the family as imported does, serve it, or serve it wrongly (another function, invalid
JSON, a leaked `</mm:think>`, reasoning over the budget, broken stream deltas): `tests/bench/test_engine_api.py` checks
that each case skips or fails a refusal by `ENGINE_API_TOOLS`, passes a correct service and fails a wrong one in both
modes. `tests/bench/test_drafts.py` runs `drafts.py` against its own stand-in that streams as the engine does and
writes a rank 0 log: the acceptance lines parsed and matched to their requests, the compare arithmetic, a greedy
mismatch failing, the reference stop, a gzipped log, DRY_RUN. CPU only, no model.
