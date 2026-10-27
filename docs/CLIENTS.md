# Clients: connecting to the server

The pair serves an OpenAI-compatible API on the head Spark. Everything below assumes the defaults of
`config/prod.env.example`; your operator may have changed the port, the context or the key.

| Setting | Value |
| --- | --- |
| Base URL | `http://127.0.0.1:8000/v1` (loopback on the head; see "Other machines") |
| Model | `MiniMax-M3.1-Flash-EXL3` (`SERVED_NAME`) |
| Context window | read it from `/v1/models` (`context_length`); 131,072 tokens on the measured profile (larger ones, with the KV tier, are opt-in: docs/OPERATIONS.md) |
| Max output | 32,768 tokens when a request sets none (`MAX_TOKENS`); a request may ask for more within the context |
| Requests at once | `PARALLEL` (1 on the measured profile); more wait in line |
| API key | none by default; with `API_KEY=1`, `Authorization: Bearer <key>` or `x-api-key: <key>` |

The context counts the prompt **plus** `max_tokens`: the reply is reserved up front, and a request that does not fit is
refused with HTTP 400 rather than cut. When memory does not fit the configured window, the server serves the most
it can and says so in that error and in `/v1/models`. Set your client's context window to the server's `context_length`, not to the
model's advertised maximum, and keep `max_tokens` modest on long prompts.

```bash
curl -s http://127.0.0.1:8000/v1/models     # id, context_length, max_model_len
curl -s http://127.0.0.1:8000/health        # liveness (scripts/serve.sh ready --json for the full verdict)
```

## Endpoints

| Route | Use |
| --- | --- |
| `POST /v1/chat/completions` | chat, streaming (`"stream": true`), tools, reasoning |
| `POST /v1/completions` | raw prompts (no chat template) |
| `GET /v1/models` | the served model and its context |
| `POST /v1/messages`, `/v1/messages/count_tokens` | an Anthropic-style Messages route, translated onto chat completions |
| `POST /v1/responses` | an OpenAI Responses-style route, translated onto chat completions |
| `POST /tokenize`, `/detokenize` | the model's tokenizer |
| `GET /health`, `/metrics` | liveness; Prometheus counters |

The chat route is the one the recipe's canary and harnesses check on every start (`scripts/canary.py`, `bench/`). The
Messages and Responses routes are the engine's translations and are not exercised by the recipe's tests: check your
client against them before relying on them.

## What a reply looks like

- `choices[0].message.content`: the answer.
- `choices[0].message.reasoning_content`: the model's thinking, when it thinks. MiniMax-M3.1's template is adaptive:
  the model decides per turn. The server keeps the thinking out of `content`; raw `<mm:think>` or `]<]minimax[>[` text in
  `content` is a server bug (please report it with the request).
- `choices[0].message.tool_calls`: parsed tool calls (OpenAI format, arguments as a JSON string).
- `usage`: prompt and completion tokens; `tensorfold`: the server's own figures (tokens a second for that request).

Sampling defaults when a request sets none: temperature 1.0, top_p 0.95 (the pack's generation config). Use
`"temperature": 0` for repeatable answers.

## Request options

| Field | Effect |
| --- | --- |
| `tools`, `tool_choice: "auto"` | tool calling; the model may answer with text or calls, several in one turn |
| `tool_choice: "required"` or a named function | forces a call through a grammar (\*) |
| `response_format` (`json_object`, `json_schema`) | JSON only, validated by a grammar (\*) |
| `chat_template_kwargs: {"enable_thinking": false}` | no thinking for this request (\*) |
| `reasoning_effort` | `low`, `medium`, `high`, `xhigh`, `max` (`minimal` runs as `low`; another word is HTTP 400) (\*) |
| `thinking_budget` | caps the thinking in tokens; the answer follows the close (\*) |
| `max_tokens`, `stop`, `seed`, `top_k`, `min_p` | as in the OpenAI API |
| `image_url` content parts | only when the operator runs `VISION=1`; otherwise HTTP 400 |

(\*) These depend on the engine build (the `mm3/api-tools` features). The release's measured engine passed them through
the server (README "Results": smoke with `json_schema` strict and a named tool choice, the canary's forced-tool and
thinking-off probes); an operator running another build sets `ENGINE_API_TOOLS` to say which (docs/PORT.md, Engine
features). A build without them answers HTTP 400 with the reason; treat that as "not available".

## Speed to expect

Measured on the pair (README "Results"; window-all-3 at CONTEXT 131,072 x 1 unless noted, window-all-5 / 6 at 114,688;
thinking off, 256-token replies):

| | Plain decode | DSpark drafts, depth by confidence (the baseline recipe) |
| --- | ---: | ---: |
| decode, code / tool calls / math (greedy) | ~20 tok/s | 31-34 tok/s |
| decode, prose (greedy) | ~21 tok/s | ~20 tok/s (prose drafts poorly) |
| decode after a 32K prompt | | 22.9 tok/s (window-all-5) |
| prefill, 2K / 32K prompt | | 786 / 720 tok/s (window-all-6) |
| time to first token, 32K prompt | | 66.7 s (window-all-3) |
| time to first token, the same 70K prompt again (prefix cache) | | 0.62 s (109 s fresh, window-all-6) |
| four streams at once (`PARALLEL=4`), code | 67.6-69.2 tok/s in all (window-all-6b) | 49.6 tok/s in all (window-all-5) |

Drafted replies equal plain ones (verified). A warm server start loads in ~54 s (~28 s of it is timing the draft's verify
costs). `tool_choice: "none"` no longer lets tool-call markup reach `content` (measured once, window-all-5). These are not guarantees: long
contexts, concurrency and the device's thermal state (a hot Spark runs at a lower GPU limit) change them. A long prompt
costs far more than a long reply: keep prompts short and stable (below).

## Talk to the server efficiently

Only one request (or `PARALLEL`) runs at a time, so wasted work on the client side becomes queueing for everyone.

- **Stream** (`"stream": true`) for anything interactive: the first tokens arrive as they are made, and a client that
  gives up can disconnect, which cancels the request on the server. A stream that has nothing to send yet (a long
  prefill) gets `: keepalive` comments, so an idle-timeout on the client or a proxy does not cut it.
- **Reuse one client** (one `OpenAI(...)` object, one HTTP session) instead of building one a request.
- **Through a reverse proxy**, turn response buffering off for `/v1/` (nginx: `proxy_buffering off;` or the
  `X-Accel-Buffering: no` header) and set its read timeout above your longest request: a 32K-token prompt takes about a
  minute before its first token, a 1M-token one far longer.
- **Timeouts**: give the client a read timeout that matches the prompt size (first token after roughly
  prompt_tokens / 500 seconds) rather than a short fixed one; a client that retries on timeout queues the same work twice.
- **Ask only for what you use**: a `max_tokens` near what the task needs (it is reserved against the context up front),
  `logprobs` off unless you read them, `n` 1, and `stream_options: {"include_usage": true}` only when you need usage on a
  stream.
- **Concurrency**: more client threads than `PARALLEL` only wait in line; size client pools to `PARALLEL`.

## Keep prompts reusable

The engine can keep the prompt states of recent requests (its prompt cache, exact-prefix reuse; engine work in progress) and resume a new request from the longest
matching prefix, so a follow-up turn of a long conversation would cost only its new tokens instead of the whole prompt
again (about 60 s at 32K tokens). **On this recipe's profiles no cache hit has been observed yet**: the server logs
`cached=0` for every request so far, even for two identical prompts (docs/KNOWN-ISSUES.md). The habits below are what any
reuse needs, cost nothing, and keep prompts small; check `cached=` in the server log (or
`usage.prompt_tokens_details.cached_tokens`) to see whether your deployment reuses prefixes. Reuse needs the start of
the prompt to be byte for byte the same as before:

- **Keep the system prompt and the tool list stable**: the same text, the same tools in the same order, the same JSON
  schemas with the same key order. Put anything that changes per request (the date, a request id, retrieved snippets)
  after the stable part, or in the latest user message, never at the top.
- **Append, do not rewrite**: keep earlier turns exactly as sent, including the assistant's reasoning and tool calls as
  the server returned them. Editing, reformatting or re-summarising old turns changes the prefix and loses the cache from
  that point on.
- **Bound the history deliberately**: when a conversation has to shrink, cut it rarely and in large steps (drop the
  oldest turns or replace them with one summary, then keep appending), not a sliding window that changes the start of the
  prompt on every turn. Truncating changes what the model sees: check that your task still works after a cut.
- **Bound tool results**: a tool that returns whole files or logs grows every later prompt; return what the task needs
  (a range, a match, a summary) and let the model ask for more.

## Client examples

**OpenAI Python SDK**

```python
import os

from openai import OpenAI

# api_key: any text without API_KEY=1; with it, the key file's value (read it from API_KEY_FILE, never hard-code it)
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key=os.environ.get("MM31_API_KEY", "<none>"))
r = client.chat.completions.create(
    model="MiniMax-M3.1-Flash-EXL3",
    messages=[{"role": "user", "content": "Reverse a linked list in Python."}],
    max_tokens=1024,
)
print(r.choices[0].message.content)
```

Read the key from its file in your own code (`API_KEY_FILE`, mode 600); never put it on a command line.

**opencode** (`opencode.json`, a provider entry):

```json
{
  "provider": {
    "minimax-spark": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8000/v1" },
      "models": {
        "MiniMax-M3.1-Flash-EXL3": { "limit": { "context": 131072, "output": 32768 } }
      }
    }
  }
}
```

**Anthropic Python SDK** (the `/v1/messages` route):

```python
import os

import anthropic

client = anthropic.Anthropic(base_url="http://127.0.0.1:8000", api_key=os.environ.get("MM31_API_KEY", "<none>"))
r = client.messages.create(
    model="MiniMax-M3.1-Flash-EXL3",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Reverse a linked list in Python."}],
    # thinking={"type": "enabled", "budget_tokens": 512},   # off when absent; adaptive lets the model choose
)
print(r.content[-1].text)
```

On this route thinking is **off** unless the request sends `thinking`: `{"type": "adaptive"}` lets the model decide per
turn, `{"type": "enabled", "budget_tokens": N}` asks for it with N under `max_tokens` (the budget needs the engine build
noted above). Thinking comes back as `thinking` blocks, tool calls as `tool_use` blocks, both streamed as typed events.
The base URL has no `/v1`: the SDK adds it. Any `model` name is served by the one model the server runs.

**Terminal coding tools.** Tools of this kind send a long, fixed system prompt and dozens of tool schemas on every
turn, and keep the whole session as history, so the "Keep prompts reusable" habits above matter most for them.

- OpenAI-compatible ones (opencode and similar): the provider entry below; the model's context from `/v1/models`.
- Ones that speak the Anthropic Messages API and read `ANTHROPIC_BASE_URL`: set it to `http://127.0.0.1:8000`, set
  `ANTHROPIC_API_KEY` to the key (or any text without `API_KEY=1`), and point every model setting the tool has (main,
  small or fast) at `MiniMax-M3.1-Flash-EXL3`; names the server does not know are served by it anyway, so a tool that
  sends its own model names still works.
- Give the tool the server's real window, not a larger one: a tool that thinks it has 1M tokens on a 131,072-token
  profile is refused when its history outgrows the server, instead of compacting first. Let the tool compact or summarise
  history itself when it approaches the window (see "Bound the history deliberately").
- Keep `max_tokens` (or the tool's output limit) at what a turn needs; 32,768 is the server default.
- With `PARALLEL=1` (the measured profile) one request runs at a time: a tool's background calls (titles, summaries, a second
  model for small tasks) wait behind the main one. Turn them off or point them elsewhere if latency matters.

**Continue, Cline, Open WebUI and similar**: choose an "OpenAI-compatible" provider, base URL
`http://127.0.0.1:8000/v1`, model `MiniMax-M3.1-Flash-EXL3`, context window from `/v1/models`, max output 32,768. If
the client shows a separate reasoning field, point it at `reasoning_content`.

## Other machines

The server listens on the head's loopback only and has no key by default. To serve other machines, the operator either
puts a reverse proxy with authentication in front of port 8000, or sets `HOST=0.0.0.0` together with `API_KEY=1`
(docs/OPERATIONS.md). Do not expose the port to a network you do not control.

## When something goes wrong

| Symptom | Likely cause |
| --- | --- |
| HTTP 400 "This server's maximum context length is N tokens ..." | prompt + `max_tokens` over `context_length`: shorten one of them (the wording is OpenAI's, so clients that compact on it do) |
| HTTP 400 naming `tool_choice`, `response_format`, `thinking_budget` or `reasoning_effort` | the engine build lacks that feature (above), or the value is not one it takes |
| HTTP 401 `invalid_api_key` | `API_KEY=1` and the key is missing or wrong |
| HTTP 503, or the connection is refused | the server is starting, stopped, or held by a thermal or memory stop; the operator runs `scripts/serve.sh status` |
| Slow replies for a while | another request is running (`PARALLEL`), or a Spark is hot and running at its lower GPU limit |
