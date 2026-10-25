#!/usr/bin/env bash
# Drafted decode must equal plain decode. Capture the same requests from a server started plain and from one started with a
# draft (dspark or DFlash2), then compare:
#   DRAFTS=0 ./start.sh restart                    # plain decode
#   scripts/check-drafted.sh capture /tmp/mm3-plain
#   DRAFTS=1 DRAFTER=dspark ./start.sh restart     # copies + the dspark draft
#   scripts/check-drafted.sh capture /tmp/mm3-drafted
#   scripts/check-drafted.sh compare /tmp/mm3-plain /tmp/mm3-drafted
#   DRAFTS=dflash2 ./start.sh restart              # copies + the DFlash2 draft (DFLASH2_DIR; docs/DFLASH2.md)
#   scripts/check-drafted.sh capture /tmp/mm3-dflash2
#   scripts/check-drafted.sh compare /tmp/mm3-plain /tmp/mm3-dflash2
# A drafting start needs the MEM_GATE_GIB scripts/quant_budget.py --serve --drafter dspark|dflash2 gives for the profile.
# Greedy requests (temperature 0) and keyed sampled ones (a fixed seed) must give the same text, finish reason and token count;
# the long prompts cross the 2048-row prompt chunks, and the repeated ones resume from the prompt cache (the draft's ring
# snapshot). Any difference is a bug in the draft path, never an acceptable loss of quality: the target verifies every draft.
# URL defaults to http://127.0.0.1:${PORT:-8000}; API_KEY_FILE (if the server asks for a key) is read, never printed.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[[ -f "$here/config.sh" ]] && source "$here/config.sh" 2>/dev/null || true

cmd=${1:-}
case "$cmd" in
  capture) out=${2:?capture needs an output folder}; url=${3:-http://127.0.0.1:${PORT:-8000}} ;;
  compare) a=${2:?compare needs two folders}; b=${3:?compare needs two folders} ;;
  *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

if [[ $cmd == capture ]]; then
  mkdir -p "$out"
  key=""
  # the file's first key, without a "label: " prefix (the form config.sh's api_header_file reads); kept out of argv
  [[ -n "${API_KEY_FILE:-}" && -r "${API_KEY_FILE:-}" ]] && key=$(head -n 1 "$API_KEY_FILE" | sed 's/^[^:]*: *//')
  MM3_URL=$url MM3_OUT=$out MM3_KEY=$key MM3_NAME=${SERVED_NAME:-MiniMax-M3.1-Flash-EXL3} python3 -I - <<'PY'
import json, os, sys, time, urllib.request

url, out, key, name = os.environ["MM3_URL"], os.environ["MM3_OUT"], os.environ["MM3_KEY"], os.environ["MM3_NAME"]

def filler(n):
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima"]
    return " ".join("%s%d" % (words[(i * 7 + i // 12) % 12], i % 97) for i in range(n))

long_doc = "Reference table (read it, then answer):\n" + "\n".join("row %d: %s" % (i, filler(14)) for i in range(420))
prompts = [
    ("code", "Write a Python module with a Fraction class (add, sub, mul, div, comparison, repr) and tests.", 0.0, None, 300),
    ("repeat", "Repeat this line exactly 12 times, numbered 1 to 12: 'the quick brown fox jumps over the lazy dog'.", 0.0, None, 260),
    ("json", "List 25 programming languages as a JSON array of objects with name, year and paradigm.", 0.0, None, 400),
    ("long-a", long_doc + "\n\nWhich row has the word kilo12 in it? Answer with the row number.", 0.0, None, 120),
    ("long-b", long_doc + "\n\nHow many rows are there? Answer with a number.", 0.0, None, 120),
    ("sampled", "Tell a short story about a lighthouse keeper and a storm.", 0.8, 11, 220),
    ("sampled-long", long_doc + "\n\nSummarize the table's pattern in two sentences.", 0.7, 5, 160),
]
for tag, text, temp, seed, limit in prompts:
    body = {"model": name, "messages": [{"role": "user", "content": text}], "max_tokens": limit, "temperature": temp, "stream": False}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = json.load(r)
    ch = data["choices"][0]
    rec = {"tag": tag, "text": ch["message"].get("content"), "finish": ch.get("finish_reason"), "tokens": data.get("usage", {}).get("completion_tokens"), "seconds": round(time.time() - t0, 2)}
    with open(os.path.join(out, tag + ".json"), "w") as f:
        json.dump(rec, f, indent=1)
    print("%-13s %4s tokens %7.2fs" % (tag, rec["tokens"], rec["seconds"]))
PY
else
  MM3_A=$a MM3_B=$b python3 -I - <<'PY'
import json, os, sys

a, b = os.environ["MM3_A"], os.environ["MM3_B"]
names = sorted(n for n in os.listdir(a) if n.endswith(".json"))
if not names or sorted(n for n in os.listdir(b) if n.endswith(".json")) != names:
    sys.exit("the two folders do not hold the same captures")
bad = 0
for n in names:
    ra, rb = json.load(open(os.path.join(a, n))), json.load(open(os.path.join(b, n)))
    same = all(ra[k] == rb[k] for k in ("text", "finish", "tokens"))
    print("%-13s %s  (%ss plain, %ss drafted)" % (n[:-5], "EQUAL" if same else "DIFFERENT", ra["seconds"], rb["seconds"]))
    if not same:
        bad += 1
        ta, tb = ra["text"] or "", rb["text"] or ""
        at = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        print("    first difference at character %d: %r vs %r" % (at, ta[at:at + 60], tb[at:at + 60]))
sys.exit(1 if bad else 0)
PY
fi
