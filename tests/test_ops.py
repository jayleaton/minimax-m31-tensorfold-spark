"""CPU-only checks of the pair's operations: the memory gate's rule, the prepared per-rank weights, the boot timing record,
health and readiness, log rotation with its thermal note, the boot start, the last good image, rollback and upgrade, and
the systemd units.

    tests/run.sh      (or: python3 -m pytest -q tests)

No GPU, network, Docker or SSH: docker, ssh, curl, systemctl and nvidia-smi are fakes on PATH, and the device's limits and
state are files under a temporary directory.
"""

import gzip
import json
import os
import re
import time
import unittest
from pathlib import Path
from typing import ClassVar

from test_repo import EXAMPLE, ROOT, ScriptFixture, fake

GIB_KB = 1024 * 1024
STARTED = "2026-10-25T13:07:28.123456789Z"


class OpsFixture(ScriptFixture):
    """ScriptFixture with this recipe's state under the sandbox's HOME, and a fake curl and systemctl."""

    def setUp(self):
        super().setUp()
        self.state = self.root / ".local" / "state" / "tensorfold-minimax-m3"
        self.logs = self.root / ".cache" / "tensorfold-minimax-m3" / "logs"
        fake(self.bin, "systemctl", f'echo "systemctl $*" >> {self.calls}\nexit 1\n')
        fake(self.bin, "curl", f'echo "curl $*" >> {self.calls}\nexit 7\n')

    def ops(self, snippet, **extra):
        return self.bash(f"source scripts/ops.sh; {snippet}", **extra)

    def serve(self, *argv, **extra):
        return self.run_script(ROOT / "scripts" / "serve.sh", *argv, **extra)

    def called(self):
        return self.calls.read_text() if self.calls.exists() else ""


class MemGateTests(OpsFixture, unittest.TestCase):
    """The gate requires what the profile needs: MemAvailable after a drop, and a MemFree floor MEM_GATE_SLACK_GIB under it."""

    def check(self, free_gib, avail_gib, gate=116, slack=4):
        snippet = f"MEM_GATE_GIB={gate}; MEM_GATE_SLACK_GIB={slack}; mem_gate_check {int(free_gib * GIB_KB)} {int(avail_gib * GIB_KB)}"
        return self.bash(snippet)

    def test_the_measured_case_passes_now(self):
        # today's two slow starts: MemFree 112-113 GiB after the drop, the rest reclaimable, gate 116: no 4-6 minute wait
        result = self.check(112.4, 117.0)
        self.assertEqual((result.returncode, result.stdout), (0, ""), result.stderr)

    def test_memfree_under_the_floor_waits_and_says_why(self):
        result = self.check(110.0, 117.0)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "MemFree 110.0 GiB < 112 (MEM_GATE_GIB - MEM_GATE_SLACK_GIB 4)")

    def test_memavailable_under_the_gate_waits_and_says_why(self):
        result = self.check(115.0, 115.5)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout.strip(), "MemAvailable 115.5 GiB < MEM_GATE_GIB 116")
        both = self.check(100.0, 101.0)
        self.assertIn("MemAvailable 101.0 GiB < MEM_GATE_GIB 116; MemFree 100.0 GiB < 112", both.stdout)

    def test_the_edges_and_an_unreadable_node(self):
        self.assertEqual(self.check(112, 116).returncode, 0, "exactly the gate and the floor pass")
        self.assertEqual(self.check(112, 116, slack=0).returncode, 1)
        unreadable = self.bash("MEM_GATE_GIB=116; mem_gate_check '' ''")
        self.assertEqual((unreadable.returncode, unreadable.stdout.strip()), (1, "meminfo unreadable"))

    def gate(self, here, worker, **extra):
        snippet = ("drop_caches() { echo dropped >&2; }; "
                   f"meminfo_kb() {{ if [[ $1 == head ]]; then echo '{here}'; else echo '{worker}'; fi; }}; "
                   'MEM_GATE_GIB=116; MEM_GATE_TIMEOUT=0; if mem_gate; then echo "PASS $HERE_GB $WORKER_GB"; '
                   'else echo "WAIT $MEM_GATE_WHY"; fi')
        return self.bash(snippet, **extra)

    def test_the_loop_passes_or_names_the_node_and_the_reason(self):
        ok = f"{int(113 * GIB_KB)} {int(117 * GIB_KB)}"
        result = self.gate(ok, ok)
        self.assertIn("PASS 113 113", result.stdout, result.stderr)
        self.assertIn("Memory gate passed", result.stdout)
        short = f"{int(109.5 * GIB_KB)} {int(117 * GIB_KB)}"
        result = self.gate(ok, short)
        self.assertIn("WAIT user@203.0.113.2: MemFree 109.5 GiB < 112", result.stdout, result.stderr)
        self.assertNotIn("here:", result.stdout)
        result = self.gate(ok, "")
        self.assertIn("WAIT user@203.0.113.2: meminfo unreadable", result.stdout)

    def test_start_checks_the_slack(self):
        gate = re.search(r"^MEM_GATE_GIB=(\d+)", EXAMPLE.read_text(), re.MULTILINE).group(1)   # a slack not under the gate
        result = self.run_script(ROOT / "start.sh", DRY_RUN="1", MEM_GATE_SLACK_GIB=gate)
        self.assertEqual(result.returncode, 1)
        self.assertIn("MEM_GATE_SLACK_GIB is a whole number of GiB under MEM_GATE_GIB", result.stderr)

    def test_the_documented_gate_is_the_profiles(self):
        # AGENTS.md section 7, docs/OPERATIONS.md section 3 and the example name the same production gate
        gate = re.search(r"^MEM_GATE_GIB=(\d+)", EXAMPLE.read_text(), re.MULTILINE).group(1)
        self.assertIn(f"`MEM_GATE_GIB={gate}`", (ROOT / "AGENTS.md").read_text())
        self.assertIn(f"| `MEM_GATE_GIB` | {gate} GiB", (ROOT / "docs" / "OPERATIONS.md").read_text())
        self.assertRegex(EXAMPLE.read_text(), r"(?m)^# MEM_GATE_SLACK_GIB=4 ")


class PreparedTests(OpsFixture, unittest.TestCase):
    PREP_ID = "0123456789abcdef"

    def folder(self, *names, mtime_step=100):
        d = self.root / ".cache" / "tensorfold-minimax-m3" / "prepared"
        d.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for i, name in enumerate(names):
            (d / name).write_bytes(b"x" * (i + 1))
            os.utime(d / name, (now - mtime_step * (len(names) - i), now - mtime_step * (len(names) - i)))
        return d

    def test_the_switch_reaches_both_ranks_with_one_mount_each(self):
        env = self.bash('echo "$TF_MM3_PREPARED $TF_MM3_PREPARED_WRITE $(prepared_here)"', PREPARED="1")
        self.assertEqual(env.stdout.split(), ["/prepared", "1", f"{self.root}/.cache/tensorfold-minimax-m3/prepared"],
                         env.stderr)
        result = self.run_script(ROOT / "start.sh", DRY_RUN="1", PREPARED="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout
        rank1, rank0 = out[out.index("[dry-run] rank 1"):out.index("[dry-run] rank 0")], out[out.index("[dry-run] rank 0"):]
        for rank in (rank0, rank1):
            self.assertIn("TF_MM3_PREPARED=/prepared", rank)
            self.assertIn("TF_MM3_PREPARED_WRITE=1", rank)
            self.assertIn(":/prepared ", rank)
        self.assertIn(f"-v {self.root}/.cache/tensorfold-minimax-m3/prepared:/prepared ", rank0)
        off = self.run_script(ROOT / "start.sh", DRY_RUN="1").stdout
        self.assertNotIn("/prepared", off)

    def test_a_leading_tilde_is_each_nodes_own_home(self):
        here = self.bash("prepared_here", HEAD_PREPARED="~/pp")
        self.assertEqual(here.stdout, f"{self.root}/pp\n", here.stderr)
        fake(self.bin, "ssh", f'echo "ssh $*" >> {self.calls}\nprintf /var/lib/rootish\n')
        worker = self.bash("worker_prepared_dir")
        self.assertEqual(worker.stdout, "/var/lib/rootish/.cache/tensorfold-minimax-m3/prepared\n", worker.stderr)
        self.assertEqual(self.bash("worker_prepared_dir", WORKER_PREPARED="/srv/prep").stdout, "/srv/prep\n")

    def test_the_variables_without_the_switch_are_refused(self):
        result = self.bash("true", TF_MM3_PREPARED="/prepared")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("without PREPARED=1", result.stderr)
        self.assertIn("is 0 or 1", self.bash("true", PREPARED="yes").stderr)
        self.assertIn("an absolute path or one under ~/", self.bash("true", WORKER_PREPARED="rel/dir").stderr)

    def test_stale_files_are_all_but_the_newest_of_each_rank(self):
        lines = "\n".join([f"F mm3-r0of2-{'a' * 16}.tfprep 10 1000.5", f"F mm3-r0of2-{'b' * 16}.tfprep 10 2000.1",
                           f"F mm3-r1of2-{'c' * 16}.tfprep 10 1500.0", f"F mm3-r0of1-{'d' * 16}.tfprep 10 900.0",
                           "F notes.txt 3 3000.0", "F mm3-r0of2-short.tfprep 1 4000.0", "D 123456789"])
        stale = self.bash(f"prepared_stale <<'EOF'\n{lines}\nEOF")
        self.assertEqual(stale.stdout.split(), [f"mm3-r0of2-{'a' * 16}.tfprep"], stale.stderr)
        every = self.bash(f"prepared_stale all <<'EOF'\n{lines}\nEOF").stdout.split()
        self.assertEqual(sorted(every), sorted(f"mm3-{r}-{k * 16}.tfprep" for r, k in
                                               (("r0of2", "a"), ("r0of2", "b"), ("r1of2", "c"), ("r0of1", "d"))))

    def test_the_node_reports_its_own_ranks_newest_file_or_its_free_space(self):
        none = self.bash("prepared_node head", PREPARED="1")
        self.assertRegex(none.stdout.strip(), r"^none [0-9]+$", none.stderr)
        self.assertFalse((self.root / ".cache" / "tensorfold-minimax-m3" / "prepared").exists(), "read-only without mkdir")
        self.folder(f"mm3-r0of2-{'1' * 16}.tfprep", f"mm3-r1of2-{'2' * 16}.tfprep", f"mm3-r0of2-{self.PREP_ID}.tfprep")
        have = self.bash("prepared_node head", PREPARED="1")
        self.assertRegex(have.stdout.strip(), rf"^file mm3-r0of2-{self.PREP_ID}\.tfprep 3 [0-9]+$", have.stderr)

    def test_list_and_clean_touch_only_the_engines_files(self):
        d = self.folder(f"mm3-r0of2-{'1' * 16}.tfprep", "keep-me.txt", f"mm3-r0of2-{self.PREP_ID}.tfprep")
        listed = self.serve("prepared", "list", PREPARED="1")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertRegex(listed.stdout, rf"head +mm3-r0of2-{'1' * 16}\.tfprep .* stale")
        self.assertRegex(listed.stdout, rf"head +mm3-r0of2-{self.PREP_ID}\.tfprep .* newest of rank 0")
        cleaned = self.serve("prepared", "clean", PREPARED="1")
        self.assertEqual(cleaned.returncode, 0, cleaned.stderr)
        self.assertEqual(sorted(p.name for p in d.iterdir()), sorted(["keep-me.txt", f"mm3-r0of2-{self.PREP_ID}.tfprep"]))
        self.serve("prepared", "clean", "all", PREPARED="1")
        self.assertEqual([p.name for p in d.iterdir()], ["keep-me.txt"])
        self.assertIn("usage", self.serve("prepared", "wipe").stderr)


class BootTimesTests(OpsFixture, unittest.TestCase):
    MARKS: ClassVar[dict] = {"begin": 1000.0, "gate": 1010.0, "gate_done": 1010.5, "launch": 1011.0, "answered": 1035.0,
             "smoke": 1035.4, "smoke_done": 1036.0, "live": 1066.0}

    def record(self, marks, rc=0):
        fake(self.bin, "docker", 'case "$*" in logs*) echo "x [tensorfold] prepared: read mm3-r0of2-k.tfprep 93.8 GiB" ;;\n'
                                  '  "image inspect"*) echo "[sha256:aa] [PATH=/x]" ;; *) exit 1 ;; esac\n')
        fake(self.bin, "ssh", 'case "$*" in *"docker logs"*) echo "y [tensorfold] prepared: wrote mm3-r1of2-k.tfprep" ;;\n'
                               "  *) exit 255 ;; esac\n")
        f = self.root / "marks"
        f.write_text("".join(f"{k} {v}\n" for k, v in marks.items()))
        self.state.mkdir(parents=True, exist_ok=True)
        result = self.ops(f"ops_boot_record {f} restart {rc}")
        path = self.state / "boot-times.jsonl"
        return result, [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_a_live_start_has_every_phase(self):
        result, recs = self.record(self.MARKS)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["seconds"], {"checks": 10.5, "gate": 0.5, "load": 24.0, "smoke": 1.0, "first_reply": 0.6,
                                        "canary": 30.0, "total": 66.0})
        self.assertEqual((r["ok"], r["failed_at"], r["mode"]), (True, None, "restart"))
        self.assertEqual(r["prepared"], {"r0": "read", "r1": "wrote"})
        self.assertEqual((r["image"], r["model_revision"]), ("tensorfold-minimax-m3:v1.1.2",
                                                             "5bff3d732e9dfa236a41b6f3054dd8550bef595f"))
        self.assertRegex(r["image_id"], r"^[0-9a-f]{12}$")
        self.assertRegex(r["utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertIn("boot times (restart): checks 10.5 s, gate wait 0.5 s, load 24.0 s, smoke 1.0 s (first reply 0.6 s), "
                      "canary 30.0 s, total 66.0 s; prepared r0 read, r1 wrote", result.stdout)

    def test_a_start_that_stopped_at_the_gate_is_recorded_as_such(self):
        result, recs = self.record({"begin": 1000.0, "gate": 1009.0}, rc=1)
        self.assertEqual(recs[0]["ok"], False)
        self.assertEqual(recs[0]["failed_at"], "the memory gate")
        self.assertEqual(recs[0]["seconds"]["total"], 9.0)
        self.assertIn("FAILED at the memory gate", result.stdout)

    def test_a_start_refused_before_the_gate_records_nothing(self):
        _, recs = self.record({"begin": 1000.0})
        self.assertEqual(recs, [])

    def test_boot_times_prints_the_last_starts(self):
        self.record(self.MARKS)
        self.record({"begin": 1000.0, "gate": 1300.0}, rc=1)
        out = self.serve("boot-times", "5")
        self.assertEqual(out.returncode, 0, out.stderr)
        lines = out.stdout.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertRegex(lines[1], r" yes +10\.5 +0\.5 +24\.0 +1\.0 +30\.0 +66\.0 read/wrote ")
        self.assertIn("(failed at the memory gate)", lines[2])
        self.assertIn("a count of starts", self.serve("boot-times", "x").stderr)

    def test_start_marks_every_phase_and_up_is_start(self):
        text = (ROOT / "start.sh").read_text()
        for phase in ("gate", "gate_done", "launch", "answered", "smoke", "smoke_done", "live"):
            self.assertRegex(text, rf"(?m)^\s*mark {phase}$")
        serve = (ROOT / "scripts" / "serve.sh").read_text()
        self.assertIn("start|up|restart)", serve)
        self.assertIn('ops_boot_record "$marks" "$mode" "$rc"', serve)


class HealthReadyTests(OpsFixture, unittest.TestCase):
    def pair(self, healthy=True):
        fake(self.bin, "docker", 'case "$*" in *Running*) echo true ;; *StartedAt*) echo ' + STARTED + ' ;; *) exit 1 ;; esac\n')
        fake(self.bin, "ssh", 'case "$*" in *Running*) echo true ;; *thermal-stop*) ;; */state*) echo normal ;; *) exit 0 ;; esac\n')
        code = 200 if healthy else 503
        fake(self.bin, "curl", 'url=${@: -1}\ncase "$url" in\n'
                               f'  */health) printf \'{{"ok":true}}\'; [[ "$*" != *-w* ]] || printf "\\n{code}" ;;\n'
                               '  */v1/models) printf \'{"data":[{"id":"MiniMax-M3.1-Flash-EXL3"}]}\' ;;\nesac\n')
        self.state.mkdir(parents=True, exist_ok=True)

    def canary(self, started=STARTED, result="pass"):
        (self.state / "canary-last").write_text(f"started={started} result={result} at=2026-10-25T13:08:30Z\n")

    def test_ready_when_every_check_passes(self):
        self.pair()
        self.canary()
        human = self.serve("ready")
        self.assertEqual(human.returncode, 0, human.stdout + human.stderr)
        self.assertIn("ready: ok", human.stdout)
        machine = self.serve("ready", "--json")
        self.assertEqual(machine.returncode, 0)
        verdict = json.loads(machine.stdout)
        self.assertEqual((verdict["check"], verdict["ok"]), ("ready", True))
        self.assertEqual(set(verdict["checks"]), {"rank0", "rank1", "health", "models", "canary", "thermal_head",
                                                  "thermal_worker", "thermal_stop", "failfast"})

    def test_not_ready_without_this_starts_canary(self):
        self.pair()
        self.canary(started="2026-10-24T01:00:00Z")
        result = self.serve("ready", "--json")
        self.assertEqual(result.returncode, 1)
        checks = json.loads(result.stdout)["checks"]
        self.assertFalse(checks["canary"]["ok"])
        self.assertIn("no canary record for this rank 0 container", checks["canary"]["detail"])
        self.canary(result="fail")
        self.assertIn("ready: canary: the canary of this start: fail", self.serve("ready").stdout)

    def test_not_ready_on_a_hot_node_or_a_record(self):
        self.pair()
        self.canary()
        (self.thermal_run / "state").write_text("throttled\n")
        checks = json.loads(self.serve("ready", "--json").stdout)["checks"]
        self.assertEqual(checks["thermal_head"], {"ok": False, "detail": "device thermal state throttled"})
        (self.thermal_run / "state").write_text("normal\n")
        (self.state / "failfast").write_text("1700000000 rank 1 exited with code 137\n")
        result = self.serve("ready")
        self.assertEqual(result.returncode, 1)
        self.assertIn("ready: failfast: rank 1 exited with code 137", result.stdout)

    def test_health_is_liveness_only(self):
        self.pair()
        result = self.serve("health", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(json.loads(result.stdout)["checks"]), {"rank0", "rank1", "health", "models"})
        self.assertIn("health: ok (both ranks up, /health 200, MiniMax-M3.1-Flash-EXL3 listed)", self.serve("health").stdout)
        self.pair(healthy=False)
        down = self.serve("health")
        self.assertEqual(down.returncode, 1)
        self.assertIn("health: health: /health 503", down.stdout)


class LogRotateTests(OpsFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.limits = self.root / "thermal.env"
        # values no default holds, so a number in the note can only have come from the node's own file
        self.limits.write_text("GUARD_PAUSE_C=81\nGUARD_RESUME_C=75\nGUARD_KILL_C=88\nGUARD_KILL_HOLD_S=12\n")
        zone = self.root / "zone"
        zone.mkdir()
        (zone / "temp").write_text("71000\n")
        self.extra = {"SPARK_THERMAL_ENV": str(self.limits), "MEMGUARD_THERMAL_GLOB": str(zone / "temp"),
                      "LOG_ROTATE_MIB": "1", "LOG_KEEP": "2"}

    def rotate(self, **extra):
        return self.run_script(ROOT / "scripts" / "logrotate.sh", **{**self.extra, **extra})

    def test_copytruncate_keeps_log_keep_generations(self):
        self.logs.mkdir(parents=True)
        log = self.logs / "memguard.log"
        for n in range(3):
            log.write_bytes(str(n).encode() * (1024 * 1024 + 10))
            result = self.rotate()
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(log.stat().st_size, 0, "truncated in place, the same file")
        self.assertEqual(gzip.decompress((self.logs / "memguard.log.1.gz").read_bytes())[:1], b"2")
        self.assertEqual(gzip.decompress((self.logs / "memguard.log.2.gz").read_bytes())[:1], b"1")
        self.assertFalse((self.logs / "memguard.log.3.gz").exists())
        small = self.logs / "pairwatch.log"
        small.write_text("short\n")
        self.rotate()
        self.assertEqual(small.read_text(), "short\n")
        self.assertIn("worker's memguard.log was not rotated", self.rotate().stderr)

    def test_the_dry_run_changes_nothing(self):
        self.logs.mkdir(parents=True)
        (self.logs / "memguard.log").write_bytes(b"x" * (2 * 1024 * 1024))
        result = self.run_script(ROOT / "scripts" / "logrotate.sh", "--dry-run", **self.extra)
        self.assertIn("would rotate", result.stdout)
        self.assertEqual((self.logs / "memguard.log").stat().st_size, 2 * 1024 * 1024)
        self.assertFalse((self.logs / "thermal.log").exists())

    def test_the_thermal_note_takes_the_limits_from_each_nodes_file(self):
        self.rotate()
        lines = (self.logs / "thermal.log").read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r"Z head state normal, hottest 71 C, limits pause 81 C, resume 75 C, stop 88 C held 12 s "
                                   r"\(.*thermal\.env sha256 [0-9a-f]{12}\)$")
        self.assertRegex(lines[1], r"Z worker unreachable over ssh$")
        # the worker's file lacks a limit: the note says so, and no number stands in for it
        fake(self.bin, "ssh", 'case "$*" in *"cat "*thermal.env) printf "GUARD_PAUSE_C=81\\nGUARD_RESUME_C=75\\n'
                               'GUARD_KILL_HOLD_S=12\\n" ;; */state*) echo normal ;; *sha256sum*) echo ' + "f" * 64 +
                               ' ;; *) exit 0 ;; esac\n')
        self.rotate()
        worker = (self.logs / "thermal.log").read_text().splitlines()[-1]
        self.assertIn("worker state normal, hottest unreadable, limits unreadable (no whole GUARD_KILL_C)", worker)
        self.assertNotIn("stop", worker.split("limits", 1)[1])

    def test_no_limit_value_is_written_in_the_script(self):
        text = (ROOT / "scripts" / "logrotate.sh").read_text()
        self.assertFalse(re.findall(r"GUARD_[A-Z_]+=\d|\b(8[0-9]|9[0-9])\b", text), "no temperature in the script")


class BootStartTests(OpsFixture, unittest.TestCase):
    def boot(self, **extra):
        return self.run_script(ROOT / "scripts" / "boot-start.sh", BOOT_STDOUT="1", BOOT_DRY_RUN="1", BOOT_WAIT="0", **extra)

    def serving(self, boot_id):
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "serving").write_text(f"{boot_id} 1729860000\n")

    def test_not_serving_before_means_no_start(self):
        result = self.boot()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("was not serving before this boot", result.stdout)
        self.assertNotIn("docker", self.called())

    def test_a_marker_of_this_boot_means_started_already(self):
        self.serving(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        self.assertIn("the serving marker is from this boot", self.boot().stdout)

    def test_serving_before_the_reboot_waits_for_the_worker_and_the_link(self):
        self.serving("an-earlier-boot")
        fake(self.bin, "docker", f'echo "docker $*" >> {self.calls}\ncase "$*" in info*) exit 0 ;; *) exit 1 ;; esac\n')
        result = self.boot()
        self.assertEqual(result.returncode, 1)
        self.assertIn("the pair was serving before this reboot", result.stdout)
        self.assertIn("not ready after 0s: leaving it to the watchdog", result.stdout)
        self.assertNotIn("serve.sh", self.called())

    def test_rank_0_running_or_a_fresh_lease_means_nothing(self):
        self.serving("an-earlier-boot")
        fake(self.bin, "docker", 'case "$*" in info*) exit 0 ;; *Running*) echo true ;; *) exit 1 ;; esac\n')
        self.assertIn("rank 0 already running", self.boot().stdout)
        lease = self.root / "lease"
        lease.touch()
        self.assertIn("is fresh: not starting", self.boot(BOOT_LEASE=str(lease)).stdout)

    def test_stop_keeps_the_marker_only_at_a_system_shutdown(self):
        text = (ROOT / "stop.sh").read_text()
        self.assertIn('"$(systemctl is-system-running 2>/dev/null)" == stopping', text)
        self.assertIn('rm -f "$STATE_DIR/serving"', text)
        self.assertIn('> "$STATE_DIR/serving"', (ROOT / "start.sh").read_text())


class LastGoodTests(OpsFixture, unittest.TestCase):
    def image_fakes(self, worker_layers="[sha256:aa]"):
        fake(self.bin, "docker", f'echo "docker $*" >> {self.calls}\ncase "$*" in *StartedAt*) echo {STARTED} ;;\n'
                                  '  "image inspect -f {{.RootFS"*) echo "[sha256:aa]" ;; "image inspect"*) echo abc ;;\n'
                                  '  tag*) exit 0 ;; *) exit 1 ;; esac\n')
        fake(self.bin, "ssh", f'echo "ssh $*" >> {self.calls}\ncase "$*" in *"image inspect"*) echo "{worker_layers}" ;;\n'
                               '  *"docker tag"*) exit 0 ;; *) exit 1 ;; esac\n')
        self.state.mkdir(parents=True, exist_ok=True)

    def test_image_repo(self):
        for image, repo in (("tensorfold-minimax-m3:v1.1.2", "tensorfold-minimax-m3"), ("ghcr.io/a/b:t", "ghcr.io/a/b"),
                            ("reg:5000/a/b:t", "reg:5000/a/b"), ("reg:5000/a/b", "reg:5000/a/b"), ("a/b@sha256:ff", "a/b")):
            with self.subTest(image=image):
                self.assertEqual(self.ops(f"image_repo {image}").stdout.strip(), repo)

    def test_a_passed_canary_records_and_tags_the_last_good_on_both_nodes(self):
        self.image_fakes()
        (self.state / "canary-last").write_text(f"started={STARTED} result=pass at=x\n")
        result = self.ops("ops_last_good_record")
        self.assertEqual(result.returncode, 0, result.stderr)
        good = dict(line.split("=", 1) for line in (self.state / "last-good.env").read_text().splitlines()
                    if not line.startswith("#"))
        self.assertEqual(good["IMAGE"], "tensorfold-minimax-m3:v1.1.2")
        self.assertEqual(good["LAST_GOOD_IMAGE"], "tensorfold-minimax-m3:last-good")
        self.assertEqual(good["MODEL_REVISION"], "5bff3d732e9dfa236a41b6f3054dd8550bef595f")
        self.assertRegex(good["IMAGE_IDENT"], r"^[0-9a-f]{64}$")
        calls = self.called()
        self.assertIn("docker tag tensorfold-minimax-m3:v1.1.2 tensorfold-minimax-m3:last-good", calls)
        self.assertIn("docker tag 'tensorfold-minimax-m3:v1.1.2' 'tensorfold-minimax-m3:last-good'", calls)
        self.assertEqual(self.ops("ops_last_good_check").returncode, 0)

    def test_no_record_without_a_passed_canary_and_a_changed_image_is_caught(self):
        self.image_fakes()
        (self.state / "canary-last").write_text(f"started={STARTED} result=fail at=x\n")
        self.ops("ops_last_good_record")
        self.assertFalse((self.state / "last-good.env").exists())
        (self.state / "canary-last").write_text(f"started={STARTED} result=pass at=x\n")
        self.ops("ops_last_good_record")
        self.image_fakes(worker_layers="[sha256:bb]")
        check = self.ops("ops_last_good_check")
        self.assertEqual(check.returncode, 1)
        self.assertIn("is not the recorded one on both nodes", check.stdout)

    def test_upgrade_needs_a_way_back(self):
        result = self.serve("upgrade")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no safe way back, so no upgrade: no last good start is recorded", result.stderr)
        self.assertNotIn("docker build", self.called())

    def test_the_pin_makes_every_start_run_on_the_last_good(self):
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "rollback.env").write_text("IMAGE=tensorfold-minimax-m3:last-good\n"
                                                 "MODEL_REVISION=1f404fc3c665883d0d001cfb32fb430108160589\n")
        result = self.serve("args")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rollback pin", result.stderr)
        self.assertIn("tensorfold-minimax-m3:last-good tensorfold-native serve", result.stdout)
        self.assertIn("snapshots/1f404fc3c665883d0d001cfb32fb430108160589", result.stdout)
        self.assertNotIn("tensorfold-minimax-m3:v1.1.2", result.stdout)
        self.assertIn("rollback pin dropped", self.serve("rollback", "--clear").stdout)
        self.assertFalse((self.state / "rollback.env").exists())
        self.assertIn("tensorfold-minimax-m3:v1.1.2 tensorfold-native serve", self.serve("args").stdout)

    def test_rollback_refuses_without_a_last_good(self):
        result = self.serve("rollback")
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot roll back: no last good start is recorded", result.stderr)


class UnitAndDocsTests(unittest.TestCase):
    UNITS = ROOT / "scripts" / "systemd"

    def unit(self, name):
        return (self.UNITS / name).read_text()

    def test_the_units_run_this_checkouts_scripts(self):
        workdir = re.search(r"^WorkingDirectory=(.*)$", self.unit("tf-watchdog.service"), re.MULTILINE).group(1)
        for name in ("mm31-serve.service", "mm31-boot-start.service", "mm31-logrotate.service"):
            with self.subTest(unit=name):
                text = self.unit(name)
                self.assertIn(f"WorkingDirectory={workdir}\n", text)
                self.assertIn("Type=oneshot\n", text)
                for script in re.findall(r"^Exec(?:Start|Stop)=/usr/bin/env bash (\S+)", text, re.MULTILINE):
                    self.assertTrue(os.access(ROOT / script, os.X_OK), script)
        serve = self.unit("mm31-serve.service")
        self.assertIn("RemainAfterExit=yes\n", serve)
        self.assertIn("ExecStart=/usr/bin/env bash scripts/serve.sh start\n", serve)
        self.assertIn("ExecStop=/usr/bin/env bash scripts/serve.sh stop\n", serve)
        self.assertGreaterEqual(int(re.search(r"^TimeoutStartSec=(\d+)", serve, re.MULTILINE).group(1)), 600 + 2400 + 900)
        self.assertNotIn("[Install]", serve, "the boot start decides; this unit is not enabled for every boot")
        self.assertIn("OnCalendar=hourly", self.unit("mm31-logrotate.timer"))

    def test_the_new_scripts_settings_are_documented(self):
        read = set()
        for name in ("scripts/ops.sh", "scripts/boot-start.sh", "scripts/logrotate.sh"):
            read |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):?-", (ROOT / name).read_text()))
        documented = set(re.findall(r"(?:^#?\s*|\s)([A-Z][A-Z0-9_]*)=", EXAMPLE.read_text(), re.MULTILINE))
        self.assertEqual(sorted(read - documented - {"MM31_PINNED"}), [])

    def test_the_docs_name_every_command(self):
        ops = (ROOT / "docs" / "OPERATIONS.md").read_text()
        agents = (ROOT / "AGENTS.md").read_text()
        for word in ("serve.sh ready", "--json", "boot-times", "prepared clean", "upgrade", "rollback", "logrotate.sh",
                     "mm31-serve.service", "mm31-boot-start.service", "mm31-logrotate.timer", "MEM_GATE_SLACK_GIB"):
            with self.subTest(word=word):
                self.assertIn(word, ops)
        for word in ("serve.sh ready", "health --json"):
            self.assertIn(word, agents)
        port = (ROOT / "docs" / "PORT.md").read_text()
        for word in ("Prepared per-rank weights", "Boot timing", "`serve.sh ready`", "mm31-serve.service", "Log rotation",
                     "`serve.sh upgrade`"):
            self.assertIn(word, port)


if __name__ == "__main__":
    unittest.main()
