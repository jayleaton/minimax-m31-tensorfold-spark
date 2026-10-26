"""CPU-only checks of the recipe: the serving config, script syntax and refusals, the rank contract, the pair watch,
check-public, the calculator, and that the engine patches stay as imported.

    tests/run.sh      (or: python3 -m pytest -q tests)

No GPU, network, Docker or SSH is used: docker, ssh and nvidia-smi are replaced by fakes on PATH where a script calls them.
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "guard"))
sys.path.insert(0, str(ROOT / "scripts"))
import quant_budget as qb  # noqa: E402
from common import POLICY  # noqa: E402

EXAMPLE = ROOT / "config" / "prod.env.example"
SMOKE = ROOT / "config" / "smoke.env.example"
# What the smoke profile may set differently from the production profile (config/smoke.env.example says why)
SMOKE_DIFFERS = {"WORKER", "WEIGHTS", "MODEL_DIR", "ENGINE_PATCHES", "ENGINE_COMMIT", "ENGINE_API_TOOLS", "TF_MM3_GRAPHS_TP",
                 # the smoke load is the all-resident 131072 x 1 profile: production's 4 x 1M needs the KV tier
                 "CONTEXT", "PARALLEL", "MEM_GATE_GIB", "KV_TIER_DIR", "TF_MM3_KV_TIER", "TF_MM3_KV_TIER_GIB",
                 "TF_MM3_KV_HORIZON", "TF_MM3_KV_CANDIDATES", "KV_TIER_KEEP_GIB",
                 # production is the measured profile: the dspark draft and the RoCE switches on, the smoke load plain
                 "DRAFTS", "DRAFTER", "TF_MM3_DSPARK_STATS", "TF_MM3_DSPARK_CHANCES", "TF_MM3_CALIB", "TF_MM3_ROCE_SUM",
                 "TF_MM3_SHARE_INPLACE", "TF_ROCE_FAST"}
DRAFTS_PROFILE = ROOT / "config" / "drafts.env.example"
# What the drafts profile may set differently from the smoke profile, its drafts-off control (config/drafts.env.example)
DRAFTS_DIFFERS = {"DRAFTS", "DRAFTER", "TF_MM3_DSPARK_STATS", "TF_MM3_DSPARK_CHANCES", "TF_MM3_CALIB", "MEM_GATE_GIB"}
SLOTS4 = ROOT / "config" / "slots4.env.example"
# What the four-slot profile may set differently from the drafts profile (config/slots4.env.example says why)
SLOTS4_DIFFERS = {"PARALLEL", "CONTEXT", "MEM_GATE_GIB", "ENGINE_PATCHES", "ENGINE_COMMIT",
                  # measured at four streams at depth 3, depth by confidence at one stream only
                  "TF_MM3_DRAFT_DEPTH", "TF_MM3_DSPARK_CHANCES", "TF_MM3_CALIB"}
# Engine switches no profile of this recipe may set
FORBIDDEN_ENGINE_ENV = ("KV_SPLIT_DEDUP", "KV_SPLIT_EARLY", "TF_TP_ROCE_CHUNKED_KB")
SHELL_SCRIPTS = sorted([ROOT / "start.sh", ROOT / "stop.sh", *(ROOT / "scripts").glob("*.sh"),
                        *(ROOT / "bench").rglob("*.sh"), *(ROOT / "tests").glob("*.sh")])
# MiaAI-Lab's patches/ at fd1d379 (checked identical with diff -r): sha256 over "<sha256>  <path>" lines sorted by path.
# The engine patches belong to the engine branch; this recipe must not change them (docs/PORT.md).
PATCHES_TREE_SHA256 = "1f10a178f6fffaea5a795c4e4280e351361b94470a5525a86ce7edaac5c7bf98"
# Values the scripts compute, not settings; and the device's limits, read from /etc/spark/thermal.env, never from the recipe
COMPUTED = {"HEAD_ADDR", "HEAD_HCA", "HERE_GB", "HF_HOME", "IP", "SERVE_PROFILE", "WORKER_ADDR", "WORKER_GB",
            "TIER_HERE_GIB", "TIER_WORKER_GIB", "MM31_BOOT_MARKS", "MM31_PINNED",
            "GUARD_PAUSE_C", "GUARD_RESUME_C", "GUARD_KILL_C", "GUARD_KILL_HOLD_S"}
# The operator's device files, copied verbatim (the spark-health copies and both nodes' installed files)
DEVICE_SHA256 = {'20-from-thermal-env.conf': '764c86c6349aceade73a26122e801ce39f91e92951a686774ebeb120fbfeb1ff', 'dgx-gpu-clock-cap.service': 'fa82eb44f907af02f938cfd51c5c27f1d99cc349ea1e0c851585f515fdbede12', 'spark-thermal-guard': '48311306ebdea7861b23051a9e84ae052bf2ee70463ae95f2592ad08172667a2', 'spark-thermal-guard.service': '5267b101ced2d73511a649db81980e8fde00ea3846f0db359103554050f659c9', 'thermal.env': '2a172210e67f529b7da7b612f9d19d599280b3616ba35d2771a25a7ab426bae1'}


def example_env(path=EXAMPLE):
    """The example's assignments as config.sh reads them (KEY=value, quotes stripped, # comments dropped)."""
    env = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line)
        if m:
            value = m.group(2)
            q = re.match(r'^"([^"]*)"', value)
            env[m.group(1)] = q.group(1) if q else value.split("#", 1)[0].rstrip()
    return env


def profile_gate(e, vision=None):
    """The calculator's serving terms for a profile's settings; ``vision`` None: the profile's VISION."""
    on = e["VISION"] == "1" if vision is None else vision
    return qb.serving(199.9, int(e["CONTEXT"]), int(e["PARALLEL"]), e["KV"], float(e["TENSORFOLD_MEMORY_RESERVE_GIB"]),
                      bool(e["DRAFTER"]), vision_image_tokens=int(e["VISION_IMAGE_TOKENS"]) if on else 0)


def fake(bin_dir, name, body):
    path = Path(bin_dir) / name
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


class ConfigExampleTests(unittest.TestCase):
    def setUp(self):
        self.env = example_env()

    def test_every_line_is_a_comment_blank_or_assignment(self):
        for number, line in enumerate(EXAMPLE.read_text().splitlines(), 1):
            with self.subTest(line=number):
                self.assertRegex(line, r"^(|#.*|[A-Z][A-Z0-9_]*=.*)$")

    def test_only_the_worker_is_a_placeholder_and_the_pack_is_pinned(self):
        self.assertEqual([k for k, v in self.env.items() if "<" in v], ["WORKER"])
        self.assertEqual(self.env["MODEL_ID"], "jayleaton/MiniMax-M3.1-Flash-EXL3-3.55bpw-mm31")
        self.assertRegex(self.env["MODEL_REVISION"], r"^[0-9a-f]{40}$")
        self.assertEqual(self.env["HOST"], "127.0.0.1")
        self.assertEqual(self.env["ENGINE_PATCHES"], "engine/patches")

    def test_the_engine_is_the_exported_series(self):
        series = sorted(p.name for p in (ROOT / "engine" / "patches").glob("*.patch"))
        self.assertEqual(series, ["0001-minimax-m3-family.patch", "0002-minimax-m3-engine.patch"])
        # 0001 is MiaAI-Lab's family patch itself, not a copy
        self.assertEqual((ROOT / "engine/patches/0001-minimax-m3-family.patch").resolve(),
                         (ROOT / "patches/0001-minimax-m3-family.patch").resolve())
        export = (ROOT / "engine" / "SOURCE-EXPORT.md").read_text()
        self.assertRegex(self.env["ENGINE_COMMIT"], r"^[0-9a-f]{40}$")
        self.assertIn(f"from engine commit `{self.env['ENGINE_COMMIT']}`", export)
        self.assertIn(self.env["ENGINE_COMMIT"][:12], (ROOT / "engine/patches/0002-minimax-m3-engine.patch").read_text()[:600])

    def test_every_setting_the_scripts_read_is_documented(self):
        read = set()
        for name in ("scripts/config.sh", "start.sh", "stop.sh", "scripts/serve.sh", "scripts/prepare.sh",
                     "scripts/watch.sh", "scripts/nodes.sh"):
            read |= set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):?-", (ROOT / name).read_text()))
        documented = set(re.findall(r"(?:^#?\s*|\s)([A-Z][A-Z0-9_]*)=", EXAMPLE.read_text(), re.M))
        self.assertEqual(sorted(read - documented - COMPUTED), [])

    def test_mem_gate_is_the_calculators_for_the_profile(self):
        e = self.env
        context, parallel, reserve = int(e["CONTEXT"]), int(e["PARALLEL"]), float(e["TENSORFOLD_MEMORY_RESERVE_GIB"])
        for vision in (False, True):  # the images come out of the KV pool (the tier's slice cache): the gate is the same
            with self.subTest(vision=vision):
                if not e.get("KV_TIER_DIR"):
                    s = profile_gate(e, vision)
                    self.assertEqual(int(e["MEM_GATE_GIB"]), s["mem_gate"])
                    self.assertGreaterEqual(s["spare"], 0, "the profile's CONTEXT must fit the calculator's pool")
                    continue
                images = int(e["VISION_IMAGE_TOKENS"]) if vision else 0
                s = qb.serving(199.9, 0, parallel, e["KV"], reserve, bool(e["DRAFTER"]), vision_image_tokens=images)
                t = qb.tier(context, parallel, s["kv_room"], int(e["TF_MM3_KV_HORIZON"]), int(e["TF_MM3_KV_CANDIDATES"]))
                self.assertEqual(int(e["MEM_GATE_GIB"]), int(-(-(s["fixed"] + t["device_min"]) // 1)))
                self.assertTrue(t["ok"], "the slice cache left by the profile must hold twice its pinned slices")
                self.assertGreaterEqual(int(e["TF_MM3_KV_TIER_GIB"]), t["nvme"], "the NVMe budget holds every live page")

    def test_images_are_off_until_kv_offload_with_full_size_limits(self):
        e = self.env
        self.assertEqual((e["VISION"], e["VISION_URLS"]), ("0", "0"))
        self.assertEqual(int(e["VISION_IMAGE_TOKENS"]), int(e["VISION_MAX_IMAGES"]) * 576, "every image at 576 tokens")
        self.assertAlmostEqual(qb.vision_gib(int(e["VISION_IMAGE_TOKENS"])), 2.27 + 2304 * 6144 * 2 / 2**30)
        resident = example_env(SMOKE)   # the all-resident 131,072 x 1 profile
        text, images = profile_gate(resident, False), profile_gate(resident, True)
        self.assertEqual((text["mem_gate"], images["mem_gate"]), (113, 113))
        self.assertEqual((text["pool_tokens"], images["pool_tokens"]), (238_336, 185_984))
        self.assertEqual(text["pages"] - images["pages"], 409)
        self.assertEqual(images["pool_tokens"] % qb.PAGE_TOKENS, 0)

    def test_a_profile_past_the_room_has_the_kv_tier(self):
        e = self.env
        s = qb.serving(199.9, int(e["CONTEXT"]), int(e["PARALLEL"]), e["KV"], float(e["TENSORFOLD_MEMORY_RESERVE_GIB"]))
        if s["spare"] < 0:
            self.assertTrue(e.get("KV_TIER_DIR"), "CONTEXT x PARALLEL does not fit the room: it needs the KV tier")
        if e.get("KV_TIER_DIR"):
            self.assertEqual(e["TF_MM3_KV_TIER"], "/kvtier")
            self.assertEqual(e["KV"], "fp8")
            self.assertGreaterEqual(int(e["KV_TIER_KEEP_GIB"]), 300)
            self.assertRegex(e["KV_TIER_DIR"], r"^(/|~/)")

    def test_memory_levels_are_ordered_above_earlyoom(self):
        e = self.env
        earlyoom = qb.MEMTOTAL_GIB * 2 / 100          # `earlyoom -m 2,1` on the nodes: SIGTERM under 2% of MemTotal
        self.assertGreater(float(e["MEMGUARD_HARD_GIB"]), earlyoom)
        self.assertGreater(float(e["MEMGUARD_GIB"]), float(e["MEMGUARD_HARD_GIB"]))
        self.assertGreater(POLICY["memory_floor_gib"], float(e["MEMGUARD_GIB"]))
        self.assertGreaterEqual(float(e["TENSORFOLD_MEMORY_RESERVE_GIB"]), POLICY["memory_floor_gib"])

    def test_engine_floors_sit_between_the_guard_and_the_reserve(self):
        e = self.env
        hard, target, transient = (float(e[k]) for k in ("TF_MM3_FLOOR_HARD_GIB", "TF_MM3_FLOOR_GIB",
                                                         "TF_MM3_FLOOR_TRANSIENT_GIB"))
        self.assertEqual(hard, POLICY["memory_floor_gib"])
        self.assertGreater(hard, float(e["MEMGUARD_GIB"]))
        self.assertLessEqual(hard, target)
        self.assertLessEqual(target + transient, float(e["TENSORFOLD_MEMORY_RESERVE_GIB"]))
        self.assertEqual(e["TF_ROCE_MARK"], "/cache/roce-failed")

    def test_the_heat_limits_are_not_recipe_settings(self):
        self.assertFalse([k for k in self.env if k.startswith("GUARD_") and k != "GUARD_CONFIG"])

    def test_the_device_files_are_the_operators_verbatim(self):
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "device").iterdir())}
        self.assertEqual(files, DEVICE_SHA256)
        self.assertTrue(os.access(ROOT / "device" / "spark-thermal-guard", os.X_OK))

    def test_guard_example_matches_the_policy_and_the_slots(self):
        import json

        guard = json.loads((ROOT / "config" / "guard.example.json").read_text())
        self.assertEqual(guard["policy"], POLICY)
        self.assertEqual(guard["limits"]["max_inflight"], int(self.env["PARALLEL"]))
        self.assertEqual(set(guard["containers"].values()), {self.env["CONTAINER_NAME"]})


class SmokeProfileTests(unittest.TestCase):
    """config/smoke.env.example: the production profile on the mm3/integrate engine, graphs off between the ranks."""

    def setUp(self):
        self.env = example_env(SMOKE)
        self.prod = example_env(EXAMPLE)

    def test_every_line_is_a_comment_blank_or_assignment(self):
        for number, line in enumerate(SMOKE.read_text().splitlines(), 1):
            with self.subTest(line=number):
                self.assertRegex(line, r"^(|#.*|[A-Z][A-Z0-9_]*=.*)$")

    def test_the_worker_and_the_engine_commit_are_the_placeholders(self):
        self.assertEqual(sorted(k for k, v in self.env.items() if "<" in v), ["ENGINE_COMMIT", "WORKER"])

    def test_the_smoke_values(self):
        e = self.env
        self.assertEqual((e["CONTEXT"], e["PARALLEL"], e["KV"], e["DRAFTS"], e["DRAFTER"]), ("131072", "1", "fp8", "0", ""))
        self.assertEqual((e["ENGINE_API_TOOLS"], e["TF_MM3_GRAPHS_TP"], e["WEIGHTS"]), ("1", "0", "dir"))
        self.assertTrue(e["MODEL_DIR"].startswith(("/", "~/")))

    def test_the_rest_is_the_production_profile(self):
        # every production setting is set here (config.sh's defaults differ, e.g. MEM_GATE_GIB=0), with its value; the
        # production-only switches (SMOKE_DIFFERS: the draft, RoCE) are unset here, which leaves them off
        self.assertEqual(sorted(set(self.prod) - set(self.env) - SMOKE_DIFFERS), [])
        changed = {k for k in self.prod if self.env.get(k) != self.prod[k]} | (set(self.env) - set(self.prod))
        self.assertLessEqual(changed, SMOKE_DIFFERS)

    def test_mem_gate_is_the_calculators_for_the_profile(self):
        for vision in (False, True):
            with self.subTest(vision=vision):
                self.assertEqual(int(self.env["MEM_GATE_GIB"]), profile_gate(self.env, vision)["mem_gate"])

    def test_images_are_off_and_the_vision_run_fits_at_110_gib(self):
        import math

        self.assertEqual(self.env["VISION"], "0")
        run = dict(w.split("=") for w in re.search(r"^# VISION_RUN: (.*)$", SMOKE.read_text(), re.M).group(1).split())
        self.assertEqual(run["VISION"], "1")
        ctx, page = int(run["CONTEXT"]), qb.PAGE_TOKENS
        s = qb.serving(199.9, ctx, int(self.env["PARALLEL"]), self.env["KV"], 8, available_gib=110.0,
                       vision_image_tokens=int(self.env["VISION_IMAGE_TOKENS"]))
        # the largest page multiple whose pages plus the engine's one-page verify window fit the pool
        self.assertEqual(ctx % page, 0)
        self.assertLessEqual(ctx + page, s["pool_tokens"])
        self.assertGreater(ctx + 2 * page, s["pool_tokens"])
        self.assertEqual((ctx, s["pool_tokens"]), (23_680, 23_808))
        self.assertEqual(int(run["MEM_GATE_GIB"]), math.ceil(s["fixed"] + s["vision"] + s["kv_need"]))
        self.assertLess(qb.serving(199.9, 131072, 1, "fp8", 8, available_gib=110.0, vision_image_tokens=2304)["spare"], 0,
                        "the full context does not fit there with images: why VISION is off")


class DraftsProfileTests(unittest.TestCase):
    """config/drafts.env.example: the smoke profile with the dspark draft on; config/smoke.env is its matched control."""

    def setUp(self):
        self.env = example_env(DRAFTS_PROFILE)
        self.smoke = example_env(SMOKE)

    def test_every_line_is_a_comment_blank_or_assignment(self):
        for number, line in enumerate(DRAFTS_PROFILE.read_text().splitlines(), 1):
            with self.subTest(line=number):
                self.assertRegex(line, r"^(|#.*|[A-Z][A-Z0-9_]*=.*)$")

    def test_the_placeholders_are_the_smoke_profiles(self):
        self.assertEqual(sorted(k for k, v in self.env.items() if "<" in v), ["ENGINE_COMMIT", "WORKER"])
        self.assertEqual(self.env["ENGINE_COMMIT"], self.smoke["ENGINE_COMMIT"])

    def test_only_the_draft_differs_from_the_control(self):
        self.assertEqual(sorted(set(self.smoke) - set(self.env)), [])
        changed = {k for k in self.smoke if self.env[k] != self.smoke[k]} | (set(self.env) - set(self.smoke))
        self.assertEqual(changed, DRAFTS_DIFFERS)
        e = self.env
        self.assertEqual((e["DRAFTS"], e["DRAFTER"], e["TF_MM3_DSPARK_STATS"]), ("1", "dspark", "1"))
        self.assertEqual((e["CONTEXT"], e["PARALLEL"], e["KV"]), (self.smoke["CONTEXT"], "1", "fp8"))
        self.assertEqual((self.smoke["DRAFTS"], self.smoke["DRAFTER"]), ("0", ""))

    def test_the_gate_prices_the_draft_and_the_context_fits(self):
        s = profile_gate(self.env)
        self.assertGreater(s["draft"], 0)
        self.assertEqual(int(self.env["MEM_GATE_GIB"]), s["mem_gate"])
        self.assertEqual(s["mem_gate"], 115)
        self.assertGreaterEqual(s["spare"], 0, "CONTEXT must fit the pool beside the draft")
        self.assertGreaterEqual(s["pool_tokens"], int(self.env["CONTEXT"]) + qb.PAGE_TOKENS, "and the verify window")

    def test_the_narrow_run_is_the_calculators_and_fits_at_113_gib(self):
        run = dict(w.split("=") for w in re.search(r"^# NARROW_RUN: (.*)$", DRAFTS_PROFILE.read_text(), re.M)
                   .group(1).split())
        narrow = dict(self.env, CONTEXT=run["CONTEXT"])
        self.assertEqual(int(run["MEM_GATE_GIB"]), profile_gate(narrow)["mem_gate"])
        at113 = qb.serving(199.9, int(run["CONTEXT"]), 1, "fp8", 8, True, available_gib=113.0)
        self.assertGreaterEqual(at113["pool_tokens"], int(run["CONTEXT"]) + qb.PAGE_TOKENS)
        self.assertLess(qb.serving(199.9, 131072, 1, "fp8", 8, True, available_gib=113.0)["spare"], 0,
                        "the full context does not fit beside the draft there: why the narrow run exists")

    def test_no_profile_sets_a_forbidden_engine_switch(self):
        for path in (EXAMPLE, SMOKE, DRAFTS_PROFILE):
            with self.subTest(path=path.name):
                self.assertFalse([k for k in example_env(path) if k in FORBIDDEN_ENGINE_ENV])
                self.assertNotRegex(path.read_text(), "|".join(FORBIDDEN_ENGINE_ENV))


class ScriptSyntaxTests(unittest.TestCase):
    def test_bash_n(self):
        for script in SHELL_SCRIPTS:
            with self.subTest(script=script.name):
                result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("shellcheck"), "shellcheck not installed")
    def test_shellcheck(self):
        result = subprocess.run(["shellcheck", "-x", "-S", "error", *map(str, SHELL_SCRIPTS)], capture_output=True,
                                text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_every_script_run_by_path_is_executable(self):
        # serve.sh is the operator's entry point and start.sh runs pairwatch.sh / memguard.sh by path: a checkout must keep +x
        for script in [ROOT / "start.sh", ROOT / "stop.sh", *(ROOT / "scripts").glob("*.sh"), ROOT / "tests" / "run.sh"]:
            with self.subTest(script=script.name):
                self.assertTrue(os.access(script, os.X_OK), f"{script.relative_to(ROOT)} is not executable")

    def test_python_compiles(self):
        for path in sorted([*ROOT.glob("scripts/**/*.py"), *ROOT.glob("bench/**/*.py")]):
            with self.subTest(path=str(path.relative_to(ROOT))):
                compile(path.read_text(), str(path), "exec")


class ScriptFixture:
    """A copy-free sandbox: fakes on PATH, a config of its own, state under a temporary directory."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.calls = self.root / "calls"
        fake(self.bin, "docker", f'echo "docker $*" >> {self.calls}\nexit 1\n')
        fake(self.bin, "ssh", f'echo "ssh $*" >> {self.calls}\nexit 255\n')
        fake(self.bin, "nvidia-smi", "exit 0\n")
        self.thermal_run = self.root / "spark-thermal"
        self.thermal_run.mkdir()
        (self.thermal_run / "state").write_text("normal\n")
        self.config = self.root / "prod.env"
        text = EXAMPLE.read_text().replace("WORKER=\"<user@worker-link-address>\"", "WORKER=user@203.0.113.2")
        self.config.write_text(text)

    def env(self, **extra):
        return {**{k: v for k, v in os.environ.items() if not k.startswith(("TF_", "TENSORFOLD_", "NCCL_"))},
                "PATH": f"{self.bin}:{os.environ['PATH']}", "CONFIG": str(self.config), "HOME": str(self.root),
                "SPARK_THERMAL_ENV": str(ROOT / "device" / "thermal.env"), "SPARK_THERMAL_RUN": str(self.thermal_run),
                **extra}

    def run_script(self, *argv, **extra):
        return subprocess.run(["bash", *map(str, argv)], cwd=ROOT, env=self.env(**extra), capture_output=True,
                              text=True, timeout=60)

    def bash(self, snippet, **extra):
        return subprocess.run(["bash", "-c", f"set -euo pipefail; source scripts/config.sh; source scripts/nodes.sh; "
                               f"{snippet}"], cwd=ROOT, env=self.env(**extra), capture_output=True, text=True,
                              timeout=60)


class ConfigLoadingTests(ScriptFixture, unittest.TestCase):
    def test_the_untouched_example_is_refused(self):
        shutil.copy(EXAMPLE, self.config)
        result = self.run_script(ROOT / "scripts" / "serve.sh", "status")
        self.assertEqual(result.returncode, 2)
        self.assertIn("set WORKER", result.stderr)

    def test_serve_without_a_config_says_how_to_make_one(self):
        result = self.run_script(ROOT / "scripts" / "serve.sh", "status", CONFIG=str(self.root / "none.env"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("cp config/prod.env.example config/prod.env", result.stderr)

    def test_the_device_file_wins_over_the_environment(self):
        result = self.bash('echo "$GUARD_PAUSE_C $GUARD_RESUME_C $GUARD_KILL_C $GUARD_KILL_HOLD_S $SPARK_CLOCK_MAX_MHZ"',
                           GUARD_PAUSE_C="50")
        self.assertEqual(result.stdout.split(), ["92", "86", "93", "30", "2100"], result.stderr)

    def test_a_recipe_config_may_not_define_the_limits(self):
        with self.config.open("a") as f:
            f.write("GUARD_PAUSE_C=95\n")
        result = self.bash("true")
        self.assertEqual(result.returncode, 2)
        self.assertIn("live on the device", result.stderr)

    def test_no_device_file_no_start(self):
        result = self.bash("need_device_limits", SPARK_THERMAL_ENV=str(self.root / "none.env"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("install-device-limits.sh", result.stderr)

    def test_the_installer_plans_without_sudo(self):
        fake(self.bin, "sudo", f'echo "sudo $*" >> {self.calls}\nexit 1\n')
        result = self.run_script(ROOT / "scripts" / "install-device-limits.sh")
        self.assertNotIn("sudo", self.calls.read_text() if self.calls.exists() else "")
        self.assertIn("Plan only", result.stdout)

    def test_the_environment_wins_over_the_config_file(self):
        result = self.bash('echo "$CONTEXT $KV $HOST $PORT"', CONTEXT="65536")
        self.assertEqual(result.stdout.strip(), "65536 fp8 127.0.0.1 8000", result.stderr)


class RankContractTests(ScriptFixture, unittest.TestCase):
    def rank_env(self, rank, **extra):
        snippet = ("LINK_HEAD_ADDR[1]=192.0.2.1; NODE_GID=(3 3); SERVE_PROFILE=context=131072,kv=fp8; "
                   f"rank_env {rank}")
        return self.bash(snippet, **extra)

    def test_the_family_reads_tf_mm3_and_nccl_comm_id(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                result = self.rank_env(rank)
                self.assertEqual(result.returncode, 0, result.stderr)
                words = result.stdout.split()
                for word in ("TF_MM3_WORLD=2", f"TF_MM3_RANK={rank}", "NCCL_COMM_ID=192.0.2.1:29620",
                             "TF_ROCE_GID_INDEX=3", "TENSORFOLD_MEMORY_RESERVE_GIB=8",
                             "MM31_SERVE_PROFILE=context=131072,kv=fp8"):
                    self.assertIn(word, words)
                self.assertFalse([w for w in words if w.startswith(("TF_TP_", "TF_COMM_BACKEND"))], words)

    def test_roce_off_and_cap_reach_both_ranks(self):
        words = self.rank_env(1, ROCE="0", ROCE_MAX_KB="1024").stdout.split()
        self.assertIn("TF_ROCE=0", words)
        self.assertIn("TF_ROCE_MAX_KB=1024", words)

    def test_a_credential_never_rides_in_the_environment(self):
        result = self.rank_env(0, TENSORFOLD_API_KEY="x" * 24)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("looks like a credential", result.stderr)
        self.assertEqual(self.rank_env(0, TF_MM3_POOL_TOKENS="1").returncode, 0, "TOKENS is a count, not a token")

    def test_start_refuses_a_key_on_the_command_line(self):
        result = self.run_script(ROOT / "start.sh", "--api-key", "x" * 24, DRY_RUN="1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("do not pass --api-key", result.stderr)

    def test_dry_run_prints_both_ranks_with_the_profile(self):
        result = self.run_script(ROOT / "start.sh", DRY_RUN="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout
        rank1 = out[out.index("[dry-run] rank 1"):out.index("[dry-run] rank 0")]
        rank0 = out[out.index("[dry-run] rank 0"):]
        e = example_env()
        tier = ("TF_MM3_KV_TIER=/kvtier", ":/kvtier") if e.get("KV_TIER_DIR") else ()
        drafts = (("--no-drafts",) if e["DRAFTS"] == "0" else
                  ("--drafter ", "/dspark", *(f"{k}={e[k]}" for k in ("TF_MM3_DRAFT_DEPTH", "TF_MM3_DSPARK_CHANCES",
                                                                      "TF_MM3_CALIB") if e.get(k)))
                  if e["DRAFTER"] == "dspark" else ())
        roce = tuple(f"{k}={e[k]}" for k in ("TF_MM3_ROCE_SUM", "TF_MM3_SHARE_INPLACE", "TF_ROCE_FAST", "TF_MM3_GRAPHS_TP") if e.get(k))
        for part in ("TF_MM3_RANK=1", f"--context {e['CONTEXT']}", f"--parallel {e['PARALLEL']}", "--kv-cache fp8",
                     *drafts, *roce, "TENSORFOLD_MEMORY_RESERVE_GIB=8", "NCCL_CUMEM_ENABLE=0", *tier):
            self.assertIn(part, rank1)
        if not tier:
            self.assertNotIn("/kvtier", rank1, "no tier folder is mounted without KV_TIER_DIR")
        self.assertNotIn("--port", rank1, "rank 1 serves no HTTP")
        for part in ("TF_MM3_RANK=0", "--host 127.0.0.1", "--port 8000", "--name MiniMax-M3.1-Flash-EXL3",
                     f"MM31_SERVE_PROFILE=context={e['CONTEXT']}", *tier):
            self.assertIn(part, rank0)
        self.assertNotIn("docker run", (self.calls.read_text() if self.calls.exists() else ""), "nothing started")

    def test_images_are_off_by_default(self):
        self.assertNotIn("--vision", self.run_script(ROOT / "start.sh", DRY_RUN="1").stdout)  # the profile's VISION=0
        bare = self.root / "bare.env"   # a config without the VISION lines: scripts/config.sh's own default
        bare.write_text("".join(line + "\n" for line in self.config.read_text().splitlines()
                                if not line.startswith("VISION")))
        result = self.bash('echo "$VISION $VISION_URLS"', CONFIG=str(bare))
        self.assertEqual(result.stdout.split(), ["0", "0"], result.stderr)
        self.assertNotIn("--vision", self.run_script(ROOT / "start.sh", DRY_RUN="1", CONFIG=str(bare)).stdout)

    def test_images_reach_both_ranks_and_urls_only_rank_0(self):
        out = self.run_script(ROOT / "start.sh", DRY_RUN="1", VISION="1", VISION_URLS="1").stdout
        rank1 = out[out.index("[dry-run] rank 1"):out.index("[dry-run] rank 0")]
        rank0 = out[out.index("[dry-run] rank 0"):]
        for rank in (rank0, rank1):
            self.assertIn("--vision --vision-max-images 4 --vision-image-tokens 2304", rank)
            self.assertIn("vision=1,vision_max_images=4,vision_image_tokens=2304,vision_urls=1", rank.replace("\\", ""))
        self.assertIn("--vision-urls", rank0)
        self.assertNotIn("--vision-urls", rank1)
        run = self.run_script(ROOT / "start.sh", DRY_RUN="1", VISION="1", CONTEXT="23680").stdout
        self.assertIn("--context 23680", run)
        self.assertIn("--vision ", run)

    def test_image_settings_are_checked(self):
        for bad, why in (({"VISION": "yes"}, "VISION and VISION_URLS are 0 or 1"),
                         ({"VISION_MAX_IMAGES": "0"}, "positive counts"),
                         ({"VISION": "0", "VISION_URLS": "1"}, "VISION_URLS=1 needs VISION=1")):
            with self.subTest(bad=bad):
                result = self.bash("true", **bad)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(why, result.stderr)


class SmokeScriptTests(ScriptFixture, unittest.TestCase):
    """The smoke profile through the scripts: its placeholders refuse, WEIGHTS=dir mounts each node's folder at /model."""

    def setUp(self):
        super().setUp()
        self.config.write_text(SMOKE.read_text().replace('WORKER="<user@worker-link-address>"', "WORKER=user@203.0.113.2"))

    def test_the_engine_commit_placeholder_is_refused(self):
        result = self.bash("true")
        self.assertEqual(result.returncode, 2)
        self.assertIn("set ENGINE_COMMIT (still <mm3/integrate head>", result.stderr)

    def test_a_branch_name_is_not_an_engine_commit(self):
        result = self.bash("true", ENGINE_COMMIT="mm3/integrate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ENGINE_COMMIT is the commit hash", result.stderr)
        self.assertEqual(self.bash("true", ENGINE_COMMIT="542c144e1").returncode, 0)

    def test_weights_dir_needs_a_folder(self):
        result = self.bash("true", ENGINE_COMMIT="542c144e1", MODEL_DIR="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("WEIGHTS=dir needs MODEL_DIR", result.stderr)
        result = self.bash("true", ENGINE_COMMIT="542c144e1", MODEL_DIR="relative/pack")
        self.assertIn("is an absolute path or one under ~/", result.stderr)

    def test_paths_under_home_resolve_on_this_node(self):
        result = self.bash('echo "$(snapshot_dir) $(model_in_container) $(head_model_mount) $(engine_patches_dir)"',
                           ENGINE_COMMIT="542c144e1")
        pack = f"{self.root}/work/minimax-m3/pack-mm31"
        self.assertEqual(result.stdout.split(), [pack, "/model", f"{pack}:/model:ro",
                                                 f"{self.root}/work/minimax-m3/engine-patches/mm3-integrate"],
                         result.stderr)

    def test_the_workers_folder_resolves_in_the_ssh_users_home(self):
        # a worker reached as root (WORKER=root@...): ~/ is root's home there, so a pack under another account's home needs
        # an absolute WORKER_MODEL_DIR, which is used as it is (no ssh to resolve it)
        fake(self.bin, "ssh", f'echo "ssh $*" >> {self.calls}\nprintf /var/lib/rootish\n')
        tilde = self.bash("worker_model_dir", ENGINE_COMMIT="542c144e1", WORKER="root@203.0.113.2")
        self.assertEqual(tilde.stdout, "/var/lib/rootish/work/minimax-m3/pack-mm31\n", tilde.stderr)
        self.calls.unlink()
        absolute = self.bash("worker_model_dir", ENGINE_COMMIT="542c144e1", WORKER="root@203.0.113.2",
                             WORKER_MODEL_DIR="/srv/packs/pack-mm31")
        self.assertEqual(absolute.stdout, "/srv/packs/pack-mm31\n", absolute.stderr)
        self.assertFalse(self.calls.exists(), "an absolute folder needs no ssh")

    def test_the_manifest_lists_what_a_rank_reads(self):
        pack = self.root / "work" / "minimax-m3" / "pack-mm31"
        (pack / "dspark").mkdir(parents=True)
        for name, size in (("model-00001-of-00002.safetensors", 5), ("config.json", 2), ("LICENSE", 3),
                           ("SHA256SUMS", 4), ("merges.txt", 1), ("README.md", 9), ("pull.log", 9), (".shard_stream.json", 7),
                           ("dspark/x.json", 6)):
            (pack / name).write_bytes(b"x" * size)
        lines = self.bash("model_dir_manifest head", ENGINE_COMMIT="542c144e1").stdout.splitlines()
        self.assertEqual(lines, ["LICENSE 3", "SHA256SUMS 4", "config.json 2", "merges.txt 1",
                                 "model-00001-of-00002.safetensors 5"])
        with_draft = self.bash("model_dir_manifest head", ENGINE_COMMIT="542c144e1", DRAFTS="1", DRAFTER="dspark")
        self.assertIn("dspark/x.json 6", with_draft.stdout.splitlines())

    def test_dry_run_mounts_each_folder_and_passes_graphs_off_to_both_ranks(self):
        result = self.run_script(ROOT / "start.sh", DRY_RUN="1", ENGINE_COMMIT="542c144e1")
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout
        rank1 = out[out.index("[dry-run] rank 1"):out.index("[dry-run] rank 0")]
        rank0 = out[out.index("[dry-run] rank 0"):]
        for rank in (rank0, rank1):
            for part in ("TF_MM3_GRAPHS_TP=0", ":/model:ro", "serve /model ", "--context 131072", "--parallel 1",
                         "--kv-cache fp8", "--no-drafts"):
                self.assertIn(part, rank)
        self.assertIn(f"-v {self.root}/work/minimax-m3/pack-mm31:/model:ro", rank0)
        self.assertNotIn("/hf", rank0 + rank1)


class Slots4ProfileTests(unittest.TestCase):
    """config/slots4.env.example: the drafts profile at four streams, each able to hold its whole CONTEXT at once."""

    def setUp(self):
        self.env = example_env(SLOTS4)
        self.drafts = example_env(DRAFTS_PROFILE)

    def test_every_line_is_a_comment_blank_or_assignment(self):
        for number, line in enumerate(SLOTS4.read_text().splitlines(), 1):
            with self.subTest(line=number):
                self.assertRegex(line, r"^(|#.*|[A-Z][A-Z0-9_]*=.*)$")

    def test_only_the_slots_differ_from_the_drafts_profile(self):
        self.assertEqual(sorted(set(self.drafts) - set(self.env)), [])
        changed = {k for k in self.drafts if self.env[k] != self.drafts[k]} | (set(self.env) - set(self.drafts))
        self.assertEqual(changed, SLOTS4_DIFFERS)
        self.assertEqual(sorted(k for k, v in self.env.items() if "<" in v), ["ENGINE_COMMIT", "WORKER"])
        e = self.env
        self.assertEqual((e["PARALLEL"], e["DRAFTS"], e["DRAFTER"], e["KV"], e["KV_TIER_DIR"]), ("4", "1", "dspark", "fp8", ""))

    def test_every_slot_holds_its_context_at_once_beside_the_draft(self):
        e = self.env
        s = profile_gate(e)
        self.assertEqual(int(e["MEM_GATE_GIB"]), s["mem_gate"])
        self.assertGreater(s["draft"], profile_gate(self.drafts)["draft"], "four rings, not one")
        need = int(e["PARALLEL"]) * -(-(int(e["CONTEXT"]) + 16) // qb.PAGE_TOKENS)
        self.assertLessEqual(need, s["pages"] * 0.93, "with the engine's pool about 7 % under the calculator's")


class DraftsScriptTests(ScriptFixture, unittest.TestCase):
    """The drafts profile through start.sh: the draft and the stats switch on both ranks, and nothing else changed
    against the smoke profile run as its control."""

    def dry(self, profile):
        self.config.write_text(profile.read_text().replace('WORKER="<user@worker-link-address>"', "WORKER=user@203.0.113.2"))
        result = self.run_script(ROOT / "start.sh", DRY_RUN="1", ENGINE_COMMIT="542c144e1")
        self.assertEqual(result.returncode, 0, result.stderr)
        out = result.stdout.replace("\\", "")
        rank1 = out[out.index("[dry-run] rank 1"):out.index("[dry-run] rank 0")]
        return rank1, out[out.index("[dry-run] rank 0"):]

    def test_four_slots_and_the_draft_reach_both_ranks(self):
        for rank in self.dry(SLOTS4):
            for part in ("--drafter /model/dspark", "TF_MM3_DRAFT_DEPTH=3", "--parallel 4", "--context 40960"):
                self.assertIn(part, rank)
            self.assertNotIn("--no-drafts", rank)

    def test_the_draft_and_its_stats_reach_both_ranks(self):
        for rank in self.dry(DRAFTS_PROFILE):
            for part in ("--drafter /model/dspark", "TF_MM3_DSPARK_STATS=1", "TF_MM3_DSPARK_CHANCES=1", "TF_MM3_CALIB=auto",
                         "drafts=1,drafter=dspark", "--context 131072"):
                self.assertIn(part, rank)
            self.assertNotIn("TF_MM3_DRAFT_DEPTH", rank)
            self.assertNotIn("--no-drafts", rank)

    def test_the_smoke_profile_is_the_matched_control(self):
        on, off = self.dry(DRAFTS_PROFILE), self.dry(SMOKE)
        for drafted, plain in zip(on, off):
            self.assertNotIn("TF_MM3_DSPARK_STATS", plain)
            back = (drafted.replace("--drafter /model/dspark", "--no-drafts").replace("-e TF_MM3_DSPARK_STATS=1 ", "")
                    .replace("-e TF_MM3_DSPARK_CHANCES=1 ", "").replace("-e TF_MM3_CALIB=auto ", "")
                    .replace("drafts=1,drafter=dspark", "drafts=0,drafter="))
            self.assertEqual(back.strip(), plain.strip())

    def test_the_stats_switch_is_checked(self):
        self.config.write_text(DRAFTS_PROFILE.read_text().replace('WORKER="<user@worker-link-address>"',
                                                                   "WORKER=user@203.0.113.2"))
        result = self.bash("true", ENGINE_COMMIT="542c144e1", TF_MM3_DSPARK_STATS="yes")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TF_MM3_DSPARK_STATS is 1", result.stderr)


class DFlash2ProfileTests(ScriptFixture, unittest.TestCase):
    """DRAFTS=dflash2: the DFlash2 draft's folder on each Spark, its class, its pins, its mount, and 0 / 1 / dspark as before."""

    PINS = {"config.json": "64c32db3c3bc5de64a048d949c5acd91352ae9b0f02ef636e3382c838f1d0815",
            "README.md": "8ec2de932fdd0e2d72a36ba17a0c593cfcba053609661a22fcb841047ae93282",
            "model.safetensors": "ad861f4ddd8f7204cda1f407837403e9095e1fb1f36d4c9f4188185261cc4fd1"}

    def draft(self, cls="DFlash2DraftModel", name="dflash2"):
        folder = self.root / name
        folder.mkdir()
        # spread over lines as the published config.json is: the class check reads it without python
        (folder / "config.json").write_text('{\n  "architectures": [\n    "%s"\n  ],\n  "block_size": 8\n}\n' % cls)
        return folder

    def dry(self, **extra):
        extra.setdefault("DRAFTER", "")   # each case states its drafts; the profile's own DRAFTER=dspark is not one of them
        result = self.run_script(ROOT / "start.sh", DRY_RUN="1", HF_HOME=str(self.root / "hf"), **extra)
        out = result.stdout
        if "[dry-run] rank 1" not in out:
            return result, "", ""
        return result, out[out.index("[dry-run] rank 1"):out.index("[dry-run] rank 0")], out[out.index("[dry-run] rank 0"):]

    def test_the_pins_are_the_incoai_revision(self):
        result = self.bash('echo "$DFLASH2_REPO $DFLASH2_REVISION"; printf "%s\\n" "$DFLASH2_PINS"')
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "incoai/MiniMax-M3.1-Flash-DFlash2 ac07a7c1ecaae26a7ace595756124dca974b432d")
        pins = {f: sha for sha, _, f in (line.split() for line in lines[1:])}
        self.assertEqual(pins, self.PINS)
        self.assertIn("ad861f4ddd8f7204cda1f407837403e9095e1fb1f36d4c9f4188185261cc4fd1 4247776488 model.safetensors", lines)

    def test_the_folder_defaults_beside_the_pack(self):
        cache = self.bash('echo "$DFLASH2_DIR"', HF_HOME=str(self.root / "hf"))
        self.assertEqual(cache.stdout.strip(), f"{self.root}/hf/dflash2/MiniMax-M3.1-Flash-DFlash2", cache.stderr)
        beside = self.bash('echo "$DFLASH2_DIR $(dflash2_dir_here)"', WEIGHTS="dir", MODEL_DIR="~/packs/pack-mm31/")
        self.assertEqual(beside.stdout.split(), ["~/packs/MiniMax-M3.1-Flash-DFlash2",
                                                 f"{self.root}/packs/MiniMax-M3.1-Flash-DFlash2"], beside.stderr)
        given = self.bash('echo "$DFLASH2_DIR"', DFLASH2_DIR="/srv/drafts/d2")
        self.assertEqual(given.stdout.strip(), "/srv/drafts/d2")

    def test_the_workers_folder_follows_its_pack(self):
        fake(self.bin, "ssh", f'echo "ssh $*" >> {self.calls}\nprintf /var/lib/rootish\n')
        given = self.bash("worker_dflash2_dir", DFLASH2_DIR="~/d2")
        self.assertEqual(given.stdout, "/var/lib/rootish/d2\n", given.stderr)
        beside = self.bash("worker_dflash2_dir", WEIGHTS="dir", MODEL_DIR="/a/pack", WORKER_MODEL_DIR="/b/pack")
        self.assertEqual(beside.stdout, "/b/MiniMax-M3.1-Flash-DFlash2\n", beside.stderr)
        own = self.bash("worker_dflash2_dir", DFLASH2_DIR="/a/d2", WORKER_DFLASH2_DIR="/c/d2")
        self.assertEqual(own.stdout, "/c/d2\n")

    def test_a_relative_folder_is_refused(self):
        result = self.bash("true", DFLASH2_DIR="drafts/d2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DFLASH2_DIR is an absolute path or one under ~/", result.stderr)

    def test_the_class_is_read_from_config_json(self):
        good = self.draft()
        result = self.bash("dflash2_class head", DFLASH2_DIR=str(good))
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "DFlash2DraftModel"), result.stderr)
        dspark = self.draft("DSparkMiniMaxDraftModel", "dspark")
        result = self.bash("dflash2_class head", DFLASH2_DIR=str(dspark))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("names DSparkMiniMaxDraftModel in architectures, not a DFlash2 draft", result.stdout)
        result = self.bash("dflash2_class head", DFLASH2_DIR=str(self.root / "none"))
        self.assertIn("no " + str(self.root / "none") + "/config.json", result.stdout)

    def test_the_files_are_checked_against_the_pins(self):
        folder = self.draft()
        for name in ("README.md", "model.safetensors"):
            (folder / name).write_bytes(name.encode())
        sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        size = (folder / "model.safetensors").stat().st_size
        pins = "\n".join(f"{sums[f]} {size if f == 'model.safetensors' else '-'} {f}" for f in sorted(sums))
        snippet = f'DFLASH2_PINS="{pins}"; bash -c "$(dflash2_verify_cmd {folder})" && bash -c "$(dflash2_verify_cmd {folder} quick)" && echo matched'
        self.assertEqual(self.bash(snippet).stdout.strip(), "matched")
        (folder / "README.md").write_text("changed")
        self.assertNotEqual(self.bash(snippet).returncode, 0, "a changed file fails the full check")
        self.assertNotEqual(self.bash(f'bash -c "$(dflash2_verify_cmd {folder})"').returncode, 0, "the real pins")

    def test_drafts_is_0_1_or_dflash2(self):
        result, _, _ = self.dry(DRAFTS="2")
        self.assertEqual(result.returncode, 1)
        self.assertIn("DRAFTS is 0 (off), 1 (copy drafts, DRAFTER optional) or dflash2", result.stderr)
        result, _, _ = self.dry(DRAFTS="dflash2", DRAFTER="dspark")
        self.assertEqual(result.returncode, 1)
        self.assertIn("DRAFTS=dflash2 loads the DFlash2 draft of DFLASH2_DIR: leave DRAFTER empty", result.stderr)
        result, _, _ = self.dry(DRAFTS="0", DRAFTER="dflash2")
        self.assertIn("needs DRAFTS=1", result.stderr)

    def test_a_folder_that_is_not_a_dflash2_draft_is_refused(self):
        result, _, _ = self.dry(DRAFTS="dflash2", DFLASH2_DIR=str(self.draft("DSparkMiniMaxDraftModel", "dspark")))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not a DFlash2 draft (DFlash2DraftModel MiniMaxDFlash2DraftModel)", result.stderr)
        self.assertIn("scripts/prepare.sh --dflash2 fetches it", result.stderr)

    def test_dry_run_mounts_the_draft_on_both_ranks(self):
        folder = self.draft()
        result, rank1, rank0 = self.dry(DRAFTS="dflash2", DFLASH2_DIR=str(folder))
        self.assertEqual(result.returncode, 0, result.stderr)
        for rank in (rank0, rank1):
            self.assertIn("--drafter /dflash2", rank)
            self.assertIn(":/dflash2:ro", rank)
            self.assertNotIn("--no-drafts", rank)
            self.assertIn("MM31_SERVE_PROFILE=", rank)
            self.assertIn("drafts=dflash2,drafter=dflash2,dflash2=ac07a7c1ecaa,", rank.replace("\\", ""))  # %q escapes the commas
        self.assertIn(f"-v {folder}:/dflash2:ro", rank0)
        self.assertIn(f"DFlash2 draft on the head: {folder} (DFlash2DraftModel)", result.stdout)
        # DRAFTS=1 DRAFTER=dflash2 is the same draft
        result, rank1, _ = self.dry(DRAFTS="1", DRAFTER="dflash2", DFLASH2_DIR=str(folder))
        self.assertIn("--drafter /dflash2", rank1)

    def test_a_missing_folder_is_a_warning_in_a_dry_run(self):
        result, rank1, _ = self.dry(DRAFTS="dflash2", DFLASH2_DIR=str(self.root / "not-yet"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY_RUN: no " + str(self.root / "not-yet") + "/config.json on the head", result.stderr)
        self.assertIn("--drafter /dflash2", rank1)

    def test_plain_copies_and_dspark_are_as_before(self):
        result, rank1, rank0 = self.dry(DRAFTS="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--no-drafts", rank1)
        self.assertNotIn("dflash2", rank0 + rank1)
        result, rank1, rank0 = self.dry(DRAFTS="1")
        self.assertNotIn("--no-drafts", rank1)
        self.assertNotIn("--drafter", rank0 + rank1)
        result, rank1, rank0 = self.dry(DRAFTS="1", DRAFTER="dspark")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(rank1, r"--drafter /hf/hub/models--[^ ]+/snapshots/[0-9a-f]{40}/dspark")
        self.assertNotIn("/dflash2", rank0 + rank1)
        state = self.bash("prepared_state 2>/dev/null || true; DRAFTS=dflash2; prepared_state 2>/dev/null || true")
        plain, drafted = [line for line in state.stdout.splitlines() if line.strip()][:2]
        self.assertNotIn("dflash2=", plain)
        self.assertIn("dflash2=ac07a7c1ecaae26a7ace595756124dca974b432d:", drafted)

    def test_prepare_has_the_dflash2_step_without_a_token(self):
        text = (ROOT / "scripts" / "prepare.sh").read_text()
        self.assertIn("--dflash2) DFLASH2_ONLY=1", text)
        self.assertIn("env -u HF_TOKEN HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=0", text)
        self.assertIn('--revision "$DFLASH2_REVISION"', text)
        self.assertIn("if dflash2_wanted; then fetch_dflash2; fi", text)
        self.assertIn("rsync -a --partial", text.split("fetch_dflash2() {")[1].split("\n}\n")[0])


class PairWatchTests(ScriptFixture, unittest.TestCase):
    def watch(self, worker_reply, worker_rc=0, seconds=4):
        fake(self.bin, "docker", f'echo "docker $*" >> {self.calls}\ncase "$*" in *inspect*) echo "true 0" ;; esac\n')
        fake(self.bin, "ssh", f'echo "ssh $*" >> {self.calls}\necho "{worker_reply}"\nexit {worker_rc}\n')
        state = self.root / "state"
        proc = subprocess.Popen(["bash", "scripts/pairwatch.sh", "run", "c", "w", "1", str(state),
                                 str(self.root / "pw.log")], cwd=ROOT, env=self.env())
        try:
            proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5)
        return state, proc.returncode

    def test_a_lost_rank_stops_the_other_and_leaves_a_record(self):
        state, _ = self.watch("false 137")
        record = (state / "failfast").read_text()
        self.assertIn("rank 1 exited with code 137", record)
        self.assertIn("docker stop -t 10 c", self.calls.read_text())

    def test_an_unreachable_worker_is_not_an_exit(self):
        state, _ = self.watch("", worker_rc=255, seconds=3)
        self.assertFalse((state / "failfast").exists())
        self.assertNotIn("docker stop", self.calls.read_text())
        self.assertIn("unreachable", (self.root / "pw.log").read_text())


class ThermalStopTests(ScriptFixture, unittest.TestCase):
    """memguard's hard stop: the hottest of GPU and ACPI at or above the limit for the hold kills the rank."""

    def guard(self, zone_c, gpu_c, seconds=7, hold=2, device="normal", stale=False):
        zone = self.root / "thermal_zone0"
        zone.mkdir()
        (zone / "temp").write_text(f"{zone_c * 1000}\n")
        (self.root / "meminfo").write_text("MemAvailable:   104857600 kB\n")
        fake(self.bin, "nvidia-smi", f"echo {gpu_c}\n")
        state = self.root / "state"
        state.mkdir()
        limits = self.root / "thermal.env"
        limits.write_text(f"GUARD_PAUSE_C=92\nGUARD_RESUME_C=86\nGUARD_KILL_C=93\nGUARD_KILL_HOLD_S={hold}\n")
        (self.thermal_run / "state").write_text(device + "\n")
        (self.thermal_run / "temp").write_text(f"{max(zone_c, gpu_c)}\n")
        if stale:
            os.utime(self.thermal_run / "state", (time.time() - 120, time.time() - 120))
        env = self.env(MEMGUARD_THERMAL_GLOB=str(self.root / "thermal_zone*/temp"),
                       MEMGUARD_MEMINFO=str(self.root / "meminfo"), MEMGUARD_THERMAL_ENV=str(limits),
                       MEMGUARD_THERMAL_RUN=str(self.thermal_run))
        proc = subprocess.Popen(["bash", "scripts/memguard.sh", "run", "c", "3", str(self.root / "mg.log"),
                                 str(state / "memguard.pid"), "4", "2.75"], cwd=ROOT, env=env)
        time.sleep(seconds)
        proc.terminate()
        proc.wait(timeout=5)
        return state / "thermal-stop", self.calls.read_text() if self.calls.exists() else ""

    def test_a_held_reading_at_the_limit_kills_and_leaves_a_record(self):
        record, calls = self.guard(93, 60)
        self.assertIn("docker kill c", calls)
        self.assertRegex(record.read_text(), r"^[0-9]+ 93 C$")

    def test_the_gpu_counts_as_much_as_the_zones(self):
        record, calls = self.guard(50, 95)
        self.assertIn("docker kill c", calls)
        self.assertTrue(record.exists())

    def test_under_the_limit_nothing_is_killed(self):
        record, calls = self.guard(92, 92, seconds=6)
        self.assertNotIn("docker kill", calls)
        self.assertFalse(record.exists())

    def test_a_device_emergency_stops_the_rank(self):
        record, calls = self.guard(80, 80, seconds=4, hold=30, device="emergency")
        self.assertIn("docker stop -t 10 c", calls)
        self.assertRegex(record.read_text(), r"^[0-9]+ 80 C$")

    def test_a_stale_device_state_is_not_an_emergency(self):
        record, calls = self.guard(80, 80, seconds=4, hold=30, device="emergency", stale=True)
        self.assertNotIn("docker stop", calls)
        self.assertIn("stale", (self.root / "mg.log").read_text())

    def test_memguard_refuses_without_device_limits(self):
        result = subprocess.run(["bash", "scripts/memguard.sh", "start", "c", "3", str(self.root / "mg.log"),
                                 str(self.root / "mg.pid")], cwd=ROOT, capture_output=True, text=True,
                                env=self.env(MEMGUARD_THERMAL_ENV=str(self.root / "none.env")), timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("install-device-limits.sh", result.stderr)

    def test_the_hold_must_elapse(self):
        record, calls = self.guard(94, 94, seconds=5, hold=30)
        self.assertNotIn("docker kill", calls)
        self.assertIn("hold starts", (self.root / "mg.log").read_text())


class CoherentStopTests(ScriptFixture, unittest.TestCase):
    """Exit 70 (the engine's coherent stop of both ranks) heals once, then the watchdog leaves the pair down."""

    def tick(self):
        fake(self.bin, "docker", 'case "$*" in *Running*) echo false ;; *ExitCode*) echo 70 ;; *) exit 0 ;; esac\n')
        fake(self.bin, "ssh", 'case "$*" in *Running*) echo false ;; *ExitCode*) echo 70 ;; *) exit 0 ;; esac\n')
        healed = self.root / "healed"
        return subprocess.run(["bash", "-c", 'source scripts/watch.sh; watch_tick'], cwd=ROOT, capture_output=True,
                              text=True, timeout=60,
                              env=self.env(STATE_DIR=str(self.root / "state"), WATCH_HEAL="1", WATCH_PROBE="0",
                                           WATCH_RESTART_CMD=f"echo healed >> {healed}")), healed

    def test_heals_once_then_alerts(self):
        first, healed = self.tick()
        self.assertIn("coherent stop", first.stderr)
        time.sleep(1)
        self.assertTrue(healed.exists(), first.stderr)
        second, _ = self.tick()
        self.assertIn("again within", second.stderr)
        time.sleep(1)
        self.assertEqual(healed.read_text().count("healed"), 1)


class CheckPublicTests(unittest.TestCase):
    def test_ai_tool_credit_passes_only_as_miaai_labs_verbatim_lines(self):
        # Mia's own credit text is kept verbatim in her two files; the same words anywhere else fail
        name = "Claude" + " Code"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "docs").mkdir()
            for f in ("check-public.sh", "check_secrets.py", "redact.py", "check-secrets-allow.txt"):
                shutil.copy(ROOT / "scripts" / f, root / "scripts")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            run = lambda: subprocess.run(["bash", "scripts/check-public.sh"], cwd=root, capture_output=True, text=True)
            (root / "docs" / "CREDITS-MIAAI-LAB.md").write_text(f"Every patch was written by Mia's AI Lab with {name}.\n")
            self.assertEqual(run().returncode, 0, "her verbatim credit line passes")
            for path, line in (("docs/CLIENTS.md", f"Written by Mia's AI Lab with {name}."),
                               ("docs/CREDITS-MIAAI-LAB.md", f"We used {name} for this."),
                               ("README.md", f"Built with {name}.")):
                with self.subTest(path=path):
                    (root / "docs" / "CREDITS-MIAAI-LAB.md").write_text(f"Every patch was written by Mia's AI Lab with {name}.\n")
                    target = root / path
                    target.write_text((target.read_text() if target.exists() and path.endswith("MIAAI-LAB.md") else "") + line + "\n")
                    self.assertEqual(run().returncode, 1, path)
                    if path != "docs/CREDITS-MIAAI-LAB.md":
                        target.unlink()

    def test_repo_is_clean(self):
        result = subprocess.run(["bash", str(ROOT / "scripts" / "check-public.sh")], capture_output=True, text=True,
                                cwd=ROOT, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_flags_each_planted_finding(self):
        # Each value is split so this file stays clean under the scan it tests; the key is built at run time.
        key = "sk-" + "t" * 4 + "".join(chr(97 + (i * 7) % 26) for i in range(32))
        planted = {
            "megahertz": "Cap it at 2000 M" "Hz.",
            "home": "Copy it from /ho" "me/alice/models/x.",
            "ip": "Rank 1 is at 192." "168.7.20.",
            "attribution": "Co-Authored" "-By: someone",
            "credential": f"export API_KEY={key}",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            for name in ("check-public.sh", "check_secrets.py", "redact.py", "check-secrets-allow.txt"):
                shutil.copy(ROOT / "scripts" / name, root / "scripts")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            clean = subprocess.run(["bash", "scripts/check-public.sh"], cwd=root, capture_output=True, text=True)
            self.assertEqual(clean.returncode, 0, clean.stdout)
            for name, line in planted.items():
                with self.subTest(name=name):
                    (root / "planted.md").write_text(line + "\n")
                    result = subprocess.run(["bash", "scripts/check-public.sh"], cwd=root, capture_output=True,
                                            text=True)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn("planted.md", result.stdout)
                    self.assertNotIn(key, result.stdout + result.stderr, "a finding prints the value masked")


class CalculatorTests(unittest.TestCase):
    def test_reproduces_the_release_and_hits_its_targets(self):
        # model.safetensors.index.json total_size at 54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e
        self.assertEqual(qb.release_bytes(), 250_103_411_200)
        for target in qb.TARGETS:
            self.assertAlmostEqual(qb.total(qb.pack(qb.solve_routed(target))) / 1e9, target, places=3)
        self.assertAlmostEqual(qb.solve_routed(qb.PRIMARY_GB), 3.551, places=3)
        self.assertEqual((qb.KV_RANK_BF16, qb.KV_RANK_FP8), (76_800, 47_040))

    def test_the_engine_default_reserve_leaves_this_pack_little_kv(self):
        default = qb.serving(199.9, 131072, 1, "fp8")
        chosen = qb.serving(199.9, 131072, 1, "fp8", 8)
        self.assertAlmostEqual(default["reserve"], qb.MEMTOTAL_GIB / 10)
        self.assertLess(default["kv_room"], 7)
        self.assertGreater(chosen["max_tokens"], 200_000)
        self.assertAlmostEqual(chosen["weights"], 92.66, places=1)

    def test_the_kv_tier_slice_is_the_fp8_token_cost_by_pages(self):
        self.assertEqual((qb.SLICE_RANK_FP8, qb.SLICE_NVME, qb.IK_TOKEN_RANK), (100_352, 102_400, 15_360))
        self.assertEqual(qb.SLICE_RANK_FP8 * qb.LAYERS // qb.PAGE, qb.KV_RANK_FP8)
        self.assertEqual(qb.SUMM_SLICE, 512)

    def test_the_kv_tier_prices_4_x_1m(self):
        room = qb.serving(199.9, 0, 4, "fp8", 8)["kv_room"]
        t = qb.tier(1_048_576, 4, room)
        self.assertAlmostEqual(t["summaries"], 0.94, places=2)
        self.assertAlmostEqual(t["horizon_ik"], 1.875, places=3)
        self.assertAlmostEqual(t["nvme_live"], 187.5, delta=0.1)
        self.assertTrue(t["ok"])
        self.assertEqual(t["exact_to"], 32768 + 16 * 128)
        self.assertTrue(qb.tier(1_048_576, 4, 7.63)["ok"], "the room measured at the first TP2 load holds it too")
        self.assertFalse(qb.tier(1_048_576, 4, 4.0)["ok"], "a room that leaves too small a slice cache refuses")

    def test_the_draft_costs_kv(self):
        self.assertLess(qb.serving(199.9, 131072, 1, "fp8", 8, True)["max_tokens"],
                        qb.serving(199.9, 131072, 1, "fp8", 8)["max_tokens"])

    def test_the_draft_term_includes_its_ring_snapshots(self):
        # dspark_config.cost with one stream (budget.draftBytes) plus native.draftCost's 8 ring snapshots of 21,053,440 B
        self.assertEqual(qb.DSPARK_SNAPSHOT_BYTES, 8 * 21_053_440)
        self.assertAlmostEqual(qb.DSPARK_RANK_GIB, (2_180_405_250 + 168_427_520) / 2**30)
        self.assertAlmostEqual(qb.serving(199.9, 131072, 1, "fp8", 8, True)["draft"], 2.1875, places=4)

    def test_the_dflash2_draft_is_priced_as_the_engine_prices_it(self):
        # the engine's dflash2_config tests: the whole draft's weights equal the safetensors data bytes; a rank of two
        # holds half the heads and MLP with fc, the convs and the selector whole
        self.assertEqual(sum(qb.dflash2_weights(world=1).values()), 4_247_767_552)
        w = qb.dflash2_weights()
        self.assertEqual(w["layers"] * 2, qb.dflash2_weights(world=1)["layers"])
        self.assertEqual(sum(w.values()), 2_549_074_432)
        self.assertAlmostEqual(sum(w.values()) / qb.GIB, 2.37, places=2)
        # a ring: (2048 + 16) positions x 5 layers x K,V x 4 kv heads x 128 x bf16
        self.assertEqual(qb.dflash2_ring(), (2048 + 16) * 5 * 2 * 4 * 128 * 2)
        self.assertEqual(qb.dflash2_ring(), 21_135_360)
        cost = qb.dflash2_cost()
        self.assertEqual(cost["runtime"], (512 << 20) + qb.dflash2_scratch() + 8 * 21_135_360)
        self.assertEqual(qb.dflash2_scratch(), 59_568_384)
        self.assertLess(sum(w.values()), qb.DFLASH2_FILE_BYTES)

    def test_the_drafter_switch_keeps_dspark(self):
        self.assertEqual(qb.draft_gib(False, 4), 0.0)
        self.assertAlmostEqual(qb.draft_gib(True, 4) - qb.DSPARK_RANK_GIB, 3 * 21_053_440 / qb.GIB)  # a ring a stream
        self.assertEqual(qb.draft_gib("dspark", 1), qb.DSPARK_RANK_GIB)
        one, four = qb.draft_gib("dflash2", 1), qb.draft_gib("dflash2", 4)
        self.assertAlmostEqual(four - one, 3 * 21_135_360 / qb.GIB)
        self.assertAlmostEqual(one, 3.11, places=2)
        d2 = qb.serving(199.9, 131072, 1, "fp8", 8, "dflash2")
        self.assertLess(d2["max_tokens"], qb.serving(199.9, 131072, 1, "fp8", 8, "dspark")["max_tokens"])
        self.assertEqual(d2["mem_gate"], 116)
        with self.assertRaises(ValueError):
            qb.draft_gib("radixark", 1)

    def test_the_calculator_cli_takes_dflash2(self):
        base = [sys.executable, "-I", str(ROOT / "scripts" / "quant_budget.py"), "--serve", "--context", "1048576",
                "--parallel", "4", "--kv", "fp8", "--reserve-gib", "8", "--kv-tier"]
        d2 = subprocess.run([*base, "--drafter", "dflash2"], capture_output=True, text=True, timeout=60)
        self.assertEqual(d2.returncode, 0, d2.stdout + d2.stderr)
        self.assertIn("MEM_GATE_GIB=115", d2.stdout)
        self.assertIn("DFlash2 draft (2.37 weights", d2.stdout)
        dspark = subprocess.run([*base, "--drafter"], capture_output=True, text=True, timeout=60)
        self.assertIn("draft dspark", dspark.stdout)


class ProvenanceTests(unittest.TestCase):
    def test_the_engine_patches_are_as_imported(self):
        files = sorted(p for p in (ROOT / "patches").rglob("*") if p.is_file())
        lines = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n" for p in files)
        self.assertEqual(hashlib.sha256(lines.encode()).hexdigest(), PATCHES_TREE_SHA256)

    def test_mia_is_credited_and_her_notice_kept(self):
        notice = (ROOT / "NOTICE").read_text()
        self.assertIn("MiaAI-Lab/MiniMax-M3.1-Flash-EXL3-2x-DGX-Sparks-TensorFold", notice)
        self.assertIn("  | Copyright 2026 MiaAI-Lab", notice)
        for name in ("LICENSE", "docs/CREDITS-MIAAI-LAB.md", "docs/CREDITS.md", "docs/LICENSING.md"):
            self.assertTrue((ROOT / name).is_file(), name)
        for name in ("README.md", "docs/CREDITS.md", "docs/LICENSING.md"):
            self.assertIn("MiaAI-Lab", (ROOT / name).read_text(), name)

    def test_port_table_has_a_verdict_and_a_reason_on_every_row(self):
        features = (ROOT / "docs" / "PORT.md").read_text().split("## Engine features")[0]
        rows = [line for line in features.splitlines()
                if line.startswith("| ") and not line.startswith(("| Feature", "| ---"))]
        self.assertGreater(len(rows), 20)
        for row in rows:
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]   # "\|" is a literal bar
            with self.subTest(row=cells[0]):
                self.assertIn(cells[2].lower().strip("*"), {"ported", "adapted", "not applicable"})
                self.assertTrue(cells[3])

    def test_no_planned_engine_names_are_left(self):
        for path in [ROOT / "start.sh", *ROOT.glob("scripts/**/*.sh"), *ROOT.glob("scripts/**/*.py"), EXAMPLE]:
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertNotRegex(path.read_text(), r"TF_MINIMAX_|ENGINE_FOLLOW_ARGS")


if __name__ == "__main__":
    unittest.main()
