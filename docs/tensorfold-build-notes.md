# TensorFold v1.1.2 build notes (GB10 / sm_121)

Tree: `~/TensorFold/tensorfold-1.1.2` (tag v1.1.2, shallow clone, sources unmodified).
Zig: pinned 0.17.0 (`.zig-version`); `~/.local/bin/zig` is 0.17.0, so it matches. `build.zig` refuses any other Zig.
Host: aarch64 Linux, nvcc 13.0.88 at `/usr/local/cuda/bin/nvcc` (`/usr/local/cuda` -> cuda-13.0). nvidia-smi fails (driver down): no GPU tests.
Keep caches inside the clone: `export ZIG_GLOBAL_CACHE_DIR=$PWD/.zig-global-cache` (local `.zig-cache`, `zig-out` are already in the clone).

## 1. Build the native CUDA binary for GB10

```sh
cd ~/TensorFold/tensorfold-1.1.2
export ZIG_GLOBAL_CACHE_DIR=$PWD/.zig-global-cache
zig build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dversion=1.1.2 -Doptimize=ReleaseSafe
# -> zig-out/native/bin/tensorfold-native   (also zig-out/bin/tensorfold, tf-cuda-test)
./zig-out/native/bin/tensorfold-native --version   # tensorfold-native 1.1.2
```

- Needs nvcc: only to compile the kernel fatbins (`-Dnvcc`, SASS list `-Dsm`, default 121). No GPU or driver needed to build.
  Alternative: `-Dfatbins=/abs/dir` embeds prebuilt `<name>.fatbin` files (no nvcc). `zig build fatbins -Dnvcc=... -Dsm=121` builds them alone into `zig-out/fatbin/`.
- Without `-Dnvcc`/`-Dfatbins` the binary has no CUDA images (host-only / verification).
- Release archives: `zig build dist -Dversion=X -Ddist-fatbins=DIR` (packaging/README.md; needs sm 80,86,120,121 fatbins; macOS target needs Xcode).
- Runtime needs `libcuda.so.1` (driver) only to run, not to build. Optional `-Dstrip=true` drops build paths.
- Result on this machine: built OK in ~2m40s (68 MB binary).

## 2. Host-only tests run here (no GPU)

| Command | Result |
| --- | --- |
| `zig test zig/src/core/exl3_format.zig` | `All 2 tests passed.` |
| `zig build test-exl3-cuda-host` | `Build Summary: 7/7 steps succeeded; 568/568 tests passed` |
| `zig build test-kernel-contract` | `Build Summary: 11/11 steps succeeded; 72/72 tests passed` |
| `zig build test-cuda-host` | `Build Summary: 155/155 steps succeeded; 1217/1218 tests passed (1 skipped)` |
| `zig build check-native-cuda-host` | compiles native CUDA server with no kernels, success |
| `zig build native -Dnvcc=... -Dsm=121` | success (see section 1) |

Notes: `failed command: ./.zig-cache/.../test` lines and "peer OOM" / "libjpeg-turbo not found" / "another spelling ... declares a different width" lines
in the output are expected negative-test noise; the summary line is the verdict. Not run: full `zig build test` (also pulls Metal/macOS and other-backend
steps), `kernel-contract` (runs on a GPU), `tf-cuda-test ...` (needs GPU), `test-golden`.

## 3. Cheat sheet: families, kernel_contract, exl3

Build steps (all in `zig/build/*.zig`, wired from `build.zig`):
- Family/host (CPU): `test-cuda-host` (umbrella, `cuda.zig:hostTests`), `test-exl3-cuda-host`, `test-exl3-prompt-host`, `test-flashnext-contract-host`,
  `test-experimental-family-host` (release opt-in policy), `test-gemma4-contract-host`, `test-glm-*-host`, `test-dsv41-*`, `test-cuda-readiness-host`,
  `test-runtime-api`, `check-native-cuda-host`, `compile-cuda-tests` (builds `tf-cuda-test`).
- Kernel contract: `test-kernel-contract` (host gates + every registered launch vs. every chip; `kernel_contract.zig`), `test-fatbin-zstd-host`,
  `test-delta-slots-contract-host`, `test-contract-images-host`, `test-kernel-symbols-host`; GPU: `kernel-contract` (`-Dnvcc` required for controls),
  `kernel-contract-fatbins`, then `tf-cuda-test kernel-contract-limits zig-out/fatbin`; Metal: `tf-kernel-contract`.
- EXL3: CUDA `test-exl3-cuda-host` (`cuda.zig`), `test-exl3-prompt-host` (`glm_family_host.zig`), `tf-cuda-test exl3` / `exl3-tp2` (GPU);
  Metal `tf-exl3-bench`, `test-exl3`, `test-exl3-gpu` (`exl3.zig`); plus `exlr8.zig`. Single-file: `zig test zig/src/core/exl3_format.zig`.
- Core EXL3 sources: `zig/src/core/exl3_{format,layout,load,prepare,projection,projection_decode,metal_load}.zig`; `core/eaq_container.zig` (EAQ packs).

Registering a new family (CUDA, per the repo; Zig 0.17 conventions, one change per PR, open an issue first, files <= 600 lines, lean check `python3 -I tools/zig/lean_check.py`):
1. Code in `zig/src/families/<name>/` (config, weights, layer graph, kernels only for what is its own). Core owns lane kernels/rounds/prompt cache/engine_api.
2. Export from `zig/src/tensorfold.zig` (`pub const <name> = @import("families/<name>/<name>.zig");` plus test-only `_ = @import(...)` lines).
3. CUDA serving adapter: `zig/src/families/<name>/cuda*.zig` exposing `native` (with `model_type`, `formats`, open/lane backend; optional `healthy(ctx)`);
   add it to `registry` in `zig/src/native/cuda.zig` (line ~28: `nemotron.native, glm.native, qwen35.native, flashnext.native, kolibri1.native ...`);
   `families` and the count test at the bottom of that file derive from it. Metal: `zig/src/native/metal.zig` (model_type/format table ~l.41, `open` dispatch ~l.176) and
   `zig/src/native/<name>.zig`. Hub titles: `zig/src/cli/hub.zig` (`.model_type`, `.title`).
4. Kernels: `.cu` under `zig/kernels/cuda/`, listed in the `kernels` catalog (`zig/build/cuda_kernels.zig`, re-exported by `zig/build/cuda.zig`) (name becomes `<name>.fatbin`; dist also lists them in packaging/README.md);
   Metal sources in `zig/build/metal_source_lists.zig`. Add module wiring in `zig/build/cuda.zig` (`family()`, `engines()`, `glmCuda`-style helper) if it needs its own import.
5. Kernel contract: register each launch in `zig/src/core/contract/registry.zig` (symbol, image, threads, shared bytes, per-SM usage records for sm_80/86/89/120/121)
   and give its test a `contract.spec.Spec` with FP64 reference (docs/kernel-contract.md); post each backend's `SUMMARY` line in the PR.
6. Host tests: add a `test-<name>-host` step in a new `zig/build/<name>_host.zig` and hook it to `test-cuda-host` / `test`.
7. Docs: recipe in `docs/recipes/`, README qualified-models row, CHANGELOG, credits in `docs/credits.md`.
Gates before claiming support: drafted == plain tokens, resumed == fresh, concurrent == solo, FP64 references per projection class (CONTRIBUTING.md).
