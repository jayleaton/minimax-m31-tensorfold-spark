# MiniMax-M3.1-Flash weight loader speed (task 0015)

Branch `mm3-loadspeed` (one commit on `mm3-int`), patch `patches/tasks/0015-minimax-m3-loader-speed.patch`. Host only: no GPU
was used, no container or server was touched, and nothing was mounted.

## What was slow

First real TP2 start: the load took 1874 s. Rank 1 read the shards over NFS at about 60 MiB/s and rank 0 from local NVMe at
about 160 MiB/s, while the hardware does GB/s. The three hypotheses, tested against the real index and headers of the pinned
snapshot `fd3466a0...` (25 shards, 67,523 text tensors, read with `mm3-loadbench stats`, which builds the real `Plan` with
`Share.of(c, rank, 2)` and reads nothing but the headers):

1. **Many small non-contiguous runs read one at a time (confirmed, this is the cause).** A TP2 rank plans 5,771,542 byte
   runs for 91.08 GiB. Under a column cut a trellis tensor is one run per K-tile row (`Span.count = k_tiles`, `stride` = the
   whole row), so every routed expert's `w1` and `w3` trellis is 384 runs of 9,216 to 12,288 bytes with a gap the same size
   between them. The old streamer (`runSerial`, kept in the code) reads each run with its own aligned `pread`, queue depth 1:
   5,771,961 payload reads of 16.5 KiB on average, plus 22,309 marker reads, 5.79 million reads in all. 1874 s / 5.79 M is 0.32
   ms a read, which is a network round trip; 91 GiB at the local 160 MiB/s is 0.1 ms a read. The device is never busy, the
   request latency is. Both ranks have the same shape (rank 1 differs only in the last bytes of `head`).
2. **Marker checks (real but small).** 22,309 four-byte reads, each a 4 KiB aligned read, one at a time: about 0.23 ms each
   locally, 5 s of the load, 0.3 % of the reads. Over NFS roughly 7 s. Worth batching, not the cause.
3. **Small bounce buffer (not a limit).** The 8 MiB buffer was never the cap: 98 % of runs are 4-16 KiB, so reads were 16 KiB
   long whatever the buffer was. A bigger buffer alone changes nothing.

### Run lengths under the TP2 cut, rank 0 (rank 1 is identical in every count)

Run-length buckets count runs. `old reads` is one per run; sizes are the byte length of one run.

| kind | tensors | runs | MiB | run bytes | runs < 4K | 4-16K | 16-64K | 1-8M | >= 8M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q.trellis | 60 | 23,040 | 1,080.0 | 49,152 | | | 23,040 | | |
| k.trellis, v.trellis (each) | 60 | 23,040 | 67.5 | 3,072 | 23,040 | | | | |
| o.trellis | 60 | 60 | 1,080.0 | 18,874,368 | | | | | 60 |
| dense gate/up trellis (each) | 3 | 1,152 | 67.5 | 61,440 | | | 1,152 | | |
| dense down trellis | 3 | 3 | 67.5 | 23,592,960 | | | | | 3 |
| shared gate/up trellis (each) | 57 | 21,888 | 320.6 | 15,360 | | 21,888 | | | |
| shared down trellis | 57 | 57 | 320.6 | 5,898,240 | | | | 57 | |
| w1.trellis, w3.trellis (each) | 7,296 | 2,801,664 | 28,800.0 | 9,216 to 12,288 | | 2,801,664 | | | |
| w2.trellis | 7,296 | 7,296 | 28,656.0 | 3.5 to 4.7 M | | | | 7,296 | |
| head.trellis | 1 | 384 | 439.9 | 1,201,152 | | | | 384 | |
| suh and svh of all matrices | 44,618 | 44,618 | 327.4 | 512 to 200,192 | 22,179 | 22,438 | | | |
| embed | 1 | 1 | 2,344.5 | 2,458,386,432 | | | | | 1 |
| index and router projections | 177 | 177 | 441.0 | 1.5 to 3.1 M | | | | 177 | |
| norms and router bias | 418 | 418 | 1.4 | 256 to 12,288 | 297 | 121 | | | |
| **all** | 67,523 | 5,771,542 | 93,269.6 | 256 to 2.4 G | 68,556 | 5,669,663 | 25,344 | 7,914 | 64 |

The table has no 64-256K column: one run, `head.svh`, falls there. Bytes by run length: under 4 KiB 199.8 MiB, 4-16 KiB 58,505 MiB (62.7 %), 16-64 KiB 1,215 MiB, 1-8 MiB 29,858 MiB, 8 MiB and
over 3,492 MiB. So 98.2 % of the runs and 63 % of the bytes are 4-16 KiB runs; only 7,978 runs (0.14 %) are a megabyte or
more, and those are the row-cut tensors (`w2`, `o`, `shared_down`, `dense_down`), which were already large reads. The full
per-projection table (`mm3-loadbench stats <dir> <rank> 2`) has one row per projection and part. With the old 8 MiB buffer
the 5,771,542 runs became 5,771,961 reads (the 64 oversize runs split), 16.5 KiB per read; 22,309 marker reads on top.

## The fix

`zig/src/families/minimax_m3/stream.zig` (`Streamer`, same `init(gpa, cat, chunk)` / `run(plan, sink)` / `verifyMarkers`
interface, `weights_device.feed` and `native.zig` unchanged):

* **Schedule.** `run` walks the plan's runs once in (shard, offset) order and folds them into windows: runs less than `gap`
  (default 1 MiB) apart share one read, a window is at most one buffer (default 16 MiB, so reads of up to 16 MiB less 8 KiB),
  never crosses a shard, and a run longer than the window is cut where the window ends. The bytes between runs are read and
  dropped. Real plan, rank 0: 15,586 payload reads of 9.8 MiB (against 5.77 M reads of 16.5 KiB), 149.3 GiB read for 91.08 GiB
  wanted (1.64x: the half of each column-cut row that belongs to the other rank is read too).
* **Reads in flight.** `threads` reader threads (default 6) claim windows in order and `pread` them (O_DIRECT, 4 KiB aligned,
  through `dio.File.read`, so the short-read retry and the "O_DIRECT refused, go through the cache" fallbacks are the old
  ones) into a ring of `threads + 2` buffers (128 MiB at the defaults). Over NFS a 16 MiB direct read is 16 parallel 1 MiB
  RPCs spread over the 8 `nconnect` connections, so six reads in flight keep about 96 RPCs outstanding.
* **Double buffering.** The calling thread is the only one that touches the sink (the CUDA context stays on it): it waits for
  window i, copies it, and releases its buffer, while the readers fill the next ones. A futex bell (`Bell`) wakes threads,
  nothing spins.
* **Packed puts.** Inside a window the wanted bytes are packed in place (runs arrive in increasing offset, so the destination
  never overtakes the source) and a cut tensor is sent as a few puts of at most `put_max` (4 MiB) instead of one put per run:
  101,148 puts for one layer became 1,315. Norm weights are folded (`foldGemma`) exactly as before and never split.
* **Markers.** `verifyMarkers` collects every matrix' marker, sorts by (shard, offset) and reads them through the same
  scheduler with reads in flight, still before the first byte reaches the sink (a bad marker leaves every slot untouched; the
  test that asserts it passes). Markers sit a whole tensor apart, so they do not coalesce: 22,309 reads, but six at a time.
* **Fail closed, unchanged.** Short or refused reads, end of file, a run past its slot (`BadPlan`), a fold run that is not
  whole (`BadPlan`), a mul1 marker that is not mul1 (`BadCodebookMarker`) all stop the stream with that error; a read error
  on a worker thread is carried to the copy loop and returned there, with the other threads joined first.
* **Memory.** Read-side buffers are `(threads + 2) x chunk` = 128 MiB by default (was 8 MiB), freed with the streamer. The
  page cache is not used by O_DIRECT; if a file system refuses O_DIRECT the old per-shard `dropCache` hook still runs after each
  shard, and now also after every 512 MiB read of a shard.
* **Tuning knobs, no rebuild** (read by `Streamer.init`): `TF_MM3_LOAD_THREADS`, `TF_MM3_LOAD_CHUNK_MIB`,
  `TF_MM3_LOAD_GAP_KIB`, `TF_MM3_LOAD_PUT_MIB`. Use them on the first real run if NFS wants more reads in flight.
* The old strategy stays in the file as `runSerial` / `verifyMarkersSerial`: the tests compare against it and the benchmark
  uses it as the baseline.

### Tests (host, no GPU)

`zig build test-minimax-load` (`stream_test.zig`): on the synthetic mini checkpoint, for world 1 and 2, every rank, 1/2/3/5
shards, chunk 16 KiB to 16 MiB, 0 to 8 threads, gap 0 to 1 MiB, put limit 5,000 bytes to 4 MiB: every slot buffer and every
written-byte count equals the `runSerial` output byte for byte, and equals the cut tiles, scales and folded norms the
existing `expectStreamed` computes from the full tensors; the stats agree with the plan (`wanted_bytes`, marker count) and
with coalescing happening on a TP2 cut. A second test empties the shards under open descriptors and asserts the streamer
returns the read error. The existing tests (bad marker stops before any put; truncation, duplicate, misshape, index
mismatch) pass unchanged.

## Measurements

Host spark1, local NVMe (`/dev/nvme0n1p2`, ext4), O_DIRECT through the pinned snapshot (read-only), live MiniMax server
running on the same machine and GPU (so the numbers are noisy by 10 % and the CPU is shared). Tool:
`zig build mm3-loadbench` installs `zig-out/bin/mm3-loadbench` (ReleaseSafe). All buffers together stay within 512 MiB (the tool refuses more), the tool stops when MemAvailable is under 9
GiB and checks it every 256 MiB (two of the longer runs stopped themselves that way, see below); tool RSS peaked at 300 MiB.
Each test reads at most 8 GiB (the tool computes the plan's read volume first and refuses above `--limit-gib`).

Device ceiling (`raw`, sequential O_DIRECT, 3000 MiB of one shard): 16 MiB blocks 10,600 MiB/s with 1 thread and 10,500 with 4;
4 MiB blocks 8,600 MiB/s with 1 thread, 10,000 with 16; 1 MiB blocks 6,900 MiB/s with 1 thread, 11,500 with 32.

Strided pattern (`pattern`, 10,240-byte runs every 20,480 bytes of shard 3, 1 GiB wanted, which is a w1 trellis cut shape):

| strategy | MiB/s of wanted bytes | reads | time |
|---|---:|---:|---:|
| old (one read per run, QD1) | 42.4 and 44.7 | 104,857 | 24.2 s, 22.9 s |
| new | 4,871 and 4,948 | 129 (16 MiB) | 0.21 s |

Real plan slices (`model`, the plan cut to a layer or two, without embed and head, sink does no device copy):

| slice | strategy | MiB/s wanted | MiB/s read | reads | seconds |
|---|---|---:|---:|---:|---:|
| rank 0, layer 10 (1.75 GiB) | old | 69.8, 74.2 | | 101,539 | 25.7, 24.2 |
| | new | 5,485, 5,586 | 9,100 | 655 | 0.33, 0.32 |
| rank 1, layer 10 | old | 75.5, 76.2 | | 101,539 | 23.8, 23.6 |
| | new | 5,528, 5,483 | 9,178 | 655 | 0.32, 0.33 |
| rank 0, layers 20-21 (2.66 GiB) | old | 54.4, 52.3 | | 203,078 | 50.2, 52.2 |
| | new | 6,149 | 10,196 | 1,310 | 0.44 |
| rank 1, layers 20-21 | old | 54.9 | | 203,078 | 49.7 |
| | new | 5,961 | 9,885 | 1,310 | 0.46 |

That is 75x to 110x on the local NVMe, and the new path is at the device ceiling (10 GB/s read): the remaining time is the
in-place packing. Marker pass: 0.09 to 0.19 s old and 0.02 to 0.04 s new per 391 to 782 markers, so about 5 s against about 1
s for the 22,309 of a whole rank. The old local rate here (52-76 MiB/s) is lower than the 160 MiB/s seen in the real start;
the server shares the CPU and the box was busy, so read the ratio, not the absolute old figure.

Thread, chunk and gap sweep on rank 1 layer 10, new strategy only (MiB/s wanted, single runs, noise about 10 %): threads 1:
4,363; 2: 5,946; 4: 6,359; 6: 5,483; 8: 5,607; 12: 5,690. Chunk 2 MiB 6,366; 4: 6,496; 8: 6,698; 16: 5,500; 32: 5,392. Gap 0
(no coalescing, 101,139 reads on six threads): 301; 16 KiB: 5,329; 64 KiB: 5,464; 256 KiB: 5,505; 4 MiB: 5,496. Local NVMe
cannot rank these (it is saturated from two threads and 2 MiB blocks); the defaults (6 threads, 16 MiB, 1 MiB gap) are chosen
for the network: more RPCs in flight, a gap near the bandwidth-delay product. Note gap 0 with six threads is still 300 MiB/s,
four times the old speed, so queue depth alone helps, but coalescing is what reaches the device.

Projected whole-model load (not measured end to end): 91 GiB wanted, 149 GiB read per rank, at the 5.5 GB/s the local slices
reached that is about 17 s of reading for rank 0, against 583 s at the 160 MiB/s seen before.

### What I could not measure

* **NFS.** Not measured at all. `/mnt/spark1` on spark2 exists but is empty (nothing mounted there); the real export is
  mounted on spark2 as a Docker volume under `/var/lib/docker/volumes/minimax-m3-flash-hf/_data` (nfs4, `vers=4.2`,
  `rsize=wsize=1048576`, `nconnect=8`, addr <head-address>), which is root only (`Permission denied` for the `<user>` user), and the
  rules forbade sudo, mounting and touching Docker. So the NFS figure is reasoning from the settings: the old run did 0.32 ms a
  read, six large reads of 16 MiB in flight should be limited by the link, the nconnect pool and the server's page cache
  reads, not by round trips. Expect rank 1 to read 149 GiB over the 200 Gb/s link (the NFS server, nfsd on spark1, reads
  through its own page cache, which is reclaimable but will churn) and rank 0 the same locally, so both ranks together read
  about 300 GiB from one NVMe: at 5 to 10 GB/s that is 30 to 60 s. Treat "a few minutes at worst" as an estimate, not a
  result.
* **The device copy.** The sink copies each put through `copy_source` (a staged pinned copy, then `cuMemcpyHtoDAsync`, a
  stream sync every 256 MiB). Puts went from about 100,000 of 10 KiB per layer to about 1,300 of up to 4 MiB; that should be
  cheaper, but I could not run it (no GPU use allowed), so the load's end to end time includes an unmeasured upload path
  that may now be the limit (a pageable H2D copy is a few GB/s on GB10). `TF_MM3_LOAD_PUT_MIB` changes the put size.
* **A whole-rank run.** Each test was capped at about 8 GiB and the rank-wide numbers above are extrapolated from the
  per-layer slices and from the full plan's run statistics. Two of my longer runs stopped themselves at the 9 GiB MemAvailable
  guard (the server's memory moved, my tool held 300 MiB and no page cache), so those two have fewer repeats.
* The reads of the over-read ratio on NFS (1.64x network bytes) were not weighed against a smaller gap on the real mount.

## Reproduce

```
cd tf-wt-loadspeed && export ZIG_GLOBAL_CACHE_DIR=~/TensorFold/tensorfold-1.1.2/.zig-global-cache
zig build mm3-loadbench
S=~/.cache/huggingface/hub/models--Mia-AiLab--MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw/snapshots/fd3466a016989e0e1fceb901453046fdd7247dec
zig-out/bin/mm3-loadbench stats $S 0 2                       # run lengths, reads old and new (headers only)
zig-out/bin/mm3-loadbench model $S 1 2 --layers 1 --first 10 # real plan slice, old and new, MiB/s
zig-out/bin/mm3-loadbench pattern $S/model-00003-of-00025.safetensors 10240 20480 1024
zig-out/bin/mm3-loadbench raw $S/model-00003-of-00025.safetensors 16384 4 3000
```

Over NFS run the same on spark2 with `$S` on the mount (a mount readable by the user is needed; none was).
