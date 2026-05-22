# CLAUDE.local.md (gitignored, machine-specific)

> **Session-handoff document.** Force-committed (`git add -f`) so the user
> can pick this up on another machine. CLAUDE.local.md is still in
> `.gitignore`; this commit just makes the current contents reachable on
> the other end.

## Current Phase: AI job-ad annotation pipeline (Milestone 1 smoke)

The annotation pipeline (Pass 0 + six LLM passes) is fully built and on the
`annotation-pipeline` branch. The Milestone-1 smoke run on 5000 ads from the
existing `production_5m` exposure run keeps failing on Isambard for the same
underlying reason every time: the runner-setup step produces a torch/vllm
combination on `/projects/a5u/ai-index-v3/.venv/` that is either internally
ABI-incompatible or corrupted by partial Lustre writes. The user has explicitly
authorised a **full rewrite of the Isambard provisioning** (anything under
`src/isambard_utils/` and `src/llm_runner/assets/`) if that's what it takes to
end this bug. **Do not keep patching individual broken files.**

### What we want, in order

1. **End the Isambard provisioning bug.** Get a single SBATCH chunk to actually
   complete and write its parquet, with the runner setup as a stable,
   reproducible step. Treat anything in `src/isambard_utils/` and
   `src/llm_runner/assets/` as fair game for rewriting from scratch.
2. **Smoke run completes.** `uv run remote-run-bg run-ai-ad-annotations
   annotation_smoke_5k_full` runs end-to-end: universe → Pass 0 → routing →
   all six LLM passes → join → `assemble_wide_annotation` writes
   `ai_job_ad_annotations_wide_v1.parquet` with 5000 rows × ~60 cols.
3. **Production run.** Same DAG with `sample_n = -1` (run name
   `annotation_production_5m`), all 5M ads. Route mode for Batches 2/3 may be
   `post_filter` (Pass-1-positive only) once we've measured the prefilter
   miss-rate on the smoke.
4. **Merge annotations into the existing index.** The annotation outputs are
   keyed on `ad_id` so they left-join cleanly onto the existing
   `store/pipeline/production_5m/compute_job_ad_exposure/ad_exposure.parquet`.

### Branch state

Branch `annotation-pipeline`. Five commits ahead of `main`:

```
0930f75 fix(isambard): pin torch+vllm versions to fix recurring ABI mismatch
354356a ops(annotations): bump netrun retries 3->10 and retry_wait 10s->30s
a7270e5 feat(annotations): add Batches 2 and 3 (Passes 2/3/4/6)
509545f fix(annotations): use adzuna.ads.id (not ad_id) as the join key
3dd0393 Add AI job-ad annotation pipeline (Milestone 1: smoke run scaffolding)
```

All 65 schema + prefilter tests pass; `uv run netrun validate -c
config/netrun_annotations.json` returns valid (12 nodes, 16 edges). The
annotation pipeline code is sound; the problem is purely Isambard provisioning.

## Diagnostic: the Isambard runner-setup bug

### Symptom timeline

Every smoke run goes through the same arc:

1. `build_annotation_universe` → 5000 ads selected (out of 4.99M).
2. `pass0_ai_keyword_prefilter` → 72/5000 keyword hits (1.44%).
3. `build_annotation_routing` → all 6 passes eligible (route=full).
4. All 6 LLM passes dispatch their first 4 chunks (max_concurrent=4) to Isambard.
5. **Failure mode rotates** between two classes:
   - **Class A — Partial-file corruption on Lustre.** Variants seen:
     `networkx-3.6.1.dist-info/METADATA` missing,
     `cuda_bindings-12.9.4.dist-info/METADATA` missing,
     `vllm/version.py` missing. Lustre apparently leaves partial dist-info
     dirs after an interrupted `uv pip install`. `_afix_lustre_hardlinks`
     in `src/isambard_utils/env.py` was written specifically to prevent
     this (it forces `UV_LINK_MODE=copy` so each compute node sees a real
     file, not a stale hardlink) but does not catch every case.
   - **Class B — torch/vllm ABI mismatch.** When jobs finally run on GPU
     they crash with:
     ```
     vllm/_C.abi3.so: undefined symbol:
       _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
     ```
     This is c10::cuda::c10_cuda_check_implementation — its signature
     changed between torch 2.8 (what vllm 0.11.0 was compiled against)
     and torch 2.12 (what the cu126 index ships as latest).
6. The pipeline cycles retries (hardened to `retries: 10` in
   `config/netrun_annotations.json`), eventually exhausts them, raises
   `EpochError`, and the whole net dies.

### Root cause

The runner setup as currently designed is a fragile post-install reinstall
dance. `asetup_runner` (in `src/isambard_utils/orchestrate.py`) does:

```
1. mkdir on Isambard
2. _aensure_uv          # install uv if missing
3. upload remote_pyproject.toml + remote_uv.lock + llm_runner src
4. _aensure_venv        # uv sync --no-dev --no-install-project
5. _aensure_cuda_torch  # if torch lacks +cu, reinstall from cu126 index
6. _afix_lustre_hardlinks  # if torch linked via hardlinks, reinstall with copy mode
7. averify_runner_env   # NEW (just added): import torch+vllm sanity check
```

The fragility lives in steps 5–6: those `uv pip install --reinstall-package`
calls **mutate the venv after sync** and (a) sometimes leave partial dist-info
dirs on Lustre when interrupted (class A), (b) historically grabbed the cu126
index's latest torch with no version pin, which drifted past vllm's
compiled-against ABI (class B).

The just-committed `fix(isambard): pin torch+vllm versions…` (commit `0930f75`)
attempted to fix both classes by:

- **Class B fix**: rewriting `remote_pyproject.toml` to use
  `[[tool.uv.index]]` + `[tool.uv.sources]` so torch is sourced directly from
  the cu126 index at lock time. Bumped vllm to `>=0.20` because its aarch64
  wheel pins `torch==2.11.0` (verified by downloading the wheel and reading
  the METADATA `Requires-Dist` line), and `2.11.0+cu126` is one of the
  versions the cu126 index actually ships for aarch64+cp312 (the index does
  NOT have 2.7.0 or 2.8.0). Lock now resolves to:
  - `torch 2.11.0+cu126`
  - `torchvision 0.26.0+cu126`
  - `vllm 0.20.2`
  Also added `[tool.uv]` `environments = ["sys_platform == 'linux' and
  platform_machine == 'aarch64'"]` to stop uv from falling back to ancient
  vllm versions trying to satisfy macOS arm64.
- **Belt-and-braces version pinning** inside `_aensure_cuda_torch`: read the
  lock-resolved torch + vllm versions BEFORE reinstalling, then pass
  `'torch==<version>'` / `'vllm==<version>'` explicitly so even the safety-net
  path can't drift.
- **`averify_runner_env`** fail-fast import test as the last step of
  `asetup_runner`. Should surface any ABI/corruption issue in `bg-job.log`
  within seconds.

### What's blocking right now

I wiped `/projects/a5u/ai-index-v3/.venv` on Isambard and relaunched. The bg log
on Hetzner shows the new runner setup is hitting **`uv sync failed, retrying...`
repeatedly** during step 4 (`_aensure_venv`). I did not yet diagnose the
underlying uv sync error before the user asked to hand off. The hand-off step
should be:

```
ssh root@46.224.166.250 \
  "ssh -o ConnectTimeout=60 a5u.aip2.isambard \
    'cd /projects/a5u/ai-index-v3 && \
     export UV_CACHE_DIR=/projects/a5u/ai-index-v3/.uv_cache && \
     export UV_LINK_MODE=copy && \
     uv sync --no-dev --no-install-project 2>&1 | tail -50'"
```

…to see the actual stderr. Could be a transient SSH issue, could be a new
package-resolution problem with the bumped vllm, could be a Lustre write
failure during install of the much larger updated dependency set (torch
2.11.0+cu126 + vllm 0.20.2 + their CUDA libs are a few GB).

The remote smoke job is still alive (PID `907880` on Hetzner) in a retry loop;
the next agent should `uv run remote-bg-kill` it before redeploying so it
doesn't fight the next launch.

## Authorisation: rewrite the Isambard provisioning if that helps

The user wrote:

> I want to instruct to basically have the leeway to do a full rewrite of the
> Isambard provisioning if that helps to get rid of this bug.

So the next agent is **authorised** to:

- Delete or replace any file under `src/isambard_utils/` and
  `src/llm_runner/assets/`.
- Change `asetup_runner` from a multi-step post-install dance into a single
  deterministic install, e.g. just `uv sync --frozen` from the locked
  cu126-sourced lockfile with zero post-install reinstalls. If the lock
  installs the right torch+vllm directly, `_aensure_cuda_torch` and
  `_afix_lustre_hardlinks` become unnecessary.
- Switch from `uv pip install --reinstall-package` to a cleaner
  delete-and-reinstall idiom if Lustre + hardlinks are the cause of partial
  dist-info corruption.
- Bake an explicit `import torch; import vllm; assert torch.cuda.is_available()`
  smoke into the deploy-time setup (not just runner setup) so corruption is
  caught at deploy time, before any chunk is dispatched.
- Move the venv off Lustre entirely if there's a writable per-job scratch
  filesystem available — Lustre's known issues with rename/uninstall during
  partial writes is the root of Class A failures, and avoiding Lustre for the
  venv directory would eliminate that class entirely.

Things to NOT touch unless deliberately:

- The annotation pipeline code itself (anything under `src/ai_annotations/`,
  `pts/ai_annotations/`, `config/netrun_annotations.json`,
  `config/run_defs_annotations.toml`, `config/prompt_library/ai_job_ad_annotations/`).
  It is verified working end-to-end up to the LLM call, and the failure mode
  is squarely on the Isambard side.
- The exposure pipeline (`config/netrun.json`, `src/ai_index/nodes/`,
  `store/pipeline/production_5m/` outputs). These are the 5M production
  baseline the annotation pipeline reads from and must not be touched.

## Key files & paths

### Annotation pipeline (working)

- DAG: `config/netrun_annotations.json` (12 nodes, 16 edges, validates clean)
- Run defs: `config/run_defs_annotations.toml`
  - `[runs.annotation_smoke_5k]`: 5000 ads, default routes
  - `[runs.annotation_smoke_5k_full]`: 5000 ads, all-passes-on-all-ads
  - `[runs.annotation_production_5m]`: -1 (all 5M)
- CLI: `uv run run-ai-ad-annotations <run_name>` (Hetzner) or
  `uv run remote-run-bg run-ai-ad-annotations <run_name>` (remote)
- Nodes: `pts/ai_annotations/nodes/` → exported to `src/ai_annotations/nodes/`
- Schemas: `src/ai_annotations/schemas.py` (Pydantic, strict=True)
- Prompts: `config/prompt_library/ai_job_ad_annotations/pass{1,2,3,4,5,6}_*/`

### Isambard provisioning (the broken part)

- Orchestrator: `src/isambard_utils/orchestrate.py:asetup_runner` (lines ~482-551)
- Env steps: `src/isambard_utils/env.py`
  - `_aensure_uv`, `_aensure_venv`, `_aensure_cuda_torch`,
    `_afix_lustre_hardlinks`, `averify_runner_env` (new)
- Remote project: `src/llm_runner/assets/remote_pyproject.toml`
- Remote lock: `src/llm_runner/assets/remote_uv.lock`
- SSH wrapper: `src/isambard_utils/ssh.py`
- Slurm runner CLI on Isambard: `src/llm_runner/cli.py` (uploaded by deploy)

### Hetzner deploy

- Server IP: `46.224.166.250` (root@…)
- Server alias: `aisi-exposure-index`
- Repo path on Hetzner: `/root/aisi-exposure-index`
- Bg job log: `/root/bg-job.log`
- Bg job PID file: `/root/bg-job.pid`
- CLI: `uv run remote-deploy-pipeline`, `remote-run-bg`, `remote-bg-log`,
  `remote-bg-kill`, `remote-run-cmd <cmd>`, `remote-clifton-auth`, etc. See
  `pyproject.toml` `[project.scripts]`.

### Isambard

- Host alias: `a5u.aip2.isambard` (Hetzner already has SSH access; local
  machine likely does too if Clifton cert is valid)
- Project dir: `/projects/a5u/ai-index-v3`
- Project venv: `/projects/a5u/ai-index-v3/.venv/lib/python3.12/site-packages/`
- HF cache: `/projects/a5u/ai-index-v3/.hf_cache`
- Runner cache (content-addressed per-chunk submission cache):
  `/projects/a5u/ai-index-v3/.runner_cache/`
- Logs dir: `/projects/a5u/ai-index-v3/.logs`

### Sample SSH commands the next agent will want

```bash
# Verify pipeline is alive on Hetzner
ssh root@46.224.166.250 'kill -0 $(cat /root/bg-job.pid) 2>/dev/null && echo alive || echo dead'

# See the last 100 non-noise log lines
uv run remote-bg-log 100

# Kill the smoke
uv run remote-bg-kill

# SSH straight through to Isambard from local
ssh -o ConnectTimeout=15 root@46.224.166.250 'ssh -o ConnectTimeout=60 a5u.aip2.isambard "<cmd>"'

# Inspect the Isambard venv
ssh root@46.224.166.250 \
  'ssh a5u.aip2.isambard "ls /projects/a5u/ai-index-v3/.venv/lib/python3.12/site-packages/ | head -30"'

# Try uv sync directly to see the actual error
ssh root@46.224.166.250 \
  'ssh a5u.aip2.isambard \
    "cd /projects/a5u/ai-index-v3 && \
     export UV_CACHE_DIR=/projects/a5u/ai-index-v3/.uv_cache && \
     export UV_LINK_MODE=copy && \
     uv sync --no-dev --no-install-project 2>&1 | tail -50"'

# Slurm queue state
ssh root@46.224.166.250 'ssh a5u.aip2.isambard "squeue --me --format=\"%t %i %M %R\""'

# Live tail of bg-job.log via Monitor
# (Use the Monitor tool with filter for: "OK torch=|Runner env sanity check failed|pass[1-6]: wrote|assemble_wide: wrote|logs saved|Traceback|EpochError|RuntimeError|No module named|undefined symbol")
```

### Production data (for context — DO NOT TOUCH)

- `store/outputs/production_5m/production_5m/geo_lad.csv` — 373 LADs, 13 columns
- `store/outputs/onet_exposure_scores/scores.csv` — 861 occupations, 7 scores
- `store/outputs/onet_exposure_scores/score_task_exposure_bt/gpt-4.1-mini/scores.csv` — BT scores
- `store/outputs/onet_exposure_scores/score_task_exposure/gpt-5.2/task_results.parquet`
- `store/outputs/onet_exposure_scores/score_task_exposure_bt/gpt-4.1-mini/task_bt_scores.parquet`
- `store/inputs/onet/db_30_0_text/Occupation Data.txt`
- `store/analysis/LAD_Dec_2022_UK_BGC.geojson`
- `store/pipeline/production_5m/compute_job_ad_exposure/ad_exposure.parquet` (405 MB)
  — this is the canonical 5M ad_id list the annotation pipeline reads from

## CRITICAL: Iterative execution rule (still applies to any analysis work)

**NEVER create analysis notebooks blindly.** All code must be executed cell by
cell. Every output (especially plots) must be viewed, analyzed, and adjusted
if needed. Iterative process:

1. Write a cell
2. Execute it
3. View the output / plot
4. Note qualitative behavior
5. If the output is wrong or uninformative, adjust and re-execute
6. Add markdown observations discovered during execution

This is non-negotiable. The user has emphasised this as "ONE VERY CRUCIAL THING
TO KEEP IN MIND" and it applies whenever an analysis notebook is being authored
(though the immediate task is fixing the Isambard provisioning, not analysis).
