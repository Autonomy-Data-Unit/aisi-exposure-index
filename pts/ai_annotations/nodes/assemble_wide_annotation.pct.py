# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.assemble_wide_annotation
#
# Build the convenience wide annotation table by left-joining the universe,
# routing, Pass 0, and Passes 1–6 outputs on `ad_id`. Pass-specific parquet
# files remain the primary deliverable; this wide table exists for downstream
# analysis convenience.
#
# Input port `passes_dict` is the synchronisation barrier from the join node;
# its value is `{"pass1": list[int], "pass2": list[int], ..., "pass6": list[int]}`.
# The actual data is read from the on-disk parquet files (which may be empty
# if that pass had no eligible ads under the configured route mode).

# %%
#|default_exp assemble_wide_annotation
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print, passes_dict: dict) -> {
    'wide_path': str
}:
    """Assemble the wide left-joined annotation table across all completed passes."""
    ...

# %%
from dev_utils import *
run_name = 'annotation_smoke_5k'
set_node_func_args('assemble_wide_annotation', run_name=run_name)
show_node_vars('assemble_wide_annotation', run_name=run_name)

# %% [markdown]
# # Function body

# %%
#|export
import json

import pandas as pd

from ai_index import const

run_name = ctx.vars["run_name"]

annotations_dir = const.pipeline_store_path / run_name / "ai_job_ad_annotations"
routing_dir = annotations_dir / "routing"
batch1_dir = annotations_dir / "batch1_core_salience_seniority"
batch2_dir = annotations_dir / "batch2_requirements_context"
batch3_dir = annotations_dir / "batch3_worker_use_quality"

universe_path = routing_dir / "annotation_universe_v1.parquet"
pass0_path = routing_dir / "pass0_ai_keyword_prefilter_v1.parquet"
routing_path = routing_dir / "annotation_routing_v1.parquet"

# Per-pass outputs (may not all exist if a batch wasn't run)
pass_paths = {
    "pass1": batch1_dir / "pass1_boolean_ai_detection_v1.parquet",
    "pass5": batch1_dir / "pass5_seniority_management_v1.parquet",
    "pass2": batch2_dir / "pass2_ai_mention_context_v1.parquet",
    "pass3": batch2_dir / "pass3_ai_requirement_v1.parquet",
    "pass4": batch3_dir / "pass4_worker_ai_tool_use_v1.parquet",
    "pass6": batch3_dir / "pass6_quality_remote_agency_v1.parquet",
}

wide_path = annotations_dir / "ai_job_ad_annotations_wide_v1.parquet"
meta_path = annotations_dir / "wide_meta.json"

print(f"assemble_wide: passes_dict keys = {list(passes_dict.keys())}")

# %% [markdown]
# ## Left-join everything on ad_id
#
# Universe is the spine. Routing/Pass 0 cover every universe row. Pass outputs
# may cover only a subset (under post_filter) or be empty entirely — the left
# join preserves the universe rows in every case.

# %%
#|export
universe = pd.read_parquet(universe_path, columns=["ad_id", "n_chars", "n_tokens"])
pass0 = pd.read_parquet(pass0_path)
routing = pd.read_parquet(routing_path)

print(f"  universe={len(universe)}, pass0={len(pass0)}, routing={len(routing)}")

wide = universe.merge(pass0, on="ad_id", how="left")
_routing_cols = ["ad_id"] + [c for c in routing.columns if c.startswith(("eligible_pass", "batch", "pass", "route_pass1"))]
wide = wide.merge(routing[_routing_cols], on="ad_id", how="left")

# Per-pass left joins. Rename pass-specific metadata columns to keep them
# distinguishable in the wide table.
_pass_summary = {}
for pass_name, path in pass_paths.items():
    if not path.exists():
        print(f"  {pass_name}: no parquet at {const.rel(path)}, skipping")
        _pass_summary[pass_name] = "absent"
        continue
    df = pd.read_parquet(path)
    print(f"  {pass_name}: {len(df)} rows from {const.rel(path)}")
    _pass_summary[pass_name] = len(df)
    if len(df) == 0:
        continue  # empty file, no merge needed
    renamed = df.rename(columns={
        "model_name": f"{pass_name}_model_name",
        "prompt_version": f"{pass_name}_prompt_version",
        "run_timestamp": f"{pass_name}_run_timestamp",
        "parse_success": f"{pass_name}_parse_success",
    })
    wide = wide.merge(renamed, on="ad_id", how="left")

print(f"  wide table: {len(wide)} rows x {len(wide.columns)} cols")

# %% [markdown]
# ## Write wide parquet and meta

# %%
#|export
wide.to_parquet(wide_path, index=False)
print(f"assemble_wide: wrote {const.rel(wide_path)}")

meta = {
    "n_rows": len(wide),
    "n_cols": len(wide.columns),
    "columns": list(wide.columns),
    "pass_row_counts": _pass_summary,
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"assemble_wide: wrote {const.rel(meta_path)}")
str(wide_path) #|func_return_line
