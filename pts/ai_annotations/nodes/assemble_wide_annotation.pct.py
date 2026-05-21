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
# routing, Pass 0, Pass 1 and Pass 5 outputs on `ad_id`. Pass-specific parquet
# files remain the primary deliverable; this wide table exists for the smoke
# run's downstream analysis.
#
# Input port `passes_dict` is the synchronisation barrier from the join node;
# its value is `{"pass1": list[int], "pass5": list[int]}`. The actual data is
# read from the on-disk parquet files.

# %%
#|default_exp assemble_wide_annotation
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print, passes_dict: dict) -> {
    'wide_path': str
}:
    """Assemble the wide left-joined annotation table."""
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

universe_path = routing_dir / "annotation_universe_v1.parquet"
pass0_path = routing_dir / "pass0_ai_keyword_prefilter_v1.parquet"
routing_path = routing_dir / "annotation_routing_v1.parquet"
pass1_path = batch1_dir / "pass1_boolean_ai_detection_v1.parquet"
pass5_path = batch1_dir / "pass5_seniority_management_v1.parquet"

wide_path = annotations_dir / "ai_job_ad_annotations_wide_v1.parquet"
meta_path = annotations_dir / "wide_meta.json"

print(f"assemble_wide: passes_dict keys = {list(passes_dict.keys())}")

# %% [markdown]
# ## Left-join everything on ad_id
#
# Universe is the spine. Routing/Pass 0 cover the full universe. Pass 1 / Pass 5
# may be a subset if non-`full` routes were used; the left join preserves universe
# rows even when an LLM pass didn't process that ad.

# %%
#|export
universe = pd.read_parquet(universe_path, columns=["ad_id", "n_chars", "n_tokens"])
pass0 = pd.read_parquet(pass0_path)
routing = pd.read_parquet(routing_path)
pass1 = pd.read_parquet(pass1_path)
pass5 = pd.read_parquet(pass5_path)

print(f"  universe={len(universe)}, pass0={len(pass0)}, routing={len(routing)}, pass1={len(pass1)}, pass5={len(pass5)}")

wide = universe.merge(pass0, on="ad_id", how="left")
wide = wide.merge(
    routing[["ad_id", "eligible_pass1", "eligible_pass5", "batch1_route_mode", "pass5_route_mode"]],
    on="ad_id", how="left",
)

# Rename pass-specific metadata so the wide table preserves which pass each field came from.
_pass1_renamed = pass1.rename(columns={
    "model_name": "pass1_model_name",
    "prompt_version": "pass1_prompt_version",
    "run_timestamp": "pass1_run_timestamp",
    "parse_success": "pass1_parse_success",
})
wide = wide.merge(_pass1_renamed, on="ad_id", how="left")

_pass5_renamed = pass5.rename(columns={
    "model_name": "pass5_model_name",
    "prompt_version": "pass5_prompt_version",
    "run_timestamp": "pass5_run_timestamp",
    "parse_success": "pass5_parse_success",
})
wide = wide.merge(_pass5_renamed, on="ad_id", how="left")

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
    "n_pass1_present": int(wide["pass1_parse_success"].notna().sum()),
    "n_pass5_present": int(wide["pass5_parse_success"].notna().sum()),
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"assemble_wide: wrote {const.rel(meta_path)}")
str(wide_path) #|func_return_line
