# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.build_annotation_routing
#
# Build (or refresh) `annotation_routing_v1.parquet`, the per-ad eligibility
# table that downstream LLM passes consult to decide which ads they should
# attempt. Idempotent: re-running after a later pass completes refreshes the
# Pass-1-positive eligibility columns used by Batches 2 and 3, but those flags
# are not consumed in the Batch-1-only smoke run.
#
# For the smoke build (`batch1_route = "full"`) every ad in the universe is
# eligible for both Pass 1 and Pass 5.
#
# Per-node variables:
# - `batch1_route`: "full" or "post_filter". Drives `eligible_pass1`.
# - `pass5_route`: "full" or "post_filter". Independent of `batch1_route` because
#   Pass 5 is seniority-for-all-ads, not AI-keyword gated. Defaults to "full".

# %%
#|default_exp build_annotation_routing
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print, ad_ids: list[int]) -> {
    'ad_ids': list[int]
}:
    """Write the annotation routing parquet describing per-ad pass eligibility."""
    ...

# %%
from dev_utils import *
run_name = 'annotation_smoke_5k'
set_node_func_args('build_annotation_routing', run_name=run_name)
show_node_vars('build_annotation_routing', run_name=run_name)

# %% [markdown]
# # Function body

# %%
#|export
import json

import pyarrow as pa
import pyarrow.parquet as pq

from ai_index import const

run_name = ctx.vars["run_name"]
batch1_route = ctx.vars["batch1_route"]
pass5_route = ctx.vars["pass5_route"]

if batch1_route not in ("full", "post_filter"):
    raise ValueError(f"batch1_route must be 'full' or 'post_filter', got {batch1_route!r}")
if pass5_route not in ("full", "post_filter"):
    raise ValueError(f"pass5_route must be 'full' or 'post_filter', got {pass5_route!r}")

routing_dir = const.pipeline_store_path / run_name / "ai_job_ad_annotations" / "routing"
universe_path = routing_dir / "annotation_universe_v1.parquet"
pass0_path = routing_dir / "pass0_ai_keyword_prefilter_v1.parquet"
pass1_path = (
    const.pipeline_store_path / run_name / "ai_job_ad_annotations"
    / "batch1_core_salience_seniority" / "pass1_boolean_ai_detection_v1.parquet"
)
routing_path = routing_dir / "annotation_routing_v1.parquet"
meta_path = routing_dir / "routing_meta.json"

# %% [markdown]
# ## Read universe + Pass 0
#
# Joining the universe ad_id list against Pass 0 gives us `route_pass0_keyword_positive`.

# %%
#|export
print(f"build_annotation_routing: reading {const.rel(universe_path)} and {const.rel(pass0_path)}")
_universe_ids = pq.read_table(universe_path, columns=["ad_id"])
_pass0 = pq.read_table(pass0_path, columns=["ad_id", "prefilter_ai_keyword_hit"])

# Join on ad_id, ordered by universe order (which is sorted ascending)
import pyarrow.compute as pc
_pass0_sorted = _pass0.take(pc.sort_indices(_pass0, sort_keys=[("ad_id", "ascending")]))
_universe_sorted = _universe_ids.take(pc.sort_indices(_universe_ids, sort_keys=[("ad_id", "ascending")]))

# The two should have identical ad_id order since both came from the same upstream
_u_ids = _universe_sorted.column("ad_id").to_pylist()
_p0_ids = _pass0_sorted.column("ad_id").to_pylist()
if _u_ids != _p0_ids:
    raise RuntimeError(
        "build_annotation_routing: universe and pass0 ad_id lists differ. "
        f"|universe|={len(_u_ids)}, |pass0|={len(_p0_ids)}. This should never happen "
        "if pass0 ran on the universe ad_ids."
    )

ad_ids_arr = _pass0_sorted.column("ad_id")
route_pass0_keyword_positive = _pass0_sorted.column("prefilter_ai_keyword_hit")
n_ads = len(ad_ids_arr)
n_pass0_pos = int(pc.sum(route_pass0_keyword_positive).as_py())
print(f"  {n_pass0_pos} / {n_ads} Pass-0 keyword-positive ({100 * n_pass0_pos / n_ads:.2f}%)")

# %% [markdown]
# ## Read Pass 1 if it exists (for Batch 2/3 routing refresh)
#
# On the very first routing build (before Pass 1 has run), `route_pass1_ai_positive`
# is all false. A later re-run of this node, after Pass 1 has been written, will
# fill in the real values, which Batches 2 and 3 then consume.

# %%
#|export
if pass1_path.exists():
    print(f"  found existing Pass 1 output at {const.rel(pass1_path)}, refreshing pass1 eligibility")
    _pass1 = pq.read_table(pass1_path, columns=["ad_id", "mentions_ai_anywhere"])
    _pass1_map = dict(zip(_pass1.column("ad_id").to_pylist(), _pass1.column("mentions_ai_anywhere").to_pylist()))
    route_pass1_ai_positive = pa.array(
        [bool(_pass1_map.get(int(aid), False)) for aid in ad_ids_arr.to_pylist()],
        type=pa.bool_(),
    )
    n_pass1_pos = int(pc.sum(route_pass1_ai_positive).as_py())
    print(f"  {n_pass1_pos} / {n_ads} Pass-1 AI-positive")
else:
    print("  no Pass 1 output yet; route_pass1_ai_positive set to false for all rows")
    route_pass1_ai_positive = pa.array([False] * n_ads, type=pa.bool_())

# %% [markdown]
# ## Compute eligibility flags per pass
#
# Batch 1: Pass 1 follows batch1_route; Pass 5 follows its independent override.

# %%
#|export
if batch1_route == "full":
    eligible_pass1 = pa.array([True] * n_ads, type=pa.bool_())
else:  # post_filter
    eligible_pass1 = route_pass0_keyword_positive

if pass5_route == "full":
    eligible_pass5 = pa.array([True] * n_ads, type=pa.bool_())
else:  # post_filter — same as Pass 1's keyword-positive set
    eligible_pass5 = route_pass0_keyword_positive

n_elig_p1 = int(pc.sum(eligible_pass1).as_py())
n_elig_p5 = int(pc.sum(eligible_pass5).as_py())
print(f"  eligible: pass1={n_elig_p1} (route={batch1_route}), pass5={n_elig_p5} (route={pass5_route})")

# %% [markdown]
# ## Write routing parquet and meta

# %%
#|export
routing_table = pa.table({
    "ad_id": ad_ids_arr,
    "annotation_universe_included": pa.array([True] * n_ads, type=pa.bool_()),
    "route_pass0_keyword_positive": route_pass0_keyword_positive,
    "route_pass1_ai_positive": route_pass1_ai_positive,
    "eligible_pass1": eligible_pass1,
    "eligible_pass5": eligible_pass5,
    "batch1_route_mode": pa.array([batch1_route] * n_ads, type=pa.string()),
    "pass5_route_mode": pa.array([pass5_route] * n_ads, type=pa.string()),
})
pq.write_table(routing_table, routing_path)
print(f"build_annotation_routing: wrote {const.rel(routing_path)}")

meta = {
    "n_ads": n_ads,
    "n_pass0_positive": n_pass0_pos,
    "n_pass1_positive": int(pc.sum(route_pass1_ai_positive).as_py()),
    "n_eligible_pass1": n_elig_p1,
    "n_eligible_pass5": n_elig_p5,
    "batch1_route": batch1_route,
    "pass5_route": pass5_route,
    "pass1_source": str(const.rel(pass1_path)) if pass1_path.exists() else None,
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"build_annotation_routing: wrote {const.rel(meta_path)}")
ad_ids_out = [int(x) for x in ad_ids_arr.to_pylist()]
ad_ids_out #|func_return_line
