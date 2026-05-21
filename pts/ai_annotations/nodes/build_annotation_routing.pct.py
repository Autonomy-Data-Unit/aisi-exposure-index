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
# attempt. Idempotent: re-running after Pass 1 completes refreshes the
# Pass-1-positive eligibility columns that Batches 2 and 3 consume.
#
# Per-batch route modes:
# - `batch1_route` drives `eligible_pass1`.
# - `batch2_route` drives `eligible_pass2` and `eligible_pass3`.
# - `batch3_route` drives `eligible_pass4` and `eligible_pass6` (with
#   `pass6_route_override` for the spec's "Pass 6 may be `full` even when the
#   rest of Batch 3 is `post_filter`" option).
# - `pass5_route` drives `eligible_pass5` independently (Pass 5 is seniority
#   for all ads, never gated by the AI keyword filter).
#
# Under `full`, every ad in the universe is eligible. Under `post_filter`,
# Pass 1 follows Pass 0 keyword-positive; Passes 2/3/4/6 follow Pass-1-positive
# (all-false until Pass 1 has been written).

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
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ai_index import const

run_name = ctx.vars["run_name"]
batch1_route = ctx.vars["batch1_route"]
batch2_route = ctx.vars["batch2_route"]
batch3_route = ctx.vars["batch3_route"]
pass5_route = ctx.vars["pass5_route"]
pass6_route_override = ctx.vars["pass6_route_override"]

_VALID_ROUTES = ("full", "post_filter")
for _name, _val in [
    ("batch1_route", batch1_route),
    ("batch2_route", batch2_route),
    ("batch3_route", batch3_route),
    ("pass5_route", pass5_route),
]:
    if _val not in _VALID_ROUTES:
        raise ValueError(f"{_name} must be one of {_VALID_ROUTES}, got {_val!r}")
if pass6_route_override not in (*_VALID_ROUTES, "inherit"):
    raise ValueError(f"pass6_route_override must be 'full', 'post_filter', or 'inherit', got {pass6_route_override!r}")

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
# Joining the universe ad_id list against Pass 0 gives us
# `route_pass0_keyword_positive`.

# %%
#|export
print(f"build_annotation_routing: reading {const.rel(universe_path)} and {const.rel(pass0_path)}")
_universe_ids = pq.read_table(universe_path, columns=["ad_id"])
_pass0 = pq.read_table(pass0_path, columns=["ad_id", "prefilter_ai_keyword_hit"])

_pass0_sorted = _pass0.take(pc.sort_indices(_pass0, sort_keys=[("ad_id", "ascending")]))
_universe_sorted = _universe_ids.take(pc.sort_indices(_universe_ids, sort_keys=[("ad_id", "ascending")]))

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
# On the very first routing build (before Pass 1 has run),
# `route_pass1_ai_positive` is all false. A later re-run of this node, after
# Pass 1 has been written, fills in the real values, which Batches 2 and 3
# then consume.

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

# %%
#|export
_all_true = pa.array([True] * n_ads, type=pa.bool_())

def _eligibility_for_batch(route_mode: str, post_filter_source: pa.Array) -> pa.Array:
    """`full` -> all true; `post_filter` -> the supplied source column."""
    return _all_true if route_mode == "full" else post_filter_source

# Pass 1: gated by Pass 0 under post_filter
eligible_pass1 = _eligibility_for_batch(batch1_route, route_pass0_keyword_positive)

# Pass 5: independent, follows its own pass5_route. post_filter for Pass 5
# means "the same Pass-0 keyword-positive set" (it never uses Pass-1 since
# Pass 5 is conceptually upstream of any AI-content judgement).
eligible_pass5 = _eligibility_for_batch(pass5_route, route_pass0_keyword_positive)

# Passes 2, 3, 4: gated by Pass-1-positive under post_filter
eligible_pass2 = _eligibility_for_batch(batch2_route, route_pass1_ai_positive)
eligible_pass3 = _eligibility_for_batch(batch2_route, route_pass1_ai_positive)
eligible_pass4 = _eligibility_for_batch(batch3_route, route_pass1_ai_positive)

# Pass 6: by default inherits batch3_route, but can be overridden to `full`
# even when the rest of Batch 3 is post_filter (per the spec).
pass6_effective_route = batch3_route if pass6_route_override == "inherit" else pass6_route_override
eligible_pass6 = _eligibility_for_batch(pass6_effective_route, route_pass1_ai_positive)

_eligibilities = {
    "pass1": eligible_pass1,
    "pass2": eligible_pass2,
    "pass3": eligible_pass3,
    "pass4": eligible_pass4,
    "pass5": eligible_pass5,
    "pass6": eligible_pass6,
}
_n_elig = {k: int(pc.sum(v).as_py()) for k, v in _eligibilities.items()}
print(f"  eligible: " + ", ".join(f"{k}={n}" for k, n in _n_elig.items()))
print(f"  routes: batch1={batch1_route}, batch2={batch2_route}, batch3={batch3_route}, pass5={pass5_route}, pass6_override={pass6_route_override}")

# %% [markdown]
# ## Write routing parquet and meta

# %%
#|export
routing_table = pa.table({
    "ad_id": ad_ids_arr,
    "annotation_universe_included": _all_true,
    "route_pass0_keyword_positive": route_pass0_keyword_positive,
    "route_pass1_ai_positive": route_pass1_ai_positive,
    "eligible_pass1": eligible_pass1,
    "eligible_pass2": eligible_pass2,
    "eligible_pass3": eligible_pass3,
    "eligible_pass4": eligible_pass4,
    "eligible_pass5": eligible_pass5,
    "eligible_pass6": eligible_pass6,
    "batch1_route_mode": pa.array([batch1_route] * n_ads, type=pa.string()),
    "batch2_route_mode": pa.array([batch2_route] * n_ads, type=pa.string()),
    "batch3_route_mode": pa.array([batch3_route] * n_ads, type=pa.string()),
    "pass5_route_mode": pa.array([pass5_route] * n_ads, type=pa.string()),
    "pass6_route_mode": pa.array([pass6_effective_route] * n_ads, type=pa.string()),
})
pq.write_table(routing_table, routing_path)
print(f"build_annotation_routing: wrote {const.rel(routing_path)}")

meta = {
    "n_ads": n_ads,
    "n_pass0_positive": n_pass0_pos,
    "n_pass1_positive": int(pc.sum(route_pass1_ai_positive).as_py()),
    "n_eligible": _n_elig,
    "routes": {
        "batch1": batch1_route,
        "batch2": batch2_route,
        "batch3": batch3_route,
        "pass5": pass5_route,
        "pass6_override": pass6_route_override,
        "pass6_effective": pass6_effective_route,
    },
    "pass1_source": str(const.rel(pass1_path)) if pass1_path.exists() else None,
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"build_annotation_routing: wrote {const.rel(meta_path)}")
ad_ids_out = [int(x) for x in ad_ids_arr.to_pylist()]
ad_ids_out #|func_return_line
