# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.build_annotation_universe
#
# Build the canonical row list and join spine for the annotation branch.
#
# Reads the production run's `compute_job_ad_exposure/ad_exposure.parquet` for the
# ad_id list (guaranteeing "exact same ads as the previous 5M run") and joins to
# `store/inputs/adzuna.duckdb` for title and description. Writes
# `annotation_universe_v1.parquet` with columns `ad_id`, `title`, `description`,
# `n_chars`, `n_tokens` (heuristic token count using `len(text) // 4`).
#
# Node variables (per-node):
# - `input_run_name`: name of the upstream production run to source ad_ids from.
# - `sample_n`: -1 for all ads, otherwise take the first N after sorting by ad_id.

# %%
#|default_exp build_annotation_universe
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print) -> {
    'ad_ids': list[int]
}:
    """Build the canonical annotation universe parquet and emit its ad_ids list."""
    ...

# %% [markdown]
# Retrieve input arguments (notebook-only)

# %%
from dev_utils import *
run_name = 'annotation_smoke_5k'
set_node_func_args('build_annotation_universe', run_name=run_name)
show_node_vars('build_annotation_universe', run_name=run_name)

# %% [markdown]
# # Function body
# ## Read node variables and paths

# %%
#|export
import json

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ai_index import const

run_name = ctx.vars["run_name"]
input_run_name = ctx.vars["input_run_name"]
sample_n = ctx.vars["sample_n"]

input_exposure_path = const.pipeline_store_path / input_run_name / "compute_job_ad_exposure" / "ad_exposure.parquet"
adzuna_db_path = const.adzuna_db_path

output_dir = const.pipeline_store_path / run_name / "ai_job_ad_annotations" / "routing"
output_dir.mkdir(parents=True, exist_ok=True)
universe_path = output_dir / "annotation_universe_v1.parquet"
meta_path = output_dir / "universe_meta.json"

# %% [markdown]
# ## Read canonical ad_id list
#
# `ad_exposure.parquet` is the canonical 5M ad list from the previous production
# run. Sourcing ad_ids from this file is what guarantees the annotation pipeline
# operates on the exact same ads as the exposure pipeline.

# %%
#|export
print(f"build_annotation_universe: sourcing ad_ids from {const.rel(input_exposure_path)}")
if not input_exposure_path.exists():
    raise FileNotFoundError(
        f"Input run {input_run_name!r} has no ad_exposure.parquet at {input_exposure_path}. "
        "The annotation pipeline requires a completed upstream run to source the canonical ad_id list from."
    )

_exposure_ids = pq.read_table(input_exposure_path, columns=["ad_id"])
all_ad_ids = sorted(int(x) for x in _exposure_ids.column("ad_id").to_pylist())
print(f"  upstream run contains {len(all_ad_ids)} ad_ids")

if sample_n == -1:
    selected_ad_ids = all_ad_ids
else:
    selected_ad_ids = all_ad_ids[:sample_n]
print(f"  selected {len(selected_ad_ids)} ad_ids for the annotation universe (sample_n={sample_n})")

# %% [markdown]
# ## Join to adzuna for title and description

# %%
#|export
print(f"build_annotation_universe: querying adzuna.duckdb for {len(selected_ad_ids)} ads")
con = duckdb.connect(str(adzuna_db_path), read_only=True)
con.register("selected", pa.Table.from_pydict({"ad_id": selected_ad_ids}))
universe_table = con.execute(
    """
    SELECT a.ad_id, a.title, a.description
    FROM ads a
    INNER JOIN selected s ON s.ad_id = a.ad_id
    ORDER BY a.ad_id
    """
).fetch_arrow_table()
con.close()
print(f"  retrieved {len(universe_table)} rows from adzuna.ads")

if len(universe_table) != len(selected_ad_ids):
    _retrieved_ids = set(int(x) for x in universe_table.column("ad_id").to_pylist())
    _missing = [aid for aid in selected_ad_ids if aid not in _retrieved_ids]
    raise RuntimeError(
        f"build_annotation_universe: requested {len(selected_ad_ids)} ad_ids but adzuna returned "
        f"{len(universe_table)}. {len(_missing)} ad_ids missing (first 5: {_missing[:5]}). "
        "This suggests the upstream run's ad_id list is out of sync with adzuna.duckdb."
    )

# %% [markdown]
# ## Compute n_chars and n_tokens
#
# `n_tokens` is a 4-chars-per-token heuristic, sufficient for budgeting. Loading
# the production tokenizer just for this estimate isn't worth the dependency.

# %%
#|export
_titles = universe_table.column("title")
_descs = universe_table.column("description")
_joined_lens = pc.add(pc.utf8_length(_titles), pc.utf8_length(_descs))
n_chars_arr = pc.add(_joined_lens, pa.scalar(1, type=pa.int64()))  # +1 for the '\n' separator
n_tokens_arr = pc.divide(n_chars_arr, pa.scalar(4, type=pa.int64()))

universe_table = universe_table.append_column("n_chars", n_chars_arr.cast(pa.int64()))
universe_table = universe_table.append_column("n_tokens", n_tokens_arr.cast(pa.int64()))

# %% [markdown]
# ## Write universe parquet and meta

# %%
#|export
pq.write_table(universe_table, universe_path)
print(f"build_annotation_universe: wrote {const.rel(universe_path)}")

meta = {
    "input_run_name": input_run_name,
    "sample_n_requested": sample_n,
    "n_ads": len(universe_table),
    "n_chars_total": int(pc.sum(universe_table.column("n_chars")).as_py()),
    "n_tokens_total_estimate": int(pc.sum(universe_table.column("n_tokens")).as_py()),
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"build_annotation_universe: wrote {const.rel(meta_path)}")
ad_ids_out = [int(x) for x in universe_table.column("ad_id").to_pylist()]
ad_ids_out #|func_return_line
