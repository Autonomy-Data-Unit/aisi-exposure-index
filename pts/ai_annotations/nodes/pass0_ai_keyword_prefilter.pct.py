# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # nodes.pass0_ai_keyword_prefilter
#
# Deterministic AI keyword prefilter over the annotation universe. Pure regex,
# no LLM. Designed for recall: false positives are cheap (Pass 1 will correct
# them); false negatives are invisible under post_filter routing, so when in
# doubt, the keyword filter includes.
#
# Writes `pass0_ai_keyword_prefilter_v1.parquet` with columns
# `ad_id`, `prefilter_ai_keyword_hit`, `prefilter_keyword_families` (list[str]),
# `prefilter_matched_terms` (list[str]).
#
# No per-node variables.

# %%
#|default_exp pass0_ai_keyword_prefilter
#|export_as_func true

# %%
#|set_func_signature
def main(ctx, print, ad_ids: list[int]) -> {
    'ad_ids': list[int]
}:
    """Apply the AI keyword prefilter to the annotation universe."""
    ...

# %%
from dev_utils import *
run_name = 'annotation_smoke_5k'
set_node_func_args('pass0_ai_keyword_prefilter', run_name=run_name)
show_node_vars('pass0_ai_keyword_prefilter', run_name=run_name)

# %% [markdown]
# # Function body

# %%
#|export
import json

import pyarrow as pa
import pyarrow.parquet as pq

from ai_index import const
from ai_annotations.keyword_prefilter import prefilter_text

run_name = ctx.vars["run_name"]

universe_path = const.pipeline_store_path / run_name / "ai_job_ad_annotations" / "routing" / "annotation_universe_v1.parquet"
output_dir = const.pipeline_store_path / run_name / "ai_job_ad_annotations" / "routing"
output_dir.mkdir(parents=True, exist_ok=True)
pass0_path = output_dir / "pass0_ai_keyword_prefilter_v1.parquet"
meta_path = output_dir / "pass0_meta.json"

# %% [markdown]
# ## Apply the regex prefilter

# %%
#|export
print(f"pass0: reading universe from {const.rel(universe_path)}")
universe = pq.read_table(universe_path, columns=["ad_id", "title", "description"])

n_rows = len(universe)
print(f"pass0: scanning {n_rows} ads")

# Apply prefilter row-by-row. Each row is one keyword scan; 5000 rows is trivial.
# At full 5M scale this is still seconds, since the regex is one alternation per family.
_ad_ids = universe.column("ad_id").to_pylist()
_titles = universe.column("title").to_pylist()
_descs = universe.column("description").to_pylist()

hit_col = []
families_col = []
terms_col = []
n_hits = 0
for title, desc in zip(_titles, _descs):
    combined = (title or "") + "\n" + (desc or "")
    hit, families, terms = prefilter_text(combined)
    hit_col.append(hit)
    families_col.append(families)
    terms_col.append(terms)
    if hit:
        n_hits += 1

print(f"pass0: {n_hits} / {n_rows} hits ({100 * n_hits / n_rows:.2f}%)")

# %% [markdown]
# ## Write pass0 parquet and meta

# %%
#|export
pass0_table = pa.table({
    "ad_id": pa.array(_ad_ids, type=pa.int64()),
    "prefilter_ai_keyword_hit": pa.array(hit_col, type=pa.bool_()),
    "prefilter_keyword_families": pa.array(families_col, type=pa.list_(pa.string())),
    "prefilter_matched_terms": pa.array(terms_col, type=pa.list_(pa.string())),
})
pq.write_table(pass0_table, pass0_path)
print(f"pass0: wrote {const.rel(pass0_path)}")

meta = {
    "n_ads": n_rows,
    "n_hits": n_hits,
    "hit_rate": n_hits / max(n_rows, 1),
}
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"pass0: wrote {const.rel(meta_path)}")
ad_ids_out = [int(x) for x in _ad_ids]
ad_ids_out #|func_return_line
