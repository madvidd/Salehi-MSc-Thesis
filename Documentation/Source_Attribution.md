# Source Attribution

This repository studies modifications of published forecasting models; it does
not claim authorship of the underlying SEAM, SHARP or Mamba implementations.
Original notices and citations in bundled source files are retained.

| Source | Repository use | Local provenance |
|---|---|---|
| SEAM | Endpoint-aware streaming baseline and the basis for SEAM ablations | [Archive provenance](../Studies/SEAM/State_Space_Integration/SOURCE_PROVENANCE.md), [controlled-source audit](../Studies/SEAM/Context_Attention_Ablation/SOURCE_PROVENANCE.md) |
| SHARP | Short-window streaming baseline and the basis for SHARP ablations | [Bundled source](../Studies/SHARP/Attention_Operators/sharp_original/README.md), [source/configuration alignment](../Studies/SHARP/Architecture_Integration/ARTICLE_CODE_ALIGNMENT.md) |
| Mamba | Selective state-space modules used in the documented insertion and replacement experiments | Study setup scripts and recorded environment specifications |

The root [MIT licence](../LICENSE) is retained unchanged. It does not replace
third-party source or dataset terms. Source archive hashes, upstream citations
and dependency versions remain available in the study records.

Scientific definitions and hypotheses are described in the
[modification guide](../Results/Info.md); empirical claims link to local result
records rather than treating an implementation hypothesis as a measured gain.
