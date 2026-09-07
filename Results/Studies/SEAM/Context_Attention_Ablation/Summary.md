# SEAM Context and Attention Ablation

[Study summary and source evidence](../../../../Studies/SEAM/Context_Attention_Ablation/Summary.md).

All three independent interventions improve selected minADE6 against the matched control: uncertainty -2.60%, geometry -1.43% and QKNorm -3.27%. QKNorm achieves the lowest selected error, 0.703845. These independent gains do not establish that their composition will be additive.

For relative geometry, selected minADE6 is 0.717261 while the final vector contains 0.718735. These are not the same checkpoint. The earlier batch-32 attempt is retained separately and is not pooled with this batch-48 study.
