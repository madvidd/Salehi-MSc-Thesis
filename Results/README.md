# Experimental Results

Curated summaries are regenerated from retained source records. Each model is compared with the control from its own study; selected checkpoint metrics and final log vectors are distinguished.

- [SEAM State-Space Integration](../Studies/SEAM/State_Space_Integration/Summary.md): Future-head replacement reduces selected minADE6 from 0.662859 to 0.648480 (-2.17%). Agent-history addition gives 0.664814 (+0.29%). The result supports sequence modelling at the future-coordinate head in this study, rather than a universal benefit from adding recurrence.
- [SEAM Context and Attention Ablation](../Studies/SEAM/Context_Attention_Ablation/Summary.md): All three independent interventions improve selected minADE6 against the matched control: uncertainty -2.60%, geometry -1.43% and QKNorm -3.27%. QKNorm achieves the lowest selected error, 0.703845. These independent gains do not establish that their composition will be additive.
- [SEAM Combined Extension](../Studies/SEAM/Combined_Extension/Summary.md): This implementation combines all three context/attention mechanisms with future-head Mamba. Accuracy claims require its own completed evaluation; they cannot be inferred from independent ablations.
- [SHARP Attention Operators](../Studies/SHARP/Attention_Operators/Summary.md): QKNorm improves selected minADE6 from 0.673460 to 0.669777 (-0.55%). Talking-Heads alone gives +0.23%, and QKNorm with head mixing gives +0.15%. The combined operator has the lowest MR in the retained vectors, but not the lowest trajectory-average error.
- [SHARP Architectural Ablation](../Studies/SHARP/Architecture_Ablation/Summary.md): Relative geometry gives the lowest selected minADE6 (0.749679, -0.35%), followed by uncertainty-aware context (0.749961, -0.32%). The remaining interventions do not improve this primary error. Endpoint refinement improves minFDE1 despite its higher minADE6, illustrating that trajectory-average and endpoint criteria need not move together.
- [SHARP State-Space Integration](../Studies/SHARP/State_Space_Integration/Summary.md): Moving recurrence to the ordered agent history reduces selected minADE6 from 0.680362 to 0.673736 (-0.97%) relative to the scene-token placement. The result motivates preserving temporal order and a small residual pathway; it is not evidence that arbitrary recurrence improves every SHARP configuration.
- [SHARP Architecture Integration](../Studies/SHARP/Architecture_Integration/Summary.md): The combined model improves MR, b-minFDE6, minFDE1 and minFDE6. Selected minADE6 changes from 0.679282 to 0.680589 (+0.19%), so the composition does not improve every accuracy criterion. MR changes from 0.155955 to 0.155515 (-0.28%). Independent screening gains are therefore not assumed to sum when mechanisms are composed.

## Supporting Files

- [Complete study registry](Main/All_Run_Registry.md)
- [Modification guide](Info.md)
- [Study-specific CSV tables](Studies/)
- [Evidence and filename conventions](../Documentation/Repository_Guide.md)

`Metrics.csv` files are derived comparisons with explicit source paths, not replacements for raw evaluations. Run `python Tools/Repository/build_results_catalog.py --check` from the repository root to check that the maintained summaries agree with the retained records.
