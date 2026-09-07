# SHARP Architectural Ablation

[Study summary and source evidence](../../../../Studies/SHARP/Architecture_Ablation/Summary.md).

Relative geometry gives the lowest selected minADE6 (0.749679, -0.35%), followed by uncertainty-aware context (0.749961, -0.32%). The remaining interventions do not improve this primary error. Endpoint refinement improves minFDE1 despite its higher minADE6, illustrating that trajectory-average and endpoint criteria need not move together.

The captured summary derives selected minADE6 from checkpoint filenames and uses retained or nearest validation records for other metrics. Treat the complete vector as reported evidence, not a newly re-evaluated selected checkpoint.
