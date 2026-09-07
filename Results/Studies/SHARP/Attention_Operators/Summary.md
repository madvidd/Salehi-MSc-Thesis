# SHARP Attention Operators

[Study summary and source evidence](../../../../Studies/SHARP/Attention_Operators/Summary.md).

QKNorm improves selected minADE6 from 0.673460 to 0.669777 (-0.55%). Talking-Heads alone gives +0.23%, and QKNorm with head mixing gives +0.15%. The combined operator has the lowest MR in the retained vectors, but not the lowest trajectory-average error.

The MHA vector is final-epoch validation, whereas the alternatives have retained selected-checkpoint evaluations. The primary comparison therefore uses the saved selected minADE6 for every operator; do not attribute all secondary-metric changes to a fully matched checkpoint evaluation.
