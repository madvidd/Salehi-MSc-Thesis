# SHARP State-Space Integration

[Study summary and source evidence](../../../../Studies/SHARP/State_Space_Integration/Summary.md).

Moving recurrence to the ordered agent history reduces selected minADE6 from 0.680362 to 0.673736 (-0.97%) relative to the scene-token placement. The result motivates preserving temporal order and a small residual pathway; it is not evidence that arbitrary recurrence improves every SHARP configuration.

Only the scene-token placement has a full-precision final vector in this comparison. No secondary metrics are invented for the temporal-agent checkpoint. These historical placement runs are not merged with the independent architecture-integration control.
