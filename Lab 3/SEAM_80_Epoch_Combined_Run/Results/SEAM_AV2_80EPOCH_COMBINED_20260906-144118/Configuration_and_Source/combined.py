from .seam import Seam
from .layers.mamba_layers import MambaTrajectoryHead


class CombinedSeam(Seam):
    """The three SEAM controls together with the previous future-head ablation."""

    def __init__(self, **kwargs):
        super().__init__(variant="combined", **kwargs)
        # Install after SEAM's recursive initialisation to retain Mamba's dt/A
        # initialisation. Context streaming and trajectory relay are unchanged.
        self.decoder.loc = MambaTrajectoryHead(
            embed_dim=self.embed_dim,
            future_steps=self.future_steps,
            depth=2,
            d_state=16,
            d_conv=4,
            expand=2,
        )
