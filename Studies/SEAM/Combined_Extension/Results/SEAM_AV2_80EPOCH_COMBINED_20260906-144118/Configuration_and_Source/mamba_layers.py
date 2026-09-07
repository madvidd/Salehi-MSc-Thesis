import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


def _build_mamba(embed_dim, d_state, d_conv, expand):
    if Mamba is None:
        raise RuntimeError(
            "The selected model variant requires mamba-ssm. "
            "Run the supplied environment installer before training."
        )
    return Mamba(
        d_model=embed_dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        use_fast_path=True,
    )


class ResidualMambaBlock(nn.Module):
    """Pre-norm Mamba refinement with a conservative learnable residual gate."""

    def __init__(self, embed_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mixer = _build_mamba(embed_dim, d_state, d_conv, expand)
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, valid_mask=None):
        mixer_input = self.norm(x)
        if valid_mask is not None:
            mixer_input = mixer_input * valid_mask.unsqueeze(-1)

        update = self.mixer(mixer_input)
        if valid_mask is not None:
            update = update * valid_mask.unsqueeze(-1)

        return x + torch.sigmoid(self.residual_gate) * update


class MambaTrajectoryHead(nn.Module):
    """Generate each mode as an ordered future sequence instead of one flat MLP."""

    def __init__(
        self,
        embed_dim,
        future_steps,
        depth=2,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.time_embed = nn.Parameter(torch.empty(1, future_steps, embed_dim))
        self.blocks = nn.ModuleList(
            ResidualMambaBlock(embed_dim, d_state, d_conv, expand)
            for _ in range(depth)
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, 2)

        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, mode_query):
        batch_size, modes, embed_dim = mode_query.shape
        sequence = mode_query.reshape(batch_size * modes, 1, embed_dim)
        sequence = sequence + self.time_embed

        for block in self.blocks:
            sequence = block(sequence)

        sequence = self.out_proj(self.out_norm(sequence))
        return sequence.view(batch_size, modes, self.future_steps, 2)
