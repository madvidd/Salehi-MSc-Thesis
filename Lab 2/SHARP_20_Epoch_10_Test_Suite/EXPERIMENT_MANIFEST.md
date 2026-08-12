# SHARP AV2 Controlled 20-Epoch Ten-Test Suite

## Controlled training configuration

The ten tests use the SHARP AV2 single-agent architecture and training protocol from the paper and official source, with one requested schedule change: 20 epochs instead of 80.

| Setting | Value |
|---|---:|
| Dataset | Argoverse 2 single-agent, SHARP processed train/validation data |
| Epochs | 20 |
| Paper epochs | 80 |
| Warm-up epochs | 13 (preserved from the paper) |
| Optimizer | AdamW |
| Peak learning rate | 1e-4 |
| Minimum learning rate | 1e-5 |
| Weight decay | 1e-2 |
| Gradient clipping | 5, norm |
| Global batch size | 32 |
| Seed | 2333 |
| Augmentation | None |
| Feature dimension | 128 |
| Attention heads | 8 |
| Encoder depth | 4 |
| Prediction modes | 6 |
| Dual training | Enabled |
| Instance-aware streaming | Enabled |
| Target-centric context | Enabled |
| Trajectory relay | Enabled |

The paper used one RTX 8000. Lab 2 uses four RTX 2080 Ti GPUs with DDP, batch size 8 per process, global batch size 32, and synchronized BatchNorm. This changes hardware execution but preserves the optimization batch size.

## Isolated tests

1. `baseline`: unmodified SHARP architecture.
2. `confidence_gated_memory`: learns a reliability gate between current tokens and instance-aware streamed updates.
3. `cross_window_consistency`: adds a 0.05-weight Smooth L1 consistency term between confidence-weighted current and transformed previous predictions.
4. `learned_temporal_pool`: replaces temporal max pooling with masked learned attention pooling.
5. `uncertainty_target_context`: adapts endpoint-centric RoI radius and feature gating using previous mode probabilities.
6. `relative_geometry_bias`: adds learned per-head relative position and heading bias to the four scene-attention blocks.
7. `kinematic_motion_stem`: adds masked velocity, acceleration, and speed-change embeddings before temporal encoding.
8. `endpoint_refinement_decoder`: adds endpoint-conditioned trajectory/logit refinement and a 0.2-weight auxiliary coarse prediction loss.
9. `lane_topology_graph`: adds sparse geometry-derived lane-neighbor message passing before scene encoding.
10. `agent_temporal_mamba`: replaces all four temporal agent-history MHA blocks with four unidirectional Mamba blocks (`d_state=16`, `d_conv=4`, `expand=2`) and retains temporal max pooling. It does not add Mamba to scene context or streaming memory.

Tests 2-9 each contain only their named change relative to test 1. Test 10 removes the original temporal agent-attention stack and replaces it with Mamba. No test stacks the proposed modifications.

## Recovery and preservation

- Each variant has a fixed result directory.
- Every completed epoch writes a numbered checkpoint.
- `last.ckpt` contains model, optimizer, scheduler, epoch, and global-step state.
- Re-running the suite skips completed variants and resumes the first incomplete variant from `last.ckpt`.
- A failed variant stops the sequence. This avoids wasting days on later tests after a shared runtime failure.
- Previous Lab 2 code, results, checkpoints, and logs are never deleted or overwritten.

### Conditional DDP compatibility

`uncertainty_target_context` is inactive before a streamed window contains prior
mode probabilities. Its four parameter tensors (five scalar parameters) are
therefore connected to every uncertainty-variant training loss through an
exactly zero-valued term. This leaves predictions, loss values, gradients for
active parameters, the DDP strategy, and every training hyperparameter unchanged
while satisfying DDP's fixed-graph requirement. The recovery launcher audits
this behavior before it starts training and archives the pre-patch model,
Lightning module, and failed attempt.

The uncertainty variant also uses a checked, out-of-place `torch.index_copy` to
map compressed target features back to their original token positions. It is
forward- and gradient-equivalent to SHARP's Boolean assignment, but avoids the
Boolean CUDA write that produced an illegal memory access on Lab 2. Recovery
must pass randomized equivalence checks and a 256-batch real-data, four-GPU
smoke test before the controlled suite is allowed to resume. The smoke retains
the controlled 20-epoch scheduler configuration and stops after 256 optimizer
steps, preventing the scheduler from being reparameterized for the diagnostic.
SHARP's wrapper expects every fit to produce a best checkpoint; the checkpoint-
free smoke therefore accepts that specific post-fit exception only after all 256
steps completed and the log passed explicit CUDA, DDP, OOM, and remap-error scans.

This 20-epoch suite is a controlled screening study. Its absolute metrics are not directly comparable to the paper's fully trained 80-epoch result; the valid comparison is primarily among these ten identically shortened tests.
