# BlindGrasping

Tactile blind grasping with ShadowHand, based on
[IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).

## Installation

Use the repository's Ubuntu 20.04 / Python 3.8 environment and install Isaac Gym
and IsaacGymEnvs. Run the commands below from `BlindGrasping/`.

## Released objects and checkpoints

Only **Cube, Egg, Rock, and Sodacan** are supported by this release. Their
canonical assets are `data/meshdata/<object>/coacd/`. Each directory contains the
URDF and the exact mesh used for both physical contact and tactile calculation.

The earlier Sodacan `coacd` asset came from separate LeapHand experiments and
was larger. The retained ShadowHand asset is approximately 5.49 cm in diameter
and 12.73 cm long, rather than 8.62 cm and 20.01 cm. The ShadowHand versions now
use the standard `coacd` name; duplicate assets and the other grasp objects have
been removed.

| Object | Checkpoint | Release profile | Observations | Actions |
| --- | --- | --- | ---: | ---: |
| Cube | `runs/cube-ckpt/nn/ShadowHand.pth` | `+release=cube` | 2940 | 24 |
| Egg | `runs/egg-ckpt/nn/ShadowHand.pth` | `+release=egg` | 1335 | 24 |
| Rock | `runs/rock-ckpt/nn/ShadowHand.pth` | `+release=rock` | 1335 | 24 |
| Sodacan | `runs/sodacan-ckpt/nn/ShadowHand.pth` | `+release=sodacan` | 1335 | 24 |

The existing Cube checkpoint is unchanged. The non-Cube checkpoints use the
original 400 tactile readout locations on a fixed surface layout. Their profiles
select those locations in the saved order and preserve the original scaling.
The propagation-model weights are the same weights used by these policies.
Do not load the non-Cube policies with the default Cube observation layout.

Each checkpoint directory includes:

- `config.yaml`: complete, portable configuration for this release;
- `training-config.yaml`: archived original training configuration, for provenance;
- `metadata.json`: source run, saved epoch, expected input/output sizes and SHA-256.

The archived training configuration includes old names and paths and is **not**
the configuration to launch. Use the release profile or `config.yaml` instead.
An aggregate manifest is available at `runs/checkpoints.json`.

## Evaluate a released checkpoint

The release profiles set the object, checkpoint, tactile layout, and evaluation
mode together. Both launch routes use the checkpoint's saved training seed;
override `seed=...` to evaluate other initial random samples:

```bash
python train.py +release=cube
python train.py +release=egg
python train.py +release=rock
python train.py +release=sodacan
```

For a headless run, add `headless=True`. For example:

```bash
python train.py +release=sodacan num_envs=64 headless=True
```

Alternatively, launch a complete checkpoint configuration directly:

```bash
python train.py --config-path runs/egg-ckpt --config-name config num_envs=64
```

## Train a new policy

```bash
python train.py task=ShadowHand task.env.objectType=sodacan num_envs=256
```

To train with a released non-Cube observation layout, reuse its profile but
explicitly clear the checkpoint and disable evaluation:

```bash
python train.py +release=egg checkpoint='' test=False num_envs=256
```

New checkpoints and logs are saved under `runs/EXPERIMENT_NAME/`. Inspect logs
with `tensorboard --logdir=runs/EXPERIMENT_NAME/summaries`.

## Reward, initialization, and success definition

All four objects share the existing reward and lift trigger: hand-to-object
distance at most 0.12 m, the **sum** of the five fingertip-to-object-center
distances at most 0.25 m, and active contact. No object-specific reward thresholds
are introduced by this release.

Initial hand heights are Cube 0.06 m, Egg 0.07 m, Rock 0.08 m, and Sodacan 0.05 m.
Initial object position and orientation are randomized; Sodacan's long axis is
initialized horizontally. Physical-parameter domain randomization is disabled
in the supplied configurations.

The implementation records an episode as successful when the **hand root**
reaches at least `target_height - 0.05 m` while contact is active. There is no
minimum holding duration. Success does not itself reset the episode. Later loss
of contact or excessive hand-object separation can incur penalties and a reset,
but a success already recorded for that episode is not revoked. The proximity
requirements for lifting rewards are not additional success conditions.

This is different from the appendix's object-height wording, 0.03 m tolerance,
and treatment of post-lift drops. Those descriptions should not be interpreted
as the implemented evaluation rule. This release does not change that rule.

## Verification and interpretation of training records

Run the asset/configuration/checkpoint checks without Isaac Gym:

```bash
python scripts/check_release.py
```

With PyTorch installed, run the CPU regression checks:

```bash
python -m unittest discover -s tests -v
```

The saved-epoch training success records are Egg 97.46%, Rock 83.79%, and
Sodacan 94.55%. These are smoothed records from the source runs, not an independent
evaluation, and not a new reproduction of the paper's table. Isaac Gym episodes
have **not** been rerun as part of this release preparation; evaluate the supplied
profiles in the supported Isaac Gym environment before comparing success rates.
