# Sodacan checkpoint

This checkpoint uses **ETac's predicted three-dimensional displacement fields
at 400 fixed tactile readout locations**. These locations are selected, in the
original saved order, from the 1,865 surface nodes in the supplied layout.
The accompanying configuration also preserves the input scaling used in training.

## Run

Run from the repository's `BlindGrasping/` directory in the supported Isaac Gym
environment:

```bash
python train.py +release=sodacan
```

This selects the Sodacan object, checkpoint, and matching 400-location tactile
configuration together. For a headless run, append `headless=True`.

Alternatively, load the complete configuration in this directory:

```bash
python train.py --config-path runs/sodacan-ckpt --config-name config
```

Use the supplied configuration rather than changing a point count to 400 or
selecting an arbitrary subset of the Cube layout. The positions, order, and
scaling must match those expected by the checkpoint.

## Included files

- [nn/ShadowHand.pth](nn/ShadowHand.pth): trained policy.
- [config.yaml](config.yaml): complete configuration to run this checkpoint.
- [training-config.yaml](training-config.yaml): original training configuration,
  retained for reference; it is not the configuration to launch.
- [metadata.json](metadata.json): checkpoint provenance and referenced tactile files.

The shared tactile files are included in the repository:

- [Surface locations](../../data/tactile/shadow-2000.json).
- [Ordered 400-location selection](../../data/tactile/shadow-400-indices.json).

The saved-epoch training success record is 94.55%.
This is a smoothed training record, not a fresh independent evaluation.
Full Isaac Gym grasping evaluations have not been rerun as part of this release update.

See the [BlindGrasping instructions](../../README.md) for installation,
object assets, and the implemented success criterion.
