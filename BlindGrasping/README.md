# BlindGrasping

Blind grasping environment based on [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).

## Installation

System: Ubuntu 20.04

Python: 3.8

### Install Isaac Gym & IsaacGymEnvs

Details can be found [here](https://docs.robotsfan.com/isaacgym/index.html). Please install Isaac Gym and IsaacGymEnvs following the official guide.

## Usage

The training process is the same as [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs). You can check configuration details in this repository under `cfg/`.

From the `BlindGrasping` directory, run:

```bash
python train.py task=ShadowHand num_envs=256
```

To check the results after training, go to `runs/EXPERIMENT_NAME` and run:

```bash
tensorboard --logdir=./summaries
```

The trained checkpoint will be saved under `runs/EXPERIMENT_NAME/nn/`.

We provide an example checkpoint for cube. Run:

```bash
python train.py task=ShadowHand num_envs=64 checkpoint=runs/cube-ckpt/nn/ShadowHand.pth test=True
```

If your GPU memory is limited, reduce `num_envs` (for example, 64 or 32).

