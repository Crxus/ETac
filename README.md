# ETac: A Lightweight and Efficient Tactile Simulation Framework for Learning Dexterous Manipulation

> Official code repository for the paper  
> **“ETac: A Lightweight and Efficient Tactile Simulation Framework for Learning Dexterous Manipulation”**.

🔗 **Project Page:**  [https://lassford.github.io/ETac/](https://lassford.github.io/ETac/)

📄 **Paper:**  [Accepted by 2026 IEEE International Conference on Robotics and Automation (ICRA)]

**System:** Ubuntu 20.04

**Python:** 3.8 (recommended by `environment.yaml`)


## Project Structure

```
ETac/
├── ETac/
│   ├── etac.py                     # Definition of ETac
│   ├── utils
│   │   ├──hand_model.py            # Hand model for contact calculation
│   │   ├──object_model.py          # Object model for contact calculation
│   │   ├──PropagationModel.py      # ETac's propagation model
│   │   ├──...
├── Propagation_calib/              # Calibration code of the propagation model
│   ├── main.py
│   ├── data/
│   ├── checkpoints/
│   ├── ...
├── BlindGrasping                   # Blind grasping environment
│   ├── train.py                    
│   ├── tasks
│   │   ├── shadow_hand.py          # Training code of the blind grasping policy
│   ├── cfg                         # Configurations
│   │   ├── task
│   │   │   ├──ShadowHand.yaml      # Configurations of the task & environment
│   │   ├── train
│   │   │   ├──ShadowHandPPO.yaml   # Configuration of PPO
│   │   ├── ...
│   ├── runs                        # Checkpoints
│   │   ├── ...
│   ├── ...
├── README.md                          
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ETac
```

### 2. Create the Conda environment:

Create a new environment using the provided `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate etac
pip install -e .
```

If `pip install -e .` fails because of build tools, first run:

```bash
pip install --upgrade pip setuptools wheel
```

### 3. Modified Kaolin for SDF calculation

We use a modified [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) by [wrc042](https://github.com/wrc042) for faster SDF calculation.

```bash
cd ETac/thirdparty/kaolin
python setup.py develop
```

### 4. Pytorch Kinematics

A modified [pytorch_kinematics](https://github.com/UM-ARM-Lab/pytorch_kinematics) by [wrc042](https://github.com/wrc042). 

```bash
cd ETac/thirdparty/pytorch_kinematics
pip install -e .
```

Then go back to the repository root:

```bash
cd ../../..
```

## Usage

### 1. Calibrate a propagation model

Calibrate the propagation model using FEM-simulated indentation data.

In this work, each sample is saved as a pickle file with the format:

```python
{
    "uppers": tensor[n, 3],   # surface nodes of elastomer in FEM simulation
    "cps": tensor[k, 3]       # active contact nodes between elastomer and indenter
}
```

If you are trying to collect your own data, name files in order (for example: `0.pkl`, `1.pkl`, ...).

We provide some sample data [here](https://drive.google.com/drive/folders/1YHKxW1SBe_AGB1KD3qEHnShBY-G4HZ0H?usp=sharing). Download and put them under `Propagation_calib/data/calib_data`.

Run

```bash
cd Propagation_calib
python main.py --data_id <DATA_ID> --num_files <NUM_FILES> --save_id <SAVE_ID>
```
to calibrate a propagation model. The checkpoint will be saved under `checkpoints/<SAVE_ID>/`.

For example, run

```bash
python main.py --data_id 1 --num_files 200 --save_id 01
```
to calibrate using data under `1`.

### 2. Test the propagation model & visualize

To test the trained propagation, run

```bash
python main.py --test True --data_id <DATA_ID> --num_files <NUM_FILES> --save_id <SAVE_ID>
```

The predicted data (including ground truth and predicted data) will be saved under `Propagation_calib/data/vis_data/<SAVE_ID>/test_results.pth`.

When using sample data, after calibration, run

```bash
python main.py --test True --data_id 7 --num_files 200 --save_id 01
```

to test the checkpoint on indentation data collected from unseen indenter.

We provide a quick example for visualizing predicted data in `visualization.ipynb`. Download `example` floder from [the drive](https://drive.google.com/drive/folders/1YHKxW1SBe_AGB1KD3qEHnShBY-G4HZ0H?usp=sharing) and put it under `dara/vis_data`. 

You may use your own predicted data for visualization.

### 3. Tactile-based dexterous manipulation

See `BlindGrasping`