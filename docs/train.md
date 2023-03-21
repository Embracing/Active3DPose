## Use Active3DPose with RLlib

---
We use [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html#) to train RL agents to control the movement of
aerial cameras. Please follow the step belows if you wish to reproduce the results shown in our paper:

### Step 1: Apply Patch to RLlib
We made a small but necessary change in the way how info is handled in RLlib.
```bash
conda activate {'YOUR_CONDA_ENV'}
cd {'PATH/TO/PROJECT/DIRECTORY'}
pip3 install ray['rllib']==1.13.0
bash run/scripts/apply-ray-patch.sh
```

### Step 2: Setting Up the Experiment File
We use [tune.Experiment API](https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-experiment) to manage our experiments in favor of its maintainability over passing arguments in the
terminal. The RLlib experiment file called ``mappo_wdl_ctcr.py`` is under the `experiment` folder. Within the
`make_ctcr_wdl_experiment` method you should be able to find the these two settings:
```
# ===== Experiment Settings =====
NUM_CAMERAS = n_cams        # (IMPORTANT) The amount of controllable aerial cameras, crucial to your experiment setup
INDEPENDENT = False         # Independent learning or paramete sharing, usually set to False
```

The following are a couple of settings (where you can find ) that you should adjust based on the computational
resources available on your machine.

```
# ===== Resource Settings =====
NUM_CPUS_FOR_DRIVER = 5     # Trainer CPU amount
TRAINER_GPUS = 0.5          # Trainer GPU amount
NUM_WORKERS = 7             # Number of remote workers for sampling
NUM_GPUS_PER_WORKER = 0.5   # Worker GPU amount, must be non-zero as Unreal binary needs GPU
NUM_ENVS_PER_WORKER = 4     # Number of vectoized environment per worker
NUM_CPUS_PER_WORKER = 1     # Worker CPU amount, 1 is usually enough
```

Note these resource settings will affect the sampling process and subsequently affect the model training. It is
advised to stick with the default settings to reproduce our results (... so you would need a machine with at least 4
GPUs each with 11
Gibs VRAM and 12 CPU cores). Though you can
tune your resource
settings to
achieve the same sampler settings as ours. Changing training batch size and SGD minibatch size can greatly impact the
training efficiency as suggested in [(Baker et al., 2019)](https://arxiv.org/pdf/1909.07528.pdf) and [(Yu et al,
2021)](https://arxiv.org/pdf/2103.01955.pdf).

```
# ===== Sampler Settings =====
ROLLOUT_FRAGMENT_LENGTH = 25
NUM_SAMPLING_ITERATIONS = 1
TRAIN_BATCH_SIZE = NUM_SAMPLING_ITERATIONS * ROLLOUT_FRAGMENT_LENGTH * NUM_WORKERS * NUM_ENVS_PER_WORKER    # default: 700
SGD_MINIBATCH_SIZE = TRAIN_BATCH_SIZE // 2    # default: 350
```

### Step 3 (optional): track experiment with Wandb
Make sure that you have ``wandb`` installed. Run ``pip install wandb`` in case if you haven't.

Create an API key file under the project directory: `touch wandb_api_key`.

Copy your wandb API key from https://wandb.ai/settings and paste into the key file.

### Step 4: Run Experiment with `tune.run_experiment`
Run the `train.py` file under the project directory:
```bash
python train.py --num-cams 5 --exp-mode MAPPO+CTCR+WDL
```
This example reruns the "5 Cameras MAPPO + CTCR + WDL" experiment shown in our paper. You can also run other experiments, e.g. "3 Cameras MAPPO + CTCR" by specifying `--num-cams 3` and `--exp-mode MAPPO+CTCR`. Currently supported experiment modes are `[MAPPO+CTCR, MAPPO+CTCR+WDL, MAPPO+WDL, MAPPO]` and number of cameras between 2 and 5. User may configure for more cameras by creating a new environment setup in the `.\activepose\env_config.py` file.

If you wish to use along with wandb logging, you can specify `project`, `group` and `tags` arguments:
```bash
--project  {'PROJECT_NAME'}
--group    {'GROUP_NAME'}
--tags     {'TAG_1'} {'TAG_2'}{'TAG_2'}
```


#### Q: Too many abandoned workers after a stopped experiment?
A: This is a known issue. We provided a script to kill all abandoned workers. Run the following command in the
project
directory:
```bash
bash run/scripts/kill-abandoned-workers.sh
```
