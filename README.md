# Installation

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Run : 
```bash
uv run test_install.py
```

# Training

If you want to use the [imitation reward](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf), you can generate reference motion with [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator)

Then copy `polynomial_coefficients.json` in `env/locomotion/open_duck_mini_v2/` or the relevant robot

You'll also have to set `USE_IMITATION_REWARD=True` in it's `joystick.py` file

Run: 

```bash
uv run train.py
```

## Tensorboard

```bash
uv run tensorboard --logdir=<yourlogdir>
```

# Eval

Infer mujoco

```bash
uv run mujoco_infer.py -o <path_to_.onnx> (-k)
```

# MISC

(Optional) To create a venv:

`uv venv`

`source .venv/bin/activate`