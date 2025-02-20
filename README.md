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

Run: 

```bash
uv run test_train.py
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