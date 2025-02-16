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
uv run mujoco_infer <path_to_.onnx> (-k)
```

# MISC

(Optional) To create a venv:

`uv venv`

`source .venv/bin/activate`