Install `uv`:
`curl -LsSf https://astral.sh/uv/install.sh | sh`

`uv run test_install.py` to check your installation
`uv run test_train.py` to run a training

To create a venv:
`uv venv`
`source .venv/bin/activate`

Export onnx : 
`uv run utils/export_onnx.py <absolute_path_to_ckpt>`

Infer mujoco
`uv run mujoco_infer <path_to_.onnx>
