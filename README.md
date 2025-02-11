Install `uv`:
`curl -LsSf https://astral.sh/uv/install.sh | sh`

`uv run test_install.py` to check your installation
`uv run test_train.py` to run a training

(Optional) To create a venv:
`uv venv`
`source .venv/bin/activate`

Infer mujoco
`uv run mujoco_infer <path_to_.onnx> (-k)` 

Export onnx (the onnx is already exported along with each chekpoint in `./ONNX.onnx`): 
`uv run export_onnx.py <absolute_path_to_ckpt>`

