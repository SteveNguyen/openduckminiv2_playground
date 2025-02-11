from env.locomotion.open_duck_mini_v2 import joystick as open_duck_mini_v2_joystick
from env.locomotion.open_duck_mini_v2 import randomize as open_duck_mini_v2_randomize

from mujoco_playground._src import registry
from mujoco_playground._src import locomotion

import functools

import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

import mediapy as media
import matplotlib.pyplot as plt

import sys

from datetime import datetime
import os
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from orbax import checkpoint as ocp

from mujoco_playground import wrapper
# from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


#@title Rollout and Render
from mujoco_playground._src.gait import draw_joystick_command





# Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU rendering:')
# get_ipython().run_line_magic('env', 'MUJOCO_GL=egl')
os.environ['MUJOCO_GL']='egl'
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags





locomotion.register_environment('OpenDuckMiniV2JoystickFlatTerrain', functools.partial(open_duck_mini_v2_joystick.Joystick, task="flat_terrain"), open_duck_mini_v2_joystick.default_config)
#Hack to add a new env... I would have prefered not to come to this...
setattr(locomotion, 'ALL',list(locomotion._envs.keys()))
locomotion._randomizer["OpenDuckMiniV2JoystickFlatTerrain"]=open_duck_mini_v2_randomize.domain_randomize
env_name = 'OpenDuckMiniV2JoystickFlatTerrain'






env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
# ppo_params = locomotion_params.brax_ppo_config(env_name)
ppo_params = locomotion_params.brax_ppo_config('BerkeleyHumanoidJoystickFlatTerrain') #TODO









randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )
print(f"PPO PARAMS: {ppo_training_params}")

ckpt_path = epath.Path(sys.argv[1])
#ckpt_path.mkdir(parents=True, exist_ok=True)


# env = registry.load(env_name)
eval_env = registry.load(env_name)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)

def progress(num_steps, metrics):
  print(f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}')


train_fn = functools.partial(
    ppo.train, #**dict(ppo_training_params),
    num_timesteps=0,
    episode_length=1000,
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    restore_checkpoint_path=ckpt_path
)

make_inference_fn, params, metrics = train_fn(
  environment=env,
  eval_env=registry.load(env_name, config=env_cfg),
  wrap_env_fn=wrapper.wrap_for_brax_training,

)

jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(1)

rollout = []
modify_scene_fns = []

x_vel = 1.0  #@param {type: "number"}
y_vel = 0.0  #@param {type: "number"}
yaw_vel = 0.0  #@param {type: "number"}
command = jp.array([x_vel, y_vel, yaw_vel])

phase_dt = 2 * jp.pi * eval_env.dt * 1.5
phase = jp.array([0, jp.pi])

for j in range(10):
  print(f"episode {j}")
  state = jit_reset(rng)
  state.info["phase_dt"] = phase_dt
  state.info["phase"] = phase
  for i in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    if state.done:
      break
    state.info["command"] = command
    rollout.append(state)

    xyz = np.array(state.data.xpos[eval_env.mj_model.body("base").id])
    xyz += np.array([0, 0.0, 0])
    x_axis = state.data.xmat[eval_env._torso_body_id, 0]
    yaw = -np.arctan2(x_axis[1], x_axis[0])
    modify_scene_fns.append(
        functools.partial(
            draw_joystick_command,
            cmd=state.info["command"],
            xyz=xyz,
            theta=yaw,
            scl=np.linalg.norm(state.info["command"]),
        )
    )

render_every = 1
fps = 1.0 / eval_env.dt / render_every
print(f"fps: {fps}")
traj = rollout[::render_every]
mod_fns = modify_scene_fns[::render_every]

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

frames = eval_env.render(
    traj,
    camera="track",
    scene_option=scene_option,
    width=640*2,
    height=480,
    modify_scene_fns=mod_fns,
)
# media.show_video(frames, fps=fps, loop=False)
media.write_video(images=frames, path="test_duck.mp4",fps=fps)
