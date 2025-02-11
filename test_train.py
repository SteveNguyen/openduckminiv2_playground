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

from jax import config as jaxconfig
# jaxconfig.update("jax_debug_nans", True)
# jax.config.update('jax_default_matmul_precision', 'high')
# jax.config.update("jax_enable_x64", True)

#ON CPU?
# jax.config.update('jax_default_device', jax.devices('cpu')[0])



#####
import os
import subprocess

if subprocess.run('nvidia-smi').returncode:
  raise RuntimeError(
      'Cannot communicate with GPU. '
      'Make sure you are using a GPU Colab runtime. '
      'Go to the Runtime menu and select Choose runtime type.'
  )

# Add an ICD config so that glvnd can pick up the Nvidia EGL driver.
# This is usually installed as part of an Nvidia driver package, but the Colab
# kernel doesn't install its driver via APT, and as a result the ICD is missing.
# (https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
  with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
    f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")

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




# env_name = 'BerkeleyHumanoidJoystickFlatTerrain'



env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
# ppo_params = locomotion_params.brax_ppo_config(env_name)
ppo_params = locomotion_params.brax_ppo_config('BerkeleyHumanoidJoystickFlatTerrain') #TODO

print("ENVIRONEMENT LOADED")




############################

#To test just some step of the env

# define the jit reset/step functions
# jit_reset = jax.jit(env.reset)
# jit_step = jax.jit(env.step)
# # initialize the state
# state = jit_reset(jax.random.PRNGKey(0))
# rollout = [state]

# # grab a trajectory
# for i in range(2):
#   ctrl = -0.1 * jp.ones(env.mj_model.nu)
#   state = jit_step(state, ctrl)
#   rollout.append(state)

# # media.show_video(env.render(rollout), fps=1.0 / env.dt)
# stop
###########################







x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):

  # clear_output(wait=True)
  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  # plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  # plt.xlabel("# environment steps")
  # plt.ylabel("reward per episode")
  # plt.title(f"y={y_data[-1]:.3f}")
  # plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

  # # display(plt.gcf())
  # plt.draw()
  print(f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}')


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

ckpt_path = epath.Path('/home/antoine/MISC/openduckminiv2_playground/openduckminiv2_playground/ckpts')
ckpt_path.mkdir(parents=True, exist_ok=True)

def policy_params_fn(current_step, make_policy, params):
  # save checkpoints

  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  d=datetime.now().strftime("%Y_%m_%d_%H%M%S")
  path = ckpt_path / f'{d}_{current_step}'
  print(f'Saving checkpoint (step: {current_step}): {path}')
  orbax_checkpointer.save(path, params, force=True, save_args=save_args)

print("NETWORK CREATED")

print("TRAINING")
train_fn = functools.partial(
  ppo.train, **dict(ppo_training_params),
  network_factory=network_factory,
  randomization_fn=randomizer,
  progress_fn=progress,
  policy_params_fn=policy_params_fn,
  # save_checkpoint_path='checkpoints', #avaible on the github version...
)



make_inference_fn, params, metrics = train_fn(
  environment=env,
  eval_env=registry.load(env_name, config=env_cfg),
  wrap_env_fn=wrapper.wrap_for_brax_training,

)
print("TRAINED")
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
plt.xlabel("# environment steps")
plt.ylabel("reward per episode")
plt.title(f"y={y_data[-1]:.3f}")
plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
plt.savefig('training.pdf')
