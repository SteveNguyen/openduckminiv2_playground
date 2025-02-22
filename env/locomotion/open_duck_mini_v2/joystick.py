# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from typing import Any, Dict, Optional, Union
import json
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
import os

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import open_duck_mini_v2_constants as consts
from . import base as open_duck_mini_v2_base
from reference_motion import process_reference_motion, get_closest_reference_motion
from poly_reference_motion import PolyReferenceMotion
# from jax import config

# config.update("jax_enable_x64", True)





# TO TRY : 
# - reduce action_scale
# - curriculum training (less push, less noise, less delay, only forward at start etc)
# https://chatgpt.com/share/67b0df9c-6a68-8011-8f04-a72fae95ba63


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=450,
      action_repeat=1,
      action_scale=1.0,
      dof_vel_scale=0.05,
      history_len=0,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          action_min_delay=0,  # env steps
          action_max_delay=3,  # env steps
          imu_min_delay=0,  # env steps
          imu_max_delay=3,  # env steps
          scales=config_dict.create(
              hip_pos=0.03,  # rad #for each hip joint
              kfe_pos=0.05,  # kfe=Knee Pitch
              ffe_pos=0.08,  # ffe=Ankle pitch
              # faa_pos=0.03, #ffa=Ankle Roll #FIXME!
              joint_vel=2.5,  # rad/s # Was 1.5
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,  # angvel. # was 0.2
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking related rewards.
              tracking_lin_vel=1.5,
              tracking_ang_vel=0.5,
              # Base related rewards.
              lin_vel_z=0.0,
              ang_vel_xy=0.0,
              orientation=-0.5,
              base_height=0.0,
              base_y_swing=0.0,
              # Energy related rewards.
              torques=-1.0e-3,
              action_rate=-1.5,  # was -1.5
              energy=0.0,
              # Feet related rewards.
              feet_clearance=0.0,
              feet_air_time=0.0,
              feet_slip=0.0,
              feet_height=-0.5,
              # feet_phase=0.0,
              # Other rewards.
              stand_still=-0.5,  # was -1.0 TODO try to relax this a bit ?
              alive=20.0,
              termination=0.0,
              imitation=0.5,
              # Pose related rewards.
              joint_deviation_knee=0.0,  # -0.1
              joint_deviation_hip=0.0,  # -0.25
              dof_pos_limits=0.0,  # -1.0
              pose=0.0,  # -1.0
          ),
          tracking_sigma=0.01,  # was working at 0.01
          max_foot_height=0.03,  # 0.1,
          base_height_target=0.15,  # 0.5,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 1.0],
      ),
      lin_vel_x=[-0.1, 0.15],
      lin_vel_y=[-0.2, 0.2],
      ang_vel_yaw=[-0.5, 0.5],
      # lin_vel_x=[0.0, 0.1],
      # lin_vel_y=[-0.1, 0.1],
      # ang_vel_yaw=[-0.3, 0.3],
  )


class Joystick(open_duck_mini_v2_base.OpenDuckMiniV2Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()


  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self.PRM = PolyReferenceMotion("polynomial_coefficients.json")

    # ( 
    #   self.reference_data_array,
    #   self.dx_range,
    #   self.dy_range,
    #   self.dtheta_range,
    #   self.dxs,
    #   self.dys,
    #   self.dthetas,
    #   self.ref_motion_fps,
    #   self.period,
    #   # self.root_pos_slice,
    #   # self.root_quat_slice,
    #   # self.linear_vel_slice,
    #   # self.angular_vel_slice,
    #   # self.joint_pos_slice,
    #   # self.joint_vels_slice,
    #   # self.left_toe_pos_slice,
    #   # self.right_toe_pos_slice

    # ) = process_reference_motion(os.getcwd()+"/new_ref_motion")
    # self.nb_frames_in_one_walk_cycle = int((1/self._config.ctrl_dt) * self.period)
    self.nb_frames_in_reference_motion = 450  # TODO extract this from the reference motion data

    # Note: First joint is freejoint.
    # get the range of the joints
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    # get the indices of the hip joints
    hip_indices = []
    hip_joint_names = ["hip_yaw", "hip_roll", "hip_pitch"] #original implementaion seems to omit hip_pitch
    for side in ["left", "right"]:
      for joint_name in hip_joint_names:
        hip_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}").qposadr - 7 # -7 is to remove the 7 first corresponding to the floating base
        )
    # print(f"HIP INDICES: {hip_indices}")
    self._hip_indices = jp.array(hip_indices)

    # get the indices of the knee joints
    knee_indices = []
    for side in ["left", "right"]:
      knee_indices.append(self._mj_model.joint(f"{side}_knee").qposadr - 7)
    # print(f"KNEE INDICES: {knee_indices}")
    self._knee_indices = jp.array(knee_indices)
    # weights for computing the cost of each joints compared to a reference pose
    # fmt: off
    # self._weights = jp.array([
    #     1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # left leg.
    #     1.0, 1.0, 0.01, 0.01, 1.0, 1.0,  # right leg.
    # ])
    self._weights = jp.array([
        1.0, 1.0, 0.01, 0.01, 1.0,   # left leg.
      # 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, #head
        1.0, 1.0, 0.01, 0.01, 1.0,  # right leg.
    ])

    # fmt: on
    # self._joint_names=["left_hip_yaw","left_hip_roll","left_hip_pitch","left_knee","left_ankle", "right_hip_yaw","right_hip_roll","right_hip_pitch","right_knee","right_ankle"]
    self._njoints = 10
    # self ._joint_range=
    # self._mj_model.joint()

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._site_id = self._mj_model.site("imu").id

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    # noise in the simu?
    qpos_noise_scale = np.zeros(self._njoints)

    #horrible
    hip_ids = [0, 1, 2, 5, 6, 7] #left/right hip_yaw/roll/pitch
    kfe_ids = [3, 8] #left/right knee
    ffe_ids = [4, 9] #left/right ankle pitch
    # faa_ids = [5, 11] #left_right ankle roll


    qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
    qpos_noise_scale[kfe_ids] = self._config.noise_config.scales.kfe_pos
    qpos_noise_scale[ffe_ids] = self._config.noise_config.scales.ffe_pos
    # qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
    self._qpos_noise_scale = jp.array(qpos_noise_scale)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q

    qvel = jp.zeros(self.mjx_model.nv)

    #init position/orientation in environment
    # x=+U(-0.05, 0.05), y=+U(-0.05, 0.05), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    #init joint position
    # qpos[7:]=*U(0.0, 0.1)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (self._njoints,), minval=0.5, maxval=1.5)
    )

    #init joint vel
    # d(xyzrpy)=U(-0.05, 0.05)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

    # Phase, freq=U(0.5, 2.5)
    # rng, key = jax.random.split(rng)
    # gait_freq = jax.random.uniform(key, (1,), minval=1.9, maxval=2.1)
    # gait_freq = 1 / self.period
    # phase_dt = 2 * jp.pi * self.dt * gait_freq
    # phase = jp.array([0, jp.pi])

    base_y_swing_freq = 1.5  # hz
    base_y_swing_amplitude = 0.06  # 6 centimeters
    base_t = 0.0

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    current_reference_motion = self.PRM.get_reference_motion(cmd[0], cmd[1], cmd[2], 0)
    # current_reference_motion = get_closest_reference_motion(
    #   self.reference_data_array,
    #   cmd[0],
    #   cmd[1],
    #   cmd[2],
    #   0,
    #   self.dx_range,
    #   self.dy_range,
    #   self.dtheta_range,
    #   self.dxs,
    #   self.dys,
    #   self.dthetas
    # )

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "last_last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        "base_y_swing_freq": base_y_swing_freq,
        "base_y_swing_amplitude": base_y_swing_amplitude,
        "base_t": base_t,
        # Phase related.
        # "phase_dt": phase_dt,
        # "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        # History related.
        "action_history": jp.zeros(self._config.noise_config.action_max_delay * self._njoints),
        "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
        # imitation related
        "imitation_i": 0,
        "current_reference_motion": current_reference_motion
    }

    metrics = {}
    for k, v in self._config.reward_config.scales.items():
      if v != 0:
        if v > 0:
          metrics[f"reward/{k}"] = jp.zeros(())
        else:
          metrics[f"cost/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

    state.info["imitation_i"] += 1
    state.info["imitation_i"] = state.info["imitation_i"] % self.nb_frames_in_reference_motion
    # state.info["imitation_i"] = state.info["imitation_i"] % self.nb_frames_in_one_walk_cycle

    state.info["current_reference_motion"] = self.PRM.get_reference_motion(
      state.info["command"][0], 
      state.info["command"][1], 
      state.info["command"][2], 
      state.info["imitation_i"]
    )
    # state.info["current_reference_motion"] = get_closest_reference_motion(
    #   self.reference_data_array, 
    #   state.info["command"][0],
    #   state.info["command"][1],
    #   state.info["command"][2],
    #   state.info["imitation_i"],
    #   self.dx_range,
    #   self.dy_range,
    #   self.dtheta_range,
    #   self.dxs,
    #   self.dys,
    #   self.dthetas
    # )

    state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(
        state.info["rng"], 4
    )

    # Handle action delay
    action_history = jp.roll(state.info["action_history"], self._njoints).at[:self._njoints].set(action)
    state.info["action_history"] = action_history
    action_idx = jax.random.randint(
        action_delay_rng,
        (1,),
        minval=self._config.noise_config.action_min_delay,
        maxval=self._config.noise_config.action_max_delay,
    )
    action_w_delay = action_history.reshape((-1, self._njoints))[action_idx[0]] # action with delay

    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    state.info["base_t"] += self.dt

    motor_targets = self._default_pose + action_w_delay * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    # jax.debug.print('STEP REWARD: {}',reward)
    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    # phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    # state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["last_last_last_act"] = state.info["last_last_act"]
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      rew_scale = self._config.reward_config.scales[k]
      if rew_scale != 0:
        if rew_scale > 0:
          state.metrics[f"reward/{k}"] = v
        else:
          state.metrics[f"cost/{k}"] = -v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data)[-1] < 0.0
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    # Handle IMU delay
    imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
    info["imu_history"] = imu_history
    imu_idx = jax.random.randint(
        noise_rng, 
        (1,),
        minval=self._config.noise_config.imu_min_delay,
        maxval=self._config.noise_config.imu_max_delay,
    )
    noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._qpos_noise_scale
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    # cos = jp.cos(info["phase"])
    # sin = jp.sin(info["phase"])
    # phase = jp.concatenate([cos, sin]) # [4]
    # phase = cos

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        # noisy_linvel,  # 3
        # noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 10
        noisy_joint_vel * self._config.dof_vel_scale,  # 10
        info["last_act"],  # 10
        info["last_last_act"],  # 10
        info["last_last_last_act"],  # 10
        contact,  # 2
        info["current_reference_motion"],
    ])

    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
        root_height,  # 1
        data.actuator_force,  # 10
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
        # info["imitation_i"],
        info["current_reference_motion"]
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.

    ret =  {
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_gravity(data)),
        "base_height": self._cost_base_height(data.qpos[2]),
        "base_y_swing": self._reward_base_y_swing(data.qvel[1], info["base_y_swing_freq"], info["base_y_swing_amplitude"], info["base_t"]),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        # "feet_phase": self._reward_feet_phase(
        #     data,
        #     info["phase"],
        #     self._config.reward_config.max_foot_height,
        #     info["command"],
        # ),
        # "both_feet_up": self._cost_both_feet_up(contact),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "imitation": self._reward_imitation(
          data.qpos,
          data.qvel,
          contact,
          info["current_reference_motion"],
          info["command"],
          # self.linear_vel_slice,
          # self.angular_vel_slice,
          # self.joint_pos_slice,
          # self.joint_vels_slice,
          # self.left_toe_pos_slice,
          # self.right_toe_pos_slice,
        ),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:], data.qvel[6:]),
        # Pose related rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
    }

    return ret

  # Tracking rewards.

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    # return jp.nan_to_num(jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma))
    y_tol = 0.1
    error_x = jp.square(commands[0] - local_vel[0])
    error_y = jp.clip(jp.abs(local_vel[1] - commands[1]) - y_tol, 0.0, None)
    lin_vel_error = error_x + jp.square(error_y)
    return jp.nan_to_num(jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma))

  # def _reward_tracking_lin_vel(self, commands, local_vel):
  #     # commands[:2] -> [x_cmd, y_cmd]
  #     # local_vel[:2] -> [x_vel, y_vel]
  #     target = commands[:2]
  #     actual = local_vel[:2]

  #     # Give a tolerance (e.g. ±0.02 m/s) for the y-velocity:
  #     tol = 0.02
  #     # Flatten out the error if |y_vel| < 0.02
  #     error_y = jp.clip(jp.abs(actual[1] - target[1]) - tol, 0.0, None)

  #     # For x-velocity, do a standard squared error
  #     error_x = jp.square(actual[0] - target[0])

  #     # Combine them
  #     lin_vel_error = error_x + jp.square(error_y)
  #     return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.nan_to_num(jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma))

  # Base-related rewards.

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    return jp.nan_to_num(jp.square(global_linvel[2]))

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(global_angvel[:2])))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(torso_zaxis[:2])))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.square(
        base_height - self._config.reward_config.base_height_target
    ))

  def _reward_base_y_swing(
      self, base_y_speed: jax.Array, freq: float, amplitude: float, t: float
  ) -> jax.Array:
    target_y_speed = amplitude * jp.sin(2 * jp.pi * freq * t)
    y_speed_error = jp.square(target_y_speed - base_y_speed)
    return jp.nan_to_num(jp.exp(-y_speed_error / self._config.reward_config.tracking_sigma))

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(torques)))
    # return jp.nan_to_num(jp.sum(jp.abs(torques)))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator)))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    c1 = jp.nan_to_num(jp.sum(jp.square(act - last_act)))
    return c1

  # Other rewards.

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.nan_to_num(jp.sum(out_of_limits))

  def _cost_stand_still(
      self,
      commands: jax.Array,
      qpos: jax.Array,
      qvel: jax.Array,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    pose_cost = jp.sum(jp.abs(qpos - self._default_pose))
    vel_cost = jp.sum(jp.abs(qvel))
    return jp.nan_to_num(pose_cost + vel_cost) * (cmd_norm < 0.01)

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def quaternion_angle(self, q1, q2):
    #  according to chatgpt :)
    dot_product = jp.dot(q1, q2)
    dot_product = jp.clip(dot_product, -1.0, 1.0)
    angle = 2 * jp.arccos(jp.abs(dot_product))
    return angle

  def _reward_imitation(
    self, 
    qpos: jax.Array,
    qvel: jax.Array,
    contacts: jax.Array,
    reference_frame: jax.Array,
    cmd: jax.Array,
    # linear_vel_slice: jax.Array,
    # angular_vel_slice: jax.Array,
    # joint_pos_slice: jax.Array,
    # joint_vels_slice: jax.Array,
    # left_toe_pos_slice: jax.Array,
    # right_toe_pos_slice: jax.Array

  ) -> jax.Array:
    # TODO don't reward for moving when the command is zero.
    cmd_norm = jp.linalg.norm(cmd)

    w_torso_pos = 1.0
    w_torso_orientation = 1.0
    w_lin_vel_xy = 1.0
    w_lin_vel_z = 1.0
    w_ang_vel_xy = 0.5
    w_ang_vel_z = 0.5
    w_joint_pos = 15.0
    w_joint_vel = 1.0e-3
    w_contact = 1.0


    #  TODO : double check if the slices are correct
    linear_vel_slice_start = 34
    linear_vel_slice_end = 37

    angular_vel_slice_start = 37
    angular_vel_slice_end = 40  

    joint_pos_slice_start = 0
    joint_pos_slice_end = 16

    joint_vels_slice_start = 16
    joint_vels_slice_end = 32

    # root_pos_slice_start = 0
    # root_pos_slice_end = 3

    # root_quat_slice_start = 3
    # root_quat_slice_end = 7

    # left_toe_pos_slice_start = 23
    # left_toe_pos_slice_end = 26

    # right_toe_pos_slice_start = 26
    # right_toe_pos_slice_end = 29

    foot_contacts_slice_start = 32
    foot_contacts_slice_end = 34


    # ref_base_pos = reference_frame[root_pos_slice_start:root_pos_slice_end]
    # base_pos = qpos[:3]

    # ref_base_orientation_quat = reference_frame[root_quat_slice_start:root_quat_slice_end]
    # ref_base_orientation_quat = ref_base_orientation_quat / jp.linalg.norm(ref_base_orientation_quat)  # normalize the quat
    # base_orientation = qpos[3:7]
    # base_orientation = base_orientation / jp.linalg.norm(base_orientation)  # normalize the quat

    ref_base_lin_vel = reference_frame[linear_vel_slice_start:linear_vel_slice_end]
    base_lin_vel = qvel[:3]

    ref_base_ang_vel = reference_frame[angular_vel_slice_start:angular_vel_slice_end]    
    base_ang_vel = qvel[3:6]

    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]
    # remove the neck and head
    ref_joint_pos = jp.concatenate([ref_joint_pos[:5], ref_joint_pos[11:]])
    joint_pos = qpos[7:]

    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]
    # remove the neck and head
    ref_joint_vels = jp.concatenate([ref_joint_vels[:5], ref_joint_vels[11:]])
    joint_vel = qvel[6:]

    # ref_left_toe_pos = reference_frame[left_toe_pos_slice_start:left_toe_pos_slice_end]
    # ref_right_toe_pos = reference_frame[right_toe_pos_slice_start:right_toe_pos_slice_end]

    ref_foot_contacts = reference_frame[foot_contacts_slice_start:foot_contacts_slice_end]


    # reward
    # torso_pos_rew = jp.exp(-200.0 * jp.sum(jp.square(base_pos[:2] - ref_base_pos[:2]))) * w_torso_pos

    # real quaternion angle doesn't have the expected  effect, switching back for now
    # torso_orientation_rew = jp.exp(-20 * self.quaternion_angle(base_orientation, ref_base_orientation_quat)) * w_torso_orientation
    # torso_orientation_rew = jp.exp(-20.0 * jp.sum(jp.square(base_orientation - ref_base_orientation_quat))) * w_torso_orientation

    lin_vel_xy_rew = jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[:2] - ref_base_lin_vel[:2]))) * w_lin_vel_xy
    lin_vel_z_rew = jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[2] - ref_base_lin_vel[2]))) * w_lin_vel_z

    ang_vel_xy_rew = jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[:2] - ref_base_ang_vel[:2]))) * w_ang_vel_xy
    ang_vel_z_rew = jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[2] - ref_base_ang_vel[2]))) * w_ang_vel_z

    joint_pos_rew = -jp.sum(jp.square(joint_pos - ref_joint_pos)) * w_joint_pos
    joint_vel_rew = -jp.sum(jp.square(joint_vel - ref_joint_vels)) * w_joint_vel

    # TODO the ref contact can oscillate a bit around 0 or 1, maybe we should add a threshold
    contact_rew = jp.sum(contacts == ref_foot_contacts) * w_contact

    # reward = torso_pos_rew + torso_orientation_rew +  lin_vel_xy_rew + lin_vel_z_rew + ang_vel_xy_rew + ang_vel_z_rew + joint_pos_rew + joint_vel_rew + contact_rew
    reward = lin_vel_xy_rew + lin_vel_z_rew + ang_vel_xy_rew + ang_vel_z_rew + joint_pos_rew + joint_vel_rew + contact_rew
    reward *= (cmd_norm > 0.01)  # No reward for zero commands.
    return jp.nan_to_num(reward)


  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  # Pose-related rewards.

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    cost = jp.sum(
        jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices])
    )
    cost *= jp.abs(cmd[1]) > 0.1
    return jp.nan_to_num(cost)

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(
        jp.abs(
            qpos[self._knee_indices] - self._default_pose[self._knee_indices]
        )
    ))

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.nan_to_num(jp.sum(jp.square(qpos - self._default_pose) * self._weights))

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    body_vel = self.get_global_linvel(data)[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return jp.nan_to_num(reward)

  def _cost_feet_clearance(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.nan_to_num(jp.sum(delta * vel_norm))

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.nan_to_num(jp.sum(jp.square(error) * first_contact))

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.1, #0.2
      threshold_max: float = 0.5,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return jp.nan_to_num(reward)

  def _reward_feet_phase(
      self,
      data: mjx.Data,
      phase: jax.Array,
      foot_height: jax.Array,
      commands: jax.Array,
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    del commands  # Unused.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    # TODO(kevin): Ensure no movement at 0 command.
    # cmd_norm = jp.linalg.norm(commands)
    # reward *= cmd_norm > 0.1  # No reward for zero commands.
    return jp.nan_to_num(reward)

  # def _cost_both_feet_up(self, contact: jax.Array) -> jax.Array:
  #   # contact [0, 0] if both feet are in the air, [1, 1] if both feet are on the ground

  #   both_up = jp.all(contact == 0)
  #   return jp.nan_to_num(both_up)

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )
