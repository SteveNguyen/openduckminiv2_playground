import mujoco
import numpy as np

import mujoco.viewer
import time
import argparse

# from mini_bdx.utils.mujoco_utils import check_contact

from utils.onnx_infer import OnnxInfer
import pickle

# from bam.model import load_model
# from bam.mujoco import MujocoController
# from mini_bdx_runtime.rl_utils import mujoco_joints_order

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
args = parser.parse_args()

# Params
linearVelocityScale = 1.0
angularVelocityScale = 1.0
dof_pos_scale = 1.0
dof_vel_scale = 1.0
action_scale = 0.5


init_pos = np.array(
    [
        0.002,
        0.053,
        -0.63,
        1.368,
        -0.784,
        # 0.0,
        # 0,
        # 0,
        # 0,
        # 0,
        # 0,
        -0.003,
        -0.065,
        0.635,
        1.379,
        -0.796,
    ]
)

# model = mujoco.MjModel.from_xml_path(
#     "/home/antoine/MISC/mini_BDX/mini_bdx/robots/open_duck_mini_v2/scene_position.xml"
# )
model = mujoco.MjModel.from_xml_path(
    "env/locomotion/open_duck_mini_v2/xmls/scene_mjx_flat_terrain.xml"
)
model.opt.timestep = 0.002
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

policy = OnnxInfer(args.onnx_model_path, awd=True)

COMMANDS_RANGE_X = [-0.2, 0.3]
COMMANDS_RANGE_Y = [-0.2, 0.2]
COMMANDS_RANGE_THETA = [-0.3, 0.3]

prev_action = np.zeros(10)
commands = [0.2, 0.0, 0.0]
decimation = 10
data.qpos[3 : 3 + 4] = [1, 0, 0.0, 0]

data.qpos[7:] = init_pos
data.ctrl[:] = init_pos

replay_index = 0
saved_obs = []


def check_contact(data, model, body1_name, body2_name):
    body1_id = data.body(body1_name).id
    body2_id = data.body(body2_name).id

    for i in range(data.ncon):
        try:
            contact = data.contact[i]
        except Exception as e:
            return False

        if (
            model.geom_bodyid[contact.geom1] == body1_id
            and model.geom_bodyid[contact.geom2] == body2_id
        ) or (
            model.geom_bodyid[contact.geom1] == body2_id
            and model.geom_bodyid[contact.geom2] == body1_id
        ):
            return True

    return False


def get_feet_contact():
    left_contact = check_contact(data, model, "foot_assembly", "floor")
    right_contact = check_contact(data, model, "foot_assembly_2", "floor")
    return [left_contact, right_contact]


# gyro_id = model.sensor_name2id("gyro")
gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
gyro_dimensions = 3
# linvel_id = model.sensor_name2id("local_linvel")
linvel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel")
linvel_dimensions = 3
imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
gait_freq = 2
control_dt = model.opt.timestep * decimation
phase_dt = 2 * np.pi * control_dt * gait_freq
current_phase = np.array([0, 0])


def get_sensor(model, data, name, dimensions):
    i = model.sensor_name2id(name)
    return data.sensordata[i : i + dimensions]


def get_gyro(data):
    return data.sensordata[gyro_id : gyro_id + gyro_dimensions]


def get_linvel(data):
    return data.sensordata[linvel_id : linvel_id + linvel_dimensions]


def get_gravity(data):
    return data.site_xmat[imu_site_id].reshape((3, 3)) @ np.array([0, 0, -1])


def get_phase():
    global current_phase
    phase_tp1 = current_phase + phase_dt
    current_phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
    cos = np.cos(current_phase)
    sin = np.sin(current_phase)
    return np.concatenate([cos, sin])


def get_obs(data, last_action, command):
    # linvel
    # gyro
    # gravity
    # command
    # joint_angles - defulat pose
    # joint vel
    # last action
    # phase

    gyro = get_gyro(data)
    linvel = get_linvel(data)
    gravity = get_gravity(data)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]
    phase = get_phase()

    obs = np.concatenate(
        [
            linvel,
            gyro,
            gravity,
            command,
            joint_angles - init_pos,
            joint_vel,
            last_action,
            phase,
        ]
    )
    # print("len linvel", len(linvel))
    # print("len gyro", len(gyro))
    # print("len gravity", len(gravity))
    # print("len command", len(command))
    # print("len joint_angles", len(joint_angles))
    # print("len joint_vel", len(joint_vel))
    # print("len last_action", len(last_action))
    # print("len phase", len(phase))

    # print("len obs", len(obs))
    # exit()

    return obs


try:
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        counter = 0
        while True:

            step_start = time.time()  # Was

            mujoco.mj_step(model, data)

            counter += 1
            if counter % decimation == 0:
                obs = get_obs(data, prev_action, commands)
                saved_obs.append(obs)

                action = policy.infer(obs)

                prev_action = action.copy()

                action = action * action_scale + init_pos
                data.ctrl = action.copy()
            viewer.sync()

            # Was
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

except KeyboardInterrupt:
    pickle.dump(saved_obs, open("mujoco_saved_obs.pkl", "wb"))
