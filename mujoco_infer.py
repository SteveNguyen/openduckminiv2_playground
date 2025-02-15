import mujoco
import pickle
import numpy as np

import mujoco.viewer
import time
import argparse

from onnx_infer import OnnxInfer
import json # TMP

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
parser.add_argument("-k", action="store_true", default=False)
args = parser.parse_args()

NUM_DOFS = 10

# reference_motion = json.load(open("reference_motion/0_processed.json"))

if args.k:
    import pygame

    pygame.init()
    # open a blank pygame window
    screen = pygame.display.set_mode((100, 100))
    pygame.display.set_caption("Press arrow keys to move robot")

# Params
linearVelocityScale = 1.0
angularVelocityScale = 1.0
dof_pos_scale = 1.0
dof_vel_scale = 1.0
action_scale = 0.35
history_len = 0

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

COMMANDS_RANGE_X = [-0.1, 0.2]
COMMANDS_RANGE_Y = [-0.1, 0.1]
COMMANDS_RANGE_THETA = [-0.4, 0.4]

prev_action = np.zeros(NUM_DOFS)
commands = [0.0, 0.0, 0.0]
decimation = 10
data.qpos[3 : 3 + 4] = [1, 0, 0.0, 0]

data.qpos[7:] = init_pos
data.ctrl[:] = init_pos



gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
gyro_dimensions = 3
linvel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "local_linvel")
linvel_dimensions = 3
# gravity_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")
# gravity_dimensions = 3


imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
gait_freq = 2
control_dt = model.opt.timestep * decimation
phase_dt = 2 * np.pi * control_dt * gait_freq
current_phase = np.array([0])

qpos_error_history = np.zeros(history_len * NUM_DOFS)
qvel_history = np.zeros(history_len * NUM_DOFS)
gravity_history = np.zeros(history_len * 3)


def get_sensor(model, data, name, dimensions):
    i = model.sensor_name2id(name)
    return data.sensordata[i : i + dimensions]


def get_gyro(data):
    return data.sensordata[gyro_id : gyro_id + gyro_dimensions]


def get_linvel(data):
    return data.sensordata[linvel_id : linvel_id + linvel_dimensions]


def get_gravity(data):
    return data.site_xmat[imu_site_id].reshape((3, 3)).T @ np.array([0, 0, -1])


def get_phase():
    global current_phase
    phase_tp1 = current_phase + phase_dt
    current_phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
    cos = np.cos(current_phase)
    sin = np.sin(current_phase)
    return np.concatenate([cos, sin])


# phases = []
def get_obs(
    data, last_action, command, qvel_history, qpos_error_history, gravity_history
):

    gyro = get_gyro(data)
    linvel = get_linvel(data)
    gravity = get_gravity(data)
    joint_angles = data.qpos[7:]
    joint_vel = data.qvel[6:]
    phase = get_phase()
    # phases.append(phase)

    if history_len > 0:
        qvel_history = np.roll(qvel_history, NUM_DOFS)
        qvel_history[:NUM_DOFS] = joint_vel

        last_motor_target = init_pos + last_action * action_scale
        qpos_error = joint_angles - last_motor_target
        qpos_error_history = np.roll(qpos_error_history, NUM_DOFS)
        qpos_error_history[:NUM_DOFS] = qpos_error

        gravity_history = np.roll(gravity_history, 3)
        gravity_history[:3] = gravity

    obs = np.concatenate(
        [
            # linvel,
            # gyro,
            gravity,
            command,
            joint_angles - init_pos,
            joint_vel,
            last_action,
            phase,
            qpos_error_history,  # is [] if history_len == 0
            qvel_history,  # is [] if history_len == 0
            gravity_history,  # is [] if history_len == 0
        ]
    )

    return obs


def key_callback(keycode):
    pass


def handle_keyboard():
    global commands
    keys = pygame.key.get_pressed()
    lin_vel_x = 0
    lin_vel_y = 0
    ang_vel = 0
    if keys[pygame.K_z]:
        lin_vel_x = COMMANDS_RANGE_X[1]
    if keys[pygame.K_s]:
        lin_vel_x = COMMANDS_RANGE_X[0]
    if keys[pygame.K_q]:
        lin_vel_y = COMMANDS_RANGE_Y[1]
    if keys[pygame.K_d]:
        lin_vel_y = COMMANDS_RANGE_Y[0]
    if keys[pygame.K_a]:
        ang_vel = COMMANDS_RANGE_THETA[1]
    if keys[pygame.K_e]:
        ang_vel = COMMANDS_RANGE_THETA[0]

    commands[0] = lin_vel_x
    commands[1] = lin_vel_y
    commands[2] = ang_vel

    print(commands)

    pygame.event.pump()  # process event queue


saved_obs = []

try:

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False, key_callback=key_callback
    ) as viewer:
        counter = 0
        # reference_i = 0
        while True:
            
            step_start = time.time()

            mujoco.mj_step(model, data)

            counter += 1

            if counter % decimation == 0:
                # reference_i += 1
                # if reference_i % len(reference_motion) == 0:
                #     reference_i = 0
                # reference = reference_motion[reference_i]
                obs = get_obs(
                    data,
                    prev_action,
                    commands,
                    qvel_history,
                    qpos_error_history,
                    gravity_history,
                )
                saved_obs.append(obs)
                action = policy.infer(obs)

                prev_action = action.copy()

                action = init_pos + action * action_scale
                data.ctrl = action.copy()
                # data.ctrl = reference

            viewer.sync()

            if args.k:
                handle_keyboard()

            # pickle.dump(phases, open("phases.pkl", "wb"))

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
except KeyboardInterrupt:
    pickle.dump(saved_obs, open("mujoco_saved_obs.pkl", "wb"))
