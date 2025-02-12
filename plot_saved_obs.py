import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, required=False, default="mujoco_saved_obs.pkl")
args = parser.parse_args()


isaac_init_pos = np.array(
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

isaac_joints_order = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]

obses = pickle.load(open(args.data, "rb"))

num_dofs = 10
dof_poses = []  # (dof, num_obs)
actions = []  # (dof, num_obs)

for i in range(num_dofs):
    print(i)
    dof_poses.append([])
    actions.append([])
    for obs in obses:
        dof_poses[i].append(obs[6 : 6 + 10][i] - isaac_init_pos[i])
        actions[i].append(obs[-(10 + 4) : -4][i])

# plot action vs dof pos

nb_dofs = len(dof_poses)
nb_rows = int(np.sqrt(nb_dofs))
nb_cols = int(np.ceil(nb_dofs / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True)

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_dofs:
            break
        axs[i, j].plot(actions[i * nb_cols + j], label="action")
        axs[i, j].plot(dof_poses[i * nb_cols + j], label="dof_pos")
        axs[i, j].legend()
        axs[i, j].set_title(f"{isaac_joints_order[i * nb_cols + j]}")

fig.suptitle(f"{args.data}")
plt.show()

obses_names = [
    "projected_gravity 0",
    "projected_gravity 1",
    "projected_gravity 2",
    # commands
    "command 0",
    "command 1",
    "command 2",
    # dof pos
    "pos_" + str(isaac_joints_order[0]),
    "pos_" + str(isaac_joints_order[1]),
    "pos_" + str(isaac_joints_order[2]),
    "pos_" + str(isaac_joints_order[3]),
    "pos_" + str(isaac_joints_order[4]),
    "pos_" + str(isaac_joints_order[5]),
    "pos_" + str(isaac_joints_order[6]),
    "pos_" + str(isaac_joints_order[7]),
    "pos_" + str(isaac_joints_order[8]),
    "pos_" + str(isaac_joints_order[9]),
    # dof vel
    "vel_" + str(isaac_joints_order[0]),
    "vel_" + str(isaac_joints_order[1]),
    "vel_" + str(isaac_joints_order[2]),
    "vel_" + str(isaac_joints_order[3]),
    "vel_" + str(isaac_joints_order[4]),
    "vel_" + str(isaac_joints_order[5]),
    "vel_" + str(isaac_joints_order[6]),
    "vel_" + str(isaac_joints_order[7]),
    "vel_" + str(isaac_joints_order[8]),
    "vel_" + str(isaac_joints_order[9]),
    # action
    "action_" + str(isaac_joints_order[0]),
    "action_" + str(isaac_joints_order[1]),
    "action_" + str(isaac_joints_order[2]),
    "action_" + str(isaac_joints_order[3]),
    "action_" + str(isaac_joints_order[4]),
    "action_" + str(isaac_joints_order[5]),
    "action_" + str(isaac_joints_order[6]),
    "action_" + str(isaac_joints_order[7]),
    "action_" + str(isaac_joints_order[8]),
    "action_" + str(isaac_joints_order[9]),
    # phase
    "phase1_cos",
    "phase1_sin",
    "phase2_cos",
    "phase2_sin",
]


# obses = [[56 obs at time 0], [56 obs at time 1], ...]

nb_obs = len(obses[0])
nb_rows = int(np.sqrt(nb_obs))
nb_cols = int(np.ceil(nb_obs / nb_rows))

fig, axs = plt.subplots(nb_rows, nb_cols, sharex=True, sharey=True)

for i in range(nb_rows):
    for j in range(nb_cols):
        if i * nb_cols + j >= nb_obs:
            break
        axs[i, j].plot([obs[i * nb_cols + j] for obs in obses])
        axs[i, j].set_title(obses_names[i * nb_cols + j])

#set ylim between -5 and 5

for ax in axs.flat:
    ax.set_ylim([-5, 5])

    


fig.suptitle(f"{args.data}")
plt.show()
