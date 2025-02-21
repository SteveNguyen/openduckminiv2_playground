from glob import glob
import os

# import numpy as np
import json
import jax.numpy as jp


def process_reference_motion(directory):

    # load all json files except metadata.json
    json_files = glob(directory + "/*.json")
    data = {}
    period = None
    fps = None
    frame_offsets = None
    dx_range = [0, 0]
    dy_range = [0, 0]
    dtheta_range = [0, 0]
    dxs = []
    dys = []
    dthetas = []
    data_array = []

    for file in json_files:

        if period is None:
            tmp_file = json.load(open(file))
            period = tmp_file["Placo"]["period"]
            fps = tmp_file["FPS"]
            frame_offsets = tmp_file["Frame_offset"][0]

        name = os.path.basename(file).strip(".json")
        split = name.split("_")
        id = float(split[0])
        dx = float(split[1])
        dy = float(split[2])
        dtheta = float(split[3])

        if dx not in dxs:
            dxs.append(dx)

        if dy not in dys:
            dys.append(dy)

        if dtheta not in dthetas:
            dthetas.append(dtheta)

        dx_range = [min(dx, dx_range[0]), max(dx, dx_range[1])]
        dy_range = [min(dy, dy_range[0]), max(dy, dy_range[1])]
        dtheta_range = [
            min(dtheta, dtheta_range[0]),
            max(dtheta, dtheta_range[1]),
        ]

        if dx not in data:
            data[dx]: dict = {}

        if dy not in data[dx]:
            data[dx][dy]: dict = {}

        if dtheta not in data[dx][dy]:
            data[dx][dy][dtheta] = json.load(open(file))["Frames"]

    dxs = sorted(dxs)
    dys = sorted(dys)
    dthetas = sorted(dthetas)

    # print("dx range: ", dx_range)
    # print("dy range: ", dy_range)
    # print("dtheta range: ", dtheta_range)

    nb_dx = len(dxs)
    nb_dy = len(dys)
    nb_dtheta = len(dthetas)

    # print("nb dx", nb_dx)
    # print("nb dy", nb_dy)
    # print("nb dtheta", nb_dtheta)

    data_array = nb_dx * [None]

    for x, dx in enumerate(dxs):
        data_array[x] = nb_dy * [None]
        for y, dy in enumerate(dys):
            data_array[x][y] = nb_dtheta * [None]
            for t, dtheta in enumerate(dthetas):
                data_array[x][y][t] = data[dx][dy][dtheta]

    # root_pos_slice = [0, 3]
    # root_quat_slice = [3, 7]
    # linear_vel_slice = [29, 32]
    # angular_vel_slice = [32, 35]
    # joint_pos_slice = [7, 23]
    # joint_vels_slice = [35, 51]
    # left_toe_pos_slice = [23, 26]
    # right_toe_pos_slice = [26, 29]

    return (
        jp.array(data_array),
        jp.array(dx_range),
        jp.array(dy_range),
        jp.array(dtheta_range),
        jp.array(dxs),
        jp.array(dys),
        jp.array(dthetas),
        jp.array(fps),
        jp.array(period),
        # jp.array(root_pos_slice),
        # jp.array(root_quat_slice),
        # jp.array(linear_vel_slice),
        # jp.array(angular_vel_slice),
        # jp.array(joint_pos_slice),
        # jp.array(joint_vels_slice),
        # jp.array(left_toe_pos_slice),
        # jp.array(right_toe_pos_slice),
    )


def vel_to_index(dx, dy, dtheta, dx_range, dy_range, dtheta_range, dxs, dys, dthetas):

    # jp.max returns the max in an array, not between two values
    # convert the following so that it works

    # dx = jp.min(jp.max(dx, dx_range[0]), dx_range[1])
    # dy = jp.min(jp.max(dy, dy_range[0]), dy_range[1])
    # dtheta = jp.min(jp.max(dtheta, dtheta_range[0]), dtheta_range[1])

    dx = jp.clip(dx, dx_range[0], dx_range[1])
    dy = jp.clip(dy, dy_range[0], dy_range[1])
    dtheta = jp.clip(dtheta, dtheta_range[0], dtheta_range[1])

    ix = jp.argmin(jp.abs(jp.array(dxs) - dx))
    iy = jp.argmin(jp.abs(jp.array(dys) - dy))
    itheta = jp.argmin(jp.abs(jp.array(dthetas) - dtheta))

    return ix, iy, itheta


def get_closest_reference_motion(
    data_array, dx, dy, dtheta, i, dx_range, dy_range, dtheta_range, dxs, dys, dthetas
):
    ix, iy, itheta = vel_to_index(
        dx, dy, dtheta, dx_range, dy_range, dtheta_range, dxs, dys, dthetas
    )
    return data_array[ix][iy][itheta][i]
