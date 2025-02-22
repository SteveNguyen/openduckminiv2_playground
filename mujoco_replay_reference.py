import mujoco
import mujoco.viewer
import argparse
from poly_reference_motion import PolyReferenceMotion
import numpy as np
import time


parser = argparse.ArgumentParser()
parser.add_argument(
    "--coefficients",
    type=str,
    default="polynomial_coefficients.json",
    help="Path to polynomial coefficients file",
)
args = parser.parse_args()

PRM = PolyReferenceMotion(args.coefficients)

model = mujoco.MjModel.from_xml_path(
    "env/locomotion/open_duck_mini_v2/xmls/scene_mjx_flat_terrain.xml"
)
model.opt.timestep = 0.002
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
decimation = 10

# Set the base position manually
# base_joint_id = model.joint("base").qposadr  # Get the index of freejoint in qpos
data.qpos[0 : 0 + 3] = np.array([0, 0, 0.5])  # Set position (x, y, z)
data.qvel[0 : 0 + 6] = np.zeros(6)

model.opt.gravity[:] = 0  # Disable gravity


counter = 0
with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:
    while True:
        step_start = time.time()
        counter += 1
        mujoco.mj_step(model, data)

        if counter % decimation == 0:
            ref = PRM.get_reference_motion(0.1, 0, 0, counter // decimation)
            ref_joints_pos = ref[0:16]
            ref_joints_pos = np.concatenate(
                [ref_joints_pos[:5], ref_joints_pos[11:]]
            )  # remove antennas
            data.ctrl = ref_joints_pos

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
