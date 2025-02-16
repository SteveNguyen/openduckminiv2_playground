import json
import time
import argparse
import os
import numpy as np

"""
Extracting only the joints positions of a walk cycle from a reference file
"""

# TODO add joint vels and contacts ? 

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="The reference file to process")
parser.add_argument("--no_head", action="store_true" ,help="remove the head joints from the file")
args = parser.parse_args()

file_name = os.path.basename(args.input).split(".")[0]
file_path = os.path.dirname(args.input)
processed_file_name = file_name + "_processed.json"
processed_file_path = os.path.join(file_path, processed_file_name)

episode = json.load(open(args.input))
fps = episode["FPS"]
period = 0.6599  # TODO write in the file
frames = episode["Frames"]
frame_offsets = episode["Frame_offset"][0]
joints_positions = slice(frame_offsets["joints_pos"], frame_offsets["left_toe_pos"])
joints_vels = slice(frame_offsets["joints_vel"], frame_offsets["left_toe_vel"])

nb_frames_in_one_walk_cycle = int(fps * period)

processed_frames = {"joints_pos" : [], "joints_vel" : []}

def remove_head_joints(frame):
    return list(frame[0:5]) + list(frame[11:16])

for i, frame in enumerate(frames):
    joints_pos = frame[joints_positions]
    joints_vel = frame[joints_vels]
    if args.no_head:
        joints_pos = remove_head_joints(joints_pos)
        joints_vel = remove_head_joints(joints_vel)
    processed_frames["joints_pos"].append(joints_pos)
    processed_frames["joints_vel"].append(joints_vel)
    if i == nb_frames_in_one_walk_cycle:
        break

with open(processed_file_path, "w") as f:
    json.dump(processed_frames, f)
