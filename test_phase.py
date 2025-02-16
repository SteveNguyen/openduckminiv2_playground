import matplotlib.pyplot as plt
import numpy as np
import time

gait_freq = 2.5
control_dt = 0.02

phase_dt = 2 * np.pi * control_dt * gait_freq
current_phase = np.array([0, np.pi])

phases = []
s = time.time()
while True:
    phase_tp1 = current_phase + phase_dt
    current_phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi
    # print(current_phase)
    cos = np.cos(current_phase)
    sin = np.sin(current_phase)
    phases.append(cos)
    time.sleep(control_dt)
    if time.time() - s > 2:
        break

phases = np.array(phases)
# plot and label each phase [phase1, phase2, phase3, phase4]

# new_phases = []
# rewards = []
# foot_zs = []
# max_foot_height = 0.03
# for phase in phases:
#     phase += 1
#     phase /= 2
#     phase *= max_foot_height
#     new_phases.append(phase)
#     foot_z = np.array([phase[0], phase[1]])
#     # foot_z = np.array([0, 0])
#     foot_zs.append(foot_z)
#     error = np.sum(np.square(foot_z - phase))
#     reward = np.exp(-error / 0.01)
#     rewards.append(reward)




# # plt.plot(new_phases, label="phases")
# plt.plot(rewards, label="rewards")
# plt.plot(foot_zs, label="foot_zs")

# plt.legend()

# plt.show()

# exit()
def get_rz(phi, swing_height=0.08):
    def cubic_bezier_interpolation(y_start, y_end, x):
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + np.pi) / (2 * np.pi)
    stance = cubic_bezier_interpolation(0, swing_height, 2 * x)
    swing = cubic_bezier_interpolation(swing_height, 0, 2 * x - 1)
    print("stance", stance)
    print("swing", swing)
    return np.where(x <= 0.5, stance, swing)

rewards = []
foot_zs = []
rzs = []
i = 0
for phase in phases:
    i += phase_dt
    # foot_z = np.array([np.cos(i), np.sin(i)]) * 0.03
    # foot_z = np.array([phase[1], phase[0]]) * 0.03
    # foot_z = phase * 0.03
    foot_z = [0.0, 0.0]
    foot_zs.append(foot_z)
    print("phase", phase)
    print("foot_z", foot_z)
    rz = get_rz(phase, swing_height=0.03)
    rzs.append(rz)
    print("rz", rz)
    error = np.sum(np.square(foot_z - rz))
    print("error", error)
    reward = np.exp(-error / 0.01)
    print("reward", reward)
    rewards.append(reward)
    print("==")


plt.plot(phases[:, 0], label="phase1")
plt.plot(phases[:, 1], label="phase2")
plt.plot(rewards, label="rewards")
plt.plot(foot_zs, label="foot_zs")
# plt.plot(rzs, label="rzs")

plt.legend()

plt.show()
