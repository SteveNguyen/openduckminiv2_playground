import matplotlib.pyplot as plt
import numpy as np
import pickle

phases = np.array(pickle.load(open("phases.pkl", "rb")))
print(phases)
# plot and label each phase [phase1, phase2, phase3, phase4]

plt.plot(phases[:, 0], label="phase1")
plt.plot(phases[:, 1], label="phase2")
# plt.plot(phases[:, 2], label="phase3")
# plt.plot(phases[:, 3], label="phase4")

plt.legend()

plt.show()
