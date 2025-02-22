import json
import jax.numpy as jp
from jax import vmap


class PolyReferenceMotion:
    def __init__(self, polynomial_coefficients: str):
        data = json.load(open(polynomial_coefficients))
        self.dx_range = [0, 0]
        self.dy_range = [0, 0]
        self.dtheta_range = [0, 0]
        self.dxs = []
        self.dys = []
        self.dthetas = []
        self.data_array = []
        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.startend_double_support_ratio = None
        self.start_offset = None
        self.nb_steps_in_period = None

        self.process(data)

    def process(self, data):
        print("[Poly ref data] Processing ...")
        _data = {}
        for name in data.keys():
            split = name.split("_")
            dx = float(split[0])
            dy = float(split[1])
            dtheta = float(split[2])

            if self.period is None:
                self.period = data[name]["period"]
                self.fps = data[name]["fps"]
                self.frame_offsets = data[name]["frame_offsets"]
                self.startend_double_support_ratio = data[name][
                    "startend_double_support_ratio"
                ]
                self.start_offset = int(self.startend_double_support_ratio * self.fps)
                self.nb_steps_in_period = int(self.period * self.fps)

            if dx not in self.dxs:
                self.dxs.append(dx)

            if dy not in self.dys:
                self.dys.append(dy)

            if dtheta not in self.dthetas:
                self.dthetas.append(dtheta)

            self.dx_range = [min(dx, self.dx_range[0]), max(dx, self.dx_range[1])]
            self.dy_range = [min(dy, self.dy_range[0]), max(dy, self.dy_range[1])]
            self.dtheta_range = [
                min(dtheta, self.dtheta_range[0]),
                max(dtheta, self.dtheta_range[1]),
            ]

            if dx not in _data:
                _data[dx] = {}

            if dy not in _data[dx]:
                _data[dx][dy] = {}

            if dtheta not in _data[dx][dy]:
                _data[dx][dy][dtheta] = data[name]

            _coeffs = data[name]["coefficients"]

            coeffs = []
            for k, v in _coeffs.items():
                coeffs.append(v)
            _data[dx][dy][dtheta] = coeffs

        self.dxs = sorted(self.dxs)
        self.dys = sorted(self.dys)
        self.dthetas = sorted(self.dthetas)

        nb_dx = len(self.dxs)
        nb_dy = len(self.dys)
        nb_dtheta = len(self.dthetas)

        self.data_array = nb_dx * [None]
        for x, dx in enumerate(self.dxs):
            self.data_array[x] = nb_dy * [None]
            for y, dy in enumerate(self.dys):
                self.data_array[x][y] = nb_dtheta * [None]
                for th, dtheta in enumerate(self.dthetas):
                    self.data_array[x][y][th] = jp.array(_data[dx][dy][dtheta])

        print("[Poly ref data] Done processing")

    def vel_to_index(self, dx, dy, dtheta):

        dx = jp.clip(dx, self.dx_range[0], self.dx_range[1])
        dy = jp.clip(dy, self.dy_range[0], self.dy_range[1])
        dtheta = jp.clip(dtheta, self.dtheta_range[0], self.dtheta_range[1])

        ix = jp.argmin(jp.abs(jp.array(self.dxs) - dx))
        iy = jp.argmin(jp.abs(jp.array(self.dys) - dy))
        itheta = jp.argmin(jp.abs(jp.array(self.dthetas) - dtheta))

        return ix, iy, itheta

    def sample_polynomial(self, t, coeffs):
        return vmap(lambda c: jp.polyval(jp.flip(c), t))(coeffs)

    def get_reference_motion(self, dx, dy, dtheta, i):
        ix, iy, itheta = self.vel_to_index(dx, dy, dtheta)
        t = i % self.nb_steps_in_period / self.nb_steps_in_period
        t = jp.clip(t, 0.0, 1.0)
        ret = self.sample_polynomial(t, self.data_array[ix][iy][itheta])
        return ret


PRM = PolyReferenceMotion(
    "/home/antoine/MISC/Open_Duck_reference_motion_generator/polynomial_coefficients.json"
)
vals = []
select_dim = -1
for i in range(PRM.nb_steps_in_period*2):
    print(i)
    vals.append(PRM.get_reference_motion(0.0, -0.05, -0.3, i)[select_dim])

# plot
import matplotlib.pyplot as plt
import numpy as np


ts = np.arange(0, PRM.nb_steps_in_period*2)
plt.plot(ts, vals)
plt.show()
