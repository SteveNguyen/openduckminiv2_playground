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
"""Domain randomization for the Open Duck Mini V2 environment. (based on Berkeley Humanoid)"""

import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
        # Floor friction: =U(0.4, 1.0).
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(key, minval=0.4, maxval=1.0)
        )

        # Scale static friction: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
            key, shape=(10,), minval=0.9, maxval=1.2  # was 0.9, 1.1
        )
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        # Scale armature: *U(1.0, 1.05).
        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(
            key, shape=(10,), minval=0.9, maxval=1.1  # was 1.0, 1.05
        )
        dof_armature = model.dof_armature.at[6:].set(armature)

        # Jitter center of mass positiion: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        dpos = jax.random.uniform(key, (3,), minval=-0.07, maxval=0.07) # was 0.05
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
            model.body_ipos[TORSO_BODY_ID] + dpos
        )

        # Scale all link masses: *U(0.9, 1.1).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        # Add mass to torso: +U(-0.2, 0.2).
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=-0.2, maxval=0.2)
        body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + dmass)

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0
        qpos0 = qpos0.at[7:].set(
            qpos0[7:] + jax.random.uniform(key, shape=(10,), minval=-0.05, maxval=0.05)
        )

        # # Randomize KP
        rng, key = jax.random.split(rng)
        factor = jax.random.uniform(key, shape=(10,), minval=0.8, maxval=1.2)
        current_kp = model.actuator_gainprm[:, 0]
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(current_kp * factor)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-current_kp * factor)

        return (
            geom_friction,
            body_ipos,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
            actuator_gainprm,
            actuator_biasprm
        )

    (
        friction,
        body_ipos,
        frictionloss,
        armature,
        body_mass,
        qpos0,
        actuator_gainprm,
        actuator_biasprm
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_ipos": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_mass": 0,
            "qpos0": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0
        }
    )

    model = model.tree_replace(
        {
            "geom_friction": friction,
            "body_ipos": body_ipos,
            "dof_frictionloss": frictionloss,
            "dof_armature": armature,
            "body_mass": body_mass,
            "qpos0": qpos0,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm
        }
    )

    return model, in_axes
