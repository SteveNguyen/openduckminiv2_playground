from env.locomotion.open_duck_mini_v2 import joystick as open_duck_mini_v2_joystick
from env.locomotion.open_duck_mini_v2 import randomize as open_duck_mini_v2_randomize
import time

from mujoco_playground._src import registry
from mujoco_playground._src import locomotion

import functools
import tensorflow as tf
from tensorflow.keras import layers
import onnxruntime as rt
import tf2onnx

import sys

import os
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
from jax import numpy as jp
import numpy as np

from mujoco_playground import wrapper

from mujoco_playground.config import locomotion_params


def export_onnx(
    params, act_size, ppo_params, obs_size, output_path="ONNX.onnx"
):
    print(" === EXPORT ONNX === ")

    # inference_fn = make_inference_fn(params, deterministic=True)

    class MLP(tf.keras.Model):
        def __init__(
            self,
            layer_sizes,
            activation=tf.nn.relu,
            kernel_init="lecun_uniform",
            activate_final=False,
            bias=True,
            layer_norm=False,
            mean_std=None,
        ):
            super().__init__()

            self.layer_sizes = layer_sizes
            self.activation = activation
            self.kernel_init = kernel_init
            self.activate_final = activate_final
            self.bias = bias
            self.layer_norm = layer_norm

            if mean_std is not None:
                self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
                self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
            else:
                self.mean = None
                self.std = None

            self.mlp_block = tf.keras.Sequential(name="MLP_0")
            for i, size in enumerate(self.layer_sizes):
                dense_layer = layers.Dense(
                    size,
                    activation=self.activation,
                    kernel_initializer=self.kernel_init,
                    name=f"hidden_{i}",
                    use_bias=self.bias,
                )
                self.mlp_block.add(dense_layer)
                if self.layer_norm:
                    self.mlp_block.add(
                        layers.LayerNormalization(name=f"layer_norm_{i}")
                    )
            if not self.activate_final and self.mlp_block.layers:
                if (
                    hasattr(self.mlp_block.layers[-1], "activation")
                    and self.mlp_block.layers[-1].activation is not None
                ):
                    self.mlp_block.layers[-1].activation = None

            self.submodules = [self.mlp_block]

        def call(self, inputs):
            if isinstance(inputs, list):
                inputs = inputs[0]
            if self.mean is not None and self.std is not None:
                print(self.mean.shape, self.std.shape)
                inputs = (inputs - self.mean) / self.std
            logits = self.mlp_block(inputs)
            loc, _ = tf.split(logits, 2, axis=-1)
            return tf.tanh(loc)

    def make_policy_network(
        param_size,
        mean_std,
        hidden_layer_sizes=[256, 256],
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        layer_norm=False,
    ):
        policy_network = MLP(
            layer_sizes=list(hidden_layer_sizes) + [param_size],
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
            mean_std=mean_std,
        )
        return policy_network

    mean = params[0].mean["state"]
    std = params[0].std["state"]

    # Convert mean/std jax arrays to tf tensors.
    mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))

    tf_policy_network = make_policy_network(
        param_size=act_size * 2,
        mean_std=mean_std,
        hidden_layer_sizes=ppo_params.network_factory.policy_hidden_layer_sizes,
        activation=tf.nn.swish,
    )

    example_input = tf.zeros((1, obs_size))
    example_output = tf_policy_network(example_input)
    print(example_output.shape)

    def transfer_weights(jax_params, tf_model):
        """
        Transfer weights from a JAX parameter dictionary to the TensorFlow model.

        Parameters:
        - jax_params: dict
        Nested dictionary with structure {block_name: {layer_name: {params}}}.
        For example:
        {
            'CNN_0': {
            'Conv_0': {'kernel': np.ndarray},
            'Conv_1': {'kernel': np.ndarray},
            'Conv_2': {'kernel': np.ndarray},
            },
            'MLP_0': {
            'hidden_0': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_1': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_2': {'kernel': np.ndarray, 'bias': np.ndarray},
            }
        }

        - tf_model: tf.keras.Model
        An instance of the adapted VisionMLP model containing named submodules and layers.
        """
        for layer_name, layer_params in jax_params.items():
            try:
                tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
            except ValueError:
                print(f"Layer {layer_name} not found in TensorFlow model.")
                continue
            if isinstance(tf_layer, tf.keras.layers.Dense):
                kernel = np.array(layer_params["kernel"])
                bias = np.array(layer_params["bias"])
                print(
                    f"Transferring Dense layer {layer_name}, kernel shape {kernel.shape}, bias shape {bias.shape}"
                )
                tf_layer.set_weights([kernel, bias])
            else:
                print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")

        print("Weights transferred successfully.")

    transfer_weights(params[1].policy["params"], tf_policy_network)

    # Example inputs for the model
    test_input = [np.ones((1, obs_size), dtype=np.float32)]

    # Define the TensorFlow input signature
    spec = [
        tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")
    ]

    tensorflow_pred = tf_policy_network(test_input)[0]
    # Build the model by calling it with example data
    print(f"Tensorflow prediction: {tensorflow_pred}")

    tf_policy_network.output_names = ["continuous_actions"]

    # opset 11 matches isaac lab.
    model_proto, _ = tf2onnx.convert.from_keras(
        tf_policy_network, input_signature=spec, opset=11, output_path=output_path
    )
    return

def main():
    # Configure MuJoCo to use the EGL rendering backend (requires GPU)
    print("Setting environment variable to use GPU rendering:")
    # get_ipython().run_line_magic('env', 'MUJOCO_GL=egl')
    os.environ["MUJOCO_GL"] = "egl"
    # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags

    locomotion.register_environment(
        "OpenDuckMiniV2JoystickFlatTerrain",
        functools.partial(open_duck_mini_v2_joystick.Joystick, task="flat_terrain"),
        open_duck_mini_v2_joystick.default_config,
    )
    # Hack to add a new env... I would have prefered not to come to this...
    setattr(locomotion, "ALL", list(locomotion._envs.keys()))
    locomotion._randomizer["OpenDuckMiniV2JoystickFlatTerrain"] = (
        open_duck_mini_v2_randomize.domain_randomize
    )
    env_name = "OpenDuckMiniV2JoystickFlatTerrain"

    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)
    # ppo_params = locomotion_params.brax_ppo_config(env_name)
    ppo_params = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )  # TODO

    obs_size = env.observation_size
    act_size = env.action_size
    print("obs size", obs_size, "action size", act_size)

    randomizer = registry.get_domain_randomizer(env_name)
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )
    print(f"PPO PARAMS: {ppo_training_params}")

    ckpt_path = epath.Path(sys.argv[1])
    # ckpt_path.mkdir(parents=True, exist_ok=True)

    # env = registry.load(env_name)
    eval_env = registry.load(env_name)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    def progress(num_steps, metrics):
        print(
            f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} reward_std: {metrics["eval/episode_reward_std"]}'
        )

    train_fn = functools.partial(
        ppo.train,  # **dict(ppo_training_params),
        num_timesteps=0,
        episode_length=1000,
        network_factory=network_factory,
        randomization_fn=randomizer,
        progress_fn=progress,
        restore_checkpoint_path=ckpt_path,
    )

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        eval_env=registry.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    print("Starting to export")
    s = time.time()
    export_onnx(params, act_size, ppo_params, int(obs_size["state"][0]))
    print("Done exporting, took", time.time() - s)


if __name__ == "__main__":
    main()
