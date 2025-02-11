from env.locomotion.open_duck_mini_v2 import joystick as open_duck_mini_v2_joystick
from env.locomotion.open_duck_mini_v2 import randomize as open_duck_mini_v2_randomize

from mujoco_playground._src import registry
from mujoco_playground._src import locomotion


import functools


locomotion.register_environment('OpenDuckMiniV2JoystickFlatTerrain', functools.partial(open_duck_mini_v2_joystick.Joystick, task="flat_terrain"), open_duck_mini_v2_joystick.default_config
)
locomotion._randomizer["OpenDuckMiniV2JoystickFlatTerrain"]=open_duck_mini_v2_randomize.domain_randomize
print(f"RAND: {locomotion._randomizer.keys()}")
setattr(locomotion, 'ALL',list(locomotion._envs.keys())) #I would have prefered not to come to this...

print(f"GET: {locomotion.get_domain_randomizer('OpenDuckMiniV2JoystickFlatTerrain')}")
# def get_domain_randomizer(
#     env_name: str,
# ) -> Optional[Callable[[mjx.Model, jax.Array], Tuple[mjx.Model, mjx.Model]]]:
#   """Get the default domain randomizer for an environment."""
#   if env_name not in _randomizer:
#     print(
#         f"Env '{env_name}' does not have a domain randomizer in the locomotion"
#         " registry."
#     )
#     return None
#   return _randomizer[env_name]


env = registry.load('OpenDuckMiniV2JoystickFlatTerrain')
