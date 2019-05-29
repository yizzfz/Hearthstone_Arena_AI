from gym.envs.registration import register

register(
    id='sabberstone-v0',
    entry_point='sabber:SabberEnv',
)
