from gym.envs.registration import register

register(
    id='binpick-v0',
    entry_point='gym_binpick.envs:BinPickEnv',
)

register(
    id='binpick-v1',
    entry_point='gym_binpick.envs:BinPickEnv1',
)

register(
    id='binpick-v2',
    entry_point='gym_binpick.envs:BinPickEnv2',
)

register(
    id='binpick-v3',
    entry_point='gym_binpick.envs:BinPickEnv3',
)

