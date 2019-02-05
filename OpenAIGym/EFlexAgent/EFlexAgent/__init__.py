from gym.envs.registration import register

register(
    id='eflex-agent-v0',
    entry_point='.envs:EFlexAgentV0',
)
register(
    id='eflex-agent-v1',
    entry_point='.envs:EFlexAgentV1',
)
