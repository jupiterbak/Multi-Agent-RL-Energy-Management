from gym.envs.registration import register

register(
    id='eflex-agent-v0',
    entry_point='gym_eflex_agent.envs:EFlexAgentV0',
)
register(
    id='eflex-agent-v1',
    entry_point='gym_eflex_agent.envs:EFlexAgentV1',
)
register(
    id='eflex-multi-agent-v0',
    entry_point='gym_eflex_agent.envs:EFlexMultiAgent',
)
