from enum import Enum

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from .EFlexAgentV0 import EFLEXAgentTransition, EFLEXAgentState


class EFlexAgentV1(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # action space specify which transition can be activated
        self.action_space = spaces.Discrete(len(EFLEXAgentTransition.list()))# {0,1,...,n-1}

        # observation is a multi discrete specifying which state is activated
        self.observation_space = spaces.MultiBinary(len(EFLEXAgentState.list()))

        # reward_range
        self.reward_range = (float(-1.0), float(1.0))

        # Simulation related variables.
        self.current_state = None
        self.current_reward = 0.0
        self.obs = None
        self.obs_pre = None

        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        last_state = self.current_state
        self._take_action(action)
        next_state = self.current_state

        reward = self._get_reward()

        ob = self._get_state()
        episode_over = self._is_episode_over()
        return ob, reward, episode_over, {'info':' From {} => {} => {}' .format(last_state, EFLEXAgentTransition(action)
                                                                                , next_state)}

    def reset(self):
        self.current_state = EFLEXAgentState.Stopped
        self.current_reward = 0.0

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self, seed=None):
        pass

    def _take_action(self, action):
        # check the action
        _act = 0
        if action:
            _act = action

        # Aborted
        if self.current_state == EFLEXAgentState.Aborted:
            if action == EFLEXAgentTransition.Clear:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = -1.0
            else:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = 0.0
        # Stopped
        elif self.current_state == EFLEXAgentState.Stopped:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -1.0
            elif action == EFLEXAgentTransition.Reset:
                self.current_state = EFLEXAgentState.Idle
                self.current_reward = 0.0
            else:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 0.0
        # Idle
        elif self.current_state == EFLEXAgentState.Idle:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -1.0
            elif action == EFLEXAgentTransition.Start:
                self.current_state = EFLEXAgentState.Execute
                self.current_reward = 0.5
            elif action == EFLEXAgentTransition.PowerOff:
                self.current_state = EFLEXAgentState.PowerOff
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Standby:
                self.current_state = EFLEXAgentState.StandBy
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Stop:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.ChangeLoad:
                self.current_state = EFLEXAgentState.Idle
                self.current_reward = 0.1
            else:
                self.current_state = EFLEXAgentState.Idle
                self.current_reward = 0.0
        # PowerOff
        elif self.current_state == EFLEXAgentState.PowerOff:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -1.0
            elif action == EFLEXAgentTransition.PowerOn:
                self.current_state = EFLEXAgentState.StartedUp
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Stop:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 0.0
            else:
                self.current_state = EFLEXAgentState.PowerOff
                self.current_reward = 0.0
        # StandBy
        elif self.current_state == EFLEXAgentState.StandBy:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -1.0
            elif action == EFLEXAgentTransition.PowerOn:
                self.current_state = EFLEXAgentState.StartedUp
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Stop:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 0.0
            else:
                self.current_state = EFLEXAgentState.StandBy
                self.current_reward = 0.0
        # StartedUp
        elif self.current_state == EFLEXAgentState.StartedUp:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -1.0
            elif action == EFLEXAgentTransition.Reset:
                self.current_state = EFLEXAgentState.Idle
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Stop:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 0.0
            else:
                self.current_state = EFLEXAgentState.StartedUp
                self.current_reward = 0.0
        # Execute
        elif self.current_state == EFLEXAgentState.Execute:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -1.0
            elif action == EFLEXAgentTransition.SC:
                self.current_state = EFLEXAgentState.Completed
                self.current_reward = 0.5
            elif action == EFLEXAgentTransition.Hold:
                self.current_state = EFLEXAgentState.Held
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Suspend:
                self.current_state = EFLEXAgentState.Suspended
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Stop:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.ChangeLoad:
                self.current_state = EFLEXAgentState.Execute
                self.current_reward = 0.0
            else:
                self.current_state = EFLEXAgentState.Execute
                self.current_reward = 0.0
        # Complete
        elif self.current_state == EFLEXAgentState.Complete:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -1.0
            elif action == EFLEXAgentTransition.SC:
                self.current_state = EFLEXAgentState.Idle
                self.current_reward = 0.5
            elif action == EFLEXAgentTransition.Stop:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 0.0
            else:
                self.current_state = EFLEXAgentState.Completed
                self.current_reward = 0.0
        # Held
        elif self.current_state == EFLEXAgentState.Held:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -1.0
            elif action == EFLEXAgentTransition.UnHold:
                self.current_state = EFLEXAgentState.Execute
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Suspend:
                self.current_state = EFLEXAgentState.Suspended
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Stop:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.ChangeLoad:
                self.current_state = EFLEXAgentState.Held
                self.current_reward = 0.0
            else:
                self.current_state = EFLEXAgentState.Held
                self.current_reward = 0.0
        # Suspended
        elif self.current_state == EFLEXAgentState.Suspended:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -1.0
            elif action == EFLEXAgentTransition.Unsuspend:
                self.current_state = EFLEXAgentState.Execute
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Hold:
                self.current_state = EFLEXAgentState.Held
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.Stop:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 0.0
            elif action == EFLEXAgentTransition.ChangeLoad:
                self.current_state = EFLEXAgentState.Suspended
                self.current_reward = 0.0
            else:
                self.current_state = EFLEXAgentState.Suspended
                self.current_reward = 0.0

    def _get_state(self):
        obs = np.zeros(self.observation_space.shape)
        obs[self.current_state.value] = 1.0
        return

    def _is_episode_over(self, action=None):
        return self.current_state == EFLEXAgentState.Aborted

    def _get_reward(self):
        return self.current_reward
