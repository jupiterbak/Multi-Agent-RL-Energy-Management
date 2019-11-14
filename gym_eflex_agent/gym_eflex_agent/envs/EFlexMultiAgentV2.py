from enum import Enum

import gym
import math
import numpy as np
import yaml
from gym import spaces
from gym.utils import seeding


class EFLEXAgentState(Enum):
    Aborted = 1
    Stopped = 0
    PowerOff = 2
    LoadChange = 3
    StandBy = 4
    StartedUp = 5
    Idle = 6
    Execute = 7
    Completed = 8
    Held = 9
    Suspended = 10


class EFLEXAgentTransition(Enum):
    SC = 0
    Abort = 1
    Clear = 2
    Reset = 3
    Stop = 4
    ChangeLoad = 5
    Hold = 6
    PowerOn = 7
    PowerOff = 8
    Standby = 9
    Start = 10
    Suspend = 11
    UnHold = 12
    Unsuspend = 13


class EFLEXAgentEnvironmentException(Exception):
    """
    Related to errors with sending actions.
    """
    pass


class EFlexMultiAgentVersion2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # action space specify which transition can be activated
        self.action_space = []  # spaces.Discrete(len(EFLEXAgentTransition))  # {0,1,...,n-1}

        # observation is a multi discrete specifying which state is activated
        self.observation_space = []  # spaces.MultiBinary(len(EFLEXAgentState))

        # reward_range
        self.reward_range = (float(-1.0), float(1.0))

        self.display = None
        self.seed_value = None

        # Simulation related variables.
        self.agent_configs = {}
        self.agents = []
        self.n = 0
        self.shared_reward = False

        # Eflex related variables
        self.daily_slot_count = 1
        self.max_allowed_power = 0
        self.current_system_power = 0.0
        self.global_reward = 0.0
        self.simulation_step = 0
        self.daily_load = [0.0] * self.daily_slot_count
        self.daily_generated = [0.0] * self.daily_slot_count
        self.daily_sell = [0.0] * self.daily_slot_count
        self.daily_buy = [0.0] * self.daily_slot_count

    def configure(self, display=None, agent_config_path=None, shared_reward=False):
        self.shared_reward = shared_reward
        self.display = display
        if agent_config_path:
            self.agent_configs = self._load_config(agent_config_path)

        # Read the environment configurations
        env_config = self.agent_configs['environement_config']
        self.max_allowed_power = env_config['max_allowed_power']
        self.daily_slot_count = env_config['daily_slot_count']

        # initialize internal variables
        self.daily_load = [0.0] * self.daily_slot_count
        self.daily_generated = [0.0] * self.daily_slot_count
        self.daily_sell = [0.0] * self.daily_slot_count
        self.daily_buy = [0.0] * self.daily_slot_count

        # initialize and configure the agents
        for agent_name, agent_conf in self.agent_configs['agents'].items():
            current_agent = None
            # initialize the agents
            module_spec = self._import_module("gym_eflex_agent", agent_conf['type'])
            if module_spec is None:
                raise EFLEXAgentEnvironmentException("The environement  config contains an unknown agent type {}"
                                                     .format(agent_conf['type']))
            else:
                current_agent = module_spec(agent_conf, agent_name, self.max_allowed_power, self.daily_slot_count)
                self.agents.append(current_agent)

            # action space
            self.action_space.append(current_agent.action_space)

            # observation space
            self.observation_space.append(current_agent.observation_space)

        # Set environment parameters
        self.n = len(self.agents)
        self.seed(self.seed_value)
        self.reset()


    def _import_module(self, module_name, class_name):
        """Constructor"""

        macro_module = __import__(module_name)
        module0 = getattr(macro_module, 'envs')
        module = getattr(module0, 'EFlexMultiAgentV2')
        my_class = getattr(module, class_name)
        return my_class

    @staticmethod
    def _load_config(_trainer_config_path):
        try:
            with open(_trainer_config_path) as data_file:
                trainer_config = yaml.load(data_file, Loader=yaml.FullLoader)
                return trainer_config
        except IOError:
            raise EFLEXAgentEnvironmentException("""Parameter file could not be found here {}.
                                            Will use default Hyper parameters"""
                                                 .format(_trainer_config_path))
        except UnicodeDecodeError:
            raise EFLEXAgentEnvironmentException("There was an error decoding Trainer Config from this path : {}"
                                                 .format(_trainer_config_path))

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
        # increase the simulation step
        self.simulation_step += 1

        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        t = self.simulation_step % self.daily_slot_count
        # set action for each agent
        for i, agent in enumerate(self.agents):
            _ob, _state_reward, _eo, _info = agent.step(action[i])
            # record observation for each agent
            obs_n.append(_ob)
            reward_n.append(_state_reward)
            done_n.append(_eo)
            info_n['n'].append(_info)
            # Collect the load of all elements
            if agent is EFlexAgent:
                self.daily_load[t] += agent.current_power
            elif agent is EFLEXEnergyGeneratorAgent:
                self.daily_generated[t] += abs(agent.current_power)
            elif agent is EFLEXEnergyMainGridAgent:
                if agent.current_state == EFLEXEnergyMainGridAgentState.Buying:
                    self.daily_buy[t] += abs(agent.current_power)
                elif agent.current_state == EFLEXEnergyMainGridAgentState.Selling:
                    self.daily_sell[t] += abs(agent.current_power)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        # done
        done = (self.simulation_step % self.daily_slot_count == 0)

        self.current_system_power = self.daily_sell[t] + self.daily_generated[t] - (self.daily_load[t]
                                                                                    - self.daily_buy[t])
        if done:
            total_cost = np.sum(np.array(self.daily_sell)) + np.sum(np.array(self.daily_generated)) \
                         - np.sum(np.array(self.daily_load)) - np.sum(np.array(self.daily_buy))
            reward_n = [0.5 * (1 - math.tanh(total_cost ** 2))] * self.n

        # # Check if the total energy is smaller than the maximum allowed energy
        # self.current_system_power = np.sum(np.array(current_power_n))
        # if self.current_system_power > self.max_allowed_power:
        #     # set a maximum negative reward to all agents
        #     reward_n = [-0.5] * self.n
        #     done = True
        # # else:
        # #     # Add a smal reward to encourage energy savings
        # #     for i, agent in enumerate(self.agents):
        # #             reward_n[i] = reward_n[i] # + (0.1 * (1 - math.tanh(self.current_system_power ** 2)))

        self.global_reward = np.mean(reward_n)
        return obs_n, reward_n, done, info_n

    def reset(self):
        self.simulation_step = 0
        obs_n = []
        for i, agent in enumerate(self.agents):
            _ob = agent.reset()
            obs_n.append(_ob)
        return obs_n

    def render(self, mode='human', close=False):
        tmp = '\t|\t'.join('{:<10}\t- State: {:<10}\t- Reward: {:.2f}'.format(agent.name,
                                                                              agent.current_state.name,
                                                                              agent.current_reward) for
                           i, agent in enumerate(self.agents))
        print('POWER: {:>6}%\t|\t GR: {:.2f}\t|\t{}'.format(self.current_system_power, self.global_reward, tmp))

    def seed(self, seed=None):
        self.seed_value = seed
        for i, agent in enumerate(self.agents):
            agent.seed(seed)

    def close(self, seed=None):
        for i, agent in enumerate(self.agents):
            agent.close()

    def _is_episode_over(self):
        for i, agent in enumerate(self.agents):
            if agent._is_episode_over():
                return True
        ## TODO: Update the signal by considering the maximum load


class EFlexAgent:
    metadata = {'render.modes': ['human']}

    def __init__(self, agent_conf, name, max_allowed_power, daily_slot_count):
        # action space specify which transition can be activated
        self.action_space = spaces.Discrete(len(EFLEXAgentTransition))  # {0,1,...,n-1}

        # observation is a multi discrete specifying which state is activated
        self.observation_space = spaces.MultiBinary(len(EFLEXAgentState) + 2)

        # reward_range
        self.reward_range = (float(-1.0), float(1.0))

        # Simulation related variables.
        self.p_min = agent_conf['p_min']
        self.p_max = agent_conf['p_max']
        self.p_slope = agent_conf['p_slope']
        self.name = name
        self.current_state = None
        self.np_random = None
        self.current_reward = 0.0
        self.obs = None
        self.obs_pre = None
        self.max_allowed_power = max_allowed_power
        self.daily_slot_count = daily_slot_count

        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()
        self.startStep = 0
        self.currentStep = 0

    def _configure(self, display=None):
        self.display = display

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, state_reward, current_power, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            state_reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            current_power (float) :
                amount of energy power consumed or produced.
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
        act_enum = EFLEXAgentTransition(action)
        self._take_action(act_enum)
        next_state = self.current_state
        reward = self._get_reward()
        ob = self._get_obs()

        episode_over = self._get_done()
        self.currentStep = self.currentStep + 1
        return ob, reward, episode_over, {'info': '{} => {} => {}'.format(last_state, EFLEXAgentTransition(action)
                                                                          , next_state)}

    def reset(self):
        # self.current_state = EFLEXAgentState(randint(0, len(EFLEXAgentState)-1))
        # while self.current_state == EFLEXAgentState.LoadChange:
        #     self.current_state = EFLEXAgentState(randint(0, len(EFLEXAgentState) -1))
        self.current_state = EFLEXAgentState.Stopped
        self.current_reward = 0.0
        self.currentStep = 0
        self.startStep = self.currentStep
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self, seed=None):
        pass

    def _take_action(self, action):
        # Aborted
        if self.current_state == EFLEXAgentState.Aborted:
            self.startStep = self.currentStep
            if action == EFLEXAgentTransition.Clear:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 1.0
            else:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = 0.0
        # Stopped
        elif self.current_state == EFLEXAgentState.Stopped:
            self.startStep = self.currentStep
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXAgentTransition.Reset:
                self.current_state = EFLEXAgentState.Idle
                self.current_reward = 0.1
            else:
                self.current_state = EFLEXAgentState.Stopped
                self.current_reward = 0.0
        # Idle
        elif self.current_state == EFLEXAgentState.Idle:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXAgentTransition.Start:
                self.startStep = self.currentStep
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
            self.startStep = self.currentStep
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -0.1
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
            self.startStep = self.currentStep
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXAgentTransition.PowerOn:
                self.current_state = EFLEXAgentState.StartedUp
                self.current_reward = 0.1
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
                self.current_reward = -0.1
            elif action == EFLEXAgentTransition.Reset:
                self.current_state = EFLEXAgentState.Idle
                self.current_reward = 0.3
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
                self.current_reward = -0.1
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
        elif self.current_state == EFLEXAgentState.Completed:
            if action == EFLEXAgentTransition.Abort:
                self.current_state = EFLEXAgentState.Aborted
                self.current_reward = -0.1
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
                self.current_reward = -0.1
            elif action == EFLEXAgentTransition.UnHold:
                self.current_state = EFLEXAgentState.Execute
                self.current_reward = 0.1
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
                self.current_reward = -0.1
            elif action == EFLEXAgentTransition.Unsuspend:
                self.current_state = EFLEXAgentState.Execute
                self.current_reward = 0.1
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

    def _get_obs(self):
        obs = np.zeros(self.observation_space.n)
        obs[self.current_state.value] = 1.0
        obs[self.observation_space.n - 2] = self.currentStep % self.daily_slot_count
        obs[self.observation_space.n - 1] = self.current_power / self.max_allowed_power

        return obs

    def _get_done(self):
        return self.current_state == EFLEXAgentState.Aborted

    def _get_reward(self):
        return self.current_reward

    @property
    def current_power(self):
        raise NotImplementedError("Please Implement this method")


class EFlexAgentPConstant(EFlexAgent):
    metadata = {'render.modes': ['human']}

    @property
    def current_power(self):
        if self.current_state == EFLEXAgentState.Aborted or self.current_state == EFLEXAgentState.Stopped:
            return 0.0
        elif self.current_state == EFLEXAgentState.Execute or self.current_state == EFLEXAgentState.Completed:
            return self.p_max
        else:
            return self.p_min


class EFlexAgentPLinear(EFlexAgent):
    metadata = {'render.modes': ['human']}

    @property
    def current_power(self):
        if self.current_state == EFLEXAgentState.Aborted or self.current_state == EFLEXAgentState.Stopped:
            return 0.0
        elif self.current_state == EFLEXAgentState.Execute or self.current_state == EFLEXAgentState.Completed:
            delta = self.currentStep - self.startStep
            c_power = self.p_min + delta * self.p_slope
            if c_power > self.p_max:
                return self.p_max
            else:
                return c_power
        else:
            return self.p_min


class EFlexAgentPLogistic(EFlexAgent):
    metadata = {'render.modes': ['human']}

    @staticmethod
    def sigmoid(x, k, l):
        return l / (1 + math.exp(- l * x))

    @property
    def current_power(self):

        if self.current_state == EFLEXAgentState.Aborted or self.current_state == EFLEXAgentState.Stopped:
            return 0.0
        elif self.current_state == EFLEXAgentState.Execute or self.current_state == EFLEXAgentState.Completed:
            delta = self.currentStep - self.startStep
            c_power = self.p_min + self.sigmoid(delta, self.p_slope, self.p_max)
            if c_power > self.p_max:
                return self.p_max
            else:
                return c_power
        else:
            return self.p_min


class EFLEXEnergyStorageAgentState(Enum):
    Aborted = 1
    Stopped = 0
    Charging = 2
    Discharging = 3


class EFLEXEnergyStorageAgentTransition(Enum):
    SC = 0
    Abort = 1
    Clear = 2
    Charge = 3
    Stop = 4
    Discharge = 5


class EFlexEnergyStorageAgent:
    metadata = {'render.modes': ['human']}

    def __init__(self, agent_conf, name, max_allowed_power, daily_slot_count):
        # action space specify which transition can be activated
        self.action_space = spaces.Discrete(len(EFLEXEnergyStorageAgentTransition))  # {0,1,...,n-1}

        # observation is a multi discrete specifying which state is activated
        self.observation_space = spaces.MultiBinary(len(EFLEXEnergyStorageAgentState) + 2)

        # reward_range
        self.reward_range = (float(-1.0), float(1.0))

        # Simulation related variables.
        self.p_min = agent_conf['p_min']
        self.p_max = agent_conf['p_max']
        self.p_slope = agent_conf['p_slope']
        self.name = name
        self.current_state = None
        self.np_random = None
        self.current_reward = 0.0
        self.obs = None
        self.obs_pre = None
        self.max_allowed_power = max_allowed_power
        self.daily_slot_count = daily_slot_count

        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()
        self.startStep = 0
        self.currentStep = 0
        self.charging_level = 0

    def _configure(self, display=None):
        self.display = display

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, state_reward, current_power, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            state_reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            current_power (float) :
                amount of energy power consumed or produced.
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
        act_enum = EFLEXEnergyStorageAgentTransition(action)
        self._take_action(act_enum)
        next_state = self.current_state
        reward = self._get_reward()
        ob = self._get_obs()

        episode_over = self._get_done()
        self.currentStep = self.currentStep + 1
        return ob, reward, episode_over, {'info': '{} => {} => {}'.format(last_state,
                                                                          EFLEXEnergyStorageAgentTransition(action)
                                                                          , next_state)}

    def reset(self):
        # self.current_state = EFLEXAgentState(randint(0, len(EFLEXAgentState)-1))
        # while self.current_state == EFLEXAgentState.LoadChange:
        #     self.current_state = EFLEXAgentState(randint(0, len(EFLEXAgentState) -1))
        self.current_state = EFLEXEnergyStorageAgentState.Stopped
        self.current_reward = 0.0
        self.currentStep = 0
        self.startStep = self.currentStep
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self, seed=None):
        pass

    def _take_action(self, action):
        # Aborted
        if self.current_state == EFLEXEnergyStorageAgentState.Aborted:
            self.startStep = self.currentStep
            if action == EFLEXEnergyStorageAgentTransition.Clear:
                self.current_state = EFLEXEnergyStorageAgentState.Stopped
                self.current_reward = 1.0
            else:
                self.current_state = EFLEXEnergyStorageAgentState.Aborted
                self.current_reward = 0.0
        # Stopped
        elif self.current_state == EFLEXEnergyStorageAgentState.Stopped:
            self.startStep = self.currentStep
            if action == EFLEXEnergyStorageAgentTransition.Abort:
                self.current_state = EFLEXEnergyStorageAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXEnergyStorageAgentTransition.Charge:
                self.current_state = EFLEXEnergyStorageAgentState.Charging
                self.current_reward = 0.1
            elif action == EFLEXEnergyStorageAgentTransition.Discharge:
                self.current_state = EFLEXEnergyStorageAgentState.Discharging
                self.current_reward = 0.1
            else:
                self.current_state = EFLEXEnergyStorageAgentState.Stopped
                self.current_reward = 0.0
        # Charging
        elif self.current_state == EFLEXEnergyStorageAgentState.Charging:
            if action == EFLEXEnergyStorageAgentTransition.Abort:
                self.current_state = EFLEXEnergyStorageAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXEnergyStorageAgentTransition.Stop:
                self.current_state = EFLEXEnergyStorageAgentState.Stopped
                self.current_reward = 0.0
            elif action == EFLEXEnergyStorageAgentTransition.Discharge:
                self.current_state = EFLEXEnergyStorageAgentState.Discharging
                val = 1 - ((self.p_max - self.charging_level) * +2 / (self.p_max ** 2))
                self.current_reward = val
            else:
                self.current_state = EFLEXEnergyStorageAgentState.Charging
                val = 1 - ((self.p_max - self.charging_level) * +2 / (self.p_max ** 2))
                self.current_reward = val

            self.charging_level = self.charging_level + self.p_slope
            if self.charging_level > self.p_max:
                self.charging_level = self.p_max
        # Discharging
        elif self.current_state == EFLEXEnergyStorageAgentState.Discharging:
            if action == EFLEXEnergyStorageAgentTransition.Abort:
                self.current_state = EFLEXEnergyStorageAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXEnergyStorageAgentTransition.Stop:
                self.current_state = EFLEXEnergyStorageAgentState.Stopped
                self.current_reward = 0.0
            elif action == EFLEXEnergyStorageAgentTransition.Charge:
                self.current_state = EFLEXEnergyStorageAgentState.Charging
                val = 1 - ((self.p_max - self.charging_level) ** 2 / (self.p_max ** 2))
                self.current_reward = val
            else:
                self.current_state = EFLEXEnergyStorageAgentState.Discharging
                val = 1 - ((self.p_max - self.charging_level) ** 2 / (self.p_max ** 2))
                self.current_reward = val

            self.charging_level = self.charging_level - self.p_slope
            if self.charging_level < self.p_min:
                self.charging_level = self.p_min

    def _get_obs(self):
        obs = np.zeros(self.observation_space.n)
        obs[self.current_state.value] = 1.0
        obs[self.observation_space.n - 2] = self.currentStep % self.daily_slot_count
        obs[self.observation_space.n - 1] = self.current_power / self.max_allowed_power

        return obs

    def _get_done(self):
        return self.current_state == EFLEXEnergyStorageAgentState.Aborted

    def _get_reward(self):
        return self.current_reward

    @property
    def current_power(self):
        if self.current_state == EFLEXEnergyStorageAgentState.Aborted or \
                self.current_state == EFLEXEnergyStorageAgentState.Stopped:
            return 0.0
        elif self.current_state == EFLEXEnergyStorageAgentState.Charging:
            if self.charging_level < self.p_max:
                return (self.p_max - self.p_min) / 2
            else:
                return 0.0
        elif self.current_state == EFLEXEnergyStorageAgentState.Discharging:
            if self.charging_level > self.p_min:
                return - (self.p_max - self.p_min) / 2
            else:
                return 0.0
        else:
            return 0


class EFLEXEnergyGeneratorAgentState(Enum):
    Aborted = 1
    Stopped = 0
    Generating = 2


class EFLEXEnergyGeneratorAgentTransition(Enum):
    SC = 0
    Abort = 1
    Clear = 2
    Generate = 3
    Stop = 4


class EFLEXEnergyGeneratorAgent:
    metadata = {'render.modes': ['human']}

    def __init__(self, agent_conf, name, max_allowed_power, daily_slot_count):
        # action space specify which transition can be activated
        self.action_space = spaces.Discrete(len(EFLEXEnergyGeneratorAgentTransition))  # {0,1,...,n-1}

        # observation is a multi discrete specifying which state is activated
        self.observation_space = spaces.MultiBinary(len(EFLEXEnergyGeneratorAgentState) + 2)

        # reward_range
        self.reward_range = (float(-1.0), float(1.0))

        # Simulation related variables.
        self.p_min = agent_conf['p_min']
        self.p_max = agent_conf['p_max']
        self.p_slope = agent_conf['p_slope']
        self.name = name
        self.current_state = None
        self.np_random = None
        self.current_reward = 0.0
        self.obs = None
        self.obs_pre = None
        self.max_allowed_power = max_allowed_power
        self.daily_slot_count = daily_slot_count

        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()
        self.startStep = 0
        self.currentStep = 0
        self.last_power = 0

    def _configure(self, display=None):
        self.display = display

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, state_reward, current_power, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            state_reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            current_power (float) :
                amount of energy power consumed or produced.
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
        act_enum = EFLEXEnergyGeneratorAgentTransition(action)
        self._take_action(act_enum)
        next_state = self.current_state
        reward = self._get_reward()
        ob = self._get_obs()

        episode_over = self._get_done()
        self.currentStep = self.currentStep + 1
        self.last_power = abs(self.current_power)
        return ob, reward, episode_over, {'info': '{} => {} => {}'.format(last_state,
                                                                          EFLEXEnergyGeneratorAgentTransition(action)
                                                                          , next_state)}

    def reset(self):
        # self.current_state = EFLEXAgentState(randint(0, len(EFLEXAgentState)-1))
        # while self.current_state == EFLEXAgentState.LoadChange:
        #     self.current_state = EFLEXAgentState(randint(0, len(EFLEXAgentState) -1))
        self.current_state = EFLEXEnergyGeneratorAgentState.Stopped
        self.current_reward = 0.0
        self.currentStep = 0
        self.startStep = self.currentStep
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self, seed=None):
        pass

    def _take_action(self, action):
        # Aborted
        if self.current_state == EFLEXEnergyGeneratorAgentState.Aborted:
            self.startStep = self.currentStep
            if action == EFLEXEnergyGeneratorAgentTransition.Clear:
                self.current_state = EFLEXEnergyGeneratorAgentState.Stopped
                self.current_reward = 1.0
            else:
                self.current_state = EFLEXEnergyGeneratorAgentState.Aborted
                self.current_reward = 0.0
        # Stopped
        elif self.current_state == EFLEXEnergyGeneratorAgentState.Stopped:
            self.startStep = self.currentStep
            if action == EFLEXEnergyGeneratorAgentTransition.Abort:
                self.current_state = EFLEXEnergyGeneratorAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXEnergyGeneratorAgentTransition.Generate:
                self.current_state = EFLEXEnergyGeneratorAgentState.Generating
                self.current_reward = 0.1
            else:
                self.current_state = EFLEXEnergyGeneratorAgentState.Stopped
                self.current_reward = 0.0
        # Generating
        elif self.current_state == EFLEXEnergyGeneratorAgentState.Generating:
            if action == EFLEXEnergyGeneratorAgentTransition.Abort:
                self.current_state = EFLEXEnergyGeneratorAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXEnergyGeneratorAgentTransition.Stop:
                self.current_state = EFLEXEnergyGeneratorAgentState.Stopped
                self.current_reward = 0.0
            else:
                self.current_state = EFLEXEnergyGeneratorAgentState.Generating
                self.current_reward = 0.2 * abs(self.current_power / self.p_max)

    def _get_obs(self):
        obs = np.zeros(self.observation_space.n)
        obs[self.current_state.value] = 1.0
        obs[self.observation_space.n - 2] = self.currentStep % self.daily_slot_count
        obs[self.observation_space.n - 1] = self.current_power / self.max_allowed_power

        return obs

    def _get_done(self):
        return self.current_state == EFLEXEnergyGeneratorAgentState.Aborted

    def _get_reward(self):
        return self.current_reward

    @property
    def current_power(self):
        if self.current_state == EFLEXEnergyGeneratorAgentState.Aborted or \
                self.current_state == EFLEXEnergyGeneratorAgentState.Stopped:
            return 0.0
        elif self.current_state == EFLEXEnergyGeneratorAgentState.Generating:
            c_power = self.last_power + self.p_slope
            if c_power > self.p_max:
                return - self.p_max
            else:
                return - c_power
        else:
            return 0


class EFLEXEnergyMainGridAgentState(Enum):
    Aborted = 0
    Stopped = 1
    Buying = 2
    Selling = 4


class EFLEXEnergyMainGridAgentTransition(Enum):
    SC = 0
    Abort = 1
    Clear = 2
    Stop = 3
    Buy = 4
    Sell = 5


class EFLEXEnergyMainGridAgent:
    metadata = {'render.modes': ['human']}

    def __init__(self, agent_conf, name, max_allowed_power, daily_slot_count):
        # action space specify which transition can be activated
        self.action_space = spaces.Discrete(len(EFLEXEnergyMainGridAgentTransition))  # {0,1,...,n-1}

        # observation is a multi discrete specifying which state is activated
        self.observation_space = spaces.MultiBinary(len(EFLEXEnergyMainGridAgentState) + 2)

        # reward_range
        self.reward_range = (float(-1.0), float(1.0))

        # Simulation related variables.
        self.p_min = agent_conf['p_min']
        self.p_max = agent_conf['p_max']
        self.p_slope = agent_conf['p_slope']
        self.name = name
        self.current_state = None
        self.np_random = None
        self.current_reward = 0.0
        self.obs = None
        self.obs_pre = None
        self.max_allowed_power = max_allowed_power
        self.daily_slot_count = daily_slot_count

        self.seed()
        self.reset()

        # Just need to initialize the relevant attributes
        self._configure()
        self.startStep = 0
        self.currentStep = 0
        self.last_power = 0

    def _configure(self, display=None):
        self.display = display

    def step(self, action):
        """

        Parameters
        ----------
        action : action to be performed

        Returns
        -------
        ob, state_reward, current_power, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            state_reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            current_power (float) :
                amount of energy power consumed or produced.
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
        act_enum = EFLEXEnergyMainGridAgentTransition(action)
        self._take_action(act_enum)
        next_state = self.current_state
        reward = self._get_reward()
        ob = self._get_obs()

        episode_over = self._get_done()
        self.currentStep = self.currentStep + 1
        self.last_power = self.current_power
        return ob, reward, episode_over, {'info': '{} => {} => {}'.format(last_state,
                                                                          EFLEXEnergyMainGridAgentTransition(action)
                                                                          , next_state)}

    def reset(self):
        self.current_state = EFLEXEnergyMainGridAgentState.Stopped
        self.current_reward = 0.0
        self.currentStep = 0
        self.startStep = self.currentStep
        return self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self, seed=None):
        pass

    def _take_action(self, action):
        # Aborted
        if self.current_state == EFLEXEnergyMainGridAgentState.Aborted:
            self.startStep = self.currentStep
            if action == EFLEXEnergyMainGridAgentTransition.Clear:
                # self.current_state = EFLEXEnergyMainGridAgentState.Stopped
                # self.current_reward = 1.0
                pass
            else:
                self.current_state = EFLEXEnergyMainGridAgentState.Aborted
                self.current_reward = 0.0
        # Stopped
        elif self.current_state == EFLEXEnergyMainGridAgentState.Stopped:
            self.startStep = self.currentStep
            if action == EFLEXEnergyMainGridAgentTransition.Abort:
                self.current_state = EFLEXEnergyMainGridAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXEnergyMainGridAgentTransition.Buy:
                self.current_state = EFLEXEnergyMainGridAgentState.Buying
                self.current_reward = 0.01
            elif action == EFLEXEnergyMainGridAgentTransition.Sell:
                self.current_state = EFLEXEnergyMainGridAgentState.Selling
                self.current_reward = 0.01
            else:
                self.current_state = EFLEXEnergyMainGridAgentState.Stopped
                self.current_reward = 0.0
        # Buying
        elif self.current_state == EFLEXEnergyMainGridAgentState.Buying:
            if action == EFLEXEnergyMainGridAgentTransition.Abort:
                self.current_state = EFLEXEnergyMainGridAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXEnergyMainGridAgentTransition.Stop:
                self.current_state = EFLEXEnergyMainGridAgentState.Stopped
                self.current_reward = -0.01
            elif action == EFLEXEnergyMainGridAgentTransition.Sell:
                self.current_state = EFLEXEnergyMainGridAgentState.Selling
                self.current_reward = 0.0
            else:
                self.current_state = self.current_state = EFLEXEnergyMainGridAgentState.Buying
                self.current_reward = -0.01
        # Selling
        elif self.current_state == EFLEXEnergyMainGridAgentState.Selling:
            if action == EFLEXEnergyMainGridAgentTransition.Abort:
                self.current_state = EFLEXEnergyMainGridAgentState.Aborted
                self.current_reward = -0.1
            elif action == EFLEXEnergyMainGridAgentTransition.Stop:
                self.current_state = EFLEXEnergyMainGridAgentState.Stopped
                self.current_reward = -0.01
            elif action == EFLEXEnergyMainGridAgentTransition.Buy:
                self.current_state = EFLEXEnergyMainGridAgentState.Buying
                self.current_reward = 0.0
            else:
                self.current_state = self.current_state = EFLEXEnergyMainGridAgentState.Selling
                self.current_reward = 0.01

    def _get_obs(self):
        obs = np.zeros(self.observation_space.n)
        obs[self.current_state.value] = 1.0
        obs[self.observation_space.n - 2] = self.currentStep % self.daily_slot_count
        obs[self.observation_space.n - 1] = self.current_power / self.max_allowed_power

        return obs

    def _get_done(self):
        return self.current_state == EFLEXEnergyMainGridAgentState.Aborted

    def _get_reward(self):
        return self.current_reward

    @property
    def current_power(self):
        if self.current_state == EFLEXEnergyMainGridAgentState.Aborted or \
                self.current_state == EFLEXEnergyMainGridAgentState.Stopped:
            return 0.0
        elif self.current_state == EFLEXEnergyMainGridAgentState.Buying:
            return - self.p_max
        elif self.current_state == EFLEXEnergyMainGridAgentState.Selling:
            return self.p_max
        else:
            return 0
