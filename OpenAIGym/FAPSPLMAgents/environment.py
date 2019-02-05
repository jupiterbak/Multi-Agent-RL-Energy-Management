import atexit
import io
import logging
import numpy as np
import zmq
import base64

from .brain import BrainInfo, BrainParameters
from .exception import FAPSPLMEnvironmentException, FAPSPLMActionException

from PIL import Image
from sys import platform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAPSPLMEnvironment(object):
    def __init__(self, worker_id=0,
                 base_port=6005):
        """
        Starts a new Faps environment and establishes a connection with the environment.
        Notice: Currently communication between Faps and Python takes place over an open socket without authentication.
        Ensure that the network where training takes place is secure.

        :string file_name: Name of Faps environment binary.
        :int base_port: Baseline port number to connect to Faps environment over. worker_id increments over this.
        :int worker_id: Number to add to communication port (5005) [0]. Used for asynchronous agent scenarios.
        """

        atexit.register(self.close)
        self.port = base_port + worker_id
        self._buffer_size = 120000
        self._loaded = False
        self._open_socket = False

        try:
            try:
                # Establish communication socket
                self._context = zmq.Context()
                self._socket = self._context.socket(zmq.PAIR)
                self._socket.bind("tcp://*:" + str(self.port))
                self._open_socket = True
                p = self._socket.recv_json()
                self._socket.send_json({"cmd": "CONFIRM"})
            except Exception as err:
                print("OS error: {0}".format(err))
                self._open_socket = True
                self.close()
                raise FAPSPLMEnvironmentException(
                    "Couldn't launch new environment because worker number {} is still in use. "
                    "Yo  u may need to manually close a previously opened environment "
                    "or use a different worker number.".format(str(worker_id)))
        except FAPSPLMEnvironmentException:
            self.close()
            raise

        self._data = {}
        self._global_done = None
        self._academy_name = p["AcademyName"]
        self._num_brains = len(p["brainParameters"])
        self._brains = {}
        self._brain_names = p["brainNames"]
        self._resetParameters = p["resetParameters"]
        for i in range(self._num_brains):
            self._brains[self._brain_names[i]] = BrainParameters(self._brain_names[i], p["brainParameters"][i])
        self._socket.send_json({"cmd": "NONE", "state": "initialized"})
        self._loaded = True
        logger.info("\n'{}' started successfully!".format(self._academy_name))

    @property
    def brains(self):
        return self._brains

    @property
    def global_done(self):
        return self._global_done

    @property
    def academy_name(self):
        return self._academy_name

    @property
    def number_brains(self):
        return self._num_brains

    @property
    def brain_names(self):
        return self._brain_names

    @staticmethod
    def _process_pixels(image_bytes=None, bw=False):
        """
        Converts bytearray observation image into numpy array, resizes it, and optionally converts it to greyscale
        :param image_bytes: input bytearray corresponding to image
        :return: processed numpy array of observation from environment
        """
        s = bytearray(image_bytes)
        image = Image.open(io.BytesIO(s))
        s = np.array(image) / 255.0
        if bw:
            s = np.mean(s, axis=2)
            s = np.reshape(s, [s.shape[0], s.shape[1], 1])
        return s

    def __str__(self):
        return '''FAPS PLM Academy name: {0}
        Number of brains: {1}
        Reset Parameters :\n\t\t{2}'''.format(self._academy_name, str(self._num_brains),
                                              "\n\t\t".join([str(self._brains[b]) for b in self._brains]))

    def _get_state_image(self, bw):
        """
        Receives observation from socket, and confirms.
        :param bw:
        :return:
        """
        # s = self._socket.recv(self._buffer_size)
        message = self._socket.recv()
        s = bytearray(base64.b64decode(message))
        s = self._process_pixels(image_bytes=s, bw=bw)
        self._socket.send_json({"cmd": "CONFIRM"})
        return s

    def _get_state_dict(self):
        """
        Receives dictionary of state information from socket, and confirms.
        :return:
        """
        state_dict = self._socket.recv_json()
        self._socket.send_json({"cmd": "CONFIRM"})
        return state_dict

    def reset(self, train_mode=True, config=None):
        """
        Sends a signal to reset the Faps environment.
        :return: A Data structure corresponding to the initial reset state of the environment.
        """
        config = config or {}
        if self._loaded:
            self._socket.send_json({"cmd": "RESET", "train_model": train_mode, "parameters": config})
            confirm_reset = self._socket.recv_json()
            for k in config:
                if (k in self._resetParameters) and (isinstance(config[k], (int, float))):
                    self._resetParameters[k] = config[k]
                elif not isinstance(config[k], (int, float)):
                    raise FAPSPLMEnvironmentException(
                        "The value for parameter '{0}'' must be an Integer or a Float.".format(k))
                else:
                    raise FAPSPLMEnvironmentException("The parameter '{0}' is not a valid parameter.".format(k))
            self._global_done = False
            return self._get_state()
        else:
            raise FAPSPLMEnvironmentException("No Simulation environment is loaded.")

    def _get_state(self):
        """
        Collects experience information from all external brains in environment at current step.
        :return: a dictionary BrainInfo objects.
        """
        self._data = {}
        self._socket.send_json({"cmd": "GET_STATE"})
        confirm_getstate = self._socket.recv_json()
        for index in range(self._num_brains):
            state_dict = self._get_state_dict()
            b = state_dict["brain_name"]
            n_agent = len(state_dict["agents"])
            try:
                if self._brains[b].state_space_type == "continuous":
                    states = np.array(state_dict["states"]).reshape((n_agent, self._brains[b].state_space_size))
                else:
                    states = np.array(state_dict["states"]).reshape((n_agent, self._brains[b].state_space_size))
            except FAPSPLMActionException:
                raise FAPSPLMActionException("Brain {0} has an invalid state. "
                                             "Expecting {1} {2} state but received {3}."
                                             .format(b, str(self._brains[b].state_space_size * n_agent),
                                                     self._brains[b].state_space_type,
                                                     len(state_dict["states"])))
            memories = np.array(state_dict["memories"]).reshape((n_agent, self._brains[b].memory_space_size))
            rewards = state_dict["rewards"]
            dones = state_dict["dones"]
            agents = state_dict["agents"]

            observations = []
            for o in range(self._brains[b].number_observations):
                obs_n = []
                for a in range(n_agent):
                    obs_n.append(self._get_state_image(self._brains[b].camera_resolutions[o]['blackAndWhite']))

                observations.append(np.array(obs_n))

            self._data[b] = BrainInfo(observations, states, memories, rewards, agents, dones)

        confirm_get_state = self._socket.recv_json()
        self._global_done = confirm_get_state["GLOBAL_DONE"] == 1
        return self._data

    def _send_action(self, action, memory, value):
        """
        Send dictionary of actions, memories, and value estimates over socket.
        :param action: a dictionary of lists of actions.
        :param memory: a dictionary of lists of of memories.
        :param value: a dictionary of lists of of value estimates.
        """
        action_message = {"action": action, "memory": memory, "value": value}
        self._socket.send_json(action_message)
        confirm_receive_step_action = self._socket.recv_json()

    @staticmethod
    def _flatten(arr):
        """
        Converts dictionary of arrays to list for transmission over socket.
        :param arr: numpy vector.
        :return: flattened list.
        """
        if isinstance(arr, (int, np.int_, float, np.float_)):
            arr = [float(arr)]
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        if len(arr) == 0:
            return arr
        if isinstance(arr[0], np.ndarray):
            arr = [item for sublist in arr for item in sublist.tolist()]
        if isinstance(arr[0], list):
            arr = [item for sublist in arr for item in sublist]
        arr = [float(x) for x in arr]
        return arr

    def step(self, action, memory=None, value=None):
        """
        Provides the environment with an action, moves the environment dynamics forward accordingly, and returns
        observation, state, and reward information to the agent.
        :param action: Agent's action to send to environment. Can be a scalar or vector of int/floats.
        :param memory: Vector corresponding to memory used for RNNs, frame-stacking, or other auto-regressive process.
        :param value: Value estimate to send to environment for visualization. Can be a scalar or vector of float(s).
        :return: A Data structure corresponding to the new state of the environment.
        """
        memory = {} if memory is None else memory
        value = {} if value is None else value
        # if self._loaded and not self._global_done and self._global_done is not None:
        if self._loaded:
            if isinstance(action, (int, np.int_, float, np.float_, list, np.ndarray)):
                if self._num_brains > 1:
                    raise FAPSPLMActionException(
                        "You have {0} brains, you need to feed a dictionary of brain names a keys, "
                        "and actions as values".format(self._num_brains))
                else:
                    action = {self._brain_names[0]: action}
            if isinstance(memory, (int, np.int_, float, np.float_, list, np.ndarray)):
                if self._num_brains > 1:
                    raise FAPSPLMActionException(
                        "You have {0} brains, you need to feed a dictionary of brain names as keys "
                        "and memories as values".format(self._num_brains))
                else:
                    memory = {self._brain_names[0]: memory}
            if isinstance(value, (int, np.int_, float, np.float_, list, np.ndarray)):
                if self._num_brains > 1:
                    raise FAPSPLMActionException(
                        "You have {0} brains, you need to feed a dictionary of brain names as keys "
                        "and state/action value estimates as values".format(self._num_brains))
                else:
                    value = {self._brain_names[0]: value}

            for b in self._brain_names:
                n_agent = len(self._data[b].agents)
                if b not in action:
                    raise FAPSPLMActionException("You need to input an action for the brain {0}".format(b))
                action[b] = self._flatten(action[b])
                if b not in memory:
                    memory[b] = [0.0] * self._brains[b].memory_space_size * n_agent
                else:
                    memory[b] = self._flatten(memory[b])
                if b not in value:
                    value[b] = [0.0] * n_agent
                else:
                    value[b] = self._flatten(value[b])
                if not (len(value[b]) == n_agent):
                    raise FAPSPLMActionException(
                        "There was a mismatch between the provided value and environment's expectation: "
                        "The brain {0} expected {1} value but was given {2}".format(b, n_agent, len(value[b])))
                if not (len(memory[b]) == self._brains[b].memory_space_size * n_agent):
                    raise FAPSPLMActionException(
                        "There was a mismatch between the provided memory and environment's expectation: "
                        "The brain {0} expected {1} memories but was given {2}"
                            .format(b, self._brains[b].memory_space_size * n_agent, len(memory[b])))
                if not ((self._brains[b].action_space_type == "discrete" and
                         len(action[b]) == self._brains[b].action_space_size * n_agent) or
                        (self._brains[b].action_space_type == "continuous" and
                         len(action[b]) == self._brains[b].action_space_size * n_agent)):
                    raise FAPSPLMActionException(
                        "There was a mismatch between the provided action and environment's expectation: "
                        "The brain {0} expected {1} {2} action(s), but was provided: {3}"
                            .format(b, str(self._brains[b].action_space_size * n_agent),
                                    self._brains[b].action_space_type,
                                    str(action[b])))

            self._socket.send_json({"cmd": "STEP"})
            confirm_step_action = self._socket.recv_json()
            self._send_action(action, memory, value)
            return self._get_state()
        elif not self._loaded:
            raise FAPSPLMEnvironmentException("No Simulation environment is loaded.")
        # elif self._global_done:
        #     raise FAPSPLMActionException("The episode is completed. Reset the simulation environment with 'reset()'")
        elif self.global_done is None:
            raise FAPSPLMActionException(
                "You cannot conduct step without first calling reset. Reset the simulation environment with 'reset()'")

    def close(self):
        """
        Sends a shutdown signal to the Faps environment, and closes the socket connection.
        """
        if self._loaded & self._open_socket:
            self._socket.send_json({"cmd": "EXIT"})
            confirm_step_action = self._socket.recv_json()
        if self._open_socket:
            self._socket.close()
            self._context.term()
            self._loaded = False
        else:
            raise FAPSPLMEnvironmentException("No Simulation environment is loaded.")
