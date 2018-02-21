import logging
import gym
import atexit
import numpy as np

from .brain import BrainInfo, BrainParameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unityagents")


class GymEnvironment(object):
    def __init__(self, env_name, log_path, render=False, skip_frames=1, record=False):
        atexit.register(self.close)
        self._academy_name = "Gym Environment"
        self._current_returns = {}
        self._last_action = []
        self.render = render
        self.env = gym.make(env_name)

        if skip_frames < 0 or not isinstance(skip_frames, int):
            logger.error("Invalid frame skip value. Frame skip deactivated.")
        elif skip_frames > 1:
            frameskip_wrapper = gym.wrappers.SkipWrapper(skip_frames)
            self.env = frameskip_wrapper(self.env)

        if record:
            self.env = gym.wrappers.Monitor(self.env, "./video", lambda x: True)

        ob_space = self.env.observation_space
        ac_space = self.env.action_space
        if isinstance(ac_space, gym.spaces.Box):
            assert len(ac_space.shape) == 1
            self.ac_space_type = "continuous"
            self.ac_space_size = ac_space.shape[0]
        elif isinstance(ac_space, gym.spaces.Discrete):
            self.ac_space_type = "discrete"
            self.ac_space_size = ac_space.n

        else:
            raise NotImplementedError

        if isinstance(ob_space, gym.spaces.Box):
            assert len(ob_space.shape) == 1
            self.ob_space_type = "continuous"
            self.ob_space_size = ob_space.shape[0]

        elif isinstance(ob_space, gym.spaces.Discrete):
            self.ob_space_type = "discrete"
            self.ob_space_size = ob_space.n
        else:
            raise NotImplementedError

        self._data = {}
        self._log_path = log_path
        self._global_done = False
        self._brains = {}
        self._brain_names = ["FirstBrain"]
        self._external_brain_names = self._brain_names
        self._parameters = {"stateSize": self.ob_space_size, "actionSize": self.ac_space_size,
                            "actionSpaceType": self.ac_space_type, "stateSpaceType": self.ob_space_type}
        self._brains[self._brain_names[0]] = BrainParameters(self._brain_names[0], self._parameters)

        self._loaded = True
        logger.info("Environment started successfully!")

    def __str__(self):
        return '''Academy name: {0}
        Actions:
        \tSize: {1},\tType: {2}
        States:
        \tSize: {3},\tType: {4}'''.format(self._academy_name,
                                          self.ac_space_size, self.ac_space_type,
                                          self.ob_space_size, self.ob_space_type)

    def _state_to_info(self):
        state = np.array(self._current_returns[self._brain_names[0]][0])

        if self.ob_space_type == "continuous":
            states = state.reshape((1, self.ob_space_size))
        else:
            states = state.reshape((1, 1))

        memories = []
        rewards = [self._current_returns[self._brain_names[0]][1]]
        agents = [self._brain_names[0]]
        dones = [self._current_returns[self._brain_names[0]][2]]
        actions = self._last_action

        self._data[self._brain_names[0]] = BrainInfo(states, memories, rewards, agents, dones, actions)

        return self._data

    def reset(self):
        """
        Sends a signal to reset the unity environment.
        :return: A Data structure corresponding to the initial reset state of the environment.
        """
        obs = self.env.reset()
        self._current_returns = {self._brain_names[0]: [obs, 0, False]}
        self._last_action = [0] * self.ac_space_size if self.ac_space_type == 'continuous' else 0
        self._global_done = False

        return self._state_to_info()

    def step(self, action=None):
        """
        Provides the environment with an action, moves the environment dynamics forward accordingly, and returns
        observation, state, and reward information to the agent.
        :param action: Agent's action to send to environment. Can be a scalar or vector of int/floats.
        :return: A Data structure corresponding to the new state of the environment.
        """
        action = {} if action is None else action
        if self._loaded and not self._global_done and self._global_done is not None:
            obs, rew, done, _ = self.env.step(action[0])
            if done:
                self._global_done = True
            self._current_returns[self._brain_names[0]] = [obs, rew, done]
            self._last_action = action
            if self.render:
                self.env.render()

            return self._state_to_info()

        elif not self._loaded:
            print("No Gym environment is loaded.")
        elif self._global_done:
            print("The episode is completed. Reset the environment with 'reset()'")
        elif self.global_done is None:
            print("You cannot conduct step without first calling reset. Reset the environment with 'reset()'")

    def close(self):
        """
        Sends a shutdown signal to the gym environment.
        """
        self.env.close()

    @property
    def logfile_path(self):
        return self._log_path

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
    def brain_names(self):
        return self._brain_names

    @property
    def external_brain_names(self):
        return self._external_brain_names
