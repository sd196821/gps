import copy
import numpy as np

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_OWN
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample
try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:  # user does not have tf installed.
    TfPolicy = None


class AgentOwn(Agent):
    """
    For own agent.
    """
    def __init__(self, hyperparams):
        config = copy.deepcopy(AGENT_OWN)
        config.update(hyperparams)
        Agent.__init__(self, config)

        self._setup_conditions()

        self._setup_world(self._hyperparams["world"],
                          self._hyperparams["target_state"],
                          self._hyperparams["render"])

    def _setup_conditions(self):
        """
        Helper method for setting some hyperparameters that may vary by
        condition.
        """
        conds = self._hyperparams['conditions']
        for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                      'noisy_body_idx', 'noisy_body_var'):
            self._hyperparams[field] = setup(self._hyperparams[field], conds)

    def _setup_world(self, world, target, render):
        """
        Helper method for handling setup of the Box2D world.
        """
        self.x0 = self._hyperparams["x0"]
        self._worlds = [world(self.x0[i], target, render)
                        for i in range(self._hyperparams['conditions'])]

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        Args:
            policy: Policy to to used in the trial.
            condition: Which condition setup to run.
            verbose: Whether or not to plot the trial.
            save: Whether or not to store the trial into the samples.
            noisy: Whether or not to use noise during sampling.
        """

        # if TfPolicy is not None:  # user has tf installed.
        #     if isinstance(policy, TfPolicy):
        #         self._init_tf(policy.dU)

        self._worlds[condition].reset()

        own_X = self._hyperparams['x0'][condition]
        new_sample = self._init_sample(condition)
        U = np.zeros([self.T, self.dU])

        # Generate noise.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            if (t + 1) < self.T:
                for _ in range(self._hyperparams['substeps']):
                    self._worlds[condition].step(U[t, :])
                own_X = self._worlds[condition].get_state()
                self._set_sample(new_sample, own_X, t)
        new_sample.set(ACTION, U)

        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self, own_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, own_X, -1)
        return sample

    def _set_sample(self, sample, own_X, t):
        for sensor in own_X.keys():
            sample.set(sensor, np.array(own_X[sensor]), t=t+1)
