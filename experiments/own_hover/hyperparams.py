# To get started, copy over hyperparams from another experiment.
# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.
""" Hyperparameters for Own Quadrotor Hover."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.own.agent_own import AgentOwn
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
# from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, ACTION
from gps.gui.config import generate_experiment_info

SENSOR_DIMS = {
    END_EFFECTOR_POINTS: 13,
    ACTION: 4
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/own_hover/'

common = {
    'experiment_name': 'own_hover' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentOwn,
    'target_state': np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    'render': False,
    'x0': np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    'dt': 0.01,
    'substeps': 1,
    'conditions': common['conditions'],
    'T': 200,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [END_EFFECTOR_POINTS], #END_EFFECTOR_POINTS
    'obs_include': [],
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
}

# algorithm['init_traj_distr'] = {
#     'type': init_pd,
#     'init_var': 5.0,
#     'pos_gains': 1.0,
#     'dQ': SENSOR_DIMS[ACTION],
#     'dt': agent['dt'],
#     'T': agent['T'],
# }

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'final_weight': 50.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1, 1, 1, 1])
}

state_cost = {
    'type': CostState,
    'data_types' : {
        END_EFFECTOR_POINTS: {
            'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
            'target_state': agent["target_state"],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1.0, 1],
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': 10,
    'num_samples': 5,
    'common': common,
    'verbose_trials': 0,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    # dQ is only for init_pd
    # 'dQ': algorithm['init_traj_distr']['dQ'],
}

common['info'] = generate_experiment_info(config)
