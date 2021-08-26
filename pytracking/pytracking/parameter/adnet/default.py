import numpy as np
from pathlib import Path
from pytracking.utils import TrackerParams
from pytracking.evaluation.environment import env_settings

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False
    params.use_gpu = True

    ## Copy parameters from init_params.m (from options/general.py)
    params.opts = {
        'imgSize' : [112, 112, 3],
        'train_dbs' : ['vot15', 'vot14', 'vot13'],
        'test_db' : 'otb',
        'train': {
            'weightDecay' : 0.0005,
            'momentum' : 0.9,
            'learningRate' : 10e-5,
            'conserveMemory' : True,
            'gt_skip' : 1,
            'rl_num_batches' : 5,
            'RL_steps' : 10
        },
        'minibatch_size' : 32,
        'numEpoch' : 30,
        'numInnerEpoch' : 3,
        'continueTrain' : False,
        'samplePerFrame_large' : 40,
        'samplePerFrame_small' : 10,
        'inputSize' : [112, 112, 3],
        'stopIou' : 0.93,
        'meta':{
            'inputSize' : [112, 112, 3]
        },
        'use_finetune' : True,
        'scale_factor' : 1.05,

        # test
        'finetune_iters' : 20,
        'finetune_iters_online' : 10,
        'finetune_interval' : 30,
        'posThre_init' : 0.7,
        'negThre_init' : 0.3,
        'posThre_online' : 0.7,
        'negThre_online' : 0.5,
        'nPos_init' : 200,
        'nNeg_init' : 150,
        'nPos_online' : 30,  # TODO why isn't this 250 like in the paper?
        'nNeg_online' : 15,
        'finetune_scale_factor' : 3.0,  # TODO try using MDNet settings (1.3)
        'redet_scale_factor' : 3.0,
        'finetune_trans' : 0.10,
        'redet_samples' : 256,

        'successThre' : 0.5,
        'failedThre' : 0.5,

        'nFrames_long' : 100, # long-term period (in matlab code, for positive samples... while current implementation just with history for now...)
        'nFrames_short' : 20, # short-term period (for negative samples)

        'nPos_train' : 150,
        'nNeg_train' : 50,
        'posThre_train' : 0.5,
        'negThre_train' : 0.3,

        'random_perturb' : {
            'x' : 0.15,
            'y' : 0.15,
            'w' : 0.03,
            'h' : 0.03
        },

        'action_move' : {
            'x' : 0.03,
            'y' : 0.03,
            'w' : 0.03,
            'h' : 0.03,
            'deltas' : np.array([
                [-1, 0, 0, 0], # left
                [-2, 0, 0, 0], # left x2
                [+1, 0, 0, 0], # right
                [+2, 0, 0, 0], # right x2
                [0, -1, 0, 0], # up
                [0, -2, 0, 0], # up x2
                [0, +1, 0, 0], # down
                [0, +2, 0, 0], # down x2
                [0,  0,  0,  0], # stop
                [0,  0, -1, -1], # smaller
                [0,  0, +1, +1]  # bigger
            ])
        },

        'num_actions' : 11,
        'stop_action' : 8,
        'num_show_actions' : 20,
        'num_action_step_max' : 20,
        'num_action_history' : 10,

        # For VGG-M input 
        'means' : [104, 117, 123]  # https://github.com/amdegroot/ssd.pytorch/blob/8dd38657a3b1df98df26cf18be9671647905c2a0/data/config.py
    }


    env = env_settings()
    params.vggm_path = Path(env.network_path) / "imagenet-vgg-m.mat"  # "imagenet-vgg-m-conv1-3.mat"
    params.checkpoints_path = Path(env.network_path) / "adnet_checkpoints"
    params.model_path = Path(env.network_path) / "adnet.pth"

    params.num_actions = 11
    params.num_action_history = 10
    params.num_action_step_max = 20

    params.action_names = [ "LEFT", "LEFT2", "RIGHT", "RIGHT2", "UP", "UP2", "DOWN", "DOWN2", "STOP", "SHRINK", "EXPAND" ]
    params.inverse_actions = [ 2, 3, 0, 1, 6, 7, 4, 5, 8, 10, 9 ]

    # How much to translate/scale when looking for positive/negative samples
    params.trans_pos = params.opts['finetune_trans']
    params.scale_pos = params.opts['finetune_scale_factor']
    params.trans_neg = 2.0  # From Matlab code (but wasn't in settings)
    params.scale_neg = 1.6  # 1.3 
    # From MDNet:
    params.trans_neg_init = 1
    params.scale_neg_init = 1.6

    # IOU overlap range for positive/negative samples
    params.overlap_pos = [params.opts['posThre_online'], 1.0]   # 0.7, 1.0
    params.overlap_neg = [0, 0.3]  # [0, params.opts['negThre_online']]  # 0, 0.5
    params.overlap_pos_init = [params.opts['posThre_init'], 1.0]  # 0.7, 1.0
    params.overlap_neg_init = [0, 0.5]  # [0, params.opts['negThre_init']]  # 0, 0.5
    params.overlap_pos_train = [0.3, 1.0]
    params.overlap_neg_train = [0, 0.3]

    params.hard_negative_mining = True
    params.batch_neg_cand = 512  # How many negative candidates to examine during hard negative mining
    params.batch_test = 512

    # Image processing 
    params.padding = 16
    params.img_size = 107  # original 112
    params.means = 128

    params.lr_update = 0.001
    params.maxiter_update = 15  # opts['finetune_iters_online']
    params.n_pos_update = 50  # nPos_online
    params.n_neg_update = 200  # nNeg_online

    params.lr_init = 0.0005
    params.maxiter_init = 40  # opts['finetune_iters']
    params.n_pos_init = 400  # nPos_init
    params.n_neg_init = 800  # nNeg_init

    params.weight_decay = params.opts['train']['weightDecay']
    params.momentum = params.opts['train']['momentum']

    # Pre-training
    # Batch size = batch_frames * (n_pos_train + n_neg_train)
    params.batch_frames = 2
    params.n_pos_train = 180
    params.n_neg_train = 40
    params.lr_backbone = 0.00005
    params.lr_train = 0.0001
    params.n_epochs_sl = 25
    params.n_epochs_rl = 600
    params.checkpoint_interval_sl = 5  
    params.checkpoint_interval_rl = 100
    params.rl_sequence_length = 10
    params.n_batches_rl = 5  # Batch this many simulations before optimizing
    params.rl_gamma = 0.5  # \gamma for RL RETURNS reward signal

    params.initial_fine_tuning = True  # Perform initial fine tuning  
    params.stop_unconfident = True  # Stop tracking procedure if below threshold
    params.redetection = True  # Perform redetection 
    params.fine_tuning = True  # Perform online fine tuning 

    params.save_visualization = False 
    params.visualization_path = Path(env.result_plot_path) / "ADNet_plots"

    return params

