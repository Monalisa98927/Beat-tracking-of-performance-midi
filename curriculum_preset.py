from note_attribute_repr import Sampler
import numpy as np
from musebert_config import pad_length, batch_size, n_epoch, with_program

# """
# default pad lengths
# """
# pad_length100 = 100
# pad_length36 = 36 # for rwc100 with beat
# pad_length28 = 28 # for rwc100, bpm = its own tempo
# pad_length150 = 150 # for pop909 chord extraction, autoenc
# pad_length = 200 # for pop909 chord extraction, corruption

"""
default R_base to R_fac conversion
# """
eo_dist0 = Sampler(0, 4, np.array([0.8, 0.08, 0.07, 0.05]))

ep_dist0 = None  # i.e., always estimate ep

w_dist0 = Sampler(-3, 4,
                  np.array([0.05, 0.05, 0.15, 0.5, 0.15, 0.05, 0.05]))

default_autoenc_dict = {
    'nmat_pad_length': pad_length,
    'atr_mat_pad_length': pad_length,
    'estimate_ep': True,
    'eo_dist': eo_dist0,
    'ep_dist': ep_dist0,
    'w_dist': w_dist0
}


"""
Preset corrupters
"""

# For light corruption where o_bt and p_hig are not corrupted.
default_corrupter_dict = {
    'corrupt_col_ids': (1, 3, 4, 5, 6),
    'pad_length': pad_length,
    'mask_ratio': 0.15,
    'unchange_ratio': 0.1,
    'unknown_ratio': 0.8,
    'relmat_cpt_ratio': 0.3
}

# For pre-trained settings where all attributes are masked.
all_corrupter_dict = {
    'corrupt_col_ids': (0, 1, 2, 3, 4, 5, 6),
    'pad_length': pad_length,
    'mask_ratio': 0.15,
    'unchange_ratio': 0.1,
    'unknown_ratio': 0.8,
    'relmat_cpt_ratio': 0.3
}

# For pre-trained settings where all attributes are masked.
musebert_v2_pretrain_corrupter_dict = {
    'corrupt_col_ids': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
    'pad_length': pad_length,
    'mask_ratio': 0.15,
    'unchange_ratio': 0.1,
    'unknown_ratio': 0.8,
    'relmat_cpt_ratio': 0.3
}

beat_ft_v2_corrupter_dict = {
    'corrupt_col_ids': (9, 10, 11, 12, 13),
    'pad_length': pad_length,
    'mask_ratio': 1.0,
    'unchange_ratio': 0.0,
    'unknown_ratio': 1.0,
    'relmat_cpt_ratio': 0.0
}

# For pre-trained settings where all attributes are masked.
program_pretrain_corrupter_dict = {
    'corrupt_col_ids': (0, 1, 2, 3, 4, 5, 6, 7, 8),
    'pad_length': pad_length,
    'mask_ratio': 0.15,
    'unchange_ratio': 0.1,
    'unknown_ratio': 0.8,
    'relmat_cpt_ratio': 0.3
}

# Program attributes masked: for fine-tuning onset predictor.
program_ft_corrupter_dict = {
    'corrupt_col_ids': (7, 8),
    'pad_length': pad_length,
    'mask_ratio': 1.0,
    'unchange_ratio': 0.0,
    'unknown_ratio': 1.0,
    'relmat_cpt_ratio': 0.0
}

# Onset attributes masked: for fine-tuning onset predictor.
onset_corrupter_dict = {
    'corrupt_col_ids': (0, 1),
    'pad_length': pad_length,
    'mask_ratio': 0.15,
    'unchange_ratio': 0.1,
    'unknown_ratio': 0.8,
    'relmat_cpt_ratio': 0.5
}

# Beat attributes masked: for fine-tuning beat predictor.
beat_corrupter_dict = {
    'corrupt_col_ids': (0, 1),
    'pad_length': pad_length,
    'mask_ratio': 1.0,
    'unchange_ratio': 0.0,
    'unknown_ratio': 1.0,
    'relmat_cpt_ratio': 0.0
}

# Pitch attributes masked: for fine-tuning pitch predictor.
pitch_corrupter_dict = {
    'corrupt_col_ids': (2, 3, 4),
    'pad_length': pad_length,
    'mask_ratio': 0.15,
    'unchange_ratio': 0.1,
    'unknown_ratio': 0.8,
    'relmat_cpt_ratio': 0.5
}

# Chord attributes masked: for fine-tuning chord predictor.
chord_corrupter_dict = {
    'corrupt_col_ids': (4,),
    'pad_length': pad_length,
    'mask_ratio': 1.0,
    'unchange_ratio': 0.0,
    'unknown_ratio': 1.0,
    'relmat_cpt_ratio': 0.0
}

# Duration attributes masked: for fine-tuning duration predictor.
duration_corrupter_dict = {
    'corrupt_col_ids': (5, 6),
    'pad_length': pad_length,
    'mask_ratio': 0.15,
    'unchange_ratio': 0.1,
    'unknown_ratio': 0.8,
    'relmat_cpt_ratio': 0.5
}


"""
Preset MuseBERT parameters
"""

# used with default corrupter
default_model_dict = {
    'loss_inds': (1, 3, 4, 5, 6)
}

# used with pre-training
all_model_dict = {
    'loss_inds': (0, 1, 2, 3, 4, 5, 6),
}

musebert_v2_pretrain_model_dict = {
    'loss_inds':  (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
}

beat_ft_v2_model_dict = {
    'loss_inds':  (9, 10, 11, 12, 13)
}

# used with pre-training with program
program_pretrain_model_dict = {
    'loss_inds': (0, 1, 2, 3, 4, 5, 6, 7, 8),
}

# used with onset, pitch, chord and duration fine-tuning.
program_ft_model_dict = {'loss_inds': (7, 8)}
onset_model_dict = {'loss_inds': (0, 1)}
beat_model_dict = {'loss_inds': (0, 1)}
pitch_model_dict = {'loss_inds': (2, 3, 4)}
chord_model_dict = {'loss_inds': (4,)}
duration_model_dict = {'loss_inds': (5, 6)}


"""
Preset learning rate parameters
"""

# pre-training parameters
default_lr_dict = {
    'lr': 5e-4,
    'final_lr_factor': 1e-2,
    'warmup': 15000,
    'n_epoch': n_epoch,
    'final_epoch': n_epoch*0.6
}


# fine-tuning parameters
ft_lr_dict = {
    'lr': 3e-4,
    'final_lr_factor': 1e-2,
    'warmup': 10000,
    'n_epoch': n_epoch,
    'final_epoch': n_epoch*0.6
}


"""
Preset training parameters
"""


# used with default corrupter
default_train_dict = {
    'batch_size': batch_size,
    'beta': (0, 1, 0, 1, 1, 0.1, 0.1),
    'lr_dict': default_lr_dict
}


# used at pre-training
all_train_dict = {
    'batch_size': batch_size,
    'beta': (1, 1, 1, 1, 1, 0.1, 0.1),
    'lr_dict': default_lr_dict
}

musebert_v2_pretrain_dict = {
    'batch_size': batch_size,
    'beta':  (1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1),
    'lr_dict': default_lr_dict
}

beat_ft_v2_dict = {
    'batch_size': batch_size,
    'beta':  (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1),
    'lr_dict': default_lr_dict
}

# used at pre-training with program
program_pretrain_dict = {
    'batch_size': batch_size,
    'beta': (1, 1, 1, 1, 1, 0.1, 0.1, 1, 1),
    'lr_dict': default_lr_dict
}


# used at general fine-tuning
all_train_ft_dict = {
    'batch_size': 32,
    'beta': (1, 1, 1, 1, 1, 0.1, 0.1),
    'lr_dict': ft_lr_dict
}

# used for fine-tuning specific attributes
program_ft_dict = {
    'batch_size': batch_size,
    'beta': (0, 0, 0, 0, 0, 0, 0, 1, 1),
    'lr_dict': default_lr_dict
}

# used for fine-tuning specific attributes
onset_train_ft_dict = {
    'batch_size': 32,
    'beta': (1, 1, 0, 0, 0, 0, 0),
    'lr_dict': ft_lr_dict
}

# used for fine-tuning specific attributes
beat_train_ft_dict = {
    'batch_size': batch_size,
    'beta': (1, 1, 0, 0, 0, 0, 0),
    'lr_dict': ft_lr_dict
}


pitch_train_ft_dict = {
    'batch_size': 32,
    'beta': (0, 0, 1, 1, 1, 0, 0),
    'lr_dict': ft_lr_dict
}


chord_train_ft_dict = {
    'batch_size': batch_size,
    'beta': (0, 0, 0, 0, 1, 0, 0),
    'lr_dict': ft_lr_dict
}


duration_train_ft_dict = {
    'batch_size': 32,
    'beta': (0, 0, 0, 0, 0, 1, 1),
    'lr_dict': ft_lr_dict
}
