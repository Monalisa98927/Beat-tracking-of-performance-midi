from note_attribute_repr import NoteAttributeAutoEncoder, Sampler
from note_attribute_corrupter import SimpleCorrupter
from dataset import PolyphonicDataLoaders, PolyphonicDataset, NoteMatrixDataset
from musebert_model import MuseBERT
from curriculum_preset import *
from musebert_config import augment, with_program, musebert_version


def prepare_data_loaders(atr_autoenc, corrupter, batch_size, task='pretrain'):
    if augment:
        train_set = PolyphonicDataset(NoteMatrixDataset.get_train_dataset(), -6, 5,
                                    atr_autoenc, corrupter, task, False)
    else:
        train_set = PolyphonicDataset(NoteMatrixDataset.get_train_dataset(), 0, 0,
                                    atr_autoenc, corrupter, task, False)        
    val_set = PolyphonicDataset(NoteMatrixDataset.get_val_dataset(), 0, 0,
                                atr_autoenc, corrupter, task, True)
    data_loaders = \
        PolyphonicDataLoaders.get_loaders(batch_size, batch_size, train_set, val_set,
                                          True, False)
    return data_loaders


def prepare_model(loss_inds, relation_vocab_sizes=(5, 5, 5, 5)):

    if musebert_version == 'v1':
        if with_program:
            return MuseBERT.\
            init_model(relation_vocab_sizes=relation_vocab_sizes,
                    in_dims=(15, 15, 15, 15, 15, 15, 15, 18, 15),
                    out_dims=(9, 7, 7, 3, 12, 5, 8, 16, 8),
                    loss_inds=loss_inds)

        return MuseBERT.\
            init_model(relation_vocab_sizes=relation_vocab_sizes,
                    loss_inds=loss_inds)
    
    elif musebert_version == 'v2':

        return MuseBERT.\
            init_model(relation_vocab_sizes=relation_vocab_sizes,
                    # in_dims=(100, 20, 20, 80, 20, 20, 15, 10, 25, 100, 100, 15, 15, 100),
                    # out_dims=(85, 11, 11, 37, 11, 11, 7, 3, 12, 54, 61, 5, 9, 61), # for fixed_100_notes with 70% overlap
                    in_dims=(100, 20, 20, 80, 20, 20, 15, 10, 25, 15, 100, 15, 15, 100), # for fixed_8_beats with 1 beat overlap
                    out_dims=(50, 11, 11, 37, 11, 11, 7, 3, 12, 9, 61, 9, 9, 61), # for fixed_8_beats with 1 beat overlap
                    loss_inds=loss_inds)


class Curriculum:

    """
    A class to handle four types of (hyper-)parameters
    - autoenc_dict: R_base <-> R_fac conversion
    - corrupter_dict: BERT-like corruption parameters
    - model_dict: MuseBERT parameters
    - train_dict: training parameters and learning rate parameters
    """

    def __init__(self, autoenc_dict, corrupter_dict, model_dict, train_dict):
        self.autoenc_dict = autoenc_dict
        self.corrupter_dict = corrupter_dict
        self.model_dict = model_dict
        self.train_dict = train_dict
        self.consistency_check()
        self.autoenc = NoteAttributeAutoEncoder(**self.autoenc_dict)
        self.corrupter = SimpleCorrupter(**self.corrupter_dict)

    def consistency_check(self):
        assert self.autoenc_dict['nmat_pad_length'] == \
               self.autoenc_dict['atr_mat_pad_length'] == \
               self.corrupter_dict['pad_length']
        assert tuple(np.where(np.array(self.train_dict['beta']) != 0)[0]) == \
               self.model_dict['loss_inds']

    def prepare_data(self, task='pretrain'):
        # prepare data_loaders
        autoenc = NoteAttributeAutoEncoder(**self.autoenc_dict)
        corrupter = SimpleCorrupter(**self.corrupter_dict)
        return prepare_data_loaders(autoenc, corrupter, self.train_dict['batch_size'], task)

    def prepare_model(self, device):
        return prepare_model(**self.model_dict).to(device)

    def reset_batch_size(self, new_bs):
        self.train_dict['batch_size'] = new_bs

    @property
    def beta(self):
        return self.train_dict['beta']

    @property
    def lr(self):
        return self.train_dict['lr_dict']

    def __call__(self, device, task='pretrain'):
        data_loaders = self.prepare_data(task)
        model = self.prepare_model(device)
        return data_loaders, model


# curriculum for used for pre-training
all_curriculum = Curriculum(default_autoenc_dict,
                            all_corrupter_dict,
                            all_model_dict,
                            all_train_dict)

musebert_v2_pretrain_curriculum = Curriculum(default_autoenc_dict,
                                             musebert_v2_pretrain_corrupter_dict,
                                             musebert_v2_pretrain_model_dict,
                                             musebert_v2_pretrain_dict)

# curriculum for used for pre-training with program
program_pretrain_curriculum = Curriculum(default_autoenc_dict,
                            program_pretrain_corrupter_dict,
                            program_pretrain_model_dict,
                            program_pretrain_dict)

# curricula for fine-tuning specific attributes
program_ft_curriculum = Curriculum(default_autoenc_dict,
                            program_ft_corrupter_dict,
                            program_ft_model_dict,
                            program_ft_dict)

onset_ft_curriculum = Curriculum(default_autoenc_dict,
                                 onset_corrupter_dict,
                                 onset_model_dict,
                                 onset_train_ft_dict)

pitch_ft_curriculum = Curriculum(default_autoenc_dict,
                                 pitch_corrupter_dict,
                                 pitch_model_dict,
                                 pitch_train_ft_dict)

duration_ft_curriculum = Curriculum(default_autoenc_dict,
                                    duration_corrupter_dict,
                                    duration_model_dict,
                                    duration_train_ft_dict)

chord_ft_curriculum = Curriculum(default_autoenc_dict,
                                 chord_corrupter_dict,
                                 chord_model_dict,
                                 chord_train_ft_dict)

beat_ft_curriculum = Curriculum(default_autoenc_dict,
                                 beat_corrupter_dict,
                                 beat_model_dict,
                                 beat_train_ft_dict)

beat_ft_v2_curriculum = Curriculum(default_autoenc_dict,
                                 beat_ft_v2_corrupter_dict,
                                 beat_ft_v2_model_dict,
                                 beat_ft_v2_dict)