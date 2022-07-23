import string
import numpy as np
import torch
from torch.utils.data import Dataset
from amc_dl.torch_plus import DataLoaders
from torch.utils.data import DataLoader
from note_attribute_repr import NoteAttributeAutoEncoder
from note_attribute_corrupter import SimpleCorrupter
from utils import augment_note_matrix
from pop909_data_prepare import merge_with_chord_atr_mat
from musebert_config import train_path, train_length_path, val_path, val_length_path, \
                        pad_length, chord_train_path, chord_valid_path, musebert_version, \
                        granularity, dataset

class NoteMatrixDataset:

    """Dataset to read files in R_base format"""

    train_path = train_path
    train_length_path = train_length_path
    val_path = val_path
    val_length_path = val_length_path
    pad_length = pad_length

    def __init__(self, data, length, pad_length):
        self.pad_length = pad_length

        self.data = data
        self.length = length

    def __len__(self):
        return len(self.length)

    def __getitem__(self, item):
        no = item * self.pad_length
        data = self.data[no: no + self.pad_length]
        length = self.length[item]
        return data.astype(np.int64), length.astype(np.int64)

    @classmethod
    def get_train_dataset(cls):
        data = np.load(cls.train_path)
        length = np.load(cls.train_length_path)
        return cls(data, length, cls.pad_length)

    @classmethod
    def get_val_dataset(cls):
        data = np.load(cls.val_path)
        length = np.load(cls.val_length_path)
        return cls(data, length, cls.pad_length)

    # @classmethod
    # def get_test_dataset(cls):
    #     data = np.load(cls.test_path)
    #     length = np.load(cls.test_length_path)
    #     return cls(data, length, cls.pad_length)


class PolyphonicDataset(Dataset):

    """Dataset class with R_fac autoencoder and corrupter"""

    dataset: NoteMatrixDataset
    repr_autoenc: NoteAttributeAutoEncoder
    corrupter: SimpleCorrupter
    is_validating: bool 

    def __init__(self, dataset, shift_low, shift_high,
                 repr_autoenc, corrupter, task='pretrain', is_validating=False):
        super(PolyphonicDataset, self).__init__()
        self.dataset = dataset
        self.shift_low = shift_low
        self.shift_high = shift_high
        self.task = task
        self.is_validating = is_validating

        self.repr_autoenc = repr_autoenc
        self.corrupter = corrupter

    @property
    def pad_length(self):
        return self.dataset.pad_length

    def generate_attention_mask(self, length):
        mask = np.zeros((self.pad_length, self.pad_length), dtype=np.int8)
        mask[0: length, 0: length] = 1
        return mask

    def __len__(self):
        return len(self.dataset) * (self.shift_high - self.shift_low + 1)

    def __getitem__(self, item):

        task = self.task

        no = item // (self.shift_high - self.shift_low + 1)
        shift = item % (self.shift_high - self.shift_low + 1) + self.shift_low
        nmat, length = self.dataset[no]

        # pitch-shift augmentation
        nmat = augment_note_matrix(nmat, length, shift)

        self.repr_autoenc.fast_mode()
        self.corrupter.fast_mode()

        # encode X_base to X_fac
        atr_mat, length = self.repr_autoenc.encode(nmat, length)

        if dataset == 'musicnet':
            # upper_bound = [85, 11, 11, 37, 11, 11, 7, 3, 12, 54, 61, 5, 9, 61] # for fixed_note
            upper_bound = [50, 11, 11, 37, 11, 11, 7, 3, 12, 9, 61, 9, 9, 61] # for fixed_8_beats_overlap musicnet
        elif dataset == 'asap':
            upper_bound = [61, 11, 11, 41, 11, 11, 7, 3, 12, 12, 12, 61, 140, 61] # for random_beats_overlap asap

        for i in range(length):
            nmat_row = nmat[i]
            
            # check if the input data = [0,0,0,0,0]
            if sum(nmat_row) == 0:
                print('Illegal data with all attribute values 0')

            atr_mat_row = atr_mat[i]
            for j in range(14):
                if not atr_mat_row[j] < upper_bound[j]:
                    print('out of bound', j, atr_mat_row[j], upper_bound[j], nmat_row, atr_mat_row)
    
        if self.task == 'chord_extraction':
            # kun: for chord extraction using pop909
            if self.is_validating:

                # valid
                atr_mat, length = merge_with_chord_atr_mat(chord_valid_path, atr_mat, length, no, shift)
            else:
                # training
                atr_mat, length = merge_with_chord_atr_mat(chord_train_path, atr_mat, length, no, shift)
            
        # corrupt X_fac and relation matrices
        cpt_atrmat, length, inds, _, cpt_relmat = self.corrupter.\
            compute_relmat_and_corrupt_atrmat_and_relmat(atr_mat, length, task=self.task, granularity=granularity)

        # square mask to mask out the pad tokens
        mask = self.generate_attention_mask(length)

        # if musebert_version == 'v2':
        #     atr_mat = nmat

        return atr_mat.astype(np.int64), cpt_atrmat.astype(np.int64), \
            cpt_relmat.astype(np.int8), mask.astype(np.int8), \
            inds.astype(bool), length, self.task


class PolyphonicDataLoaders(DataLoaders):

    def batch_to_inputs(self, batch):
        data, data_in, rel_mat, mask, inds, length, task = batch
        data = data.long().to(self.device)
        data_in = data_in.long().to(self.device)
        rel_mat = rel_mat.long().to(self.device)
        mask = mask.char().to(self.device)
        inds = inds.bool().to(self.device)
        length = length.long().to(self.device)
        task = task[0]
        return data, data_in, rel_mat, mask, inds, length, task

    @classmethod
    def get_loaders(cls, bs_train, bs_val, train_dataset, val_dataset,
                    shuffle_train=True, shuffle_val=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(train_dataset, bs_train, shuffle_train)
        val_loader = DataLoader(val_dataset, bs_val, shuffle_val)
        return cls(train_loader, val_loader, bs_train, bs_val, device)

    # @classmethod
    # def get_loaders_test(cls, bs_test, test_dataset, shuffle_test=False):
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     test_loader = DataLoader(test_dataset, bs_test, shuffle_test)
    #     return cls(test_loader, bs_test, device)
        
    @property
    def train_set(self):
        return self.train_loader.dataset

    @property
    def val_set(self):
        return self.val_loader.dataset

