from operator import index
from re import X
from unittest import result
from transformer import TransformerEncoder
from amc_dl.torch_plus.module import PytorchModel
import numpy as np
import torch.nn as nn
import torch


class MuseBERT(PytorchModel):

    tfm: TransformerEncoder

    def __init__(self, name, device, tfm, in_dims, out_dims, loss_inds):
        """
        :param name: name of the model, e.g., 'musebert'
        :param device: cpu or cuda
        :param tfm: transformer encoder
        :param in_dims: intput vocab sizes
        :param out_dims: output vocab sizes (distribution sizes)
        :param loss_inds: (which attributes to compute recon loss)
        """

        def compute_lr_inds(dims):
            dims = np.array(dims, dtype=np.int64)
            r_inds = np.cumsum(dims)
            l_inds = np.insert(r_inds[0: -1], 0, 0)
            return tuple(l_inds), tuple(r_inds)

        super(MuseBERT, self).__init__(name, device)

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.n_col = len(in_dims)

        # the left, right endpoints for output distributions of attributes
        self.lout_inds, self.rout_inds = compute_lr_inds(out_dims)

        self.loss_inds = loss_inds

        self.embeddings = \
            nn.ModuleList([nn.Embedding(ind, tfm.d_model)
                           for ind in in_dims])
        self.tfm = tfm
        self.out = nn.Linear(tfm.d_model, self.rout_inds[-1])
        self.lossf = nn.CrossEntropyLoss(reduction='none')

    @property
    def d_model(self):
        return self.tfm.d_model

    @property
    def N(self):
        return self.tfm.N

    @staticmethod
    def _truncate_data(x, length=None, truncate_len=None, mode=0):
        """Apply data truncation in three different modes."""
        truncate_len = truncate_len if truncate_len is not None \
            else (length.max() if length is not None else None)
        if truncate_len is not None:
            if mode == 0:
                return x[:, 0: truncate_len]
            elif mode == 1:
                return x[:, 0: truncate_len, 0: truncate_len]
            elif mode == 2:
                return x[:, :, 0: truncate_len, 0: truncate_len]
            else:
                raise NotImplementedError
        return x

    def _truncate_lists(self, xs, length, modes):
        truncate_len = length.max()
        return tuple(self.__class__._truncate_data(
            x, None, truncate_len, m
        ) for x, m in zip(xs, modes))

    def truncate(self, data_in, mask, rel_mat, data, inds, length):
        """truncate the input data to max(length)."""
        return self._truncate_lists([data_in, mask, rel_mat, data, inds],
                                    length,
                                    modes=[0, 1, 2, 0, 0])

    def onset_pitch_dur_embedding(self, data_in):
        """ Sum up the onset, pitch, dur embeddings """
        return sum([self.embeddings[i](data_in[:, :, i])
                    for i in range(len(self.in_dims))])

    def run(self, data_in, rel_mat, mask):
        """
        batch -> output distribution
        :param data_in: (bs, L, 3) dtype long. Last dim: onset, pitch, dur.
        :param rel_mat: (bs, k, L, L)
        :param mask: (bs, L, L) dtype long
        :return: (bs, L, pitch + dur dims)
        """
        x = self.onset_pitch_dur_embedding(data_in)
        x = self.tfm(x, rel_mat, mask=mask)
        x = self.out(x)
        return x

    def loss_function(self, recon, tgt, inds, beta, task):
        """compute reconstuction loss on corrupted attributes (inds) only."""

        def atr_loss(recon, tgt, i, task):
            if i in self.loss_inds:
                l_ind = self.lout_inds[i]
                r_ind = self.rout_inds[i]

                if task == 'chord_extraction':
                    # kun's modification: 对于chord extraction任务，只对p_hig==6(7)时的p_deg进行loss的计算
                    p_hig_6_index = []
                    for index, p_hig  in zip(range(len(tgt[:,2])),tgt[:,2]):
                        if p_hig == 6:
                            p_hig_6_index.append(index)                    
                    return (self.lossf(recon[p_hig_6_index, l_ind: r_ind],
                                    tgt[p_hig_6_index, i]) * w[p_hig_6_index]).sum()
                elif task == 'beat_detection':
                    pitch_127_index = []
                    for index, p_hig, p_reg, p_deg  in zip(range(len(tgt[:,2])),tgt[:,2],tgt[:,3],tgt[:,4]):
                        if 108 + 12 * p_reg + p_deg == 127:
                            pitch_127_index.append(index)
                    return (self.lossf(recon[pitch_127_index, l_ind: r_ind],
                                    tgt[pitch_127_index, i]) * w[pitch_127_index]).sum()

                return (self.lossf(recon[:, l_ind: r_ind],
                                tgt[:, i]) * w).sum()

            else:
                return torch.zeros([]).float().to(self.device)

        # compute weight w: so that each data sample is treated equally.
        # E.g., when bs=2, 1st sample has 1 corrupted tokens and 2nd has 2,
        #  w = [0.5, 0.25, 0.25], not [0.333, 0.333, 0.333].
        counts = inds.long().sum(-1)
        bs = inds.size(0) # bs=batch_size=24, self.n_col=7
        w = torch.cat([torch.tensor([1 / c.float()] * c.long())
                       for c in counts], 0) / bs
        w = w.to(self.device)
        recon = recon[inds]  # (*, outs)
        tgt = tgt[inds]

        # recon (recon_len * 51)   tgt (tgt_len * 7)   recon和tgt长度相等
        # losses返回7个维度的loss，index=4时表示p_deg
        losses = [atr_loss(recon, tgt, i, task) for i in range(self.n_col)]

        # beta controls the weighting for different attributes
        total_loss = sum([ls * b for ls, b in zip(losses, beta)])

        return (total_loss, *losses)

    def loss(self, data, data_in, rel_mat, mask, inds, length, task, beta):
        data_in, mask, rel_mat, data, inds = \
            self.truncate(data_in, mask, rel_mat, data, inds, length)       
        recon = self.run(data_in, rel_mat, mask)
        loss = self.loss_function(recon, data, inds, beta, task)
        return loss

    def inference(self, data, data_in, rel_mat, mask, inds,
                  length, task, beta, truncate=True):
        self.eval()
        with torch.no_grad():
            if truncate:
                data_in, mask, rel_mat, data, inds = \
                    self.truncate(data_in, mask, rel_mat, data, inds, length)                
                recon = self.run(data_in, rel_mat, mask)
                pred_and_tgt = self.get_prediction_and_target(recon, data, inds, task)

        return pred_and_tgt

    def get_prediction_and_target(self, recon, tgt, inds, task):
        recon = recon[inds]  # (*, outs)
        tgt = tgt[inds]

        def atr_prediction_and_target(recon, tgt, i, task):
            if i in self.loss_inds:
                l_ind = self.lout_inds[i]
                r_ind = self.rout_inds[i]

                # i=4表示计算p_deg的loss
                if task == 'chord_extraction' and i == 4:
                    # kun's modification: 对于chord extraction任务，只对p_hig==6(7)时的p_deg进行loss的计算
                    p_hig_6_index = []
                    for index, p_hig in zip(range(len(tgt[:,2])),tgt[:,2]):
                        if p_hig == 6: 
                            p_hig_6_index.append(index)

                    # 数据条数：n = len(p_hig_6_index) 
                    recon_atr = recon[p_hig_6_index, l_ind: r_ind] # n*12
                    tgt_atr = tgt[p_hig_6_index, i] # n*1, 值为p_deg
                    tgt_atr_p_reg = tgt[p_hig_6_index, i-1] # 值为p_reg=0,1,2 for root,chroma,bass

                    # root: [0..11]
                    # chroma: [12..23]
                    # bass: [24..35]
                    prediction_values = [int(np.argmax(r.cpu())) + int(t_reg.cpu()) * 12 for r,t_reg in zip(recon_atr,tgt_atr_p_reg)]
                    target_values = [int(t.cpu()) + int(t_reg.cpu()) * 12 for t,t_reg in zip(tgt_atr,tgt_atr_p_reg)]

                    return prediction_values, target_values
                
                elif task == 'beat_detection' and i==0:

                    pitch_127_index = []
                    for index, p_hig, p_reg, p_deg, d_hlf, d_sqv  in \
                        zip(range(len(tgt[:,2])),tgt[:,2],tgt[:,3],tgt[:,4],tgt[:,5],tgt[:,6]):
                        if (108 + 12 * p_reg + p_deg) == 127 and (d_hlf * 8 + d_sqv) == 0:
                            pitch_127_index.append(index)

                    recon_o_bt = recon[pitch_127_index, l_ind: r_ind]
                    recon_o_sub = recon[pitch_127_index, self.lout_inds[i+1]: self.rout_inds[i+1]]
                    tgt_o_bt = tgt[pitch_127_index, i]
                    tgt_o_sub = tgt[pitch_127_index, i+1]

                    prediction_values = [(int(np.argmax(o_bt.cpu())) * 4 + int(np.argmax(o_sub.cpu()))) \
                                        for o_bt, o_sub in zip(recon_o_bt, recon_o_sub)]
                    target_values = [(int(o_bt.cpu()) * 4 + int(o_sub.cpu())) \
                                    for o_bt, o_sub in zip(tgt_o_bt, tgt_o_sub)]
                    
                    return prediction_values, target_values

                else:
                    recon_atr = recon[:, l_ind: r_ind] # n*12
                    tgt_atr = tgt[:, i] # n*1

                    prediction_values = [int(np.argmax(r.cpu())) for r in recon_atr]
                    target_values = [int(t.cpu()) for t in tgt_atr]

                return prediction_values, target_values

            else:
                return [], []
        
        result = [atr_prediction_and_target(recon, tgt, i, task) for i in range(self.n_col)]

        return result


    @classmethod
    def init_model(cls, N=12, h=8, d_model=128, d_ff=512, non_linear=None,
                   relation_vocab_sizes=(5, 5, 5, 5),
                   in_dims=(15, 15, 15, 15, 15, 15, 15),
                   out_dims=(9, 7, 7, 3, 12, 5, 8), # if chord, the third dim(p_hig) 7 -> 8 
                   loss_inds=(1, 3, 4, 5, 6),
                   dropout=0.1):
        """Easier way to initialize a MuseBERT"""
        name = 'musebert'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if non_linear == 'gelu':
            non_linear = nn.GELU()
        tfm = TransformerEncoder(N, h, d_model, d_ff, non_linear=non_linear,
                                 relation_vocab_sizes=relation_vocab_sizes,
                                 dropout=dropout, attn_dropout=None)
        model = cls(name, device, tfm, in_dims, out_dims, loss_inds)
        model.to(device)
        return model

