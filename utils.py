from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import torch
import numpy as np

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                    num_training_steps, final_lr_factor,
                                    last_epoch=-1):
    """
    Copied and **modified** from: https://github.com/huggingface/transformers

    Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period during which
    it increases linearly from 0 to the initial lr set in the optimizer.

    :param optimizer: The optimizer for which to schedule the learning rate.
    :param num_warmup_steps: The number of steps for the warmup phase.
    :param num_training_steps: The total number of training steps.
    :param final_lr_factor: Final lr = initial lr * final_lr_factor
    :param last_epoch: the index of the last epoch when resuming training.
        (defaults to -1)
    :return: `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            final_lr_factor, float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def augment_note_matrix(nmat, length, shift):
    """Pitch shift a note matrix in R_base format.
       kun's modification: 如果越界，-12 或 +12   
    """
    aug_nmat = nmat.copy()
    aug_nmat[0: length, 1] += shift
    for index,pitch in enumerate(aug_nmat[0: length, 1]):
        if pitch > 127:
            aug_nmat[index, 1] -= 12
        elif pitch < 0:
            aug_nmat[index, 1] += 12
    return aug_nmat

def plot_mask(mask, name):

    plt.matshow(mask, cmap=plt.cm.Blues)
    ax = plt.gca()
    ax.set_xlim(0, len(mask[0])-1)
    ax.set_ylim(len(mask[0])-1, 0)
    miloc = plt.MultipleLocator(1)
    ax.xaxis.set_minor_locator(miloc)
    ax.yaxis.set_minor_locator(miloc)
    ax.grid(axis='x', which='minor')
    ax.grid(axis='y', which='minor')
    plt.title("relation mask")
    plt.grid()
    plt.show()
    plt.savefig(f'mask/mask_{name}.png', format='png')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True,为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), f'checkpoint_{path}.pt')     # 这里会存储迄今最优模型的参数
        # torch.save(model, 'finish_model.pkl')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss