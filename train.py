from tensorboard import program
import torch
from torch import optim
from curricula import Curriculum, all_curriculum, chord_ft_curriculum, \
                        beat_ft_curriculum, program_pretrain_curriculum, program_ft_curriculum, \
                        musebert_v2_pretrain_curriculum, beat_ft_v2_curriculum

from amc_dl.torch_plus import LogPathManager, SummaryWriters, \
    ParameterScheduler, OptimizerScheduler, \
    ConstantScheduler, TrainingInterface
from utils import get_linear_schedule_with_warmup
from typing import Union
from musebert_config import task, model_path, mode, stage, with_program, \
                            last_epoch, train_after_interrupt, last_step, musebert_version

class TrainMuseBERT(TrainingInterface):

    def _batch_to_inputs(self, batch):
        """Convert a data batch to proper data types."""

        data, data_in, rel_mat, mask, inds, length, task = batch

        # data: the ground truth X_fac
        data = data.long().to(self.device)

        # data_in: the corrupted X_fac^*
        data_in = data_in.long().to(self.device)

        # rel_mat: the corrupted R_S^*.
        rel_mat = rel_mat.long().to(self.device)

        # MuseBERT mask (masking the paddings)
        mask = mask.char().to(self.device)

        # The corrupted rows.
        inds = inds.bool().to(self.device)

        # number of notes contained in each sample.
        length = length.long().to(self.device)

        task = task[0]

        return data, data_in, rel_mat, mask, inds, length, task


def train_musebert(parallel: bool, curriculum: Curriculum,
                   model_path: Union[None, str]=None, \
                   task='pretrain', mode='default', stage='pretrain', \
                   with_program=False, last_epoch=-1, \
                   train_after_interrupt=False, musebert_version='v1'):
    """
    The main function to train a MuseBERT model.

    :param parallel: whether to use data parallel.
    :param curriculum: input parameters
    :param model_path: None or pre-trained model path.
    """

    print(f'\n\nTask: {task}\n\tMode: {mode}\n\tStage: {stage}\n')
    print(f'musebert version: {musebert_version}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    readme_fn = [__file__, 'run.sh', 'musebert_config.py', 'pop909_data_prepare.py', \
                'curriculum_preset.py', 'note_attribute_corrupter.py', \
                'note_attribute_repr.py', 'musebert_model.py', 'dataset.py', \
                'curricula.py', 'amc_dl/torch_plus/module.py']

    clip = 1
    parallel = parallel if (torch.cuda.is_available() and
                            torch.cuda.device_count() > 1) else False

    # create data_loaders and initialize model specified by the curriculum.
    data_loaders, model = curriculum(device, task)

    # load a pre-trained model if necessary.
    if model_path is not None and model_path is not '':
        model.load_model(model_path, device)

    # to handle the path to save model parameters, logs etc.
    log_path_mng = LogPathManager(readme_fn=readme_fn, log_path_name=task)

    # optimizer and scheduler
    if train_after_interrupt:
        optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': curriculum.lr['lr']}], lr=curriculum.lr['lr'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=curriculum.lr['lr'])
    
    schdl_step = len(data_loaders.train_loader) * curriculum.lr['final_epoch']
    if train_after_interrupt:
        scheduler = \
            get_linear_schedule_with_warmup(optimizer,
                                            curriculum.lr['warmup'],
                                            schdl_step,
                                            curriculum.lr['final_lr_factor'], last_epoch=last_step)
    else:
        scheduler = \
            get_linear_schedule_with_warmup(optimizer,
                                            curriculum.lr['warmup'],
                                            schdl_step,
                                            curriculum.lr['final_lr_factor'])       
    optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, clip)

    # tensorboard writers
    if musebert_version == 'v1':
        if with_program:
            writer_names = ['loss', 'o_bt', 'o_sub', 'p_hig', 'p_reg',
                        'p_deg', 'd_hlf', 'd_sqv', 'prog_type', 'prog_sub']
        else:
            writer_names = ['loss', 'o_bt', 'o_sub', 'p_hig', 'p_reg',
                        'p_deg', 'd_hlf', 'd_sqv']
    elif musebert_version == 'v2':
        writer_names = ['loss', 'o_t_1s', 'o_t_100ms', 'o_t_10ms', 'd_t_1s', 'd_t_100ms', 'd_t_10ms', \
            'p_hig', 'p_reg', 'p_deg', 'o_b_bt', 'o_b_sub', 'd_b_8bt', 'd_b_1bt', 'd_b_sub']
    tags = {'loss': None}
    summary_writers = SummaryWriters(writer_names, tags,
                                     log_path_mng.writer_path)

    # keyword training parameters
    beta_scheduler = ConstantScheduler(curriculum.beta)
    params_dic = dict(beta=beta_scheduler)
    param_scheduler = ParameterScheduler(**params_dic)

    # initialize the training interface
    musebert_train = \
        TrainMuseBERT(device, model, parallel, log_path_mng, data_loaders,
                      summary_writers, optimizer_scheduler,
                      param_scheduler, curriculum.lr['n_epoch'])

    # start training

    if stage == 'pretrain' or stage == 'finetune':
        if train_after_interrupt:
            musebert_train.run(start_epoch=last_epoch) # for pre-training/fine-tuning Training stage
        else:
            musebert_train.run()
    elif stage == 'inference':
        musebert_train.eval(task=task, mode=mode, stage=stage) # for inference


if __name__ == '__main__':

    ################## Pretraining or Fine-tuning MuseBERT #########################

    curriculum_dic = {
        'chord_extraction': chord_ft_curriculum,
        'pretrain': musebert_v2_pretrain_curriculum,#program_pretrain_curriculum, #all_curriculum,
        'beat_detection': beat_ft_curriculum,
        'arrangement': program_ft_curriculum,
        'beat_detection_v2': beat_ft_v2_curriculum
    }

    train_musebert(parallel=False, curriculum=curriculum_dic[task], \
        model_path=model_path[mode], task=task, mode=mode, stage=stage, \
        with_program=with_program, last_epoch=last_epoch, \
        train_after_interrupt=train_after_interrupt, musebert_version=musebert_version)