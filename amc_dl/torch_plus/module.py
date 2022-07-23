from logging import root
import time
import os
from tkinter.ttk import LabeledScale
from typing_extensions import Self
import torch
from torch import nn
from .train_utils import epoch_time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import datetime
from utils import EarlyStopping
from musebert_config import is_early_stopping, patience
import datetime
from note_attribute_repr import pitch_attributes_to_pitch, time_attributes_to_time

class PytorchModel(nn.Module):

    def __init__(self, name, device):
        self.name = name
        super(PytorchModel, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')
        self.device = device

    def run(self, *input):
        """A general way to run the model.
        Usually tensor input -> tensor output"""
        raise NotImplementedError

    def loss(self, *input, **kwargs):
        """Call it during training. The output is loss and possibly others to
        display on tensorboard."""
        raise NotImplementedError

    def inference(self, *input):
        """Call it during inference.
        The output is usually numpy after argmax."""
        raise NotImplementedError

    def loss_function(self, *input):
        raise NotImplementedError

    def forward(self, mode, *input, **kwargs):
        if mode in ["run", 0]:
            return self.run(*input, **kwargs)
        elif mode in ['loss', 'train', 1]:
            return self.loss(*input, **kwargs)
        elif mode in ['inference', 'eval', 'val', 2]:
            return self.inference(*input, **kwargs)
        else:
            raise NotImplementedError

    def load_model(self, model_path, map_location=None):
        if map_location is None:
            map_location = self.device
        dic = torch.load(model_path, map_location=map_location)
        for name in list(dic.keys()):
            dic[name.replace('module.', '')] = dic.pop(name)
        self.load_state_dict(dic)
        self.to(self.device)

    @classmethod
    def init_model(cls, *inputs):
        raise NotImplementedError


class TrainingInterface:

    def __init__(self, device, model, parallel, log_path_mng, data_loaders,
                 summary_writers,
                 opt_scheduler, param_scheduler, n_epoch, **kwargs):
        self.model = model
        self.model.device = device
        if parallel:
            self.model = nn.DataParallel(self.model)
        self.model.to(device)
        self.path_mng = log_path_mng
        self.summary_writers = summary_writers
        self.data_loaders = data_loaders
        self.opt_scheduler = opt_scheduler
        self.param_scheduler = param_scheduler
        self.device = device
        self.n_epoch = n_epoch
        self.epoch = 0
        self.train_step = 0
        self.val_step = 0
        self.parallel = parallel
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def name(self):
        if self.parallel:
            return self.model.module.name
        else:
            return self.model.name

    @property
    def log_path(self):
        return self.path_mng.log_path

    @property
    def model_path(self):
        return self.path_mng.model_path

    @property
    def writer_path(self):
        return self.path_mng.writer_path

    @property
    def writer_names(self):
        return self.summary_writers.writer_names

    def _init_loss_dic(self):
        loss_dic = {}
        for key in self.writer_names:
            loss_dic[key] = 0.
        return loss_dic

    def _init_prediction_dic(self):
        pred_dic = {}
        for key in self.writer_names:
            pred_dic[key] = []
        return pred_dic

    def _init_label_dic(self):
        label_dic = {}
        for key in self.writer_names:
            label_dic[key] = []
        return label_dic

    def _accumulate_loss_dic(self, loss_dic, loss_items):
        assert len(self.writer_names) == len(loss_items)
        for key, val in zip(self.writer_names, loss_items):
            loss_dic[key] += val.item()
        return loss_dic

    def _accumulate_pred_and_label_dic(self, pred_dic, label_dic, pred_and_tgt_items):

        # writer_names 的第一项是‘loss’，后面7项分别对应7个attributes
        assert len(self.writer_names) == len(pred_and_tgt_items) + 1
        for key, val in zip(self.writer_names[1:], pred_and_tgt_items):
            for pred, label in zip(val[0], val[1]):
                pred_dic[key].append(pred)
                label_dic[key].append(label)    
        return pred_dic, label_dic

    def _write_loss_to_dic(self, loss_items):
        loss_dic = {}
        assert len(self.writer_names) == len(loss_items)
        for key, val in zip(self.writer_names, loss_items):
            loss_dic[key] = val.item()
        return loss_dic

    def _batch_to_inputs(self, batch):
        return self.data_loaders.batch_to_inputs(batch)

    def train(self, **kwargs):
        self.model.train()
        self.param_scheduler.train()
        epoch_loss_dic = self._init_loss_dic()

        for i, batch in enumerate(self.data_loaders.train_loader):
            inputs = self._batch_to_inputs(batch)
            self.opt_scheduler.optimizer_zero_grad()
            input_params = self.param_scheduler.step()
            outputs = self.model('train', *inputs, **input_params)
            outputs = self._sum_parallel_loss(outputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.opt_scheduler.clip)
            self.opt_scheduler.step()
            self._accumulate_loss_dic(epoch_loss_dic, outputs)
            batch_loss_dic = self._write_loss_to_dic(outputs)
            self.summary_writers.write_task('train', batch_loss_dic,
                                            self.train_step)
            self.train_step += 1
        return epoch_loss_dic

    def _sum_parallel_loss(self, loss):
        if self.parallel:
            if isinstance(loss, tuple):
                return tuple([x.mean() for x in loss])
            else:
                return loss.mean()
        else:
            return loss

    def eval(self, task='pretrain', mode='default', stage='pretrain'):
        self.model.eval()
        self.param_scheduler.eval()
        epoch_loss_dic = self._init_loss_dic()
        epoch_prediction_dic = self._init_prediction_dic()
        epoch_label_dic = self._init_label_dic()

        for i, batch in enumerate(self.data_loaders.val_loader):

            inputs = self._batch_to_inputs(batch)
            input_params = self.param_scheduler.step()
            with torch.no_grad():
                # get loss
                outputs = self.model('train', *inputs, **input_params) 
                outputs = self._sum_parallel_loss(outputs)

                if stage == 'inference':
                    # get reconstruction: prediction and targets
                    pred_and_tgt = self.model('inference', *inputs, **input_params) # kun: 'inference' 'val'

            self._accumulate_loss_dic(epoch_loss_dic, outputs)
            batch_loss_dic = self._write_loss_to_dic(outputs)
            self.summary_writers.write_task('val', batch_loss_dic,
                                            self.val_step)

            if stage == 'inference':
                self._accumulate_pred_and_label_dic(epoch_prediction_dic, epoch_label_dic, pred_and_tgt)

            self.val_step += 1
        

        def compute_confusion_matrix(y_pred, y_label, labels, img_path, title='Default Title'):

            cm = confusion_matrix(y_label, y_pred)
            print(f'\n\nClassification Report anf Confusion Matrix for {title}\n', classification_report(y_label, y_pred, digits = 5))
            np.set_printoptions(precision=2)
            print(cm)
            # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print(cm_normalized)
            plt.figure(figsize=(20, 16), dpi=1000)
            ind_array = np.arange(len(labels))
            x, y = np.meshgrid(ind_array, ind_array)
            
            for x_val, y_val in zip(x.flatten(), y.flatten()):
                c = cm[y_val][x_val]  # cm_normalized[y_val][x_val]
                if c >= 0:
                    plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=12, va='center', ha='center')
            
            # offset the tick
            tick_marks = np.array(range(len(labels))) + 0.5
            plt.gca().set_xticks(tick_marks, minor=True)
            plt.gca().set_yticks(tick_marks, minor=True)
            plt.gca().xaxis.set_ticks_position('none')
            plt.gca().yaxis.set_ticks_position('none')
            plt.grid(True, which='minor', linestyle='-')
            plt.gcf().subplots_adjust(bottom=0.15)

            plot_confusion_matrix(cm, labels)
            plt.savefig(img_path, format='png')
            plt.show()

        def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.binary):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            xlocations = np.array(range(len(labels)))
            plt.xticks(xlocations, labels)
            plt.yticks(xlocations, labels)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        if stage == 'inference':
            if task == 'arrangement':
                ctime = datetime.datetime.now().time().strftime("%H%M%S")
                
                type_img_path = f'cm_arrangemnet_{mode}-prog-type-{ctime}.png'
                sub_img_path = f'cm_arrangemnet_{mode}-prog-sub-{ctime}.png'
                program_img_path = f'cm_arrangemnet_{mode}-prog-program-{ctime}.png'

                prog_type_label = epoch_label_dic['prog_type']
                prog_type_pred = epoch_prediction_dic['prog_type']
                prog_type_labels = list(set(prog_type_label) | set(prog_type_pred))

                prog_sub_label = epoch_label_dic['prog_sub']
                prog_sub_pred = epoch_prediction_dic['prog_sub']
                prog_sub_labels = list(set(prog_sub_label) | set(prog_sub_pred))

                program_label = [(type*8 + sub) for type,sub in zip(epoch_label_dic['prog_type'], epoch_label_dic['prog_sub'])]
                program_pred = [(type*8 + sub) for type,sub in zip(epoch_prediction_dic['prog_type'], epoch_prediction_dic['prog_sub'])]
                program_labels = list(set(program_label) | set(program_pred))

                compute_confusion_matrix(prog_type_pred, prog_type_label, prog_type_labels, type_img_path, title='Arrangement - Type')
                compute_confusion_matrix(prog_sub_pred, prog_sub_label, prog_sub_labels, sub_img_path, title='Arrangement - Sub')
                compute_confusion_matrix(program_pred, program_label, program_labels, program_img_path, title='Arrangement - Program')

            elif task == 'chord_extraction':

                ctime = datetime.datetime.now().time().strftime("%H%M%S")
                root_img_path = f'cm_chord_extraction_{mode}-root-{ctime}.png'
                chroma_img_path = f'cm_chord_extraction_{mode}-chroma-{ctime}.png'
                bass_img_path = f'cm_chord_extraction_{mode}-bass-{ctime}.png'
                all_img_path = f'cm_chord_extraction_{mode}-all-{ctime}.png'

                all_label = epoch_label_dic['p_deg']
                all_pred = epoch_prediction_dic['p_deg']
                all_labels = list(set(all_label) | set(all_pred))
            
                root_label = [r for r in epoch_label_dic['p_deg'] if r<12]
                root_pred = [r for r in epoch_prediction_dic['p_deg'] if r<12]
                root_labels = list(set(root_label) | set(root_pred))

                chroma_label = [r-12 for r in epoch_label_dic['p_deg'] if r>=12 and r<24]
                chroma_pred = [r-12 for r in epoch_prediction_dic['p_deg'] if r>=12 and r<24]
                chroma_labels = list(set(chroma_label) | set(chroma_pred))
                
                bass_label = [r-24 for r in epoch_label_dic['p_deg'] if r>=24 and r<36]
                bass_pred = [r-24 for r in epoch_prediction_dic['p_deg'] if r>=24 and r<36]
                bass_labels = list(set(bass_label) | set(bass_pred))

                # calculate confusion matrix for root,chroma,bass respectively
                compute_confusion_matrix(root_pred, root_label, root_labels, root_img_path, 'Root')
                compute_confusion_matrix(chroma_pred, chroma_label, chroma_labels, chroma_img_path, 'Chroma')
                compute_confusion_matrix(bass_pred, bass_label, bass_labels, bass_img_path, 'Bass')
                compute_confusion_matrix(all_pred, all_label, all_labels, all_img_path, 'Root, Chroma and Bass')

            elif task == 'beat_detection':
                
                # # 计算confusion matrix没必要...
                # ctime = datetime.datetime.now().time().strftime("%H%M%S")
                # img_path = f'cm_{task}_{mode}-{ctime}.png'

                onset_label = epoch_label_dic['o_bt']
                onset_pred = epoch_prediction_dic['o_bt']
                # sorted_onset_pred = []
                # for i in range(len(onset_label) // 8):
                #     pred = onset_pred[i*8: (i+1)*8]
                #     pred.sort()
                #     for item in pred:
                #         sorted_onset_pred.append(item)
                # labels = list(set(onset_label) | set(onset_pred))

                # print(sorted_onset_pred)

                # compute_confusion_matrix(sorted_onset_pred, onset_label, labels, img_path, task)


                # 计算每个2-bar的准确率，ground truth=[0,4,8,12,16,20,24,28]
                error_count = 0
                for i in range(len(onset_label) // 8):
                    label = onset_label[i*8: (i+1)*8]
                    pred = onset_pred[i*8: (i+1)*8]
                    error_count += len(list(set(label) - set(pred)))
                
                sequence_accuracy = 1.0 - error_count / len(onset_label)
                print('Error count:', error_count)
                print('Total count:', len(onset_label))
                print('Sequence accuracy: {:.2%}'.format(sequence_accuracy))

            elif task == 'beat_detection_v2':

                ctime = datetime.datetime.now().time().strftime("%H%M%S")
                
                o_b_bt_img_path = f'cm_beat_detection_v2_{mode}-o_b_bt-{ctime}.png'
                o_b_sub_img_path = f'cm_beat_detection_v2_{mode}-o_b_sub-{ctime}.png'
                d_b_8bt_img_path = f'cm_beat_detection_v2_{mode}-d_b_8bt-{ctime}.png'
                d_b_1bt_img_path = f'cm_beat_detection_v2_{mode}-d_b_1bt-{ctime}.png'
                d_b_sub_img_path = f'cm_beat_detection_v2_{mode}-d_b_sub-{ctime}.png'
                onset_beat_img_path = f'cm_beat_detection_v2_{mode}-onset_beat-{ctime}.png'
                duration_beat_img_path = f'cm_beat_detection_v2_{mode}-duration_beat-{ctime}.png'
                onset_int_beat_img_path = f'cm_beat_detection_v2_{mode}-onset_int_beat-{ctime}.png'

                o_b_bt_label = epoch_label_dic['o_b_bt']
                o_b_bt_pred = epoch_prediction_dic['o_b_bt']
                o_b_bt_labels = list(set(o_b_bt_label) | set(o_b_bt_pred))

                o_b_sub_label = epoch_label_dic['o_b_sub']
                o_b_sub_pred = epoch_prediction_dic['o_b_sub']
                o_b_sub_labels = list(set(o_b_sub_label) | set(o_b_sub_pred))

                d_b_8bt_label = epoch_label_dic['d_b_8bt']
                d_b_8bt_pred = epoch_prediction_dic['d_b_8bt']
                d_b_8bt_labels = list(set(d_b_8bt_label) | set(d_b_8bt_pred))

                d_b_1bt_label = epoch_label_dic['d_b_1bt']
                d_b_1bt_pred = epoch_prediction_dic['d_b_1bt']
                d_b_1bt_labels = list(set(d_b_1bt_label) | set(d_b_1bt_pred))

                d_b_sub_label = epoch_label_dic['d_b_sub']
                d_b_sub_pred = epoch_prediction_dic['d_b_sub']
                d_b_sub_labels = list(set(d_b_sub_label) | set(d_b_sub_pred))

                onset_beat_label = [(bt*60 + sub) for bt,sub in zip(epoch_label_dic['o_b_bt'], epoch_label_dic['o_b_sub'])]
                onset_beat_pred = [(bt*60 + sub) for bt,sub in zip(epoch_prediction_dic['o_b_bt'], epoch_prediction_dic['o_b_sub'])]
                onset_beat_labels = list(set(onset_beat_label) | set(onset_beat_pred))

                duration_beat_label = [(bt_8*480 + bt_1*60 + sub) for bt_8,bt_1,sub in zip(epoch_label_dic['d_b_8bt'], epoch_label_dic['d_b_1bt'], epoch_label_dic['d_b_sub'])]
                duration_beat_pred = [(bt_8*480 + bt_1*60 + sub) for bt_8,bt_1,sub in zip(epoch_label_dic['d_b_8bt'], epoch_prediction_dic['d_b_1bt'], epoch_prediction_dic['d_b_sub'])]
                duration_beat_labels = list(set(duration_beat_label) | set(duration_beat_pred))

                # round the onset beat to the nearest integer
                onset_int_beat_label = []
                onset_int_beat_pred = []
                for bt_label, bt_pred, sub_label, sub_pred in zip(o_b_bt_label, o_b_bt_pred, o_b_sub_label, o_b_sub_pred):
                    if sub_label == 0:
                        onset_int_beat_label.append(bt_label)
                        if sub_pred < 30:
                            onset_int_beat_pred.append(bt_pred)
                        else:
                            onset_int_beat_pred.append(bt_pred+1)
                max_beat = max(onset_int_beat_label)
                onset_int_beat_label_len = len(set(onset_int_beat_label))
                onset_int_beat_labels = list(set(onset_int_beat_label) | set(onset_int_beat_pred))
                print('There are', max_beat - onset_int_beat_label_len, 'beats which do not have note event')

                # print the result in the .o file and plot the confusion matrix
                compute_confusion_matrix(o_b_bt_pred, o_b_bt_label, o_b_bt_labels, o_b_bt_img_path, title='Beat Detection v2 - o_b_bt')
                compute_confusion_matrix(o_b_sub_pred, o_b_sub_label, o_b_sub_labels, o_b_sub_img_path, title='Beat Detection v2 - o_b_sub')
                compute_confusion_matrix(d_b_8bt_pred, d_b_8bt_label, d_b_8bt_labels, d_b_8bt_img_path, title='Beat Detection v2 - d_b_8bt')
                compute_confusion_matrix(d_b_1bt_pred, d_b_1bt_label, d_b_1bt_labels, d_b_1bt_img_path, title='Beat Detection v2 - d_b_1bt')
                compute_confusion_matrix(d_b_sub_pred, d_b_sub_label, d_b_sub_labels, d_b_sub_img_path, title='Beat Detection v2 - d_b_sub')
                compute_confusion_matrix(onset_beat_pred, onset_beat_label, onset_beat_labels, onset_beat_img_path, title='Beat Detection v2 - onset_beat')
                compute_confusion_matrix(duration_beat_pred, duration_beat_label, duration_beat_labels, duration_beat_img_path, title='Beat Detection v2 - duration_beat')
                compute_confusion_matrix(onset_int_beat_pred, onset_int_beat_label, onset_int_beat_labels, onset_int_beat_img_path, title='Beat Detection v2 - onset int beat')

                print('\n')
                print(len(epoch_label_dic['o_t_1s']), len(epoch_label_dic['o_t_100ms']), len(epoch_label_dic['o_t_10ms']), \
                    len(epoch_label_dic['d_t_1s']), len(epoch_label_dic['d_t_1s']), len(epoch_label_dic['d_t_1s']), \
                    len(epoch_label_dic['p_hig']), len(epoch_label_dic['p_reg']), len(epoch_label_dic['p_deg']))

                onset_time_label = time_attributes_to_time(epoch_label_dic['o_t_1s'], epoch_label_dic['o_t_100ms'], epoch_label_dic['o_t_10ms'])
                duration_time_label = time_attributes_to_time(epoch_label_dic['d_t_1s'], epoch_label_dic['d_t_100ms'], epoch_label_dic['d_t_10ms'])
                pitch_label = pitch_attributes_to_pitch(epoch_label_dic['p_hig'], epoch_label_dic['p_reg'], epoch_label_dic['p_deg'])
                
                print(len(onset_time_label), len(duration_time_label), len(pitch_label), len(onset_beat_label), len(onset_beat_pred))
                print('\n')

                inference_output_file = open(f'inf_output_{ctime}.txt', 'w+')
                for d_t, p, b_label, b_pred in zip(duration_time_label, pitch_label, onset_beat_label, onset_beat_pred):
                    print(d_t, p, b_label, b_pred)
                    inference_output_file.writelines([d_t, p, b_label,b_pred])
                inference_output_file.close()

        return epoch_loss_dic

    def save_model(self, fn):
        if self.parallel:
            torch.save(self.model.module.state_dict(), fn)
        else:
            torch.save(self.model.state_dict(), fn)

    def epoch_report(self, start_time, end_time, train_loss, valid_loss):
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {self.epoch + 1:02} | '
              f'Time: {epoch_mins}m {epoch_secs}s',
              flush=True)
        print(
            f'\tTrain Loss: {train_loss:.5f}', flush=True)
        print(
            f'\t Valid. Loss: {valid_loss:.5f}', flush=True)

    def run(self, start_epoch=0, start_train_step=0, start_val_step=0):
        self.epoch = start_epoch
        self.train_step = start_train_step
        self.val_step = start_val_step
        best_valid_loss = float('inf')
        if is_early_stopping:
            early_stopping = EarlyStopping(patience, verbose=True)

        for i in range(self.n_epoch):
            start_time = time.time()
            train_loss = self.train()['loss'] / \
                         len(self.data_loaders.train_loader)
            val_loss = self.eval()['loss'] / \
                       len(self.data_loaders.val_loader)
            end_time = time.time()
            self.save_model(self.path_mng.epoch_model_path(self.name))
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                self.save_model(self.path_mng.valid_model_path(self.name))
            self.epoch_report(start_time, end_time, train_loss, val_loss)
            self.epoch += 1
        
            if is_early_stopping:
                date = str(datetime.date.today())
                ctime = datetime.datetime.now().time().strftime("%H%M%S")
                path = date + '-' + ctime
                early_stopping(val_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        self.save_model(self.path_mng.final_model_path(self.name))
        print('Model saved.')




