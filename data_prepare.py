import numpy as np
import pretty_midi as pm
import mirdata
import os
import matplotlib.pyplot as plt
import mido
from tensorboard import program


NMAT_LENGTH = 32 # 每个两小节加上8个beat tokens, 这里的32不包含beat length 8 
NMAT_LENGTH_WITH_PROGRAM = 307 # 暂时不包含beat
BPM = 120

pm.pretty_midi.MAX_TICK = 1e10
rwc_midi_path = '/gpfsnyu/scratch/kf2395/RWC100/AIST.RWC-MDB-P-2001.SMF_SYNC'
rwc_beat_path = '/gpfsnyu/scratch/kf2395/RWC100/AIST.RWC-MDB-P-2001.BEAT'
melody_name_list = ['MELODY','VOCAL','MELO','Melody','MELODY 1','Melo','melody','MELODY ','Melody           ','Vocal.1','Vocal.2']

# darklist
not_in_4_4_complex = [6, 22, 28, 37, 38, 39, 43, 50, 62, 71, 72, 76, 88, 99] # 只取其中4/4拍的片段；index=38只要第一段4/4拍的
not_in_4_4_delete = [12, 34, 70, 89, 100] # 完全不是4/4拍的文件
invalid_file_list = [23, 64, 77] # 无效的空文件

INFINITE_TIME = 1000000

def midi_to_nmat_with_program(fn, with_program, bpm, mode):
    alpha = 60 / bpm
    midi = pm.PrettyMIDI(fn)

    ######################### Get all 4/4 segments ##########################

    # print('\n')
    cut_dur = 0
    start_time_list_of_4_4 = []
    end_time_list_of_4_4 = []
    time_sig_change = midi.time_signature_changes

    for t in time_sig_change:
      # print(t)
      if t.numerator == 4 and t.denominator == 4:
        cut_dur = t.time
        start_time_list_of_4_4.append(t.time)
      else:
        if len(start_time_list_of_4_4) > len(end_time_list_of_4_4):
          end_time_list_of_4_4.append(t.time)

    if len(start_time_list_of_4_4) == len(end_time_list_of_4_4) + 1:
      end_time_list_of_4_4.append(INFINITE_TIME)

    time_range_of_4_4 = []
    for start,end in zip(start_time_list_of_4_4,end_time_list_of_4_4):
      time_range_of_4_4.append([start, end])

    # print(time_range_of_4_4)

    ####################################################################

    instruments = midi.instruments

    melody_nmat = []
    melody_nmat_length = []
    program_list = []
    two_bar_index_list = []

    # print('\nA new song begins...')
    
    for instrument in instruments:

      notes = instrument.notes
      program = instrument.program

      name = instrument.name
      name = name.replace(" ", "")

      if mode=='melody':

        # judge if it is melody, when mode='melody'
        if len(name) >= 3 and (name[0:3].lower() == 'mel' or name[0:3].lower()== 'voc'):

          piano_rolls = midi_to_mel_pianoroll(notes, bpm, cut_dur, time_range_of_4_4)
          # piano_rolls = piano_rolls.reshape((-1, 32, 130))
          for pr in piano_rolls:          

            notes_of_2_bars, nmat_of_2_bars, nmat_of_2_bars_length = mel_pianoroll_to_notes_and_nmat(pr, bpm, True)
            melody_nmat += nmat_of_2_bars

            if not nmat_of_2_bars_length==0:
              melody_nmat_length.append(nmat_of_2_bars_length)

      elif mode=='all':

        piano_rolls = midi_to_mel_pianoroll(notes, bpm, cut_dur, time_range_of_4_4)

        # index==0表示第0个2-bar
        for index, pr in enumerate(piano_rolls):     

          notes_of_2_bars, nmat_of_2_bars, nmat_of_2_bars_length = mel_pianoroll_to_notes_and_nmat(pr, bpm, False)
          
          if not nmat_of_2_bars_length==0:
            program_list.append(program)
            two_bar_index_list.append(index)
            melody_nmat_length.append(nmat_of_2_bars_length)
            melody_nmat += nmat_of_2_bars

    # print(max(melody_nmat_length))

    return melody_nmat, melody_nmat_length, program_list, two_bar_index_list


def get_beat(dataset, beat_file):
  '''
    onset, pitch = 8, duration = 0, program = 0
  '''
  beat_data = dataset.load_beats(beat_file)
  return beat_data.times


def one_of_time_range_of_4_4(start, time_range_of_4_4):
  """
  判断某音符处于所有4/4拍区间中的哪一个
  """

  for range in time_range_of_4_4:
    s = range[0]
    e = range[1]
    if start >= s and start <= e:
      return range
  
  return []


def midi_to_mel_pianoroll(notes, bpm, cut_dur, time_range_of_4_4):
    alpha = 60 / bpm
    end_time = np.ceil(max([n.end for n in notes]) / (8 * alpha))
    pr = np.zeros((int(end_time * 32), 130))
    pr[:, -1] = 1
    
    for n in notes:
        
        if len(one_of_time_range_of_4_4(n.start, time_range_of_4_4)) > 0:

          range = one_of_time_range_of_4_4(n.start, time_range_of_4_4)
          cut_dur = range[0]

          s = (n.start - cut_dur) / (alpha / 4)
          e = (n.end - cut_dur) / (alpha / 4)
          p = n.pitch

          pr[int(s), int(p)] = 1
          pr[int(s) + 1: int(e) + 1, 128] = 1
          pr[int(s): int(e) + 1, -1] = 0
    
    pr = pr.reshape((-1, 32, 130))

    return pr

def mel_pianoroll_to_prmat(pr):
    '''
      input: (32,130)
    '''
    steps = pr.shape[0] # 32
    pr = pr.argmax(axis=-1) # 找到值为1的pitch
    prmat = np.zeros((steps, 128))
    dur = 0
    for i in range(steps - 1, -1, -1):
        if pr[i] == 128:
            dur += 1
        elif pr[i] < 128:
            prmat[i, int(pr[i])] = dur + 1
            dur = 0
        else:
            dur = 0
    return prmat

# 数据增强，shift为要移调的个数，shift取值为1-11，输入一个（32，130）的pianoroll
def augment_mel_pianoroll(pr, shift=0):
    pitch_part = np.roll(pr[:, 0: 128], shift, axis=-1)
    control_part = pr[:, 128:]
    augmented_pr = np.concatenate([pitch_part, control_part], axis=-1)
    return augmented_pr

def prmat_to_notes_and_nmat(prmat, bpm, with_beat, begin=0., vel=100, nmat_length=NMAT_LENGTH):
    steps = prmat.shape[0]
    alpha = 0.25 * 60 / bpm
    notes = []
    nmat = []
    nmat_count = 0
    nmat_non_zero_length = 0

    for t in range(steps):
        for p in range(128):
            
            if prmat[t, p] >= 1:
                s = alpha * t + begin
                e = alpha * (t + prmat[t, p]) + begin
                notes.append(pm.Note(int(vel), int(p), s, e))

                try:
                  assert (int(p) >=24 and int(p) < 108)
                except:
                  # print('pitch out of range(24, 108):',int(p))
                  pass

                nmat.append([t, p, int(prmat[t,p])]) # (onset, pitch, duration)
                nmat_count += 1
    
    nmat_non_zero_length = nmat_count
    if nmat_count == 0:
      return [], [], 0

    if with_beat:
      # add beat tokens, onset = 0,4,8,12,16,20,24,28; pitch = 127; duration = 0
      for beat in range(8):
        nmat.append([beat*4, 127, 0])

    while nmat_count < nmat_length:
      nmat.append([0, 0, 0])
      nmat_count += 1

    if with_beat:
      nmat_non_zero_length += 8
    
    return notes, nmat, nmat_non_zero_length


def mel_pianoroll_to_notes_and_nmat(pr, bpm=BPM, with_beat='True', begin=0., vel=100): # bpm=80
    prmat = mel_pianoroll_to_prmat(pr)
    notes, nmat, nmat_length = prmat_to_notes_and_nmat(prmat, bpm, with_beat, begin, vel)
    return notes, nmat, nmat_length


def split_rwc100(rwc_midi_path):
  '''
    for rwc_100
    8:1:1
  '''
  train = []
  validation = []
  test = []

  # for file_number in range(1,2):
  for file_number in range(1,101):
    str_file_name = str(file_number).zfill(3)
    file_prefix = 'RM-P' + str_file_name

    if  file_number not in not_in_4_4_delete and file_number not in invalid_file_list:
      if file_number <= 80:
        train.append(file_prefix)
      elif file_number <= 90:
        validation.append(file_prefix)
      else:
        test.append(file_prefix)

  return train, validation, test


def initialize_dataset_using_mirdata(dataset_path):
  '''
  Initialize the dataset`using mirdata and get beat info
  '''
  if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
  dataset = mirdata.initialize('rwc_popular', dataset_path)


def get_bpm(beat_path):
  '''
  这里get到的是wav的beat,和midi的beat是一致的
  '''

  bpm = 0
  sum_delta = 0
  beat0 = 0
  count = 0
  with open(beat_path) as file:
    for item in file:
      item = int(item.replace('\n','').split('\t')[0])/100
      if count == 0:
        beat0 = item
      else:
        sum_delta += (item - beat0)
        # print('delta:', item, beat0, item - beat0)
        beat0 = item
      count+=1
  
  bpm = round(60 / (sum_delta/(count-1)))

  # print('get_bpm: ', bpm)

  return bpm


def merge_all_programs(with_beat, nmat_item, nmat_length_item, program_list, two_bar_index_list):
  """
  将处于同一个2-bar片段的不同乐器的nmat合并在一起
  two_bar_index_list: 每一个nmat片段属于整首歌曲的第几个2-bar
  """

  assert (len(nmat_item)//NMAT_LENGTH==len(nmat_length_item))
  assert (len(nmat_item)//NMAT_LENGTH==len(program_list))
  assert (len(nmat_item)//NMAT_LENGTH==len(two_bar_index_list))

  two_bar_index_set = set(two_bar_index_list)

  nmat_dictionary = dict.fromkeys(list(two_bar_index_set))
  nmat_length_dictionary = dict.fromkeys(list(two_bar_index_set), 0)

  for index in range(len(nmat_length_item)):

    nmat = nmat_item[index*NMAT_LENGTH : (index+1)*NMAT_LENGTH] # (32, 3)
    length = nmat_length_item[index]
    program = program_list[index]
    two_bar_index = two_bar_index_list[index]

    if nmat_dictionary[two_bar_index] == None:
      nmat_dictionary[two_bar_index] = []

    for n in nmat:
      if not sum(n) == 0:
        nmat_dictionary[two_bar_index].append(n)  # add program info to nmat

    nmat_length_dictionary[two_bar_index] += length

  # 计算整首歌曲里最长的2-bar的长度
  max_length = max(zip(nmat_length_dictionary.values()))[0]
  # print(max_length)

  if with_beat:
    nmat_item = np.zeros((len(list(two_bar_index_set))*(NMAT_LENGTH_WITH_PROGRAM+8), 3),dtype=int)
  else:
    nmat_item = np.zeros((len(list(two_bar_index_set))*NMAT_LENGTH_WITH_PROGRAM, 3),dtype=int)
  nmat_length_item = np.zeros(len(list(two_bar_index_set)), dtype=int)

  for index, i in enumerate(list(two_bar_index_set)):

    if with_beat:
      nmat_item[index*(NMAT_LENGTH_WITH_PROGRAM+8): index*(NMAT_LENGTH_WITH_PROGRAM+8)+nmat_length_dictionary[i]] \
                = nmat_dictionary[i]
      nmat_item[index*(NMAT_LENGTH_WITH_PROGRAM+8)+nmat_length_dictionary[i] : index*(NMAT_LENGTH_WITH_PROGRAM+8)+nmat_length_dictionary[i]+8] \
                = [[0, 127, 0], [4, 127, 0], [8, 127, 0], [12, 127, 0], \
                  [16, 127, 0], [20, 127, 0], [24, 127, 0], [28, 127, 0]]
      nmat_length_item[index] = nmat_length_dictionary[i] + 8
    else:
      nmat_item[index*NMAT_LENGTH_WITH_PROGRAM: index*NMAT_LENGTH_WITH_PROGRAM+nmat_length_dictionary[i]] \
                = nmat_dictionary[i]
      nmat_length_item[index] = nmat_length_dictionary[i]
    
  return nmat_item, nmat_length_item, max_length


def convert_data_into_musebert_input_for_a_split(splits, split, mode='melody', with_beat=False):
  
  nmat = []
  nmat_length = []

  nmat_with_program = np.zeros((0, 3), dtype=int)
  nmat_length_with_program = np.zeros(0, dtype=int)

  for file_prefix in splits[split]:

    file_name = file_prefix + '.SMF_SYNC.MID'
    file_path = os.path.join(rwc_midi_path, file_name)

    beat_file_name = file_name.split('.')[0] + '.BEAT.TXT'
    beat_path = os.path.join(rwc_beat_path, beat_file_name)
    # beats = get_beat(dataset, beat_path)
    bpm = get_bpm(beat_path)
    # print('bpm=',bpm)

    nmat_item, nmat_length_item, program_list, two_bar_index_list = midi_to_nmat_with_program(file_path, False, bpm, mode)
    if mode=='all':
      nmat_item, nmat_length_item, max_length = merge_all_programs(with_beat, nmat_item, nmat_length_item, program_list, two_bar_index_list)
      
      nmat_with_program = np.concatenate((nmat_with_program,nmat_item), axis=0)
      nmat_length_with_program = np.concatenate((nmat_length_with_program,nmat_length_item), axis=0)

    else:
      nmat += nmat_item
      nmat_length += nmat_length_item
  
  if mode=='all':
    return nmat_with_program, nmat_length_with_program
  
  return nmat, nmat_length


def rwc100_data_preprocess(mode='melody', with_beat=False):

  # split train set
  train, validation, test = split_rwc100(rwc_midi_path)
  splits = {
    'train': train,
    'validation': validation,
    'test': test
  }

  print(splits)

  melody_nmat_train, train_length_list = convert_data_into_musebert_input_for_a_split(splits, 'train', mode, with_beat)
  melody_nmat_validation, validation_length_list = convert_data_into_musebert_input_for_a_split(splits, 'validation', mode, with_beat)
  melody_nmat_test, test_length_list = convert_data_into_musebert_input_for_a_split(splits, 'test', mode, with_beat)

  if mode == 'melody':
    assert len(melody_nmat_train) == (NMAT_LENGTH+8)*len(train_length_list)
    assert len(melody_nmat_validation) == (NMAT_LENGTH+8)*len(validation_length_list)
    assert len(melody_nmat_test) == (NMAT_LENGTH+8)*len(test_length_list)
  elif mode == 'all':
    if with_beat:
      assert len(melody_nmat_train) == (NMAT_LENGTH_WITH_PROGRAM+8)*len(train_length_list)
      assert len(melody_nmat_validation) == (NMAT_LENGTH_WITH_PROGRAM+8)*len(validation_length_list)
      assert len(melody_nmat_test) == (NMAT_LENGTH_WITH_PROGRAM+8)*len(test_length_list)
    else:
      assert len(melody_nmat_train) == (NMAT_LENGTH_WITH_PROGRAM)*len(train_length_list)
      assert len(melody_nmat_validation) == (NMAT_LENGTH_WITH_PROGRAM)*len(validation_length_list)
      assert len(melody_nmat_test) == (NMAT_LENGTH_WITH_PROGRAM)*len(test_length_list)

  # rwc without_program / with_program_no_beat / with_program_and_beat
  print(len(melody_nmat_train)) # 104040 / 1169056 / 1199520
  print(len(train_length_list)) # 2890 / 3808 / 3808
  print(len(melody_nmat_validation)) # 13968 / 156570 / 160650
  print(len(validation_length_list)) # 388 / 510 / 510
  print(len(melody_nmat_test)) # 12348 / 151965 / 155925
  print(len(test_length_list)) # 343 / 495 / 495

  if mode == 'all':
    np.save('rwc_data/no_program_but_with_beat/beat_rwc_train.npy', melody_nmat_train)
    np.save('rwc_data/no_program_but_with_beat/beat_rwc_train_length.npy', train_length_list)
    np.save('rwc_data/no_program_but_with_beat/beat_rwc_validation.npy', melody_nmat_validation)
    np.save('rwc_data/no_program_but_with_beat/beat_rwc_validation_length.npy', validation_length_list)
    np.save('rwc_data/no_program_but_with_beat/beat_rwc_test.npy', melody_nmat_test)
    np.save('rwc_data/no_program_but_with_beat/beat_rwc_test_length.npy', test_length_list)
  
  # elif mode == 'melody':
  #   np.save('rwc_data/with_beat/beat_rwc_train.npy', melody_nmat_train)
  #   np.save('rwc_data/with_beat/beat_rwc_train_length.npy', train_length_list)
  #   np.save('rwc_data/with_beat/beat_rwc_validation.npy', melody_nmat_validation)
  #   np.save('rwc_data/with_beat/beat_rwc_validation_length.npy', validation_length_list)
  #   np.save('rwc_data/with_beat/beat_rwc_test.npy', melody_nmat_test)
  #   np.save('rwc_data/with_beat/beat_rwc_test_length.npy', test_length_list)


rwc100_data_preprocess('all',True)