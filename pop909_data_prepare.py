from operator import index
from matplotlib import pyplot as plt
import numpy as np
from note_attribute_repr import onset_to_onset_attributes, dur_to_dur_attributes


def chord_event_to_pitch_attributes(pitch, type, shift):
  '''
  p_hig, p_reg, p_deg
  '''
  p_hig = 6
  p_reg = type
  p_deg = (pitch + shift) % 12

  return [p_hig, p_reg, p_deg]


def get_attributes(onset_arr, chord_event_arr, duration_arr):

  # print(onset_arr, chord_event_arr, duration_arr)

  atr_mat = np.zeros(7, dtype=np.int64)

  # output row [0: 2]: o_bt, o_sub
  atr_mat[0: 2] = onset_arr

  # output row [2: 5]: p_hig, p_reg, p_deg
  atr_mat[2: 5] = chord_event_arr

  # output row [5: 7]: d_hlf, d_sqv
  atr_mat[5:] = duration_arr

  return atr_mat


# R_fac := {o_bt, o_sub, p_hig, p_reg, p_deg, d_hlf, d_sqv}
def chord_per_beat_to_R_fac(onset, duration, root_note, chroma_note, bass_note, shift):

  # print(onset, duration, root_note, chroma_note, bass_note)

  r_fac = []

  onset_arr = onset_to_onset_attributes(onset)
  duration_arr = dur_to_dur_attributes(duration)

  r_fac.append(get_attributes(onset_arr, chord_event_to_pitch_attributes(root_note, 0, shift), duration_arr))
  for chroma in chroma_note:
    r_fac.append(get_attributes(onset_arr, chord_event_to_pitch_attributes(chroma, 1, shift), duration_arr))
  r_fac.append(get_attributes(onset_arr, chord_event_to_pitch_attributes(bass_note, 2, shift), duration_arr))

  return r_fac


def encode_chord_data_to_atr_mat(chord, number, shift):
  '''
  encode chord to nmat [onset, pitch, duration]

  input: 8 * 36; 8 for beats; 36 for root(12), chroma(12), bass(12)

  output: 8 * 36; 8 for beats; 36 for root(12), chroma(12), bass(12)
  '''
  item = chord[number]

  chord_r_fac = np.zeros((48, 7), dtype=np.int64)
  duration = 32 // len(item) # duration of a chord
  count = 0

  for one_beat, index in zip(item, range(len(item))):

    onset = index * 32 // len(item)

    root = one_beat[:12]
    chroma = one_beat[12:24]
    bass = one_beat[24:]
      
    root_note = np.where(root==1)[0][0]
    chroma_note = np.where(chroma==1)[0]
    bass_note = np.where(bass==1)[0][0]

    for event in chord_per_beat_to_R_fac(onset, duration, root_note, chroma_note, bass_note, shift):
      chord_r_fac[count] = event
      count+=1
    
  return chord_r_fac

# 暂时用不上
def merge_all_mel_and_acc_nmat(mel_nmat, acc_nmat, mel_nmat_len, acc_nmat_len):

  assert len(mel_nmat_len) == len(acc_nmat_len) 

  len_per_mel = len(mel_nmat) // len(mel_nmat_len)
  len_per_acc = len(acc_nmat) // len(acc_nmat_len)

  for i in range(len(mel_nmat_len)):

    non_zero_mel_len = mel_nmat_len[i]
    non_zero_acc_len = acc_nmat_len[i]

    print(non_zero_mel_len, non_zero_acc_len)
    
    non_zero_mel = mel_nmat[(len_per_mel * i) : (len_per_mel * i + len_per_mel)][ : non_zero_mel_len]
    non_zero_acc = acc_nmat[(len_per_acc * i) : (len_per_acc * i + len_per_acc)][ : non_zero_acc_len]

    zero_mel = mel_nmat[(len_per_mel * i) : (len_per_mel * i + len_per_mel)][non_zero_mel_len : ]
    zero_acc = acc_nmat[(len_per_acc * i) : (len_per_acc * i + len_per_acc)][non_zero_acc_len : ]

    assert sum(zero_mel).all() == 0
    assert sum(zero_acc).all() == 0

    merged_nmat = np.zeros((len_per_mel + len_per_acc, 3), dtype=np.int64)
    merged_nmat[0 : non_zero_mel_len , ] = non_zero_mel
    merged_nmat[non_zero_mel_len : (non_zero_mel_len + non_zero_acc_len) , ] = non_zero_acc

    print(merged_nmat)


def delete_invalid_nmat_for_acc_and_chord(acc_nmat, acc_nmat_len, c_nmat):

  assert len(acc_nmat_len) == len(c_nmat)

  len_per_acc = len(acc_nmat) // len(acc_nmat_len)

  new_acc_nmat_without_invalid = []
  new_acc_nmat_length = []
  new_chord_nmat_without_invalid = []

  for i in range(len(acc_nmat_len)):

    non_zero_acc_len = acc_nmat_len[i]
    
    if non_zero_acc_len <= 0:
      # print('<=0 : ',non_zero_acc_len)
      assert sum(c_nmat[i]).all() == 0
    else:
      for item in acc_nmat[(len_per_acc * i) : (len_per_acc * i + len_per_acc)]:
        new_acc_nmat_without_invalid.append(item)
    
      # pad acc_nmat from length 150 to 200
      for j in range(48):
        new_acc_nmat_without_invalid.append([0, 0, 0])

      new_acc_nmat_length.append(non_zero_acc_len)
      new_chord_nmat_without_invalid.append(c_nmat[i])

  assert len(new_acc_nmat_without_invalid) == (len_per_acc + 48) * len(new_chord_nmat_without_invalid)
  assert len(new_chord_nmat_without_invalid) == len(new_acc_nmat_length)

  return new_acc_nmat_without_invalid, new_acc_nmat_length, new_chord_nmat_without_invalid


def merge_with_chord_atr_mat(data_path, atr_mat, length, number, shift):

  chord = np.load(f'{data_path}')

  c_atr_mat = encode_chord_data_to_atr_mat(chord, number, shift)

  # merge atr_mat & c_atr_mat
  result = np.zeros((176 , 7), dtype=np.int64)    # 128(acc) + 48(chord) = 176
  result[:length] = atr_mat[:length]
  result[length:length+48] = c_atr_mat
  result[(length+48):] = atr_mat[length:(len(atr_mat)-48)]

  # calculate new length
  for c in c_atr_mat:
    if not sum(c) == 0:
      length += 1
      
  return result, length


def write_data_of_acc_and_chord_to_npy():
  '''
  val:7513 - train:58768
  '''

  data_path = '/gpfsnyu/scratch/kf2395/musebert/musebert/data/pop909-dataset'

  c_test = np.load(f'{data_path}/pop909_c_val.npy')
  c_train = np.load(f'{data_path}/pop909_c_train.npy')

  mel_train = np.load(f'{data_path}/pop909_mel_nmat_train.npy')
  mel_test = np.load(f'{data_path}/pop909_mel_nmat_val.npy')

  mel_train_len = np.load(f'{data_path}/pop909_mel_nmat_train_length.npy')
  mel_test_len = np.load(f'{data_path}/pop909_mel_nmat_val_length.npy')

  acc_train = np.load(f'{data_path}/pop909_acc_nmat_train.npy')
  acc_test = np.load(f'{data_path}/pop909_acc_nmat_val.npy')

  acc_train_len = np.load(f'{data_path}/pop909_acc_nmat_train_length.npy')
  acc_test_len = np.load(f'{data_path}/pop909_acc_nmat_val_length.npy')

  # melody：每32条表示一个2-bar， acc：每150条表示一个2-bar， 每条melody/acc数据为3*1: [onset, pitch, duration]
  # val:7513 - train:58768
  print(len(c_test), len(mel_test)/len(mel_test_len), len(mel_test_len), len(acc_test)/len(acc_test_len), len(acc_test_len))
  print(len(c_train), len(mel_train)/len(mel_train_len), len(mel_train_len), len(acc_train)/len(acc_train_len), len(acc_train_len))

  acc_test, acc_test_len, c_test = delete_invalid_nmat_for_acc_and_chord(acc_test, acc_test_len, c_test)

  # print(len(acc_test) // len(acc_test_len))

  # np.save('data/pop909-dataset/pop909_acc_test_without_invalid_data_pad_len.npy', acc_test)
  # np.save('data/pop909-dataset/pop909_acc_test_length_without_invalid_data.npy', acc_test_len)
  # np.save('data/pop909-dataset/pop909_chord_test_without_invalid_data.npy', c_test)

# write_data_of_acc_and_chord_to_npy()

def split_pop909():
  '''
  train : val = 88 : 12 约等于9 : 1
  因此将train set划分成train和validation set, 原来的val set作为test set, 80 : 10 : 10

  test:7509 - val:6527 - train:52209
  '''

  data_path = '/gpfsnyu/scratch/kf2395/musebert/musebert/data/pop909-dataset'

  c_train = np.load(f'{data_path}/pop909_chord_train_without_invalid_data.npy')
  acc_train = np.load(f'{data_path}/pop909_acc_train_without_invalid_data_pad_len_200.npy')
  acc_train_len = np.load(f'{data_path}/pop909_acc_train_length_without_invalid_data.npy')

  c_test = np.load(f'{data_path}/pop909_chord_test_without_invalid_data.npy')
  acc_test = np.load(f'{data_path}/pop909_acc_test_without_invalid_data_pad_len.npy')
  acc_test_len = np.load(f'{data_path}/pop909_acc_test_length_without_invalid_data.npy')

  print(len(c_train), len(acc_train)//len(acc_train_len), len(acc_train_len))

  item_length = len(acc_train)//len(acc_train_len)

  new_train_set_length = int(8 * len(acc_train_len) / 9)

  new_acc_train_len = acc_train_len[ : new_train_set_length]
  new_acc_train = acc_train[ : new_train_set_length * item_length]
  new_acc_valid_len = acc_train_len[new_train_set_length : ]
  new_acc_valid = acc_train[new_train_set_length * item_length : ]
  new_c_train = c_train[ : new_train_set_length]
  new_c_valid = c_train[new_train_set_length : ]

  print(len(new_c_train), len(new_acc_train), len(new_acc_train_len), len(new_acc_train) // len(new_acc_train_len))
  print(len(new_c_valid), len(new_acc_valid), len(new_acc_valid_len), len(new_acc_valid) // len(new_acc_valid_len))
  print(len(c_test), len(acc_test), len(acc_test_len), len(acc_test) // len(acc_test_len))

  # # train set
  # np.save('data/pop909-final/pop909_acc_train_final.npy', new_acc_train)
  # np.save('data/pop909-final/pop909_acc_train_length_final.npy', new_acc_train_len)
  # np.save('data/pop909-final/pop909_chord_train_final.npy', new_c_train)

  # # valid set
  # np.save('data/pop909-final/pop909_acc_valid_final.npy', new_acc_valid)
  # np.save('data/pop909-final/pop909_acc_valid_length_final.npy', new_acc_valid_len)
  # np.save('data/pop909-final/pop909_chord_valid_final.npy', new_c_valid)

  # test set
  np.save('data/pop909-final/pop909_acc_test_final.npy', acc_test)
  np.save('data/pop909-final/pop909_acc_test_length_final.npy', acc_test_len)
  np.save('data/pop909-final/pop909_chord_test_final.npy', c_test)


# split_pop909()

def truncate_data_to_specific_length(path, from_len, to_len):

  data = np.load(path)
  length = len(data) // from_len
  new_data = []
  
  for i in range(length):

    item = data[(i*from_len+to_len) : (i*from_len+from_len)]
    for line in item:
      assert sum(line)==0

    new_item = data[(i*from_len) : (i*from_len+to_len)]
    for line in new_item:
      new_data.append(list(line))
  
  path = path.replace('_final','_final_176')
  np.save(path, new_data)
  
  

def truncate():

  data_list = ['data/pop909-final/pop909_acc_train_final.npy', \
                'data/pop909-final/pop909_acc_valid_final.npy', \
                'data/pop909-final/pop909_acc_test_final.npy']
  
  from_len_list = [200, 200, 200]
  to_len_list = [176, 176, 176]
  
  for data, from_len, to_len in zip(data_list, from_len_list, to_len_list):
    truncate_data_to_specific_length(data, from_len, to_len)
  
# truncate()