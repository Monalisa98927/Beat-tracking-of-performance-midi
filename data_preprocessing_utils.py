import csv
import pretty_midi
import numpy as np

def read_midi_file(file_path):

    midi_data = pretty_midi.PrettyMIDI(file_path)
    return midi_data

def load_musicnet_csv(csv_file):
    """
    Loads a CSV file containing the MusicNet data.
    """
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def get_note_events_windows(file_path, beat_num_range):
    """
    get the note index windows for each music piece
    to ensure that each window can contain 8 - 12 beats
    """
    data = load_musicnet_csv(file_path)
    beat_list = []
    beat_dic = {}
    windows = []

    # get all the integer beats
    for index,row in enumerate(data[1:]):
        o_b = float(row[4])

        # 这里取0.1表示对某个范围内的onset beat容错，例如对于一个onset beat=8.03的note event，我们可以将其当作在第8个beat上
        tolerance = 0.10
        if abs(o_b - int(o_b)) < tolerance:
            if int(o_b) not in beat_list:
                beat_list.append(int(o_b))
                beat_dic[int(o_b)] = index - 1

    for i,start_beat in enumerate([beat for beat in beat_list if beat <= (max(beat_list) - 12 + 1)]):

        end_beat_candidates = [(beat_num + start_beat) for beat_num in beat_num_range if (beat_num + start_beat) in beat_list]
        
        # 随机取8-12个beat
        if len(end_beat_candidates) != 0:

            # 最后一个window，固定取12个beat，保证所有note events都可以涵盖到
            if start_beat == (max(beat_list) - max(beat_num_range) + 1):
                end_beat = max(beat_num_range)
            else:
                end_beat = np.random.choice(end_beat_candidates)
            windows.append([start_beat, end_beat])

        # 如果某个beat之后有较多空拍，则只能取小于8个beats，但是还是尽可能多取beats
        else:
            short_beat_num_range = range(0,min(beat_num_range))
            end_beat_candidates = [(beat_num + start_beat) for beat_num in short_beat_num_range if (beat_num + start_beat) in beat_list]
            end_beat = max(end_beat_candidates)
            windows.append([start_beat, end_beat])

    return windows, beat_dic

def write_nmat_txt(file_path, nmat, nmat_length, nmat_windows):
    '''
    write nmat, nmat_length, nmat_windows for each music piece into .txt file
    '''
    # write nmat for each music piece into .txt file
    output_nmat_path = file_path.replace('.csv', '_nmat.txt')
    output_nmat_len_path = file_path.replace('.csv', '_nmat_len.txt')
    output_nmat_windows_path = file_path.replace('.csv', '_nmat_win.txt')

    nmat_file = open(output_nmat_path, 'w+')
    nmat_len_file = open(output_nmat_len_path, 'w+')
    nmat_win_file = open(output_nmat_windows_path, 'w+')

    nmat_file.writelines([str(row) + '\n' for row in nmat])
    nmat_len_file.writelines([str(length) + '\n' for length in nmat_length])
    nmat_win_file.writelines([str(win) + '\n' for win in nmat_windows])

    nmat_file.close()
    nmat_len_file.close()
    nmat_win_file.close()

def absolute_time_to_relative_time(pad_length, nmat, nmat_length):
    '''
    calculate the relative onset time by letting each o_t minus the minimum of o_t in the corresponding segment
    finally round all the onset_time and duration_time to integers
    '''

    for i in range(len(nmat_length)):
        current_segment = nmat[i * pad_length: (i * pad_length + nmat_length[i])]
        
        # find the minimum onset time of a segment
        min_o_t = 1000000000
        for row in current_segment:
            min_o_t = min(min_o_t, row[0])

        # calculate the relative onset time by letting each o_t minus min_o_t
        for row in current_segment:
            row[0] -= min_o_t
            row[0] = round(row[0])
            row[1] = round(row[1])

    return nmat, nmat_length

def calculate_nmat_of_random_beats(dataset, file_path, pad_length, beat_num_range=range(8,13), length_range = [6, 400], sample_rate=44100, time_granularity=100, beat_granularity=60):
    """
    Calculates note matrix of each song in the MusicNet dataset.
    将所有note events分成若干个8-12 beat

    1. force the lower bound of duration_time/duration_beat to be 1 
    (to ensure all the duration more than 0...)
    2. for schubert_wanderer_fantasie which has segements longer than 700 note events, 
    we force these beat window length <= 4 (or just split them into 2)
    3. for the length of segements lower than 5, delete them...
    """

    nmat = []
    nmat_windows = []
    nmat_length = []
    nmat_rows_for_whole_song = []

    data = load_musicnet_csv(file_path)
    windows, beat_dic = get_note_events_windows(file_path, beat_num_range)

    print(file_path, len(windows))

    for row in data[1:]:

        if dataset == 'asap':

            o_t = float(row[0])
            d_t = float(row[1]) - o_t
            pitch = int(row[3])
            o_b = float(row[4])
            d_b = float(row[5])
            o_b_pos = int(row[6])
            o_b_neg = int(row[7])
            o_subdiv = float(row[8])

            nmat_row = [o_t, d_t, pitch, o_b, o_b_pos, o_b_neg, o_subdiv, d_b]
        
        elif dataset == 'musicnet':

            o_t = int(row[0])
            d_t = int(row[1]) - o_t
            pitch = int(row[3])
            o_b = float(row[4])
            d_b = float(row[5])

            nmat_row = [(o_t*time_granularity)//sample_rate, (d_t*time_granularity)//sample_rate, \
                        pitch, o_b, d_b]
        
        nmat_rows_for_whole_song.append(nmat_row)
        
    for window in windows:

        start_beat = window[0]
        end_beat = window[1]

        current_nmat_length = 0
        current_segment_nmat = []

        start_exist = False
        end_exist = False
        
        for index, row in enumerate(data[1:]):
            o_t = float(row[0])
            o_b = float(row[4])
            d_b = float(row[5])
            
            # 1. for each beat window, append all the note events whose onset_time is within the beat window
            #
            # 2. for the note events which is on the first or the last beat of the beat window,
            # randomly remove some of these note events (namely 掐头去尾)
            if o_b >= start_beat and o_b < end_beat:

                if_delete_note = False
                note_position = 'mid'

                # judge if a note event is at the beginning or the end of the current segment
                if abs(o_b - start_beat) < 1.0:
                    note_position = 'start'
                    if start_exist:
                        if np.random.rand() <= 0.5:
                            if_delete_note = True
                    else:
                        start_exist = True

                elif abs(o_b - end_beat) < 1.0:
                    note_position = 'end'
                    if end_exist:
                        if np.random.rand() <= 0.5:
                            if_delete_note = True
                    else:
                        end_exist = True

                if dataset == 'asap':
                    o_t = nmat_rows_for_whole_song[index][0] * time_granularity
                    d_t = nmat_rows_for_whole_song[index][1] * time_granularity
                    pitch = nmat_rows_for_whole_song[index][2]
                    o_b = nmat_rows_for_whole_song[index][3]
                    o_b_pos = nmat_rows_for_whole_song[index][4]
                    o_b_neg = nmat_rows_for_whole_song[index][5]
                    o_b_subdiv = round(beat_granularity * (nmat_rows_for_whole_song[index][6]))
                    d_b = round(beat_granularity * (nmat_rows_for_whole_song[index][7]))

                    # ensure d_t and d_b > 0 (point 1)
                    d_t = d_t if d_t > 0 else 1
                    d_b = d_b if d_b > 0 else 1
                    
                    new_nmat_row = [o_t, d_t, pitch, o_b, o_b_pos, o_b_neg, o_b_subdiv, d_b]
                
                elif dataset == 'musicnet':
                    new_nmat_row = [nmat_rows_for_whole_song[index][0], 
                                    nmat_rows_for_whole_song[index][1], 
                                    nmat_rows_for_whole_song[index][2],
                                    round(beat_granularity * (nmat_rows_for_whole_song[index][3] - start_beat)),
                                    round(beat_granularity * (nmat_rows_for_whole_song[index][4]))]

                if if_delete_note == False:

                    # store nmat for current segment, 
                    # once ensure its length is [6, 400], 
                    # concatenate it into nmat of the whole song
                    current_segment_nmat.append(new_nmat_row)
                    current_nmat_length += 1

        # check the length of the current segment, 
        # if legal (within length range), concatenate it to nmat of the whole song
        # if the segment is too long, split it into two pieces by the middle beat (point 2)
        # if the segment is too short, ignore it (point 3)
        if current_nmat_length >= length_range[0] and current_nmat_length <= length_range[1]:

            nmat_length.append(current_nmat_length)
            nmat_windows.append(window)
            nmat += current_segment_nmat # concatenate the current segment to the whole song

            # pad nmat to pad_length
            if current_nmat_length < pad_length:
                for i in range(pad_length - current_nmat_length):
                    nmat_row = [0, 0, 0, 0, 0, 0, 0, 0]
                    nmat.append(nmat_row)
        
        elif current_nmat_length > length_range[1]:

            # split the extremely long segments into two by the middle beat value
            mid_beat = (window[1] + window[0]) // 2

            first_split = []
            second_split = []
            first_split_len = 0
            second_split_len = 0

            for row in current_segment_nmat:
                o_b = row[3]
                if o_b < mid_beat:
                    first_split.append(row)
                    first_split_len += 1
                else:
                    second_split.append(row)
                    second_split_len += 1
                
            nmat_length.append(first_split_len)
            nmat_windows.append([window[0], mid_beat])
            nmat_length.append(second_split_len)
            nmat_windows.append([mid_beat, window[1]])

            nmat += first_split

            # pad nmat to pad_length
            if first_split_len < pad_length:
                for i in range(pad_length - first_split_len):
                    if dataset == 'asap':
                        nmat_row = [0, 0, 0, 0, 0, 0, 0, 0]
                    elif dataset == 'musicnet':
                        nmat_row = [0, 0, 0, 0, 0]
                    nmat.append(nmat_row)

            nmat += second_split

            # pad nmat to pad_length
            if second_split_len < pad_length:
                for i in range(pad_length - second_split_len):
                    if dataset == 'asap':
                        nmat_row = [0, 0, 0, 0, 0, 0, 0, 0]
                    elif dataset == 'musicnet':
                        nmat_row = [0, 0, 0, 0, 0]
                    nmat.append(nmat_row)

    # transfer the absolute onset time and duration time to relative ones
    nmat, nmat_length = absolute_time_to_relative_time(pad_length, nmat, nmat_length)

    # write nmat for each music piece into .txt file
    write_nmat_txt(file_path, nmat, nmat_length, nmat_windows)

    return nmat, nmat_length, len(nmat_length), nmat_windows
