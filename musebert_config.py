import numpy as np
#######################################################################################################

musebert_version = 'v2'
dataset = 'musicnet' # 'asap'/'musicnet'/'rwc100'

####################################################

task = 'beat_detection_v2' # curriculum; 'pretrain'/'beat_detection'/'chord_extraction'/'arrangement'/'beat_detection_v2'
mode = 'finetuned_model_v2' # model_path
stage = 'inference' # 'pretrain','finetune','inference'
with_program = False
augment = False
is_early_stopping = True
patience = 50
n_epoch = 1000

#########################################

granularity = 500 # 100 / 500

#########################################

last_epoch = -1 #1091 #163
train_after_interrupt = False
last_step = 1 #1372785

#######################################################################################################

model_path = {

    'pretrained_model': 'pretrained_model/result_2022-05-03_011703/models/musebert_valid.pt',

    # For chord extraction:
    # 对所有note event的p_deg算loss(all note)
    'ab_all_note_finetuned_model': 'abandoned-chord_extraction_result/all_note/models/musebert_valid.pt',
    # 只对chord event对应的p_deg算loss(only chord)
    'ab_only_chord_finetuned_model': 'abandoned-chord_extraction_result/only_chord/models/musebert_valid.pt',
    'ab_beat_detection_finetuned_model': 'abandoned-beat_detection_result/beat_detection_2022-05-16_155030/models/musebert_valid.pt',

    # pop909(only chord loss)
    'chord_extraction_finetuned_model': 'chord_extraction_2022-05-17_172116/models/musebert_valid.pt',
        
    # rwc
    'beat_detection_finetuned_model': 'beat_detection_2022-05-17_194715/models/musebert_valid.pt',

    'beat_final': 'beat_detection_2022-05-22_143351/models/musebert_final.pt',
    'chord_final': 'chord_extraction_2022-05-22_143103/models/musebert_valid.pt',
    'chord_final_no-aug': 'chord_extraction_2022-05-23_105156/models/musebert_valid.pt',
    'beat_early': 'beat_detection_2022-05-23_171909/models/musebert_final.pt',

    'pretrained_model_with_program': 'pretrain_2022-06-12_162454/models/musebert_final.pt',
    # 'pretrain_with_program': 'pretrain_2022-05-31_135243/models/musebert_valid.pt',
    'pretrain_with_program': 'pretrain_2022-06-08_180225/models/musebert_valid.pt',
    'arrangement_finetuned_model': 'arrangement_2022-06-17_153505/models/musebert_valid.pt',
    # 'arrangement_finetuned_model': 'arrangement_2022-06-01_223947/models/musebert_valid.pt',

    # musicnet - pretrain - musebert-v2
    'pretrain_v2': '',
    
    # 'pretrained_model_v2': 'pretrain_2022-06-26_003910/models/musebert_valid.pt', # for fixed beat
    # 'pretrained_model_v2': 'pretrain_2022-07-13_065939/models/musebert_valid.pt', # for fixed 100 notes
    'pretrained_model_v2': 'pretrain_2022-07-22_215545/models/musebert_valid.pt', # for fixed 100 notes

    # 'finetuned_model_v2': 'beat_detection_v2_2022-07-03_014534/models/musebert_valid.pt',
    # 'finetuned_model_v2': 'beat_detection_v2_fixed_8_beats_pretrain&finetune&inference/finetune/beat_detection_v2_2022-07-06_011418/models/musebert_final.pt',
    # 'finetuned_model_v2': 'beat_detection_v2_2022-07-20_223338/models/musebert_valid.pt'
    'finetuned_model_v2': 'beat_detection_v2_2022-07-23_052701/models/musebert_valid.pt',

    'pretrained_model_v2_asap': '',
    'finetuned_model_v2_asap': '',

}


chord_train_path = 'data/pop909-final/pop909_chord_train_final.npy'

if stage == 'inference':
    chord_valid_path = 'data/pop909-final/pop909_chord_test_final.npy'
else:
    chord_valid_path = 'data/pop909-final/pop909_chord_valid_final.npy'

train_path = ''
train_length_path = ''
val_path = ''
val_length_path = ''
pad_length = 0

batch_size = 32

if musebert_version == 'v1':

    if task == 'pretrain':

        if mode == 'pretrain_with_program':
            # pre-training with program and beat
            train_path = 'rwc_data/with_program_and_beat/beat_program_rwc_train.npy'
            train_length_path = 'rwc_data/with_program_and_beat/beat_program_rwc_train_length.npy'
            val_path = 'rwc_data/with_program_and_beat/beat_program_rwc_validation.npy'
            val_length_path = 'rwc_data/with_program_and_beat/beat_program_rwc_validation_length.npy'
            pad_length = 315
            batch_size = 8

            # train_path = 'rwc_data/no_program_but_with_beat/beat_rwc_train.npy'
            # train_length_path = 'rwc_data/no_program_but_with_beat/beat_rwc_train_length.npy'
            # val_path = 'rwc_data/no_program_but_with_beat/beat_rwc_validation.npy'
            # val_length_path = 'rwc_data/no_program_but_with_beat/beat_rwc_validation_length.npy'

        else:
            # pre-training
            train_path = 'data/nmat_train.npy'
            train_length_path = 'data/nmat_train_length.npy'
            val_path = 'data/nmat_val.npy'
            val_length_path = 'data/nmat_val_length.npy'
            pad_length = 100

    elif task == 'chord_extraction':

        # fine-tuning on pop909 - chord extraction
        train_path = 'data/pop909-final/pop909_acc_train_final_176.npy'
        train_length_path = 'data/pop909-final/pop909_acc_train_length_final.npy'

        if stage == 'inference':
            val_path = 'data/pop909-final/pop909_acc_test_final_176.npy'
            val_length_path = 'data/pop909-final/pop909_acc_test_length_final.npy'
        else:
            val_path = 'data/pop909-final/pop909_acc_valid_final_176.npy'
            val_length_path = 'data/pop909-final/pop909_acc_valid_length_final.npy'

        pad_length = 176
        batch_size = 28

    elif task == 'beat_detection':

        # fine-tuning on rwc100 - beat detection
        train_path = 'rwc_data/with_beat/beat_rwc_train.npy'
        train_length_path = 'rwc_data/with_beat/beat_rwc_train_length.npy'

        if stage == 'inference':
            val_path = 'rwc_data/with_beat/beat_rwc_test.npy'
            val_length_path = 'rwc_data/with_beat/beat_rwc_test_length.npy'
        else:
            val_path = 'rwc_data/with_beat/beat_rwc_validation.npy'
            val_length_path = 'rwc_data/with_beat/beat_rwc_validation_length.npy'

        pad_length = 36
        batch_size = 450

    elif task == 'arrangement':
        
        # fine-tuning on rwc100 - arrangement
        train_path = 'rwc_data/with_program_and_beat/beat_program_rwc_train.npy'
        train_length_path = 'rwc_data/with_program_and_beat/beat_program_rwc_train_length.npy'

        if stage == 'inference':
            val_path = 'rwc_data/with_program_and_beat/beat_program_rwc_test.npy'
            val_length_path = 'rwc_data/with_program_and_beat/beat_program_rwc_test_length.npy'
        else:
            val_path = 'rwc_data/with_program_and_beat/beat_program_rwc_validation.npy'
            val_length_path = 'rwc_data/with_program_and_beat/beat_program_rwc_validation_length.npy'

        pad_length = 315

        if stage == 'pretrain':
            batch_size = 10
        else:
            batch_size = 6

elif musebert_version == 'v2':

    if dataset == 'musicnet':

        if task == 'pretrain' or task == 'beat_detection_v2':
            
            if mode == 'pretrain_v2' or mode == 'pretrained_model_v2' or mode == 'finetuned_model_v2': 

                # pre-training of musebert v2
                train_path = 'musicnet_nmat/fixed_8_beats/train_nmat_list_350_1.npy'
                train_length_path = 'musicnet_nmat/fixed_8_beats/train_nmat_length_list.npy'

                if stage == 'inference':
                    val_path = 'musicnet_nmat/fixed_8_beats/test_nmat_list_350_1.npy'
                    val_length_path = 'musicnet_nmat/fixed_8_beats/test_nmat_length_list.npy'
                elif stage == 'finetune' or stage == 'pretrain':
                    val_path = 'musicnet_nmat/fixed_8_beats/validation_nmat_list_350_1.npy'
                    val_length_path = 'musicnet_nmat/fixed_8_beats/validation_nmat_length_list.npy'
                    
                pad_length = 350 # 100 #350
                batch_size = 6 #80 #6

    elif dataset == 'asap':

        if task == 'pretrain' or task == 'beat_detection_v2':
            
            if mode == 'pretrain_v2' or mode == 'pretrained_model_v2' or mode == 'finetuned_model_v2': 

                # pre-training of musebert v2
                train_path = 'asap_data/random_beats/train_nmat.npy'
                train_length_path = 'asap_data/random_beats/train_nmat_length.npy'

                if stage == 'inference':
                    val_path = 'asap_data/random_beats/test_nmat.npy'
                    val_length_path = 'asap_data/random_beats/test_nmat_length.npy'
                elif stage == 'finetune' or stage == 'pretrain':
                    val_path = 'asap_data/random_beats/validation_nmat.npy'
                    val_length_path = 'asap_data/random_beats/validation_nmat_length.npy'
                    
                pad_length = 400
                batch_size = 10