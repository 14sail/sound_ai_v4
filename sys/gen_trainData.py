import numpy as np
from collections import Counter 

import random
import  sys
import os
# sys.path.append(os.path.abspath(f'{self.opt.project_PATH}/common'))
import utils_aud as U

class TLGenerator():
    #Generates data for Keras
    def __init__(self, samples=None, labels=None, options=None,  preprocess_type=None):
        random.seed(1123);
        #Initialization
        # print("random other")
        # self.data = [(samples[i], labels[i]) for i in range (0, len(samples))]; 原本的
        
        # samples_with_label_0 = [(sample, label) for sample, label in zip(samples, labels) if label == 0]
        # samples_with_other_labels = [(sample, label) for sample, label in zip(samples, labels) if label != 0]
        # num_samples = Counter(labels).most_common(2)[1][1] #.most_common(1)[0][1]
        # if len(samples_with_label_0) >= num_samples:
        #     selected_samples_labels = random.sample(samples_with_label_0, num_samples)
        # else:
        #     raise ValueError(f"label 為 0 的樣本數不足 {num_samples}")
        # selected_samples, selected_labels = zip(*selected_samples_labels)
        # final_samples = list(selected_samples) + [sample for sample, _ in samples_with_other_labels]
        # final_labels = list(selected_labels) + [label for _, label in samples_with_other_labels]        
        # self.data = list(zip(final_samples, final_labels))
         
        self.opt = options;
        self.data = [(samples[i], labels[i]) for i in range (0, len(samples))]; # 原本的
        self.samples = samples
        self.labels = labels        

        # U = self.opt.U


        self.opt.preprocess_funcs = self.preprocess_setup(); # preprocess_type
        # 

    def len(self):
        return int(np.floor(len(self.data) / self.opt.batch_size));

    def getitem(self, batchIndex):
        #Generate one batch of data
        batchX, batchY = self.generate_batch_select_fixed_class();  # generate_batch  or  generate_batch_select_fixed_class
        batchX = np.expand_dims(batchX, axis=1);
        batchX = np.expand_dims(batchX, axis=3);
    
        return batchX, batchY

    # def generate_batch(self, batchIndex): # if you don't want to fixed the class you can use this without generate_batch_select_fixed_class
    #     #Generates data containing batch_size samples
    #     sounds = [];
    #     labels = [];
    #     indexes = None;

    #     for i in range(self.batch_size):
    #         # Training phase of BC learning
    #         # Select two training examples
    #         while True:
    #             sound1, label1 = self.data[random.randint(0, len(self.data) - 1)]
    #             sound2, label2 = self.data[random.randint(0, len(self.data) - 1)]
    #             if label1 != label2:
    #                 break
    #         sound1 = self.preprocess(sound1)
    #         sound2 = self.preprocess(sound2)

    #         # Mix two examples
    #         r = np.array(random.random())
    #         sound = U.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
    #         eye = np.eye(self.opt.nClasses)
    #         idx1 = self.mapdict[str(label1)]- 1
    #         idx2 = self.mapdict[str(label2)] - 1
    #         label = (eye[idx1] * r + eye[idx2] * (1 - r)).astype(np.float32)
    #         # label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

    #         #For stronger augmentation
    #         sound = U.random_gain(6)(sound).astype(np.float32) #################################################
    #         # print(f"sound length after U.random_gain is {len(sound)}")
    #         sounds.append(sound);
    #         labels.append(label);
    #         # print(f"---{label}---{sound.max()},{sound.min()}")
    #     sounds = np.asarray(sounds);
    #     labels = np.asarray(labels);

    #     return sounds, labels;

    def generate_batch_select_fixed_class(self):
        #Generates data containing batch_size samples
        sounds = [];
        labels = [];
        indexes = None;
        #two variables recording alarm and moaning sounds count
        # alarm_selected, help_eng_selected, help_ch_selected, help_ja_selected, help_tw_selected, help_hk_selected = 0, 0, 0, 0, 0, 0
        alarm_selected, help_eng_selected, help_ch_selected, help_ja_selected, help_tw_selected = 0, 0, 0, 0, 0
        dog_selected, cat_selected, flush_selected, glass_breaking_selected= 0, 0, 0, 0


        ###
        # samples_with_label_0 = [(sample, label) for sample, label in zip(self.samples, self.labels) if label == 0]
        # samples_with_other_labels = [(sample, label) for sample, label in zip(self.samples, self.labels) if label != 0]

        # num_samples = Counter(self.labels).most_common(2)[1][1] # .most_common()[-1][1] #.most_common(1)[0][1]
        # # print(num_samples)
        # if len(samples_with_label_0) >= num_samples:
        #     selected_samples_labels = random.sample(samples_with_label_0, num_samples)
        # else:
        #     raise ValueError(f"label 為 0 的樣本數不足 {num_samples}")
        
        
        # selected_samples, selected_labels = zip(*selected_samples_labels)
        # final_samples = list(selected_samples) + [sample for sample, _ in samples_with_other_labels]
        # final_labels = list(selected_labels) + [label_ for _, label_ in samples_with_other_labels]      
        
        # combined = list(zip(final_samples, final_labels))
        # # random.seed(1123)  
        # random.shuffle(combined)
        # final_samples, final_labels = zip(*combined)
        # final_samples = list(final_samples)
        # final_labels = list(final_labels)
        # # print(Counter(final_labels))
        # self.data_ran = list(zip(final_samples, final_labels))      
        ###



        for i in range(self.opt.batch_size):
            # print("i:",i)
            # Training phase of BC learning
            # Select two training examples

            # map_dict_train = {'other':0, 'Environment':0, 'alarm': 8,
            #                 'en_help': 1, 'ch_help': 2, 'ja_help': 3, 'tw_help': 4, #'hk_help': 5, 'yue_help':6,
            #                 'dog':5, 'cat':6, 'flush':7};

            while True:
                # print("enter while true",len(self.data))
                sound1, label1 = self.data[random.randint(0, len(self.data) - 1)]
                sound2, label2 = self.data[random.randint(0, len(self.data) - 1)]
                lbl1_int = np.int16(label1);
                lbl2_int = np.int16(label2);
                if label1 != label2:
                    # {'alarm': 0, 'en_help': 1, 'ch_help': 2, 'ja_help': 3, 'tw_help': 4, 'hk_help': 5, 'Environment': 6, 'other': 7}

                    if (lbl1_int == 1 and lbl2_int == self.opt.other_class) or (lbl1_int == self.opt.other_class and lbl2_int == 1):
                        if (help_eng_selected < alarm_selected): # and (help_eng_selected == help_ch_selected):
                            help_eng_selected += 1;
                            break;
                    if (lbl1_int == 2 and lbl2_int == self.opt.other_class) or (lbl1_int == self.opt.other_class and lbl2_int == 2):
                        if (help_ch_selected < alarm_selected) and (help_ch_selected < help_eng_selected):
                            help_ch_selected += 1;
                            break;
                    if (lbl1_int == 3 and lbl2_int == self.opt.other_class) or (lbl1_int == self.opt.other_class and lbl2_int == 3):
                        if (help_ja_selected < alarm_selected) and (help_ja_selected < help_ch_selected):
                            help_ja_selected += 1;
                            break;
                    if (lbl1_int == 4 and lbl2_int == self.opt.other_class) or (lbl1_int == self.opt.other_class and lbl2_int == 4):
                        if (help_tw_selected < alarm_selected) and (help_tw_selected < help_ch_selected):
                            help_tw_selected += 1;
                            break;
                    if (lbl1_int == 5 and lbl2_int == self.opt.other_class) or (lbl1_int == self.opt.other_class and lbl2_int ==5):
                        if (dog_selected < alarm_selected) and (dog_selected < help_ch_selected):
                            dog_selected += 1;
                            break;
                    if (lbl1_int == 6 and lbl2_int == self.opt.other_class) or (lbl1_int == self.opt.other_class and lbl2_int == 6):
                        if (cat_selected < alarm_selected) and (cat_selected < help_ch_selected):
                            cat_selected += 1;
                            break; # flush_selected
                    if (lbl1_int == 7 and lbl2_int == self.opt.other_class) or (lbl1_int == self.opt.other_class and lbl2_int == 7):
                        if (flush_selected < alarm_selected) and (flush_selected < help_ch_selected):
                            flush_selected += 1;
                            break; #                     
                    if (lbl1_int == 8 and lbl2_int == self.opt.other_class) or (lbl1_int == self.opt.other_class and lbl2_int ==8):
                        if (alarm_selected == help_ch_selected) and (alarm_selected == help_eng_selected):
                            alarm_selected += 1;
                            break;        
                    if (lbl1_int == 9 and lbl2_int == self.opt.other_class) or (lbl1_int == self.opt.other_class and lbl2_int == 9):
                        if (glass_breaking_selected < alarm_selected) and (glass_breaking_selected < help_ch_selected):
                            glass_breaking_selected += 1;
                            break; #  


                    # if (lbl1_int == 6 and lbl2_int == self.opt.other_class) or (lbl1_int == self.opt.other_class and lbl2_int == 6):
                    #     if (help_hk_selected < alarm_selected) and (help_hk_selected < help_ch_selected):
                    #         help_hk_selected += 1;                    
                    #         break;


            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix two examples
            r = np.array(random.random());
            
            eye = np.eye(self.opt.nClasses)
            idx1 = label1 # self.mapdict[str(label1)]- 1
            idx2 = label2 # self.mapdict[str(label2)] - 1
            # print(idx1, idx2 , type(idx1))
            if idx1 == np.int16(self.opt.other_class):
                if r<self.opt.threshold:
                    sound = U.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
                    label = (eye[idx1] * r + eye[idx2] * (1 - r)).astype(np.float32)
                else:
                    sound = U.mix(sound2, sound1, r, self.opt.sr).astype(np.float32)
                    label = (eye[idx2] * r + eye[idx1] * (1 - r)).astype(np.float32)
            elif idx2 == np.int16(self.opt.other_class):
                if r<self.opt.threshold:
                    sound = U.mix(sound2, sound1, r, self.opt.sr).astype(np.float32)
                    label = (eye[idx2] * r + eye[idx1] * (1 - r)).astype(np.float32)                    
                else:
                    sound = U.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
                    label = (eye[idx1] * r + eye[idx2] * (1 - r)).astype(np.float32)

            # normalize_func = U.normalize(audio_max_value)  

            #For stronger augmentation
            sound = U.random_gain(self.opt.other_class)(sound).astype(np.float32)
            # print(f"sound length after U.random_gain is {len(sound)}")
            sounds.append(sound)
            labels.append(label);

        sounds = np.asarray(sounds);
        labels = np.asarray(labels);
        # print(f"---{label}---{sound.max()},{sound.min()}")
        
        return sounds, labels;

    def preprocess_setup(self):
        funcs = []
        # if self.opt.strongAugment:
        #     funcs += [U.random_scale(1.25)]

        funcs += [
                # U.padding(self.opt.inputLength // 2),
                #   U.random_crop(self.opt.inputLength),
                #   U.regularization(audio_max_value, audio_min_value),  # 
                  U.normalize() #self.opt.audio_max_value
                  ]
        # if type == "regular":
        #     funcs += [U.normalize(audio_max_value)]
        return funcs

    def preprocess(self, sound):
        for f in self.opt.preprocess_funcs:
            sound = f(sound)
        return sound;

# project_PATH = '/home/sail/sound_project/sound_ai_v2.2.1'
# sys.path.append(os.path.abspath(f'{project_PATH}/common'))
# import utils as U

