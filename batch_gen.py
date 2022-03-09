#!/usr/bin/python2.7

import torch
import numpy as np
import random
import os
import pandas as pd
from scipy.stats import norm

import time
from torchvision import transforms


class BatchGenerator(object):
    saved_video_tensors_path = '/home/student/code-rotem/videos_dir/'
    img_normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    def __init__(self, num_classes_gestures,num_classes_tools, actions_dict_gestures,actions_dict_tools,features_path,split_num,folds_folder,frames_folder,gt_path_gestures=None, gt_path_tools_left=None, gt_path_tools_right=None, sample_rate=1,normalization="None",task="gestures", model_type='B'):
        """
        
        :param num_classes_gestures: 
        :param num_classes_tools: 
        :param actions_dict_gestures: 
        :param actions_dict_tools: 
        :param features_path: 
        :param split_num: 
        :param folds_folder: 
        :param gt_path_gestures: 
        :param gt_path_tools_left: 
        :param gt_path_tools_right: 
        :param sample_rate: 
        :param normalization: None - no normalization, min-max - Min-max feature scaling, Standard - Standard score	 or Z-score Normalization
        ## https://en.wikipedia.org/wiki/Normalization_(statistics)
        """""
        self.task =task
        self.model_type = model_type
        self.normalization = normalization
        self.folds_folder = folds_folder
        self.frames_folder = frames_folder
        self.frames_files = os.listdir(frames_folder)
        self.split_num = split_num
        self.list_of_train_examples = list()
        self.list_of_valid_examples = list()
        self.index = 0
        self.num_classes_gestures = num_classes_gestures
        self.num_classes_tools = num_classes_tools
        self.actions_dict_gestures= actions_dict_gestures
        self.action_dict_tools = actions_dict_tools
        self.gt_path_gestures = gt_path_gestures
        self.gt_path_tools_left = gt_path_tools_left
        self.gt_path_tools_right = gt_path_tools_right
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.read_data()
        self.normalization_params_read()

        # self.list_of_train_examples = self.list_of_train_examples[:1]   # overfit assertion


    def normalization_params_read(self):
        params = pd.read_csv(os.path.join(self.folds_folder, "std_params_fold_" + str(self.split_num) + ".csv"),index_col=0).values
        self.max = params[0, :]
        self.min = params[1, :]
        self.mean = params[2, :]
        self.std = params[3, :]


    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_train_examples)


    def has_next(self):
        if self.index < len(self.list_of_train_examples):
            return True
        return False


    def read_data(self):
        self.list_of_train_examples =[]
        for file in os.listdir(self.folds_folder):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "fold" in filename:
                if str(self.split_num) in filename:
                    file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                    self.list_of_valid_examples = file_ptr.read().split('\n')[:-1]
                    file_ptr.close()
                    random.shuffle(self.list_of_valid_examples)
                else:
                    file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                    self.list_of_train_examples = self.list_of_train_examples + file_ptr.read().split('\n')[:-1]
                    file_ptr.close()
                continue
            else:
                continue
        random.shuffle(self.list_of_train_examples)


    def pars_ground_truth(self,gt_source):
        contant =[]
        for line in gt_source:
            info = line.split()
            line_contant = [info[2]] * (int(info[1])-int(info[0]) +1)
            contant = contant + line_contant
        return contant




##### this is supports one and two heads and 3 heads #############

    @staticmethod
    def get_data(seq: str, features_path, saved_video_tensors_path, sample_rate, img_normalizer):
        """
        return features and video tensors for given sequence.
        :param seq: name of sequence file
        :return:
        """
        seq_name = seq.split('.')[0]
        features = np.load(features_path + seq_name + '.npy')[:, ::sample_rate]
        side_vid = torch.load(saved_video_tensors_path + seq_name + '_side.pt')  # side video setup
        side_vid = img_normalizer(side_vid.float())
        top_vid = torch.load(saved_video_tensors_path + seq_name + '_top.pt')
        top_vid = img_normalizer(top_vid.float())
        cutoff_length = min(features.shape[1], side_vid.shape[0], top_vid.shape[0])
        features = features[:, :cutoff_length]
        features = torch.from_numpy(features).float().unsqueeze(0)
        side_vid = side_vid[:cutoff_length, ...].unsqueeze(0)
        top_vid = top_vid[:cutoff_length, ...].unsqueeze(0)
        return features, side_vid, top_vid, torch.tensor([cutoff_length], dtype=torch.int64)

    def next_batch(self, batch_size):
            batch = self.list_of_train_examples[self.index:self.index + batch_size]
            self.index += batch_size

            batch_input = []
            side_vids = []
            top_vids = []
            loading_time = 0
            batch_target_gestures = []
            batch_target_left = []
            batch_target_right = []
            # convert_tensor = transforms.PILToTensor()

            for seq in batch:
                # reps = [len([t for t in self.frames_files if t.startswith(name)]) for name in set(['_'.join(s.split('_')[:2]) for s in self.frames_files])]
                # assert all([x == 2 for x in reps])
                seq_name = seq.split('.')[0]
                features = np.load(self.features_path + seq_name + '.npy')
                t0 = time.perf_counter()
                if self.model_type == 'B':
                    side_vid, top_vid = None, None
                else:
                    side_vid = torch.load(self.saved_video_tensors_path + seq_name + '_side.pt')    # side video setup
                    side_vid = self.img_normalizer(side_vid.float())
                    top_vid = torch.load(self.saved_video_tensors_path + seq_name + '_top.pt')      # top video setup
                # top_vid = self.img_normalizer(top_vid.float())
                loading_time += time.perf_counter() - t0

                if self.normalization == "Min-max":
                    numerator =features.T - self.min
                    denominator = self.max-self.min
                    features = (numerator / denominator).T
                elif self.normalization == "Standard":
                    numerator =features.T - self.mean
                    denominator = self.std
                    features = (numerator / denominator).T
                elif self.normalization == "samplewise_SD":
                    samplewise_meam = features.mean(axis=1)
                    samplewise_std = features.std(axis=1)
                    numerator =features.T - samplewise_meam
                    denominator = samplewise_std
                    features = (numerator / denominator).T

                batch_input.append(features[:, ::self.sample_rate])
                side_vids.append(side_vid)
                top_vids.append(top_vid)


                if self.task == "gestures":
                    file_ptr = open(self.gt_path_gestures + seq.split('.')[0] + '.txt', 'r')
                    gt_source = file_ptr.read().split('\n')[:-1]
                    content = self.pars_ground_truth(gt_source)
                    classes_size = min(np.shape(features)[1], len(content))

                    classes = np.zeros(classes_size)
                    for i in range(len(classes)):
                        classes[i] = self.actions_dict_gestures[content[i]]
                    batch_target_gestures.append(classes[::self.sample_rate])


                elif self.task == "tools":
                    file_ptr_right = open(self.gt_path_tools_right + seq.split('.')[0] + '.txt', 'r')
                    gt_source_right = file_ptr_right.read().split('\n')[:-1]
                    content_right = self.pars_ground_truth(gt_source_right)
                    file_ptr_left = open(self.gt_path_tools_left + seq.split('.')[0] + '.txt', 'r')
                    gt_source_left = file_ptr_left.read().split('\n')[:-1]
                    content_left = self.pars_ground_truth(gt_source_left)

                    classes_size_right = min(np.shape(features)[1], len(content_left), len(content_right))
                    classes_right = np.zeros(classes_size_right)
                    for i in range(classes_size_right):
                        classes_right[i] = self.action_dict_tools[content_right[i]]
                    batch_target_right.append(classes_right[::self.sample_rate])

                    classes_size_left = min(np.shape(features)[1], len(content_left), len(content_right))
                    classes_left = np.zeros(classes_size_left)
                    for i in range(classes_size_left):
                        classes_left[i] = self.action_dict_tools[content_left[i]]

                    batch_target_left.append(classes_left[::self.sample_rate])

                elif self.task == "multi-taks":
                    file_ptr = open(self.gt_path_gestures + seq.split('.')[0] + '.txt', 'r')
                    gt_source = file_ptr.read().split('\n')[:-1]
                    content = self.pars_ground_truth(gt_source)
                    classes_size = min(np.shape(features)[1], len(content))

                    classes = np.zeros(classes_size)
                    for i in range(len(classes)):
                        classes[i] = self.actions_dict_gestures[content[i]]
                    batch_target_gestures.append(classes[::self.sample_rate])

                    file_ptr_right = open(self.gt_path_tools_right + seq.split('.')[0] + '.txt', 'r')
                    gt_source_right = file_ptr_right.read().split('\n')[:-1]
                    content_right = self.pars_ground_truth(gt_source_right)
                    classes_size_right = min(np.shape(features)[1], len(content_right))
                    classes_right = np.zeros(classes_size_right)
                    for i in range(len(classes_right)):
                        classes_right[i] = self.action_dict_tools[content_right[i]]

                    batch_target_right.append(classes_right[::self.sample_rate])

                    file_ptr_left = open(self.gt_path_tools_left + seq.split('.')[0] + '.txt', 'r')
                    gt_source_left = file_ptr_left.read().split('\n')[:-1]
                    content_left = self.pars_ground_truth(gt_source_left)
                    classes_size_left = min(np.shape(features)[1], len(content_left))
                    classes_left = np.zeros(classes_size_left)
                    for i in range(len(classes_left)):
                        classes_left[i] = self.action_dict_tools[content_left[i]]

                    batch_target_left.append(classes_left[::self.sample_rate])

            # print("total batch pt loading time:", loading_time)
            batch_side_input = None
            batch_top_input = None
            if self.task == "gestures":
                length_of_sequences = list(map(len, batch_target_gestures))
                batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
                batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
                # mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(length_of_sequences), dtype=torch.float)
                mask = torch.zeros(len(batch_input), 1, max(length_of_sequences), dtype=torch.float)
                for i in range(len(batch_input)):
                    batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i][:,:batch_input_tensor.shape[2]])
                    batch_target_tensor[i, :np.shape(batch_target_gestures[i])[0]] = torch.from_numpy(batch_target_gestures[i])
                    # mask[i, :, :np.shape(batch_target_gestures[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target_gestures[i])[0])
                    mask[i, :, :np.shape(batch_target_gestures[i])[0]] = torch.ones(1,
                                                                                    np.shape(batch_target_gestures[i])[0])

                return batch_input_tensor, batch_side_input, batch_top_input, batch_target_tensor, mask

            elif self.task == "tools":
                length_of_sequences_left = np.expand_dims(np.array( list(map(len, batch_target_left))),1)
                length_of_sequences_right = np.expand_dims(np.array( list(map(len, batch_target_right))),1)

                length_of_sequences = list(np.min(np.concatenate((length_of_sequences_left, length_of_sequences_right),1),1))

                batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0],
                                                 max(length_of_sequences), dtype=torch.float)
                batch_target_tensor_left = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
                batch_target_tensor_right = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
                mask = torch.zeros(len(batch_target_right), self.num_classes_tools, max(length_of_sequences), dtype=torch.float)


                for i in range(len(batch_input)):
                    batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i][:,:batch_input_tensor.shape[2]])
                    batch_target_tensor_left[i, :np.shape(batch_target_left[i])[0]] = torch.from_numpy(batch_target_left[i][:batch_target_tensor_left.shape[1]])
                    batch_target_tensor_right[i, :np.shape(batch_target_right[i])[0]] = torch.from_numpy(batch_target_right[i][:batch_target_tensor_right.shape[1]])
                    mask[i, :, :np.shape(batch_target_right[i])[0]] = torch.ones(self.num_classes_tools, np.shape(batch_target_right[i])[0])



                return batch_input_tensor, batch_side_input, batch_top_input, batch_target_tensor_left ,batch_target_tensor_right, mask

            elif self.task == "multi-taks":
                length_of_sequences_left = np.expand_dims(np.array( list(map(len, batch_target_left))),1)
                length_of_sequences_right = np.expand_dims(np.array( list(map(len, batch_target_right))),1)
                length_of_sequences_gestures = np.expand_dims(np.array( list(map(len, batch_target_gestures))),1)
                if self.model_type == 'B':
                    tup = (length_of_sequences_left, length_of_sequences_right, length_of_sequences_gestures)
                else:
                    length_of_sequences_side = np.array([x.shape[0] for x in side_vids]).reshape(-1, 1)
                    length_of_sequences_top = np.array([x.shape[0] for x in top_vids]).reshape(-1,1)
                    tup = (length_of_sequences_left, length_of_sequences_right, length_of_sequences_gestures,
                           length_of_sequences_side, length_of_sequences_top)


                length_of_sequences = list(np.min(np.concatenate(tup,1),1))
                if self.model_type == 'B':
                    batch_side_input, batch_top_input = None, None
                else:
                    side_vids = [side_vids[i][:length_of_sequences[i], ...] for i in range(len(side_vids))]
                    top_vids = [top_vids[i][:length_of_sequences[i], ...] for i in range(len(top_vids))]
                    batch_side_input = torch.nn.utils.rnn.pad_sequence(side_vids, batch_first=True)
                    batch_top_input = torch.nn.utils.rnn.pad_sequence(top_vids, batch_first=True)
                max_len = max(length_of_sequences)

                batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max_len, dtype=torch.float)

                batch_target_tensor_left = torch.ones(len(batch_input), max_len, dtype=torch.long) * (-100)
                batch_target_tensor_right = torch.ones(len(batch_input), max_len, dtype=torch.long) * (-100)
                batch_target_tensor_gestures = torch.ones(len(batch_input), max_len, dtype=torch.long)*(-100)

                # mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(length_of_sequences), dtype=torch.float)
                mask = torch.zeros(len(batch_input), 1, max_len, dtype=torch.float)


                for i in range(len(batch_input)):
                    batch_input_tensor[i, :, :length_of_sequences[i]] = torch.from_numpy(batch_input[i][:,:length_of_sequences[i]])
                    batch_target_tensor_left[i, :length_of_sequences[i]] = torch.from_numpy(batch_target_left[i][:length_of_sequences[i]])
                    batch_target_tensor_right[i, :length_of_sequences[i]] = torch.from_numpy(batch_target_right[i][:length_of_sequences[i]])
                    batch_target_tensor_gestures[i, :length_of_sequences[i]] = torch.from_numpy(batch_target_gestures[i][:length_of_sequences[i]])
                    # mask[i, :, :np.shape(batch_target_gestures[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target_gestures[i])[0])
                    mask[i, :, :length_of_sequences[i]] = torch.ones(1, length_of_sequences[i])

                return batch_input_tensor, batch_side_input, batch_top_input, batch_target_tensor_left ,batch_target_tensor_right,batch_target_tensor_gestures, mask
    ##### this is supports one and two heads#############

    def next_batch_backup(self, batch_size):
        batch = self.list_of_train_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_target_left = []
        batch_target_right = []

        for seq in batch:
            features = np.load(self.features_path + seq.split('.')[0] + '.npy')
            if self.normalization == "Min-max":
                numerator =features.T - self.min
                denominator = self.max-self.min
                features = (numerator / denominator).T
            elif self.normalization == "Standard":
                numerator =features.T - self.mean
                denominator = self.std
                features = (numerator / denominator).T

            if self.task == "gestures":
                file_ptr = open(self.gt_path_gestures + seq.split('.')[0] + '.txt', 'r')
                gt_source = file_ptr.read().split('\n')[:-1]
                content = self.pars_ground_truth(gt_source)
                classes_size = min(np.shape(features)[1], len(content))

                classes = np.zeros(classes_size)
                for i in range(len(classes)):
                    classes[i] = self.actions_dict_gestures[content[i]]
                batch_input .append(features[:, ::self.sample_rate])
                batch_target.append(classes[::self.sample_rate])


            elif self.task == "tools":
                file_ptr_right = open(self.gt_path_tools_right + seq.split('.')[0] + '.txt', 'r')
                gt_source_right = file_ptr_right.read().split('\n')[:-1]
                content_right = self.pars_ground_truth(gt_source_right)
                classes_size_right = min(np.shape(features)[1], len(content_right))
                classes_right = np.zeros(classes_size_right)
                for i in range(len(classes_right)):
                    classes_right[i] = self.action_dict_tools[content_right[i]]

                batch_input.append(features[:, ::self.sample_rate])
                batch_target_right.append(classes_right[::self.sample_rate])

                file_ptr_left = open(self.gt_path_tools_left + seq.split('.')[0] + '.txt', 'r')
                gt_source_left = file_ptr_left.read().split('\n')[:-1]
                content_left = self.pars_ground_truth(gt_source_left)
                classes_size_left = min(np.shape(features)[1], len(content_left))
                classes_left = np.zeros(classes_size_left)
                for i in range(len(classes_left)):
                    classes_left[i] = self.action_dict_tools[content_left[i]]

                batch_target_left.append(classes_left[::self.sample_rate])

        if self.task == "gestures":
            length_of_sequences = list(map(len, batch_target))
            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
            batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
            mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(length_of_sequences), dtype=torch.float)
            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i][:,:batch_input_tensor.shape[2]])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
                mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target[i])[0])

            return batch_input_tensor, batch_target_tensor, mask

        elif self.task == "tools":
            length_of_sequences = list(map(len, batch_target_left))
            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0],
                                             max(length_of_sequences), dtype=torch.float)
            batch_target_tensor_left = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
            batch_target_tensor_right = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
            mask = torch.zeros(len(batch_input), self.num_classes_tools, max(length_of_sequences), dtype=torch.float)


            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor_left[i, :np.shape(batch_target_left[i])[0]] = torch.from_numpy(batch_target_left[i])
                batch_target_tensor_right[i, :np.shape(batch_target_right[i])[0]] = torch.from_numpy(batch_target_right[i])
                mask[i, :, :np.shape(batch_target_right[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target_right[i])[0])



            return batch_input_tensor, batch_target_tensor_left ,batch_target_tensor_right


    def next_batch_with_gt_tools_as_input(self, batch_size):
        batch = self.list_of_train_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_target_left = []
        batch_target_right = []

        for seq in batch:
            features = np.load(self.features_path + seq.split('.')[0] + '.npy')
            if self.normalization == "Min-max":
                numerator =features.T - self.min
                denominator = self.max-self.min
                features = (numerator / denominator).T
            elif self.normalization == "Standard":
                numerator =features.T - self.mean
                denominator = self.std
                features = (numerator / denominator).T

            file_ptr = open(self.gt_path_gestures + seq.split('.')[0] + '.txt', 'r')
            gt_source = file_ptr.read().split('\n')[:-1]
            content = self.pars_ground_truth(gt_source)
            classes_size = min(np.shape(features)[1], len(content))

            classes = np.zeros(classes_size)
            for i in range(len(classes)):
                classes[i] = self.actions_dict_gestures[content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])


            file_ptr_right = open(self.gt_path_tools_right + seq.split('.')[0] + '.txt', 'r')
            gt_source_right = file_ptr_right.read().split('\n')[:-1]
            content_right = self.pars_ground_truth(gt_source_right)
            classes_size_right = min(np.shape(features)[1], len(content_right))
            classes_right = np.zeros(classes_size_right)
            for i in range(len(classes_right)):
                classes_right[i] = self.action_dict_tools[content_right[i]]

            batch_target_right.append(classes_right[::self.sample_rate])

            file_ptr_left = open(self.gt_path_tools_left + seq.split('.')[0] + '.txt', 'r')
            gt_source_left = file_ptr_left.read().split('\n')[:-1]
            content_left = self.pars_ground_truth(gt_source_left)
            classes_size_left = min(np.shape(features)[1], len(content_left))
            classes_left = np.zeros(classes_size_left)
            for i in range(len(classes_left)):
                classes_left[i] = self.action_dict_tools[content_left[i]]

            batch_target_left.append(classes_left[::self.sample_rate])

        # for i in range(len(batch_input)):
        #     min_dim = min([batch_target_left[i].size,batch_target_right[i].size, batch_input[i].shape[1]])
        #     batch_target_left[i] = (np.expand_dims(batch_target_left[i][:min_dim], axis=1).T)/ max(self.action_dict_tools.values())
        #     batch_target_right[i] = (np.expand_dims(batch_target_right[i][:min_dim], axis=1).T)/ max(self.action_dict_tools.values())
        #     batch_input[i] = np.concatenate((batch_input[i][:,:min_dim],batch_target_right[i],batch_target_left[i]), axis=0 )

        for i in range(len(batch_input)):
            min_dim = min([batch_target_left[i].size,batch_target_right[i].size, batch_input[i].shape[1]])
            batch_target_left[i] = (np.expand_dims(batch_target_left[i][:min_dim], axis=1).T)/ max(self.action_dict_tools.values())
            batch_target_right[i] = (np.expand_dims(batch_target_right[i][:min_dim], axis=1).T)/ max(self.action_dict_tools.values())
            batch_input[i] = np.concatenate((batch_target_right[i],batch_target_left[i]), axis=0 )

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
