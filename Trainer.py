#Created by Adam Goldbraikh - Scalpel Lab Technion
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
import os

import torch

from model import *
import sys
from torch import optim
import math
import pandas as pd
from termcolor import colored, cprint

from metrics import*
import wandb
from datetime import datetime
import tqdm

from batch_gen import BatchGenerator
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from smooth_loss import SmoothLoss

class Trainer:
    def __init__(self, dim, num_classes_list,hidden_dim=64,dropout=0.4,num_layers=3, offline_mode=True, task="gestures", device="cuda",
                 network='LSTM',debagging=False):

        self.model = MT_RNN_dp(network, input_dim=dim, hidden_dim=hidden_dim, num_classes_list=num_classes_list,
                            bidirectional=offline_mode, dropout=dropout,num_layers=num_layers)


        self.debagging =debagging
        self.network = network
        self.device = device
        # self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        # self.ce = SmoothLoss(lamb=0.15, tau=4)
        self.ce = SmoothLoss(lamb=0.1, tau=3)
        self.num_classes_list = num_classes_list
        self.task =task
        self.weights = [0, 0, 1] if task == 'multi-taks' else [1]
        self.manual_batch_size = 3
        self.global_index = 0
        self.plot_gests_index = 0
        self.plot_gests_every = 1
        self.best_evaluate_accuracy = 0


    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, eval_dict, args):

        number_of_seqs = len(batch_gen.list_of_train_examples)
        number_of_batches = math.ceil(number_of_seqs / batch_size)

        eval_results_list = []
        train_results_list = []
        print(args.dataset + " " + args.group + " " + args.dataset + " dataset " + "split: " + args.split)

        if args.upload is True:
            wandb.init(project=args.project, group=args.group,
                       name="split: " + args.split,
                       reinit=True, entity="rotem_bz")
            delattr(args, 'split')
            wandb.config.update(args)

        self.model.train()
        self.model.to(self.device)
        eval_rate = eval_dict["eval_rate"]
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            n_tasks = 3 if self.task == 'multi-taks' else 1
            correct = np.zeros(n_tasks)
            total = np.zeros(n_tasks)

            while batch_gen.has_next():
                batch_input, side_input, top_input, *batch_target_gestures, mask = batch_gen.next_batch(batch_size)
                for i in range(len(batch_target_gestures)):
                    batch_target_gestures[i] = batch_target_gestures[i].to(self.device)
                batch_input, mask = batch_input.to(self.device), mask.to(self.device)
                if side_input is not None and top_input is not None:
                    side_input, top_input = side_input.to(self.device), top_input.to(self.device)

                optimizer.zero_grad()
                lengths = torch.sum(mask[:, 0, :], dim=1).to(dtype=torch.int64).to(device='cpu')
                # print(f"amount of frames: {batch_input.shape}")
                predictions1 = self.model(batch_input, side_input, top_input, lengths)
                # predictions1 = (predictions1[0] * mask).unsqueeze_(0)
                predictions1 = [(p * mask) for p in predictions1]

                loss = 0
                for p, target, class_size, weight in zip(predictions1, batch_target_gestures, self.num_classes_list, self.weights):
                    loss += weight * self.ce(p.transpose(2, 1).contiguous().view(-1, class_size), target.view(-1))


                loss = loss / self.manual_batch_size
                epoch_loss += loss.item()
                loss.backward()
                self.global_index += 1
                if self.global_index % self.manual_batch_size:
                    optimizer.step()
                _, predicted1 = torch.max(predictions1[-1].data, 1)
                for i in range(len(lengths)):
                    correct[-1] += (predicted1[i][:lengths[i]] == batch_target_gestures[-1][i][
                                                               :lengths[i]]).float().sum().item()
                    total[-1] += lengths[i]

                pbar.update(1)

            batch_gen.reset()
            pbar.close()
            if not self.debagging:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(colored(dt_string, 'green',
                          attrs=['bold']) + "  " + "[epoch %d]: train loss = %f,   train acc = %f" % (epoch + 1,
                                                                                                      epoch_loss / len(
                                                                                                          batch_gen.list_of_train_examples),
                                                                                                      float(
                                                                                                          correct[-1]) / total[-1]))
            train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                             "train acc": float(correct[-1]) / total[-1]}

            if args.upload:
                wandb.log(train_results)

            train_results_list.append(train_results)

            if (epoch) % eval_rate == 0:
                print(colored("epoch: " + str(epoch + 1) + " model evaluation", 'red', attrs=['bold']))
                results = {"epoch": epoch}
                results.update(self.evaluate(eval_dict, batch_gen))
                eval_results_list.append(results)
                if args.upload is True:
                    wandb.log(results)

        return eval_results_list, train_results_list

    def evaluate(self, eval_dict, batch_gen):
        plot_flag = False
        save_model = False
        model_save_path = '/home/student/code-rotem/model_weights'
        if save_model and not os.path.isdir(model_save_path):
            os.mkdir(model_save_path)
        model_save_path = os.path.join(model_save_path, 'saved_model')
        results = {}
        device = eval_dict["device"]
        features_path = eval_dict["features_path"]
        sample_rate = eval_dict["sample_rate"]
        actions_dict_gesures = eval_dict["actions_dict_gestures"]
        ground_truth_path_gestures = eval_dict["gt_path_gestures"]

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            list_of_vids = batch_gen.list_of_valid_examples
            # list_of_vids = batch_gen.list_of_train_examples[:3]
            recognition1_list = []

            for seq in list_of_vids:
                # print vid
                # features = np.load(features_path + seq.split('.')[0] + '.npy')
                # features = features[:, ::sample_rate]
                # input_x = torch.tensor(features, dtype=torch.float)
                # input_x.unsqueeze_(0)
                input_x = BatchGenerator.get_data(seq, features_path, batch_gen.saved_video_tensors_path, sample_rate,
                                                  batch_gen.img_normalizer)
                # input_x = input_x.to(device)
                input_x = [a.to(device) for a in input_x[:-1]] + [input_x[-1]]
                predictions1 = self.model(*input_x)
                predictions1 = predictions1[-1].unsqueeze_(0)
                predictions1 = torch.nn.Softmax(dim=2)(predictions1)

                _, predicted1 = torch.max(predictions1[-1].data, 1)
                predicted1 = predicted1.squeeze()
                if plot_flag:
                    self.plot_gests_index += 1
                    if self.plot_gests_index % self.plot_gests_every == 0:
                        self.plot_gestures(predicted1.tolist())
                    plot_flag = False


                recognition1 = []
                for i in range(len(predicted1)):
                    recognition1 = np.concatenate((recognition1, [list(actions_dict_gesures.keys())[
                                                                      list(actions_dict_gesures.values()).index(
                                                                          predicted1[i].item())]] * sample_rate))
                recognition1_list.append(recognition1)

            print("gestures results")
            results1, _ = metric_calculation(ground_truth_path=ground_truth_path_gestures,
                                             recognition_list=recognition1_list, list_of_videos=list_of_vids,
                                             suffix="gesture")
            accuracy = results1['Acc gesture']
            if accuracy > self.best_evaluate_accuracy:
                print("new best accuracy:", accuracy)
                self.best_evaluate_accuracy = accuracy
                if save_model:
                    torch.save(self.model.state_dict(), model_save_path)
            results.update(results1)


            self.model.train()
            return results

    @staticmethod
    def plot_gestures(gestures):
        print("ARRIVED")
        fig, ax = plt.subplots(1, 1)
        label_color = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y'}
        colors = [label_color[gesture] for gesture in gestures]
        segments = []
        for i in range(len(colors)):
            segments.append([(i, 0.5), (i + 1, 0.5)])
        lc = LineCollection(segments, colors=colors, linewidths=100)
        ax.add_collection(lc)
        ax.set_xlim(0, len(colors))
        ax.get_yaxis().set_visible(False)
        plt.show()

