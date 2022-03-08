#Created by Adam Goldbraikh - Scalpel Lab Technion
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.utils.rnn import pack_padded_sequence

import vtn_helper
import vtn_model




class MT_RNN_dp(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, num_classes_list, bidirectional, dropout,num_layers=2):
        super(MT_RNN_dp, self).__init__()

        self.z = None
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(dropout)

        video_nets_cfg = VideoNetsCFG()
        self.side_network = vtn_model.VTN(video_nets_cfg)
        # self.top_network = vtn_model.VTN(video_nets_cfg)
        side_dim = self.side_network.embed_dim
        ################ LSTM through images ####################
        # input_dim = side_dim
        #########################################################
        ################ backbone + kinematics concat ####################
        # input_dim = side_dim + input_dim
        #########################################################
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                               num_layers=num_layers)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=bidirectional,
                              num_layers=num_layers)
        else:
            raise NotImplemented


        # The linear layer that maps from hidden state space to tag space
        kinematics_dim = hidden_dim * 2 if bidirectional else hidden_dim
        # overall_dim = kinematics_dim + side_dim
        overall_dim = kinematics_dim

        # self.output_heads = nn.ModuleList([copy.deepcopy(
        #     nn.Sequential(
        #         nn.Linear(overall_dim, hidden_dim + overall_dim),
        #         nn.Tanh(),
        #         nn.Linear(hidden_dim + overall_dim, num_classes_list[s])
        #     )
        # ) for s in range(len(num_classes_list))])

        self.output_heads = nn.ModuleList([copy.deepcopy(
            nn.Sequential(
                nn.Linear(overall_dim, num_classes_list[s])
            )
        ) for s in range(len(num_classes_list))])

        # self.output_heads = nn.ModuleList([copy.deepcopy(
        #     nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes_list[s]))
        #     for s in range(len(num_classes_list))])


    def forward(self, rnn_inpus, side_inputs, top_inputs, lengths):
        outputs=[]
        rnn_inpus = rnn_inpus.permute(0, 2, 1)
        rnn_inpus=self.dropout(rnn_inpus)
        ################ LSTM through images ####################
        # rnn_inpus = self.side_network(side_inputs, lengths)
        #########################################################
        ################ Longformer with kinematics ####################
        # rnn_inpus = self.side_network(rnn_inpus, lengths, back_bone=False)
        # rnn_inpus = rnn_inpus[:, :max(lengths), :]
        # z = rnn_inpus
        #########################################################

        # x = side_inputs
        # B, F, C, H, W = x.shape
        # x = x.permute(0, 2, 1, 3, 4)
        # x = x.reshape(B * F, C, H, W)
        # x = self.side_network.backbone(x.float())
        # x = x.reshape(B, F, -1)
        # # rnn_inpus = torch.cat([rnn_inpus, x], dim=2)
        # rnn_inpus = x

        packed_input = pack_padded_sequence(rnn_inpus, lengths=lengths, batch_first=True, enforce_sorted=False)
        rnn_output, _ = self.rnn(packed_input)

        unpacked_rnn_out, unpacked_rnn_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, padding_value=-1, batch_first=True)
        # flat_X = torch.cat([unpacked_ltsm_out[i, :lengths[i], :] for i in range(len(lengths))])
        unpacked_rnn_out = self.dropout(unpacked_rnn_out)

        # side_out = self.side_network(side_inputs, lengths)
        # side_out = side_out[:, :unpacked_rnn_out.shape[1], :]
        # z = torch.cat([unpacked_rnn_out, side_out], dim=2)
        z = unpacked_rnn_out
        # if self.z is not None:
        #     diff = torch.sum(torch.abs(z - self.z))
        #     print("diff=", diff, "shape=", z.shape, "avg. diff=", diff / (z.shape[0]*z.shape[1]*z.shape[2]))
        # self.z = z
        # top_out = self.top_network(top_inputs, lengths)
        for output_head in self.output_heads:
            outputs.append(output_head(z).permute(0, 2, 1))
        return outputs


class VideoNetsCFG:
    class VTN:
        # embed_dim = 768
        MAX_POSITION_EMBEDDINGS = 2 * 60 * 60
        NUM_ATTENTION_HEADS = 12
        NUM_HIDDEN_LAYERS = 3
        ATTENTION_MODE = 'sliding_chunks'
        PAD_TOKEN_ID = -1
        ATTENTION_WINDOW = [36, 36]
        INTERMEDIATE_SIZE = 3072
        ATTENTION_PROBS_DROPOUT_PROB = 0.1
        HIDDEN_DROPOUT_PROB = 0.1
        PRETRAINED = False
        DROP_PATH_RATE = 0.0
        DROP_RATE = 0.0

    class MODEL:
        ARCH = 'VIT'
        # HIDDEN_DIM = 768
        # MLP_DIM = 768
        # DROPOUT_RATE = 0
        # NUM_CLASSES = 0


