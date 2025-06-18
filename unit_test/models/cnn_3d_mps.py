import torch
import torch.nn as nn
import torch.nn.functional as F
import plinio.methods.mps.quant.nn as qnn
from plinio.methods.mps.quant.quantizers import PACTAct, MinMaxWeight, QuantizerBias


class ExportedCNN3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_classes = 1
        self.in_channel = 13
        self.filter_size = 3
        self.patch_size = 7

        self.num_filter = 40
        self.num_filter_2 = int(self.num_filter * 3 / 4) + self.num_filter
        self.dilation = 1

        dilation = (self.dilation, 1, 1)

        conv1_in_a_qtz = PACTAct(precision=8)
        conv1_out_a_qtz = PACTAct(precision=8)
        self.conv1 = qnn.QuantConv3d(
            nn.Conv3d(
                1,
                self.num_filter,
                (self.filter_size, self.filter_size, self.filter_size),
                dilation=dilation,
                padding=0,
            ),
            in_quantizer=conv1_in_a_qtz,
            out_quantizer=conv1_out_a_qtz,
            w_quantizer=MinMaxWeight(precision=8, cout=self.num_filter),
            b_quantizer=QuantizerBias(precision=32, cout=self.num_filter)
            )

        pool1_out_a_qtz = PACTAct(precision=8)
        self.pool1 = qnn.QuantConv3d(
            nn.Conv3d(
                self.num_filter,
                self.num_filter,
                (self.filter_size, 1, 1),
                dilation=dilation,
                stride=(2, 1, 1),
                padding=(1, 0, 0),
            ),
            in_quantizer=conv1_out_a_qtz,
            out_quantizer=pool1_out_a_qtz,
            w_quantizer=MinMaxWeight(precision=8, cout=self.num_filter),
            b_quantizer=QuantizerBias(precision=32, cout=self.num_filter),
        )
            
        conv2_out_a_qtz = PACTAct(precision=8)
        self.conv2 = qnn.QuantConv3d(
            nn.Conv3d(
                self.num_filter,
                self.num_filter_2,
                (self.filter_size, self.filter_size, self.filter_size),
                dilation=dilation,
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                ),
            in_quantizer=pool1_out_a_qtz,
            out_quantizer=conv2_out_a_qtz,
            w_quantizer=MinMaxWeight(precision=8, cout=self.num_filter_2),
            b_quantizer=QuantizerBias(precision=32, cout=self.num_filter_2),
        )

        pool2_out_a_qtz = PACTAct(precision=8)
        self.pool2 = qnn.QuantConv3d(
            nn.Conv3d(
                self.num_filter_2,
                self.num_filter_2,
                (self.filter_size, 1, 1),
                dilation=dilation,
                stride=(2, 1, 1),
                padding=(1, 0, 0),
                ),
            in_quantizer=conv2_out_a_qtz,
            out_quantizer=pool2_out_a_qtz,
            w_quantizer=MinMaxWeight(precision=8, cout=self.num_filter_2),
            b_quantizer=QuantizerBias(precision=32, cout=self.num_filter_2),
        )

        conv3_out_a_qtz = PACTAct(precision=8)
        self.conv3 = qnn.QuantConv3d(
            nn.Conv3d(
                self.num_filter_2,
                self.num_filter_2,
                (self.filter_size, 1, 1),
                dilation=dilation,
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                ),
            in_quantizer=pool2_out_a_qtz,
            out_quantizer=conv3_out_a_qtz,
            w_quantizer=MinMaxWeight(precision=8, cout=self.num_filter_2),
            b_quantizer=QuantizerBias(precision=32, cout=self.num_filter_2),
        )
        conv4_out_a_qtz = PACTAct(precision=8)
        self.conv4 = qnn.QuantConv3d(
            nn.Conv3d(
                self.num_filter_2,
                self.num_filter_2,
                (2, 1, 1),
                dilation=dilation,
                stride=(2, 1, 1),
                padding=(1, 0, 0),
                ),
            in_quantizer=conv3_out_a_qtz,
            out_quantizer=conv4_out_a_qtz,
            w_quantizer=MinMaxWeight(precision=8, cout=self.num_filter_2),
            b_quantizer=QuantizerBias(precision=32, cout=self.num_filter_2),
        )

        self.features_size = self._get_final_flattened_size()

        fc1_out_a_qtz = PACTAct(precision=8)
        self.fc1= qnn.QuantLinear(
            nn.Linear(self.features_size, 1),
            in_quantizer=conv4_out_a_qtz,
            out_quantizer=fc1_out_a_qtz,
            w_quantizer=MinMaxWeight(precision=8, cout=1),
            b_quantizer=QuantizerBias(precision=32, cout=1)
            )

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.in_channel, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):   # , embeddings=False): --> DJP: removed to make graph static
        x = torch.unsqueeze(x, 1)  # DJP --> could be moved out of model to simplify plinification
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        # replaced view with flatten for plinification
        # emb = x.view(-1, self.features_size)
        emb = torch.flatten(x, 1)
        x = self.fc1(emb)
        #x= torch.sigmoid(x)
        # commented out to avoid dynamic DNN graph
        # if embeddings:
            # return x, emb
        return x

