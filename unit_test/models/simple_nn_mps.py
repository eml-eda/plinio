import torch.nn as nn
import torch.nn.functional as F
from plinio.methods.mps.nn import MPSConv2d
from plinio.methods.mps.nn.qtz import MPSPerLayerQtz, MPSBiasQtz
from plinio.methods.mps.quant.quantizers import PACTAct, MinMaxWeight, QuantizerBias
import plinio.methods.mps.quant.nn as qnn


class SimpleMPSNN(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""

    def __init__(self, input_shape=(3, 40, 40), num_classes=3):
        super(SimpleMPSNN, self).__init__()
        self.input_shape = input_shape
        mixprec_a_quantizer = MPSPerLayerQtz((2, 4, 8),
                                             PACTAct,
                                             {'cout': 32})
        mixprec_w_quantizer = MPSPerLayerQtz((2, 4, 8),
                                             MinMaxWeight,
                                             {'cout': 32})
        mixprec_b_quantizer = MPSBiasQtz(QuantizerBias,
                                         quantizer_kwargs={'precision': 32, 'cout': 32})
        self.conv0 = MPSConv2d(nn.Conv2d(3, 32, (3, 3), padding='same'),
                               mixprec_a_quantizer,
                               mixprec_w_quantizer,
                               mixprec_b_quantizer,
                               )
        self.bn0 = nn.BatchNorm2d(32, track_running_stats=True)
        self.pool0 = nn.AvgPool2d(2)
        self.conv1 = nn.Conv2d(32, 57, (5, 5), padding='same')
        self.bn1 = nn.BatchNorm2d(57, track_running_stats=True)
        self.pool1 = nn.AvgPool2d(2)
        self.dpout = nn.Dropout(0.5)
        self.fc = nn.Linear(57 * (input_shape[-1] // 2 // 2)**2, num_classes)
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = F.relu6(self.pool0(self.bn0(self.conv0(x))))
        x = F.relu6(self.pool1(self.bn1(self.conv1(x))))
        x = self.dpout(x.flatten(1))
        res = self.fc(x)
        return res


class SimpleExportedNN2D(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""

    def __init__(self, input_shape=(3, 40, 40), num_classes=3, bias=True):
        super(SimpleExportedNN2D, self).__init__()
        self.input_shape = input_shape
        conv0_in_a_qtz = PACTAct(precision=8)
        conv0_out_a_qtz = PACTAct(precision=8)
        self.conv0 = qnn.QuantConv2d(
            nn.Conv2d(3, 32, (3, 3), padding='same', bias=bias),
            in_quantizer=conv0_in_a_qtz,
            out_quantizer=conv0_out_a_qtz,
            w_quantizer=MinMaxWeight(precision=8, cout=32),
            b_quantizer=QuantizerBias(precision=32, cout=32)
            )
        self.pool0 = nn.AvgPool2d(2)
        conv1_out_a_qtz = PACTAct(precision=8)
        self.conv1 = qnn.QuantConv2d(
            nn.Conv2d(32, 57, (5, 5), padding='same', bias=bias),
            in_quantizer=conv0_out_a_qtz,
            out_quantizer=conv1_out_a_qtz,
            w_quantizer=MinMaxWeight(precision=8, cout=57),
            b_quantizer=QuantizerBias(precision=32, cout=57)
            )
        self.pool1 = nn.AvgPool2d(2)
        self.dpout = nn.Dropout(0.5)
        fc_out_a_qtz = PACTAct(precision=8)
        self.fc = qnn.QuantLinear(
            nn.Linear(57 * (input_shape[-1] // 2 // 2)**2, num_classes),
            in_quantizer=conv1_out_a_qtz,
            out_quantizer=fc_out_a_qtz,
            w_quantizer=MinMaxWeight(precision=8, cout=num_classes),
            b_quantizer=QuantizerBias(precision=32, cout=num_classes)
            )
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = self.pool0(self.conv0(x))
        x = self.pool1(self.conv1(x))
        x = self.dpout(x.flatten(1))
        res = self.fc(x)
        return res


class SimpleExportedNN2D_ch(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""

    def __init__(self, input_shape=(3, 40, 40), num_classes=3, bias=True):
        super(SimpleExportedNN2D_ch, self).__init__()
        self.input_shape = input_shape
        conv0_in_a_qtz = PACTAct(precision=8)
        conv0_out_a_qtz = PACTAct(precision=8)
        self.conv0 = qnn.QuantList([
            qnn.QuantConv2d(
                nn.Conv2d(3, 10, (3, 3), padding='same', bias=bias),
                in_quantizer=conv0_in_a_qtz,
                out_quantizer=conv0_out_a_qtz,
                w_quantizer=MinMaxWeight(precision=2, cout=10),
                b_quantizer=QuantizerBias(precision=32, cout=10)
            ),
            qnn.QuantConv2d(
                nn.Conv2d(3, 10, (3, 3), padding='same', bias=bias),
                in_quantizer=conv0_in_a_qtz,
                out_quantizer=conv0_out_a_qtz,
                w_quantizer=MinMaxWeight(precision=4, cout=10),
                b_quantizer=QuantizerBias(precision=32, cout=10),
            ),
            qnn.QuantConv2d(
                nn.Conv2d(3, 12, (3, 3), padding='same', bias=bias),
                in_quantizer=conv0_in_a_qtz,
                out_quantizer=conv0_out_a_qtz,
                w_quantizer=MinMaxWeight(precision=8, cout=12),
                b_quantizer=QuantizerBias(precision=32, cout=12),
            ),
        ])

        self.pool0 = nn.AvgPool2d(2)
        conv1_out_a_qtz = PACTAct(precision=8)
        self.conv1 = qnn.QuantList([
            qnn.QuantConv2d(
                nn.Conv2d(32, 57, (5, 5), padding='same', bias=bias),
                in_quantizer=conv0_out_a_qtz,
                out_quantizer=conv1_out_a_qtz,
                w_quantizer=MinMaxWeight(precision=8, cout=57),
                b_quantizer=QuantizerBias(precision=32, cout=57),
            ),
        ])
        self.pool1 = nn.AvgPool2d(2)
        self.dpout = nn.Dropout(0.5)
        fc_out_a_qtz = PACTAct(precision=8)
        self.fc = qnn.QuantList([
            qnn.QuantLinear(
                nn.Linear(57 * (input_shape[-1] // 2 // 2)**2, 2),
                in_quantizer=conv1_out_a_qtz,
                out_quantizer=fc_out_a_qtz,
                w_quantizer=MinMaxWeight(precision=8, cout=2),
                b_quantizer=QuantizerBias(precision=32, cout=2),
            ),
            qnn.QuantLinear(
                nn.Linear(57 * (input_shape[-1] // 2 // 2)**2, num_classes - 2),
                in_quantizer=conv1_out_a_qtz,
                out_quantizer=fc_out_a_qtz,
                w_quantizer=MinMaxWeight(precision=4, cout=num_classes - 2),
                b_quantizer=QuantizerBias(precision=32, cout=num_classes - 2),
            ),
        ])
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = self.pool0(self.conv0(x))
        x = self.pool1(self.conv1(x))
        x = self.dpout(x.flatten(1))
        res = self.fc(x)
        return res
