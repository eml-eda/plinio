import copy
import torch.nn as nn
import torch.nn.functional as F
from plinio.methods.mixprec.nn import MixPrec_Conv2d
from plinio.methods.mixprec.nn.mixprec_qtz import MixPrecType, \
    MixPrec_Qtz_Layer, MixPrec_Qtz_Layer_Bias
from plinio.methods.mixprec.quant.quantizers import PACT_Act, MinMax_Weight, \
    Quantizer_Bias
import plinio.methods.mixprec.quant.nn as qnn


class SimpleMixPrecNN(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(3, 40, 40), num_classes=3):
        super(SimpleMixPrecNN, self).__init__()
        self.input_shape = input_shape
        mixprec_a_quantizer = MixPrec_Qtz_Layer((2, 4, 8),
                                                PACT_Act,
                                                {'cout': 32})
        mixprec_w_quantizer = MixPrec_Qtz_Layer((2, 4, 8),
                                                MinMax_Weight,
                                                {'cout': 32})
        mixprec_b_quantizer = MixPrec_Qtz_Layer_Bias(Quantizer_Bias,
                                                     32,
                                                     mixprec_w_quantizer,
                                                     mixprec_a_quantizer=mixprec_a_quantizer,
                                                     quantizer_kwargs={'num_bits': 32,
                                                                       'cout': 32})
        self.conv0 = MixPrec_Conv2d(nn.Conv2d(3, 32, (3, 3), padding='same'),
                                    40,
                                    40,
                                    (2, 4, 8),
                                    (2, 4, 8),
                                    mixprec_a_quantizer,
                                    mixprec_w_quantizer,
                                    mixprec_b_quantizer,
                                    MixPrecType.PER_LAYER
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
    def __init__(self, input_shape=(3, 40, 40), num_classes=3):
        super(SimpleExportedNN2D, self).__init__()
        self.qargs = {'a_precision': 8,
                      'w_precision': 4,
                      'in_a_quantizer': PACT_Act,
                      'out_a_quantizer': PACT_Act,
                      'w_quantizer': MinMax_Weight,
                      'b_quantizer': Quantizer_Bias,
                      }
        self.input_shape = input_shape
        self.conv0 = qnn.Quant_Conv2d(
            nn.Conv2d(3, 32, (3, 3), padding='same'),
            **self.qargs)
        self.pool0 = nn.AvgPool2d(2)
        self.conv1 = qnn.Quant_Conv2d(
            nn.Conv2d(32, 57, (5, 5), padding='same'),
            **self.qargs)
        self.pool1 = nn.AvgPool2d(2)
        self.dpout = nn.Dropout(0.5)
        self.fc = qnn.Quant_Linear(
            nn.Linear(57 * (input_shape[-1] // 2 // 2)**2, num_classes),
            **self.qargs)
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = self.pool0(self.conv0(x))
        x = self.pool1(self.conv1(x))
        x = self.dpout(x.flatten(1))
        res = self.fc(x)
        return res


class SimpleExportedNN2D_ch(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(3, 40, 40), num_classes=3):
        super(SimpleExportedNN2D_ch, self).__init__()
        self.qargs = {'a_precision': 8,
                      'w_precision': 2,
                      'a_quantizer': PACT_Act,
                      'w_quantizer': MinMax_Weight,
                      'b_quantizer': Quantizer_Bias,
                      }
        self.input_shape = input_shape
        qargs_2 = copy.deepcopy(self.qargs)
        self.qargs.update({'w_precision': 4})
        qargs_4 = copy.deepcopy(self.qargs)
        self.qargs.update({'w_precision': 8})
        qargs_8 = copy.deepcopy(self.qargs)
        self.conv0 = qnn.Quant_List([
            qnn.Quant_Conv2d(nn.Conv2d(3, 10, (3, 3), padding='same'), **qargs_2),
            qnn.Quant_Conv2d(nn.Conv2d(3, 10, (3, 3), padding='same'), **qargs_4),
            qnn.Quant_Conv2d(nn.Conv2d(3, 12, (3, 3), padding='same'), **qargs_8)])
        self.pool0 = nn.AvgPool2d(2)
        self.conv1 = qnn.Quant_List([
            qnn.Quant_Conv2d(nn.Conv2d(32, 57, (5, 5), padding='same'), **qargs_2)])
        self.pool1 = nn.AvgPool2d(2)
        self.dpout = nn.Dropout(0.5)
        self.fc = qnn.Quant_List([
            qnn.Quant_Linear(nn.Linear(57 * (input_shape[-1] // 2 // 2)**2, 2),
                             **qargs_8),
            qnn.Quant_Linear(nn.Linear(57 * (input_shape[-1] // 2 // 2)**2, 1),
                             **qargs_4)])
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = self.pool0(self.conv0(x))
        x = self.pool1(self.conv1(x))
        x = self.dpout(x.flatten(1))
        res = self.fc(x)
        return res


class SimpleExportedNN2D_NoBias(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(3, 40, 40), num_classes=3):
        super(SimpleExportedNN2D_NoBias, self).__init__()
        self.qargs = {'a_precision': 8,
                      'w_precision': 4,
                      'a_quantizer': PACT_Act,
                      'w_quantizer': MinMax_Weight,
                      'b_quantizer': Quantizer_Bias,
                      }
        self.input_shape = input_shape
        self.conv0 = qnn.Quant_Conv2d(
            nn.Conv2d(3, 32, (3, 3), padding='same', bias=False),
            **self.qargs)
        self.pool0 = nn.AvgPool2d(2)
        self.conv1 = qnn.Quant_Conv2d(
            nn.Conv2d(32, 57, (5, 5), padding='same', bias=False),
            **self.qargs)
        self.pool1 = nn.AvgPool2d(2)
        self.dpout = nn.Dropout(0.5)
        self.fc = qnn.Quant_Linear(
            nn.Linear(57 * (input_shape[-1] // 2 // 2)**2, num_classes),
            **self.qargs)
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = self.pool0(self.conv0(x))
        x = self.pool1(self.conv1(x))
        x = self.dpout(x.flatten(1))
        res = self.fc(x)
        return res


class SimpleExportedNN2D_NoBias_ch(nn.Module):
    """Defines a simple sequential DNN used within unit tests"""
    def __init__(self, input_shape=(3, 40, 40), num_classes=3):
        super(SimpleExportedNN2D_NoBias_ch, self).__init__()
        self.qargs = {'a_precision': 8,
                      'w_precision': 4,
                      'a_quantizer': PACT_Act,
                      'w_quantizer': MinMax_Weight,
                      'b_quantizer': Quantizer_Bias,
                      }
        self.input_shape = input_shape
        self.conv0 = qnn.Quant_List([qnn.Quant_Conv2d(
            nn.Conv2d(3, 32, (3, 3), padding='same', bias=False),
            **self.qargs)])
        self.pool0 = nn.AvgPool2d(2)
        self.conv1 = qnn.Quant_List([qnn.Quant_Conv2d(
            nn.Conv2d(32, 57, (5, 5), padding='same', bias=False),
            **self.qargs)])
        self.pool1 = nn.AvgPool2d(2)
        self.dpout = nn.Dropout(0.5)
        self.fc = qnn.Quant_List([qnn.Quant_Linear(
            nn.Linear(57 * (input_shape[-1] // 2 // 2)**2, num_classes),
            **self.qargs)])
        self.foo = "non-nn.Module attribute"

    def forward(self, x):
        x = self.pool0(self.conv0(x))
        x = self.pool1(self.conv1(x))
        x = self.dpout(x.flatten(1))
        res = self.fc(x)
        return res
