import unittest
import torch
from flexnas.methods import PITSuperNet
from unit_test.models.pit_supernet_nn import StandardPITSNModule
from unit_test.models.kws_pit_sn_model import DSCnnPITSN
from unit_test.models.icl_pit_sn_model import ResNet8PITSN
from unit_test.models.vww_pit_sn_model import MobileNetPITSN


class TestPITSuperNet(unittest.TestCase):

    # StandardSNModule
    def test_standard_pitsn_module(self):
        ch_in = 32
        ch_out = 32
        in_width = 64
        in_height = 64
        out_width = 64
        out_heigth = 64
        batch_size = 1

        model = StandardPITSNModule()
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = sn_model(dummy_inp)
        self.assertEqual(out.shape, (batch_size, ch_out, out_width, out_heigth),
                         "Unexpected output shape")

    # KWS PITSN Model
    def test_kws_pitsn_target_modules(self):
        ch_in = 1
        in_width = 49
        in_height = 10

        model = DSCnnPITSN()
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        target_modules = sn_model._target_modules
        self.assertEqual(len(target_modules), 4, "Wrong target modules number")

    def test_kws_pitsn_input(self):
        ch_in = 1
        in_width = 49
        in_height = 10
        batch_size = 1

        model = DSCnnPITSN()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        model(dummy_inp)
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        sn_model(dummy_inp)

    def test_kws_pitsn_export(self):
        ch_in = 1
        in_width = 49
        in_height = 10
        batch_size = 1

        model = DSCnnPITSN()
        # exclude_names = ['inputlayer', 'conv1', 'conv2', 'conv3', 'conv4']
        exclude_names = []
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height), exclude_names=exclude_names)
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = sn_model(dummy_inp)
        sn_model.get_size()
        exp = sn_model.arch_export()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = exp(dummy_inp)
        print(out)

    # ICL PITSN Model
    def test_icl_pitsn_export(self):
        ch_in = 3
        in_width = 32
        in_height = 32
        batch_size = 1

        model = ResNet8PITSN()
        exclude_names = []
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height), exclude_names=exclude_names)
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = sn_model(dummy_inp)
        sn_model.get_size()
        exp = sn_model.arch_export()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = exp(dummy_inp)
        print(out)

    '''
    # VWW PITSN Model
    def test_vww_pitsn_export(self):
        ch_in = 3
        in_width = 96
        in_height = 96
        batch_size = 1

        model = MobileNetPITSN()
        exclude_names = []
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height), exclude_names=exclude_names)
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = sn_model(dummy_inp)
        sn_model.get_size()
        exp = sn_model.arch_export()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = exp(dummy_inp)
        print(out)
    '''


if __name__ == '__main__':
    unittest.main(verbosity=2)
