import unittest
import torch
from flexnas.methods import PITSuperNet
from unit_test.models.pit_supernet_nn import StandardPITSNModule
from unit_test.models.kws_pit_sn_model import DSCnnPITSN


class TestPITSuperNet(unittest.TestCase):

    # StandardSNModule
    def test_standard_pit_sn_module(self):
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

    # KWS SN Model
    def test_supernet_kws_sn_model_target_modules(self):
        ch_in = 1
        in_width = 49
        in_height = 10

        model = DSCnnPITSN()
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        target_modules = sn_model._target_modules
        self.assertEqual(len(target_modules), 4, "Wrong target modules number")

    def test_pit_supernet_model_input(self):
        ch_in = 1
        in_width = 49
        in_height = 10
        batch_size = 1

        model = DSCnnPITSN()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        model(dummy_inp)
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        sn_model(dummy_inp)

    def test_pit_supernet(self):
        ch_in = 1
        in_width = 49
        in_height = 10

        model = DSCnnPITSN()
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))

        print(sn_model._target_modules)
        # print(sn_model._target_modules[0][1]._pit_layers)

    def test_pit_supernet_export(self):
        ch_in = 1
        in_width = 49
        in_height = 10
        batch_size = 1

        model = DSCnnPITSN()
        # exclude_names = ['inputlayer', 'conv1', 'conv2', 'conv3', 'conv4']
        exclude_names = []
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height), exclude_names=exclude_names)

        exp = sn_model.arch_export()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = exp(dummy_inp)
        print(out)


if __name__ == '__main__':
    unittest.main(verbosity=2)
