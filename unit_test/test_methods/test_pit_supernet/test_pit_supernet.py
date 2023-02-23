import unittest
import torch
from flexnas.methods import PITSuperNet
from unit_test.models.pit_supernet_nn import StandardPITSNModule
from unit_test.models.pit_supernet_nn_gs import GumbelPITSNModule
from unit_test.models.kws_pit_sn_model import DSCnnPITSN
from unit_test.models.icl_pit_sn_model import ResNet8PITSN
from unit_test.models.vww_pit_sn_model import MobileNetPITSN


class TestPITSuperNet(unittest.TestCase):

    # StandardSNModule
    def test_standard_pitsn_module(self):
        """Test that the output of a PITSNModule has the correct shape
        """
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

    def test_standard_pitsn_module_get_size(self):
        """Test correctness of get_size() function
        """
        ch_in = 32
        in_width = 64
        in_height = 64

        model = StandardPITSNModule()
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        self.assertEqual(sn_model.get_size(), 13344)

    def test_standard_pitsn_module_get_macs(self):
        """Test correctness of get_macs() function
        """
        ch_in = 32
        in_width = 64
        in_height = 64

        model = StandardPITSNModule()
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        self.assertEqual(sn_model.get_macs(), 54657024)

    # KWS PITSN Model
    def test_kws_pitsn_target_modules(self):
        """Test that the number of PITSNModules found is correct
        """
        ch_in = 1
        in_width = 49
        in_height = 10

        model = DSCnnPITSN()
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        target_modules = sn_model._target_sn_combiners
        self.assertEqual(len(target_modules), 4, "Wrong target modules number")

    def test_kws_pitsn_input(self):
        """Test the integrity of the PITSuperNet model by forwarding a dummy input
            and checking that the output has the same shape of the output of the
            original model
        """
        ch_in = 1
        in_width = 49
        in_height = 10
        batch_size = 1

        model = DSCnnPITSN()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out1 = model(dummy_inp)
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        out2 = sn_model(dummy_inp)
        self.assertTrue(out1.shape == out2.shape, "Different output shapes")

    def test_kws_pitsn_export(self):
        """Test the arch_export() function and the integrity of the exported model
            by forwarding a dummy input and checking that the output has the same
            shape of the output of the original model
        """
        ch_in = 1
        in_width = 49
        in_height = 10
        batch_size = 1

        model = DSCnnPITSN()
        # exclude_names = ['inputlayer', 'conv1', 'conv2', 'conv3', 'conv4']
        exclude_names = []
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height), exclude_names=exclude_names)
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out1 = sn_model(dummy_inp)
        sn_model.get_size()
        exp = sn_model.arch_export()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out2 = exp(dummy_inp)
        self.assertTrue(out1.shape == out2.shape, "Different output shapes")

    # ICL PITSN Model
    def test_icl_pitsn_export(self):
        """Test the arch_export() function and the integrity of the exported model
            by forwarding a dummy input and checking that the output has the same
            shape of the output of the original model
        """
        ch_in = 3
        in_width = 32
        in_height = 32
        batch_size = 1

        model = ResNet8PITSN()
        exclude_names = []
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height), exclude_names=exclude_names)
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out1 = sn_model(dummy_inp)
        sn_model.get_size()
        exp = sn_model.arch_export()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out2 = exp(dummy_inp)
        self.assertTrue(out1.shape == out2.shape, "Different output shapes")

    # VWW PITSN Model
    def test_vww_pitsn_export(self):
        """Test the arch_export() function and the integrity of the exported model
            by forwarding a dummy input and checking that the output has the same
            shape of the output of the original model
        """
        ch_in = 3
        in_width = 96
        in_height = 96
        batch_size = 1

        model = MobileNetPITSN()
        exclude_names = []
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height), exclude_names=exclude_names)
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out1 = sn_model(dummy_inp)
        sn_model.get_size()
        exp = sn_model.arch_export()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out2 = exp(dummy_inp)
        self.assertTrue(out1.shape == out2.shape, "Different output shapes")

    def test_pitsn_summary(self):
        """Test arch_summary() function
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = StandardPITSNModule()
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        sn_model.arch_summary()

    def test_pitsn_getmacs(self):
        """Test get_macs() function
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = StandardPITSNModule()
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        macs = sn_model.get_macs()
        self.assertGreater(macs.item(), 0.0)

    def test_pitsn_icv(self):
        """Test the correct computation of icv loss
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = StandardPITSNModule()
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        eps = 1e-3
        icv = sn_model.get_total_icv(eps)
        # icv = mu^2 / sigma^2, we have 4 choices and initially all architectural parameters
        # are identical. So mu^2 = (1/4)^2 and sigma^2 = 0 + eps (stability term)
        self.assertAlmostEqual(icv.item(), (1 / (4 * 4)) / eps, places=3)

    def test_pitsn_soft_gumbel_sampling(self):
        """Test softmax gumbel sampling with hard_softmax=False
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = GumbelPITSNModule(hard_softmax=False)
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        _ = sn_model.get_size()
        _ = sn_model.get_macs()
        _ = sn_model.get_total_icv()

    def test_pitsn_hard_gumbel_sampling(self):
        """Test softmax gumbel sampling with hard_softmax=True
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = GumbelPITSNModule(hard_softmax=True)
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        _ = sn_model.get_size()
        _ = sn_model.get_macs()
        _ = sn_model.get_total_icv()

    def test_pitsn_gumbel_sampling_soft_hard_switch(self):
        """Test the update of softmax gumbel sampling passing from soft to hard
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = GumbelPITSNModule(hard_softmax=False)
        sn_model = PITSuperNet(model, (ch_in, in_width, in_height))
        size_soft = sn_model.get_size()
        macs_soft = sn_model.get_macs()
        targets = sn_model._target_sn_combiners
        theta_alpha_soft = targets[0][1].theta_alpha

        sn_model.update_softmax_options(hard=True)
        dummy_inp = torch.rand((1,) + (ch_in, in_width, in_height))
        _ = sn_model(dummy_inp)  # force forward to update sampling
        size_hard = sn_model.get_size()
        macs_hard = sn_model.get_macs()
        theta_alpha_hard = targets[0][1].theta_alpha

        self.assertTrue(torch.all(theta_alpha_soft != theta_alpha_hard),
                        "Softmax type switch not working")
        self.assertNotEqual(size_soft, size_hard, "Softmax type switch not working")
        self.assertNotEqual(macs_soft, macs_hard, "Softmax type switch not working")


if __name__ == '__main__':
    unittest.main(verbosity=2)
