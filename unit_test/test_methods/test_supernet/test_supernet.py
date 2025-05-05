import unittest
import torch
from plinio.methods import SuperNet
from plinio.methods.supernet.nn import SuperNetCombiner
from plinio.cost import params, ops
from unit_test.models.supernet_nn import StandardSNModule, StandardSNModuleV2
from unit_test.models.supernet_nn_gs import GumbelSNModule
from unit_test.models.kws_sn_model import DSCnnSN
from unit_test.models.icl_sn_model import ResNet8SN
from unit_test.models.vww_sn_model import MobileNetSN

from unit_test.models.resnet1d_ppgbp import ResNet1D

class TestSuperNet(unittest.TestCase):

    def test_standard_pitsn_module(self):
        """Test that the output of a SNModule has the correct shape
        """
        ch_in = 32
        ch_out = 32
        in_width = 64
        in_height = 64
        out_width = 64
        out_heigth = 64
        batch_size = 1
        model = StandardSNModule()
        sn_model = SuperNet(model, input_shape=(ch_in, in_width, in_height))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = sn_model(dummy_inp)
        self.assertEqual(out.shape, (batch_size, ch_out, out_width, out_heigth),
                         "Unexpected output shape")

    def test_params_trainability(self):
        """Test the effectiveness of the helpers functions `train_nas_only`, `train_net_only`
        and `train_net_and_nas`."""
        ch_in = 32
        in_width = 64
        in_height = 64
        model = StandardSNModule()
        sn_model = SuperNet(model, input_shape=(ch_in, in_width, in_height))

        sn_model.train_nas_only()
        for nas_param in sn_model.nas_parameters():
            self.assertTrue(nas_param.requires_grad, "NAS parameters not trainable")
        for net_param in sn_model.net_parameters():
            self.assertFalse(net_param.requires_grad, "Net parameters trainable")
        sn_model.train_net_only()
        for nas_param in sn_model.nas_parameters():
            self.assertFalse(nas_param.requires_grad, "NAS parameters trainable")
        for net_param in sn_model.net_parameters():
            self.assertTrue(net_param.requires_grad, "Net parameters not trainable")
        sn_model.train_net_and_nas()
        for nas_param in sn_model.nas_parameters():
            self.assertTrue(nas_param.requires_grad, "NAS parameters not trainable")
        for net_param in sn_model.net_parameters():
            self.assertTrue(net_param.requires_grad, "Net parameters not trainable")

    def test_standard_pitsn_module_custom_block(self):
        """Test the import of a SuperNet with a custom defined block
        """
        ch_in = 32
        ch_out = 32
        in_width = 64
        in_height = 64
        out_width = 64
        out_heigth = 64
        batch_size = 1
        model = StandardSNModuleV2()
        sn_model = SuperNet(model, input_shape=(ch_in, in_width, in_height))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out = sn_model(dummy_inp)
        self.assertEqual(out.shape, (batch_size, ch_out, out_width, out_heigth),
                         "Unexpected output shape")

    def test_standard_pitsn_module_params(self):
        """Test correctness of params cost model
        """
        batch_size = 32
        ch_in = 32
        in_width = 64
        in_height = 64
        model = StandardSNModule()
        sn_model = SuperNet(model, input_shape=(ch_in, in_width, in_height))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        _ = sn_model(dummy_inp)
        cost = sn_model.get_cost()
        exp_cost = ((1+2) * (3*3*32+1)*32 + (5*5*32+1)*32 + 0 ) / 4
        self.assertEqual(cost, exp_cost)

    def test_standard_pitsn_module_ops(self):
        """Test correctness of ops cost model
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = StandardSNModule()
        sn_model = SuperNet(model, cost=ops, input_shape=(ch_in, in_width, in_height))
        cost = sn_model.get_cost()
        exp_cost = ((1+2) * (3*3*32+1)*32 + (5*5*32+1)*32 + 0 ) / 4
        exp_cost = exp_cost * 64 * 64
        self.assertEqual(cost, exp_cost)

    def test_kws_pitsn_target_modules(self):
        """Test that the number of SNModules found is correct
        """
        ch_in = 1
        in_width = 49
        in_height = 10
        model = DSCnnSN()
        sn_model = SuperNet(model, input_shape=(ch_in, in_width, in_height))
        n_tgt = len([_ for _ in sn_model._leaf_modules if isinstance(_[2], SuperNetCombiner)])
        self.assertEqual(n_tgt, 4, "Wrong target modules number")

    def test_kws_pitsn_input(self):
        """Test the integrity of the SuperNet model by forwarding a dummy input
            and checking that the output has the same shape of the output of the
            original model
        """
        ch_in = 1
        in_width = 49
        in_height = 10
        batch_size = 1
        model = DSCnnSN()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out1 = model(dummy_inp)
        sn_model = SuperNet(model, input_example=dummy_inp)
        out2 = sn_model(dummy_inp)
        self.assertTrue(out1.shape == out2.shape, "Different output shapes")

    def test_kws_pitsn_export(self):
        """Test the export() function and the integrity of the exported model
            by forwarding a dummy input and checking that the output has the same
            shape of the output of the original model
        """
        ch_in = 1
        in_width = 49
        in_height = 10
        batch_size = 1
        model = DSCnnSN()
        sn_model = SuperNet(model, input_shape=(ch_in, in_width, in_height))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out1 = sn_model(dummy_inp)
        sn_model.get_cost()
        exp = sn_model.export()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out2 = exp(dummy_inp)
        self.assertTrue(out1.shape == out2.shape, "Different output shapes")


    def test_groupconv_export(self):
        """
        Test the export of a model with a group conv module that require
        supernet export_graph function to be based on best branch index instead of name.
        This can be done because all_input_nodes function returns the list of nodes
        in the args and kwargs, in that order.
        """
        nn_ut = ResNet1D(use_plinio = 'supernet') #ToyGroupedConv_1D()
        sn_model = SuperNet(nn_ut, input_shape=nn_ut.input_shape[1:])
        dummy_inp = torch.rand(nn_ut.input_shape)
        out1 = sn_model(dummy_inp)
        self.assertEqual(out1.shape, (1,2), "unexpected output shape")
        sn_model.get_cost()
        # export fails if supernet export graph is entirely based on layers' name,
        # because the last layer of a group conv branch is a cat layer
        # that will have as name just 'cat', without the branch number
        exp = sn_model.export()
        out2 = exp(dummy_inp)
        self.assertTrue(out1.shape == out2.shape, "Different output shapes")

    def test_icl_pitsn_export(self):
        """Test the arch_export() function and the integrity of the exported model
            by forwarding a dummy input and checking that the output has the same
            shape of the output of the original model
        """
        ch_in = 3
        in_width = 32
        in_height = 32
        batch_size = 1
        model = ResNet8SN()
        sn_model = SuperNet(model, cost=ops, input_shape=(ch_in, in_width, in_height))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out1 = sn_model(dummy_inp)
        sn_model.get_cost()
        exp = sn_model.export()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out2 = exp(dummy_inp)
        self.assertTrue(out1.shape == out2.shape, "Different output shapes")

    def test_vww_pitsn_export(self):
        """Test the arch_export() function and the integrity of the exported model
            by forwarding a dummy input and checking that the output has the same
            shape of the output of the original model
        """
        ch_in = 3
        in_width = 96
        in_height = 96
        batch_size = 1
        model = MobileNetSN()
        sn_model = SuperNet(model, input_shape=(ch_in, in_width, in_height))
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out1 = sn_model(dummy_inp)
        sn_model.get_cost()
        exp = sn_model.export()
        dummy_inp = torch.rand((batch_size,) + (ch_in, in_width, in_height))
        out2 = exp(dummy_inp)
        self.assertTrue(out1.shape == out2.shape, "Different output shapes")

    def test_pitsn_summary(self):
        """Test arch_summary() function
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = StandardSNModule()
        sn_model = SuperNet(model, input_shape=(ch_in, in_width, in_height))
        sn_model.summary()

    def test_pitsn_icv(self):
        """Test the correct computation of icv loss
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = StandardSNModule()
        sn_model = SuperNet(model, input_shape=(ch_in, in_width, in_height))
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
        model = GumbelSNModule(hard_softmax=False)
        sn_model = SuperNet(model, cost={'params': params, 'ops': ops},
                            input_shape=(ch_in, in_width, in_height))
        _ = sn_model.get_cost('params')
        _ = sn_model.get_cost('ops')
        _ = sn_model.get_total_icv()

    def test_pitsn_hard_gumbel_sampling(self):
        """Test softmax gumbel sampling with hard_softmax=True
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = GumbelSNModule(hard_softmax=True)
        sn_model = SuperNet(model, cost={'params': params, 'ops': ops},
                            input_shape=(ch_in, in_width, in_height))
        _ = sn_model.get_cost('params')
        _ = sn_model.get_cost('ops')
        _ = sn_model.get_total_icv()

    def test_pitsn_gumbel_sampling_soft_hard_switch(self):
        """Test the update of softmax gumbel sampling passing from soft to hard
        """
        ch_in = 32
        in_width = 64
        in_height = 64
        model = GumbelSNModule(hard_softmax=False)
        sn_model = SuperNet(model, cost={'params': params, 'ops': ops},
                            input_shape=(ch_in, in_width, in_height))
        size_soft = sn_model.get_cost('params')
        macs_soft = sn_model.get_cost('ops')
        combiners = [_[2] for _ in sn_model._unique_leaf_modules
                     if isinstance(_[2], SuperNetCombiner)]
        theta_alpha_soft = combiners[0].theta_alpha

        sn_model.update_softmax_options(hard=True)
        dummy_inp = torch.rand((1,) + (ch_in, in_width, in_height))
        _ = sn_model(dummy_inp)  # force forward to update sampling
        size_hard = sn_model.get_cost('params')
        macs_hard = sn_model.get_cost('ops')
        theta_alpha_hard = combiners[0].theta_alpha

        self.assertTrue(torch.all(theta_alpha_soft != theta_alpha_hard),
                        "Softmax type switch not working")
        self.assertNotEqual(size_soft, size_hard, "Softmax type switch not working")
        self.assertNotEqual(macs_soft, macs_hard, "Softmax type switch not working")


if __name__ == '__main__':
    unittest.main(verbosity=2)
