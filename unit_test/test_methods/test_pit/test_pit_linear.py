# *----------------------------------------------------------------------------*
# * Copyright (C) 2022 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author: Matteo Risso <matteo.risso@polito.it>                              *
# *----------------------------------------------------------------------------*
import unittest
import torch
from torch.nn import Linear
from torch.nn.parameter import Parameter
from plinio.methods.pit.nn import PITLinear
from plinio.methods.pit.nn.features_masker import PITFeaturesMasker


class TestPITLinear(unittest.TestCase):
    """Test PITConv2d functionalities"""

    # TODO: this test sometimes fails...
    def test_pitlinear_features_mask_output(self):
        """Test that the output of a PITLinear layer with some masked features is equivalent
        to the output of the correspondent nn.Module"""
        in_feat = 256
        out_feat = 128
        batch_size = 4
        fc_ut = Linear(in_feat, out_feat, bias=False)
        pitfc_ut = PITLinear(
            fc_ut,
            out_features_masker=PITFeaturesMasker(out_feat)
        )
        # Returns a tensor filled with random integers between 0 (inclusive) and 2 (exclusive)
        rnd_alpha = torch.randint(0, 2, (out_feat,), dtype=torch.float32)
        pitfc_ut.out_features_masker.alpha = Parameter(rnd_alpha)

        dummy_inp = torch.rand((batch_size,) + (in_feat,))
        w_fc_ut = fc_ut.weight
        with torch.no_grad():
            fc_ut.weight = Parameter(w_fc_ut[rnd_alpha.bool(), :])
            fc_ut_out = fc_ut(dummy_inp)
            pitfc_ut_out = pitfc_ut(dummy_inp)[:, rnd_alpha.bool()]

        self.assertTrue(torch.all(fc_ut_out == pitfc_ut_out), "Different outputs")


if __name__ == '__main__':
    unittest.main(verbosity=2)
