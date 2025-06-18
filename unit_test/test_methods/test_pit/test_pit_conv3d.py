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
# * Author: Daniele Jahier Pagliari <daniele.jahier@polito.it>                 *
# *----------------------------------------------------------------------------*
import unittest
import torch
from torch.nn import Conv3d
from torch.nn.parameter import Parameter
from plinio.methods.pit.nn import PITConv3d
from plinio.methods.pit.nn.features_masker import PITFeaturesMasker


class TestPITConv3d(unittest.TestCase):
    """Test PITConv3d functionalities"""

    def test_pitconv3d_features_mask_output(self):
        """Test that the output of a PITConv3d layer with some masked features is equivalent
        to the output of the correspondent nn.Module"""
        ch_in = 3
        ch_out = 32
        k = 3
        out_height = 64
        out_width = 64
        out_depth = 64
        batch_size = 32
        conv_ut = Conv3d(ch_in, ch_out, (k, k, k), padding='same', bias=False)
        pitconv_ut = PITConv3d(
            conv_ut,
            out_features_masker=PITFeaturesMasker(ch_out)
        )
        # Returns a tensor filled with random integers between 0 (inclusive) and 2 (exclusive)
        rnd_alpha = torch.randint(0, 2, (ch_out,), dtype=torch.float32)
        pitconv_ut.out_features_masker.alpha = Parameter(rnd_alpha)

        dummy_inp = torch.rand((batch_size, ch_in, out_height, out_width, out_depth))
        w_conv_ut = conv_ut.weight
        with torch.no_grad():
            conv_ut.weight = Parameter(w_conv_ut[rnd_alpha.bool(), :, :, :])
            conv_ut_out = conv_ut(dummy_inp)
            pitconv_ut_out = pitconv_ut(dummy_inp)[:, rnd_alpha.bool(), :, :]

        self.assertTrue(torch.all(conv_ut_out == pitconv_ut_out), "Different outputs")


if __name__ == '__main__':
    unittest.main(verbosity=2)
