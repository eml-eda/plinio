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
from torch.nn import BatchNorm1d, BatchNorm2d
from flexnas.methods.pit import PITBatchNorm1d, PITBatchNorm2d


class TestPITBatchNorm(unittest.TestCase):
    """Test PITBatchNorm1d functionalities"""

    def test_pitbn1d_features_mask_output(self):
        """Test that the output of a PITBatchNorm1d layer is equivalent
        to the output of the correspondent nn.Module"""
        ch_out = 32
        out_length = 256
        batch_size = 32
        bn_ut = BatchNorm1d(ch_out)
        pitbn_ut = PITBatchNorm1d(bn_ut)

        dummy_inp = torch.rand((batch_size,) + (ch_out, out_length))
        with torch.no_grad():
            bn_ut_out = bn_ut(dummy_inp)
            pitbn_ut_out = pitbn_ut(dummy_inp)

        self.assertTrue(torch.all(bn_ut_out == pitbn_ut_out), "Different outputs")

    def test_pitbn2d_features_mask_output(self):
        """Test that the output of a PITBatchNorm2d layer is equivalent
        to the output of the correspondent nn.Module"""
        ch_out = 32
        out_height = 64
        out_width = 64
        batch_size = 32
        bn_ut = BatchNorm2d(ch_out)
        pitbn_ut = PITBatchNorm2d(bn_ut)

        dummy_inp = torch.rand((batch_size,) + (ch_out, out_height, out_width))
        with torch.no_grad():
            bn_ut_out = bn_ut(dummy_inp)
            pitbn_ut_out = pitbn_ut(dummy_inp)

        self.assertTrue(torch.all(bn_ut_out == pitbn_ut_out), "Different outputs")


if __name__ == '__main__':
    unittest.main(verbosity=2)
