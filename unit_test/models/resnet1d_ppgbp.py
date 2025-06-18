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
# * Author:  Francesco Carlucci <francesco.carlucci@polito.it>                 *
# *----------------------------------------------------------------------------*
import torch.nn as nn
import torch
from plinio.methods.supernet import SuperNetModule
from math import floor


param_model={ #UCI dataset
  'N_epoch': 256,
  'batch_size': 256,
  'in_channel': 1,
  'base_filters': 32,
  'first_kernel_size': 13,
  'kernel_size': 5,
  'stride': 4,
  'groups': 2,
  'n_block': 8,
  'output_size': 2,
  'lr': 0.001,
  'sample_step': 1,
  'is_se': True,
  'se_ch_low': 4,
}

class ResNet1D(nn.Module):
    """
    ResNet1d architecture used for unit tests.
    Variable number of repeated instances of bn relu dropout and conv1d.

    Testing on this network verifies support for:
    - features padding
    - grouped convolutions
    - getitem support
    """
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
    """

    def __init__(self, in_channels=1, base_filters=32, first_kernel_size=9, kernel_size=5, stride=4,
                        groups=2, n_block=1, output_size=2, is_se=True, se_ch_low=4, downsample_gap=2,
                        increasefilter_gap=2, use_bn=True, use_do=True, verbose=False,
                        use_plinio=False, input_shape=(1,1,625), use_ch_conv=True, use_convlr=True):
        super(ResNet1D, self).__init__()

        self.verbose = verbose
        self.n_block = n_block
        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.is_se = is_se
        self.se_ch_low = se_ch_low

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model
        self.use_ch_conv=use_ch_conv
        self.input_shape = input_shape

        in_dim=input_shape[-1]
        # first block
        if use_plinio == "supernet":
            branches=[
                MyConv1dPadSame(
                in_channels=in_channels,
                out_channels=base_filters,
                kernel_size=self.first_kernel_size,
                groups=1,
                stride=1,
                in_dim=in_dim,
                use_convlr= use_convlr),

                DepthConv1dPadSame(
                in_channels=in_channels,
                out_channels=base_filters,
                kernel_size=self.first_kernel_size,
                groups=1,
                stride=1,
                in_dim=in_dim,
                use_convlr= use_convlr)
            ]
            if base_filters==in_channels:
                branches.append(nn.Identity())
            else:
                in_dim=branches[0].out_dim
            self.first_block_conv = SuperNetModule(branches)
        else:
            self.first_block_conv = MyConv1dPadSame(in_channels=in_channels,
                                        out_channels=base_filters,
                                        kernel_size=self.first_kernel_size,
                                        groups=1,
                                        stride=1,
                                        in_dim=in_dim)
            in_dim=self.first_block_conv.out_dim

        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.first_block_maxpool = MyMaxPool1dPadSame(kernel_size=self.stride,in_dim=in_dim)

        in_dim=self.first_block_maxpool.out_dim

        #input_shape=(*input_shape[0:2],self.first_block_maxpool.out_dim)
        out_channels = base_filters

        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels # già settato alla riga 190
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            #scelta supernet, o delegare
            """
            print( "block n.: ",i_block,"in_ch: ",in_channels,"out_ch: ",out_channels,
                                        "k: ",self.kernel_size,"s: ",self.stride,
                                        "groups: ",self.groups,"downsample: ",downsample,
                                        "use bn: ",self.use_bn,"dropout: ",self.use_do,
                                        "is_first_block: ",is_first_block,"is_se:",self.is_se,
                                        "se_ch_low:", self.se_ch_low)
            """

            tmp_block = SuperBasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                stride = self.stride,
                groups = self.groups,
                downsample=downsample,
                use_bn = self.use_bn,
                use_do = self.use_do,
                is_first_block=is_first_block,
                is_se=self.is_se,
                se_ch_low=self.se_ch_low,
                use_plinio=use_plinio,
                in_dim=in_dim,
                use_ch_conv=self.use_ch_conv,
                use_convlr=use_convlr)

            in_dim=tmp_block.in_dim

            self.basicblock_list.append(tmp_block)
        #self.basicblock_seq = nn.Sequential(*self.basicblock_list)
        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

        # Classifier
        self.main_clf = nn.Linear(out_channels, output_size)

    # def forward(self, x):
    def forward(self, x):
        #x = x['ppg']
        #assert len(x.shape) == 3 #torch fx problem

        # skip batch norm if batchsize<4:
        #if x.shape[0]<4:    #torch fx problem
        #    self.use_bn = False

        # first conv
        out = self.first_block_conv(x)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_block_maxpool(out)

        #out=self.basicblock_seq(out)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            out = net(out)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        h = self.final_relu(out)

        h = torch.mean(h,-1) #.mean(-1) # (n_batch, out_channels)
        # logger.info('final pooling', h.shape)

        # ===== Concat x_demo
        out = self.main_clf(h)
        return out

def _calc_out_dim(in_dim,k,s,p):
    n_out = floor((in_dim+p[0]+p[1]-k)/s)+1
    return n_out

def _calc_padding(in_dim,k,s):
    # compute pad shape
    #in_dim = input_shape[-1]
    #print("in_dim received",in_dim)
    out_dim = (in_dim + s - 1) // s
    p = max(0, (out_dim - 1) * s + k - in_dim)

    pad_left = p // 2
    pad_right = p - pad_left

    #if pad_left != pad_right: print("different left and right pads")

    return (pad_left, pad_right)

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, in_dim, groups=1, use_convlr= True):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_convlr= use_convlr

        #if stride==1: #default pytorch
        #    pad="same"
        #else:
        self.pad = _calc_padding(in_dim,self.kernel_size,self.stride)
        #print(in_dim,self.kernel_size,self.stride,self.pad)
        self.out_dim = _calc_out_dim(in_dim,self.kernel_size,self.stride,self.pad)

        self.padder = torch.nn.ConstantPad1d(self.pad,0)
        #self.padder_r = torch.nn.ConstantPad1d(self.pad,0)
        #print("myconv1dpadsame: ", in_channels, out_channels, kernel_size, stride, groups)

        #pit compatible groups 2 conv:
        if self.groups==2 and self.use_convlr==True:
            #f = PITFeaturesMasker(self.out_channels//2)
            #t = PITTimestepMasker(self.kernel_size)
            #d = PITDilationMasker(1)
            self.conv_l = torch.nn.Conv1d(   #PITConv1d(
                in_channels=self.in_channels//2,
                out_channels=self.out_channels//2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=1,
                padding="valid")
                #,
                #f,t,d)

            self.conv_r = torch.nn.Conv1d(  #PITConv1d(
                in_channels=self.in_channels//2,
                out_channels=self.out_channels//2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=1,
                padding="valid")
                #,
                #f,t,d)
        else: # groups==1:
            self.conv = torch.nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                padding="valid")
                #padding=pad[0])

    def forward(self, x):
        """
        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        #p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)

        p = (out_dim - 1) * self.stride + self.kernel_size - in_dim
        p = torch.nn.functional.relu(p)

        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(x, (pad_left, pad_right), "constant", 0)
        """
        if self.groups==2 and self.use_convlr==True:
            #net_l,net_r=torch.chunk(net, 2, dim=1)

            net_l=x[:,:self.in_channels//2]
            net_r=x[:,self.in_channels//2:]
            net_l = self.padder(net_l)
            net_l = self.conv_l(net_l)
            net_r = self.padder(net_r)
            net_r = self.conv_r(net_r)
            net = torch.cat((net_l,net_r), dim=1)
        else:
            net = self.padder(x)
            net = self.conv(net)

        return net

class DepthConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, in_dim, groups=1, use_convlr= True):
        super(DepthConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_convlr=use_convlr

        self.pad = _calc_padding(in_dim,self.kernel_size,self.stride)
        self.padder = torch.nn.ConstantPad1d(self.pad,0)

        #assert self.out_channels%self.in_channels==0
        #print("depthconv1dpadsame: ", in_channels, out_channels, kernel_size, stride, groups)
        self.depthwise_conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.in_channels, #apply conv separately on each channel
            padding="valid")

        if self.groups==2 and self.use_convlr==True:

            self.pointwise_conv_l = torch.nn.Conv1d(
                in_channels=self.in_channels//2,
                out_channels=self.out_channels//2,
                kernel_size=1,
                stride=1,
                groups=1, padding='same')
            self.pointwise_conv_r = torch.nn.Conv1d(
                in_channels=self.in_channels//2,
                out_channels=self.out_channels//2,
                kernel_size=1,
                stride=1,
                groups=1, padding='same')
        else:
            self.pointwise_conv = torch.nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                groups=self.groups, padding='same')

    def forward(self, x):

        #net = x

        """
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        """
        #net = F.pad(x, self.pad, "constant", 0)
        net = self.padder(x)

        net = self.depthwise_conv(net)

        if self.groups==2 and self.use_convlr==True:
            net_l=net[:,:self.in_channels//2]
            net_r=net[:,self.in_channels//2:]
            net_l = self.pointwise_conv_l(net_l)
            net_r = self.pointwise_conv_r(net_r)
            net = torch.cat((net_l,net_r), dim=1)
        else:
            net = self.pointwise_conv(net)

        return net

class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size, in_dim):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.pad = _calc_padding(in_dim,self.kernel_size,self.stride)
        #print("in_dim: ",in_dim,"kernel: ",self.kernel_size,"stride: ",self.stride,"pad: ",self.pad)
        self.out_dim=_calc_out_dim(in_dim,self.kernel_size,s = self.kernel_size,p = self.pad)

        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size) #,padding=pad[0]
        self.padder = torch.nn.ConstantPad1d(self.pad,0)

    def forward(self, x):

        #net = x
        """
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        """
        net = self.padder(x) #F.pad(x, self.pad, "constant", 0)
        net = self.max_pool(net)

        return net

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16, se_ch_low=4, in_dim=None):
        super().__init__()
        self.c=c
        self.in_dim = in_dim
        h = c // r
        if h<4: h = se_ch_low
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, h, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(h, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #bs, c, _ = x.shape
        y = torch.squeeze(self.squeeze(x), dim=2) #.view(-1, self.c) view(bs, c) is the same as squeeze(,dim=2) because is 1d and output_size==1
        y = torch.unsqueeze(self.excitation(y),dim=-1) #self.excitation(y).view(-1, self.c, 1)  .view(bs, c, 1) same as unsqueeze(,dim=-1) because y.shape is [bs,c]
        return torch.multiply(x, torch.cat([y for _ in range(self.in_dim)],dim=2)) #y.expand_as(x)

class SuperBasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample,
                 use_bn, use_do, is_first_block=False, is_se=False, se_ch_low=4,
                 use_plinio=False,in_dim=625, use_ch_conv=False, use_convlr=True):
        super(SuperBasicBlock, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.is_se = is_se
        self.use_bn = use_bn
        self.use_do = use_do

        self.use_ch_conv=use_ch_conv

        # the first conv
        if not self.is_first_block:
            if self.use_bn: self.bn1 = nn.BatchNorm1d(in_channels)
            self.relu1 = nn.ReLU()
            if self.use_do: self.do1 = nn.Dropout(p=0.5)

        self.in_dim=in_dim
        self.use_convlr=use_convlr


        #test depthwise-pointwise
        if use_plinio == "supernet":
            branches = [
                MyConv1dPadSame(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                in_dim=self.in_dim,
                groups=self.groups,
                use_convlr=self.use_convlr),

                DepthConv1dPadSame(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                in_dim=self.in_dim,
                groups=self.groups,
                use_convlr=self.use_convlr),
                ]
            #branches=[nn.Sequential(b,
            #nn.BatchNorm1d(out_channels),
            #nn.ReLU(),
            #nn.Dropout(p=0.5)) for b in branches]

            if not self.downsample and in_channels==out_channels: #self.in_dim==branches[0].out_dim
                branches.append(nn.Identity())
            else:
                self.in_dim=branches[0].out_dim

            self.conv1 = SuperNetModule(branches)
        else:
            self.conv1 = MyConv1dPadSame(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                in_dim=self.in_dim,
                groups=self.groups,
                use_convlr=self.use_convlr)
            self.in_dim=self.conv1.out_dim

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)

        if use_plinio == "supernet":
            self.conv2 = SuperNetModule([
                MyConv1dPadSame(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                in_dim=self.in_dim,
                groups=self.groups,
                use_convlr=self.use_convlr),

                DepthConv1dPadSame(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                in_dim=self.in_dim,
                groups=self.groups,
                use_convlr=self.use_convlr),

                nn.Identity()])
        else:
            #if self.groups==1:
            self.conv2 = MyConv1dPadSame(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                in_dim=self.in_dim,
                groups=self.groups,
                use_convlr=self.use_convlr)

        if self.downsample: self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride, in_dim=self.in_dim)

        ch1 = (self.out_channels-self.in_channels)//2
        ch2 = self.out_channels-self.in_channels-ch1

        if self.use_ch_conv:
            self.ch_conv=nn.Conv1d(in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                padding='same')
        else:
            self.ch_padder = torch.nn.ConstantPad1d((0, 0, ch1, ch2),0)
        # Squeeze and excitation layer
        if self.is_se:  self.se = SE_Block(out_channels, 16, se_ch_low, in_dim=self.in_dim)

    def forward(self, x):
        #if x.shape[0]<4:    self.use_bn = False

        identity = x

        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        #if self.out_channels != self.in_channels:
            #out_channels=out.size()[1]
            #in_channels=identity.size()[1]

            #ch1 = (self.out_channels-self.in_channels)//2
            #ch2 = self.out_channels-self.in_channels-ch1

            #ch1 = (out_channels-in_channels)//2
            #ch2 = out_channels-in_channels-ch1
        if self.use_ch_conv:
            identity = self.ch_conv(identity)
        else:
            identity = self.ch_padder(identity)

        # Squeeze and excitation layer
        if self.is_se:
            out = self.se(out)
        # shortcut
        out = torch.add(out, identity)

        return out