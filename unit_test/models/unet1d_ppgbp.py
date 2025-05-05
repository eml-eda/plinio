import torch.nn as nn
import functools
from math import floor
import torch

param_model={
  'N_epoch': 256,
  'batch_size': 256,
  'lr': 0.001,
  'input_size': 1,
  'output_size': 625,
  'output_channel': 128,
  'layers': [2, 2],
  'sample_step': 1,
  'loss': [ 'mse', 'peak_loss'],
  'loss_w': [1, 1, 1],
  'layer_order': 'new',
  'normalizer': 'pinst',
  'activation': 'prelu',
}

def get_normalizer(normalizer):
    if normalizer == 'inst':
        return nn.InstanceNorm1d
    elif normalizer == 'pinst':
        return functools.partial(nn.InstanceNorm1d, affine=True)
    else:
        return nn.BatchNorm1d

def cal_kernel_padding(kernel_size, ksize_plus=0):
    if isinstance(kernel_size, (list, tuple)):
        ksize = list()
        psize = list()
        stride = list()
        for k in kernel_size:
            ks = k + ksize_plus if k > 1 else 1
            ksize.append(ks)
            psize.append(ks // 2)
            stride.append(2 if k > 1 else 1)
        return tuple(ksize), tuple(psize), tuple(stride)
    else:
        ksize = kernel_size + ksize_plus if kernel_size > 1 else 1
        psize = ksize // 2
        stride = 2 if kernel_size > 1 else 1
        return ksize, psize, stride

def check_is_stride(stride, output_size=None, scaler='down'):
    if output_size is not None and scaler == 'up':
        return True
    if isinstance(stride, (list, tuple)):
        for s in stride:
            if s > 1:
                return True
        return False
    else:
        return stride > 1

def get_activator(activator, in_planes=None):
    if activator == 'prelu':
        return functools.partial(nn.PReLU, num_parameters=in_planes)
    elif activator == 'relu':
        return functools.partial(nn.ReLU, inplace=False)
    else:
        return None

def get_upscaler(stride, output_size=None):
    mode = 'linear'
    if output_size is None:
        if not isinstance(stride, (list, tuple)):
            stride = (stride, )
        layer_upsample = nn.Upsample(scale_factor=stride, mode=mode, align_corners=True)
    else:
        layer_upsample = nn.Upsample(size=output_size, mode=mode, align_corners=True)
    return layer_upsample

def _calc_out_dim(in_dim,k,s,p):
    if isinstance(p, list):
            n_out = floor((in_dim+p[0]+p[1]-k)/s)+1
    elif isinstance(p, int):
        n_out = floor((in_dim+2*p-k)/s)+1
    return n_out

class _ConvModernNd(nn.Module):
    '''N1D Modern convolutional layer
    The implementation for the modern convolutional layer. The modern
    convolutional layer is a stack of convolution, normalization and
    activation.
    '''
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, output_size=None,
                 normalizer='pinst', activator='prelu', scaler='down',
                 use_bias=True, is_last=False): #, exp_hidden=False):
        '''Initialization.
        Arguments:
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            kernel_size: the kernel size of this layer.
            stride: the stride size of this layer.
            padding: the padding size of the convolutional layer.
            output_size: the size of the output data. This option is only used
                         when "scaler=up". When setting this value, the size
                         of the up-sampling would be given explicitly and
                         the option "stride" would not be used.
            normalizer: the normalization method, could be:
                        - "batch": Batch normalization.
                        - "inst": Instance normalization.
                        - "pinst": Instance normalization with tunable
                                   rescaling parameters.
                        - "null": Without normalization, would falls back to
                                  the "convolution + activation" form.
            activator: activation method, could be:
                       - "prelu", - "relu", - "null".
            layer_order: the sub-layer composition order, could be:
                         - "new": norm + activ + conv
                         - "old": conv + norm + activ
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
        '''
        super().__init__()
        is_stride = check_is_stride(stride, output_size=output_size, scaler=scaler)
        seq = []

        if normalizer in ('batch', 'inst', 'pinst'):
            normalizer_op = get_normalizer(normalizer)

            new_activator = get_activator(activator, in_planes=in_planes)
            if new_activator is not None:
                seq.append(
                    new_activator()
                )

            if (not is_stride) or scaler == 'down':
                #if original_model:
                #    seq.append(
                #        nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
                #    )
                #else:
                seq.extend((
                    nn.ConstantPad1d(padding,0),
                    nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding="valid", bias=use_bias),
                ))
            elif is_stride and scaler == 'up':
                #if original_model:
                #    seq.extend((
                #        get_upscaler(stride=stride, output_size=output_size, order=order),
                #        nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias), #padding
                #    ))
                #else:
              seq.extend((
                  get_upscaler(stride=stride, output_size=output_size),
                  nn.ConstantPad1d(padding,0),
                  nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding="valid", bias=use_bias), #padding
              ))
            else:
                raise ValueError('modules.conv: The arguments "stride" and "scaler" should be valid.')

            if not is_last:
                seq.append(
                    normalizer_op(out_planes, track_running_stats=True)
                )
        else:
            raise ValueError('modules.conv: The arguments "normalizer"  should be valid.')
        self.mconv = nn.Sequential(*seq)

    def forward(self, x):
        x = self.mconv(x)
        return x

class _BlockConvStkNd(nn.Module):
    '''Create the 1D stacked modern convolutional block.
    Each block contains several stacked convolutional layers.
    '''
    def __init__(self, in_planes, out_planes, in_dim, hidden_planes=None,
                 kernel_size=3, padding=1, stride=1, stack_level=3, ex_planes=0,
                 scaler='down', export_hidden=False,
                 normalizer='batch', activation="prelu",use_bias=True,  is_last=False):
        '''Initialization
        Arguments:
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        Arguments (optional):
            hidden_planes: the channel number of the first hidden layer, would
                           also used as the base of the following channels. If
                           not set, would use "out_planes" as the default
                           value.
            kernel_size: the kernel size of the convolutional layers.
            padding: the padding size of the convolutional layers.
            stride: the stride size of the convolutional layers.
            stack_level: the number of convolutional layers in this block,
                         requiring to be >= 1.
            ex_planes: the channel number of the second input data. This value
                       is =0 in most time, but >0 during the decoding phase
                       of the U-Net. the extra input would be concatenated
                       with the input data.
            scaler: scaling method. Could be "down" or "up". When using "down",
                    the argument "stride" would be used for down-sampling; when
                    using "up", "stride" would be used for up-sampling
                    (equivalent to transposed convolution).
            export_hidden: whether to export the hidden layer as the second
                           output. This option is only used during the encoding
                           phase of the U-Net.
            layer_order: layer order in _ConvModernNd
            normalizer: normalizer in _ConvModernNd
        '''
        super().__init__()

        self.with_exinput = isinstance(ex_planes, int) and ex_planes > 0
        self.in_dim=in_dim
        if self.with_exinput:

            self.x0_size=self.in_dim[0]
            self.x1_size=self.in_dim[1]
            self.in_dim = min(self.in_dim[0], self.in_dim[1])
            self.get_size=self.in_dim

        self.export_hidden = export_hidden
        hidden_planes = hidden_planes if (isinstance(hidden_planes, int) and hidden_planes > 0) else out_planes

        self.conv_list = nn.ModuleList()
        for i in range(stack_level - 1):
            self.conv_list.append(
                _ConvModernNd( (in_planes + ex_planes) if i == 0 else hidden_planes, hidden_planes,
                              kernel_size=kernel_size, padding=padding, stride=1, scaler='down',
                              normalizer=normalizer, activator=activation,use_bias=use_bias)
            )
            self.in_dim=_calc_out_dim(self.in_dim,k=kernel_size,s=1,p=padding)
            self.sk_in_dim=self.in_dim
        self.conv_scale = _ConvModernNd( hidden_planes if stack_level > 1 else (in_planes + ex_planes), out_planes,
                                        kernel_size=kernel_size, padding=padding, stride=stride, scaler=scaler,
                                        normalizer=normalizer, activator=activation ,use_bias=use_bias,is_last=is_last)

        for sublayer in self.conv_scale.mconv:
            if isinstance(sublayer, nn.Upsample):
                self.in_dim=self.in_dim*sublayer.scale_factor[0]
                s=stride
                if check_is_stride(stride,scaler=scaler) and scaler == 'up':
                    s=1
                self.in_dim=_calc_out_dim(self.in_dim,k=kernel_size,s=s,p=padding)
                #is_upsample=True
                break
        else: # no break
            self.in_dim=_calc_out_dim(self.in_dim,k=kernel_size,s=stride,p=padding)

    def cropping1d(self, x0, x1):

        get_size = self.get_size
        x0_shift = (self.x0_size - get_size) // 2
        x1_shift = (self.x1_size - get_size) // 2

        return torch.cat((x0[:,:,x0_shift: x0_shift + get_size],
                          x1[:,:,x1_shift: x1_shift + get_size]), dim=1)

    def forward(self, *x):
        if self.with_exinput:
            x = self.cropping1d(x[0], x[1])
        else:
            x = x[0]
        for layer in self.conv_list:                               #PIT Sequential
            x = layer(x)
        res = self.conv_scale(x)
        if self.export_hidden:
            return res, x
        else:
            return res

class UNet1d(nn.Module):
    '''1D convolutional network based U-Net
    code inspired by mdnc library simplified, used for test.
    The network would down-sample and up-sample the input data according to
    the network depth. The depth is given by the length of the argument
    "layers".
    '''
    def __init__(self, channel=128, layers=[2,2], in_dim=625, kernel_size=3, in_planes=1,
                 out_planes=1, normalizer='pinst', activation="prelu", use_bias=True, input_shape=[1,1,625]):
        '''Initialization
        Arguments:
            order: the network dimension. For example, when order=2, the
                   nn.Conv2d would be used.
            channel: the channel number of the first layer, would also used
                     as the base of the following channels.
            layers: a list of layer numbers. Each number represents the number
                    of convolutional layers of a stage. The stage numer, i.e.
                    the depth of the network is the length of this list.
            kernel_size: the kernel size of each block.
            in_planes: the channel number of the input data.
            out_planes: the channel number of the output data.
        '''
        super(UNet1d, self).__init__()
        ConvNd = nn.Conv1d #get_convnd(order=order) #only conv1d supported, can be extended
        self.input_shape = input_shape
        normalizer_op = get_normalizer(normalizer)
        ksize_e, psize_e, _ = cal_kernel_padding(kernel_size, ksize_plus=2)
        self.conv_first = nn.Sequential(
                            #normalizer_op(in_planes, track_running_stats=True),
                            nn.ConstantPad1d(psize_e,0),
                            nn.Conv1d(in_planes, channel, kernel_size=ksize_e, stride=1, padding="valid", bias=use_bias),
                            normalizer_op(channel, track_running_stats=True),
                        )

        in_dim_down_route = list()
        in_dim=_calc_out_dim(in_dim,k=ksize_e,s=1,p=psize_e)
        in_dim_down_route.append(in_dim)

        self.conv_down_list = nn.ModuleList()
        ksize, psize, stride = cal_kernel_padding(kernel_size)

        self.conv_down_list.append(
            _BlockConvStkNd(channel, channel, in_dim, kernel_size=ksize, padding=psize,
                            stride=stride, stack_level=layers[0], ex_planes=0, scaler='down',
                            export_hidden=True, normalizer=normalizer, activation=activation ,use_bias=use_bias))

        in_dim=self.conv_down_list[-1].in_dim
        in_dim_down_route.append(self.conv_down_list[-1].sk_in_dim)

        for n_l in layers[1:-1]:
            self.conv_down_list.append(
                _BlockConvStkNd(channel, channel * 2, in_dim, kernel_size=ksize, padding=psize,
                                stride=stride, stack_level=n_l, ex_planes=0, scaler='down',
                                export_hidden=True, normalizer=normalizer, activation=activation, use_bias=use_bias))
            in_dim=self.conv_down_list[-1].in_dim
            in_dim_down_route.append(self.conv_down_list[-1].sk_in_dim)
            channel = channel * 2
        self.conv_middle_up = _BlockConvStkNd(channel, channel, in_dim, hidden_planes=channel * 2, kernel_size=ksize, padding=psize,
                                              stride=stride, stack_level=layers[-1], ex_planes=0, scaler='up',
                                              normalizer=normalizer, activation=activation ,use_bias=use_bias) #, is_next_cat=True)

        in_dim=self.conv_middle_up.in_dim

        self.conv_up_list = nn.ModuleList()
        # Up scaling route
        for n_l in layers[-2:0:-1]:
            self.conv_up_list.append(
                _BlockConvStkNd(channel, channel // 2, [in_dim,in_dim_down_route.pop()], hidden_planes=channel, kernel_size=ksize, padding=psize,
                                stride=stride, stack_level=n_l, ex_planes=channel, scaler='up',
                                normalizer=normalizer, activation=activation ,use_bias=use_bias))
            in_dim=self.conv_up_list[-1].in_dim
            channel = channel // 2
        self.conv_up_list.append(
            _BlockConvStkNd(channel, channel, [in_dim,in_dim_down_route.pop()], hidden_planes=channel, kernel_size=ksize, padding=psize,
                            stride=1, stack_level=layers[0], ex_planes=channel, scaler='down', #?
                            normalizer=normalizer, activation=activation ,use_bias=use_bias, is_last=True)) #, is_next_cat=False))
        #in_dim=self.conv_up_list[-1].in_dim

        self.conv_final = nn.Sequential(
                          nn.ConstantPad1d(psize_e,0),
                          ConvNd(channel, out_planes, kernel_size=ksize_e, stride=1, padding="valid", bias=True)
                          )
    def forward(self, x):
        x = self.conv_first(x)
        x_down_route = list()
        for layer in self.conv_down_list:
            x, x_sk = layer(x)
            x_down_route.append(x_sk)
        x = self.conv_middle_up(x)
        x_down_route.reverse()
        for layer, x_sk in zip(self.conv_up_list, x_down_route):
            x = layer(x, x_sk)
        x = self.conv_final(x)
        return x




