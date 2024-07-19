from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Wathna.config import config as cfg
from pathlib import Path
import ctypes

import numpy as np
import pickle
import math
import time

# Zip the pickle file
import bz2file as bz2
libpool = ctypes.CDLL('src/Junaid/convolution_cuda.so')
libconv = ctypes.CDLL('src/Junaid/SEFP.so')
warnings.simplefilter("ignore", UserWarning)

def Save_File(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def convert_to_hex(value):
    # We will Use Single-Precision, Truncated and Rounding into Brain Floating Point
    # IEEE754 Single-Precision: Sign=1, Exponent_Bit=8, Mantissa_Bit=23
    Exponent_Bit = 8
    Mantissa_Bit = 23
    Binary_Value1 = Floating2Binary(value, Exponent_Bit, Mantissa_Bit)
    Hexadecimal_Value1 = hex(int(Binary_Value1, 2))[2:]
    # Truncating and Rounding
    Floating_Hexadecimal = Truncating_Rounding(Hexadecimal_Value1)
    if len(Floating_Hexadecimal) < 4:
        Brain_Floating_Hexadecimal = Floating_Hexadecimal.zfill(4)
    else:
        Brain_Floating_Hexadecimal = Floating_Hexadecimal
    return Brain_Floating_Hexadecimal


def save_file(fname, data, module=[], layer_no=[], save_txt=False, save_hex=False, phase=[]):
    # print(f"Type of data: {type(data)}")
    if save_txt or save_hex:
        if type(data) is dict:
            for _key in data.keys():
                _fname = fname + f'_{_key}'
                save_file(_fname, data[_key])

        else:
            if module == [] and layer_no == []:
                Out_Path = f'Outputs_Python_2Iterations/{os.path.split(fname)[0]}'
                fname = os.path.split(fname)[1]
            else:
                Out_Path = f'Outputs_Python_2Iterations/By_Layer/'
                if layer_no != []: Out_Path += f'Layer{layer_no}/'
                if module != []: Out_Path += f'{module}/'
                if phase != []: Out_Path += f'{phase}/'
                fname = fname

            if save_txt: filename = os.path.join(Out_Path, fname + '.txt')
            if save_hex: hexname = os.path.join(Out_Path, fname + '_hex.txt')

            Path(Out_Path).mkdir(parents=True, exist_ok=True)

            if torch.is_tensor(data):
                try:
                    data = data.detach()
                except:
                    pass
                data = data.numpy()

            if save_txt: outfile = open(filename, mode='w')
            if save_txt: outfile.write(f'{data.shape}\n')

            if save_hex: hexfile = open(hexname, mode='w')
            if save_hex: hexfile.write(f'{data.shape}\n')

            if len(data.shape) == 0:
                if save_txt: outfile.write(f'{data}\n')
                if save_hex: hexfile.write(f'{data}\n')
                pass
            elif len(data.shape) == 1:
                for x in data:
                    if save_txt: outfile.write(f'{x}\n')
                    if save_hex: hexfile.write(f'{convert_to_hex(x)}\n')
                    pass
            else:
                w, x, y, z = data.shape
                # if w != 0:
                #     Out_Path += f'img{w+1}'
                for _i in range(w):
                    for _j in range(x):
                        for _k in range(y):
                            for _l in range(z):
                                _value = data[_i, _j, _k, _l]
                                if save_txt: outfile.write(f'{_value}\n')
                                if save_hex: hexfile.write(f'{convert_to_hex(_value)}\n')
                                pass

            if save_hex: hexfile.close()
            if save_txt: outfile.close()

            if save_txt: print(f'\t\t--> Saved {filename}')
            if save_hex: print(f'\t\t--> Saved {hexname}')
    # else:
    # print(f'\n\t\t--> Saved {filename}')


def save_cache(fname, data):
    if type(data) is dict:
        for _key in data.keys():
            _fname = fname + f'_{_key}'
            save_file(_fname, data[_key])

    else:
        Path(os.path.split(fname)[0]).mkdir(parents=True, exist_ok=True)
        fname = fname + '.txt'

        if torch.is_tensor(data):
            try:
                data = data.detach()
            except:
                pass
            data = data.numpy()

        outfile = open(fname, mode='w')
        outfile.write(f'{data.shape}\n')

        if len(data.shape) == 0:
            outfile.write(f'{data}\n')
        elif len(data.shape) == 1:
            for x in data:
                outfile.write(f'{x}\n')
        else:
            w, x, y, z = data.shape
            for _i in range(w):
                for _j in range(x):
                    for _k in range(y):
                        for _l in range(z):
                            outfile.write(f'{data[_i, _j, _k, _l]}\n')
        outfile.close()

        print(f'\n\t\t--> Saved {fname}')


class DeepConvNet(object):
    """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and 
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:
  
  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a Python_ReLU nonlinearity, and an optional
  pooling layer. After L-1 such macro layers, a single fully-connected layer
  is used to predict the class scores.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

    def __init__(self, input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 slowpool=True,
                 num_classes=10, weight_scale=1e-3, reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float32, device='cpu'):
        """
    Initialize a new network.

    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: List of length (L - 1) giving the number of convolutional
      filters to use in each macro layer.
    - max_pools: List of integers giving the indices of the macro layers that
      should have max pooling (zero-indexed).
    - batchnorm: Whether to include batch normalization in each macro layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights, or the string "kaiming" to use Kaiming initialization instead
    - reg: Scalar giving L2 regularization strength. L2 regularization should
      only be applied to convolutional and fully-connected weight matrices;
      it should not be applied to biases or to batchnorm scale and shifts.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'    
    """
        self.params = {}
        self.num_layers = len(num_filters) + 1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype
        self.slowpool = slowpool
        self.num_filters = num_filters
        self.save_pickle = False
        self.save_output = False
        self.save_debug_data = False
        self.save_16_data = False

        if device == 'cuda':
            device = 'cuda:0'

        ############################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights,        #
        # biases, and batchnorm scale and shift parameters should be stored in the #
        # dictionary self.params.                                                  #
        #                                                                          #
        # Weights for conv and fully-connected layers should be initialized        #
        # according to weight_scale. Biases should be initialized to zero.         #
        # Batchnorm scale (gamma) and shift (beta) parameters should be initilized #
        # to ones and zeros respectively.                                          #
        ############################################################################
        # Replace "pass" statement with your code
        filter_size = 3
        conv_param = {'stride': 1, 'pad': 1}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        pred_filters, H_out, W_out = input_dims
        HH = filter_size
        WW = filter_size
        # print('num_filters:', num_filters)
        for i, num_filter in enumerate(num_filters):
            H_out = int(1 + (H_out + 2 * conv_param['pad'] - HH) / conv_param['stride'])
            W_out = int(1 + (W_out + 2 * conv_param['pad'] - WW) / conv_param['stride'])
            if self.batchnorm:
                self.params['running_mean{}'.format(i)] = torch.zeros(num_filter, dtype=dtype, device=device)
                self.params['running_var{}'.format(i)] = torch.zeros(num_filter, dtype=dtype, device=device)
                self.params['gamma{}'.format(i)] = 0.01 * torch.randn(num_filter, device=device, dtype=dtype)
                self.params['beta{}'.format(i)] = 0.01 * torch.randn(num_filter, device=device, dtype=dtype)
            if i in max_pools:
                H_out = int(1 + (H_out - pool_param['pool_height']) / pool_param['stride'])
                W_out = int(1 + (W_out - pool_param['pool_width']) / pool_param['stride'])
            if weight_scale == 'kaiming':
                self.params['W{}'.format(i)] = kaiming_initializer(num_filter, pred_filters, K=filter_size, relu=True,
                                                                   device=device, dtype=dtype)
            else:
                self.params['W{}'.format(i)] = torch.zeros(num_filter, pred_filters, filter_size, filter_size,
                                                           dtype=dtype, device=device)
                self.params['W{}'.format(i)] += weight_scale * torch.randn(num_filter, pred_filters, filter_size,
                                                                           filter_size, dtype=dtype, device=device)
            pred_filters = num_filter

        i += 1

        if weight_scale == 'kaiming':
            self.params['W{}'.format(i)] = kaiming_initializer(125, 1024, K=1, relu=False, device=device, dtype=dtype)

        self.params['b{}'.format(i)] = torch.zeros(125, dtype=dtype, device=device)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_params object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
            for i, num_filter in enumerate(num_filters):
                self.bn_params[i]['running_mean'] = torch.zeros(num_filter, dtype=dtype, device=device)
                self.bn_params[i]['running_var'] = torch.zeros(num_filter, dtype=dtype, device=device)

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 3  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
            'reg': self.reg,
            'dtype': self.dtype,
            'params': self.params,
            'num_layers': self.num_layers,
            'max_pools': self.max_pools,
            'batchnorm': self.batchnorm,
            'bn_params': self.bn_params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def train(self, X, gt_boxes=None, gt_classes=None, num_boxes=None):
        
        if self.save_module_output:
            self.save_txt = self.save_in_dec_format
            self.save_hex = self.save_in_hex_format

        if self.forward_prop:
            out, cache, FOut = self.forward(X)
            if self.save_pickle:
                Path("Temp_Files/Python").mkdir(parents=True, exist_ok=True)
                with open('Temp_Files/Python/Forward_Out_last_layer.pickle', 'wb') as handle:
                    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open('Temp_Files/Python/Forward_cache.pickle', 'wb') as handle:
                    pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open('Temp_Files/Python/Forward_Out_all_layers.pickle', 'wb') as handle:
                    pickle.dump(FOut, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading previous files for Forward Propagation.")
            with open('Temp_Files/Python/Forward_Out_last_layer.pickle', 'rb') as handle:
                out = pickle.load(handle)
                out.requires_grad = True
                out.retain_grad()
            with open('Temp_Files/Python/Forward_Out_all_layers.pickle', 'rb') as handle:
                FOut = pickle.load(handle)
            with open('Temp_Files/Python/Forward_cache.pickle', 'rb') as handle:
                cache = pickle.load(handle)

        if self.cal_loss:
            # print(out.shape)
            loss, loss_grad = self.loss(out, gt_boxes=gt_boxes, gt_classes=gt_classes, num_boxes=num_boxes)

                
            if self.save_pickle:
                with open('Temp_Files/Python/loss.pickle', 'wb') as handle:
                    pickle.dump(loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open('Temp_Files/Python/loss_gradients.pickle', 'wb') as handle:
                    pickle.dump(loss_grad, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading previous files for Loss Calculation.")
            with open('Temp_Files/Python/loss.pickle', 'rb') as handle:
                loss = pickle.load(handle)
            with open('Temp_Files/Python/loss_gradients.pickle', 'rb') as handle:
                loss_grad = pickle.load(handle)

        if self.backward_prop:
            lDout, grads = self.backward(loss_grad, cache)
            if self.save_pickle:
                with open('Temp_Files/Python/Backward_lDout.pickle', 'wb') as handle:
                    pickle.dump(lDout, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open('Temp_Files/Python/Backward_grads.pickle', 'wb') as handle:
                    pickle.dump(grads, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Loading previous files for Backwards Propagation.")
            with open('Temp_Files/Python/Backward_lDout.pickle', 'rb') as handle:
                lDout = pickle.load(handle)
            with open('Temp_Files/Python/Backward_grads.pickle', 'rb') as handle:
                grads = pickle.load(handle)

        return out, cache, loss, loss_grad, lDout, grads

    def forward(self, X):
        # print(f'\nThis is python-based forward propagation code\n')
        y = 1
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_params in self.bn_params:
                bn_params['mode'] = mode

        scores = None
        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        slowpool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 1}
        cache = {}
        Out = {}
        self.phase = 'Forward'
        temp_Out = {}
        temp_cache = {}
        
        #0
        temp_Out[0], temp_cache['0'] = Python_Conv_Pool.forward(X,
                                                    self.params['W0'],
                                                    conv_param,
                                                    pool_param,
                                                    layer_no=0,
                                                    save_txt=False,
                                                    save_hex=False,
                                                    phase=self.phase,
                                                    )
        
        mean, var = Cal_mean_var.forward(temp_Out[0], layer_no=0, save_txt=False, save_hex=False, phase=self.phase)

        Out[0], cache['0'] = Python_Conv_BatchNorm_ReLU_Pool.forward(X,
                                                    self.params['W0'],
                                                    self.params['gamma0'],
                                                    self.params['beta0'],
                                                    conv_param,
                                                    self.bn_params[0],
                                                    mean,
                                                    var,
                                                    pool_param,
                                                    layer_no=0,
                                                    save_txt=False,
                                                    save_hex=False,
                                                    phase=self.phase,
                                                    )
        #1 
        temp_Out[1], temp_cache['1'] = Python_Conv.forward(Out[0],
                                                    self.params['W1'],
                                                    conv_param
                                                    )
        
        mean, var = Cal_mean_var.forward(temp_Out[1], layer_no=1, save_txt=False, save_hex=False, phase=self.phase)

        Out[1], cache['1'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[0],
                                                    self.params['W1'],
                                                    self.params['gamma1'],
                                                    self.params['beta1'],
                                                    conv_param,
                                                    self.bn_params[1],
                                                    mean,
                                                    var,
                                                    pool_param,
                                                    layer_no=1,
                                                    save_txt=False,
                                                    save_hex=False,
                                                    phase=self.phase,
                                                    )
        #2
        temp_Out[2], temp_cache['2'] = Python_Conv.forward(Out[1],
                                                    self.params['W2'],
                                                    conv_param
                                                    )
        
        mean, var = Cal_mean_var.forward(temp_Out[2], layer_no=2, save_txt=False, save_hex=False, phase=self.phase)

        Out[2], cache['2'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[1],
                                                    self.params['W2'],
                                                    self.params['gamma2'],
                                                    self.params['beta2'],
                                                    conv_param,
                                                    self.bn_params[2],
                                                    mean,
                                                    var,
                                                    pool_param,
                                                    layer_no=2,
                                                    save_txt=False,
                                                    save_hex=False,
                                                    phase=self.phase,
                                                    )
        #3   
        temp_Out[3], temp_cache['3'] = Python_Conv.forward(Out[2],
                                                    self.params['W3'],
                                                    conv_param
                                                    )
        
        mean, var = Cal_mean_var.forward(temp_Out[3], layer_no=3, save_txt=False, save_hex=False, phase=self.phase)

        Out[3], cache['3'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[2],
                                                    self.params['W3'],
                                                    self.params['gamma3'],
                                                    self.params['beta3'],
                                                    conv_param,
                                                    self.bn_params[3],
                                                    mean,
                                                    var,
                                                    pool_param,
                                                    layer_no=3,
                                                    save_txt=False,
                                                    save_hex=False,
                                                    phase=self.phase,
                                                    )

        #4
        temp_Out[4], temp_cache['4'] = Python_Conv.forward(Out[3],
                                                    self.params['W4'],
                                                    conv_param
                                                    )
        
        mean, var = Cal_mean_var.forward(temp_Out[4], layer_no=4, save_txt=False, save_hex=False, phase=self.phase)

        Out[4], cache['4'] = Python_Conv_BatchNorm_ReLU_Pool.forward(Out[3],
                                                    self.params['W4'],
                                                    self.params['gamma4'],
                                                    self.params['beta4'],
                                                    conv_param,
                                                    self.bn_params[4],
                                                    mean,
                                                    var,
                                                    pool_param,
                                                    layer_no=4,
                                                    save_txt=False,
                                                    save_hex=False,
                                                    phase=self.phase,
                                                    )
        
        #5
        temp_Out[5], temp_cache['5'] = Python_Conv.forward(Out[4],
                                                    self.params['W5'],
                                                    conv_param
                                                    )
        
        mean, var = Cal_mean_var.forward(temp_Out[5], layer_no=5, save_txt=False, save_hex=False, phase=self.phase)


        Out[5], cache['5'] = Python_Conv_BatchNorm_ReLU.forward(Out[4],
                                                    self.params['W5'],
                                                    self.params['gamma5'],
                                                    self.params['beta5'],
                                                    conv_param,
                                                    self.bn_params[5],
                                                    mean,
                                                    var,
                                                    layer_no=5,
                                                    save_txt=False,
                                                    save_hex=False,
                                                    phase=self.phase,
                                                    )
        #6
        temp_Out[6], temp_cache['6'] = Python_Conv.forward(Out[5],
                                                    self.params['W6'],
                                                    conv_param
                                                    )
        
        mean, var = Cal_mean_var.forward(temp_Out[6], layer_no=6, save_txt=False, save_hex=False, phase=self.phase)


        Out[6], cache['6'] = Python_Conv_BatchNorm_ReLU.forward(Out[5],
                                                    self.params['W6'],
                                                    self.params['gamma6'],
                                                    self.params['beta6'],
                                                    conv_param,
                                                    self.bn_params[6],
                                                    mean,
                                                    var,
                                                    layer_no=6,
                                                    save_txt=False,
                                                    save_hex=False,
                                                    phase=self.phase,
                                                    )

        #7
        temp_Out[7], temp_cache['7'] = Python_Conv.forward(Out[6],
                                                    self.params['W7'],
                                                    conv_param
                                                    )
        
        mean, var = Cal_mean_var.forward(temp_Out[7], layer_no=7, save_txt=False, save_hex=False, phase=self.phase)


        Out[7], cache['7'] = Python_Conv_BatchNorm_ReLU.forward(Out[6],
                                                    self.params['W7'],
                                                    self.params['gamma7'],
                                                    self.params['beta7'],
                                                    conv_param,
                                                    self.bn_params[7],
                                                    mean,
                                                    var,
                                                    layer_no=7,
                                                    save_txt=False,
                                                    save_hex=False,
                                                    phase=self.phase,
                                                    )
        
        #8
        conv_param['pad'] = 0
        temp_Out[8], temp_cache['8'] = Python_ConvB.forward(Out[7],
                                                            self.params['W8'],
                                                            self.params['b8'],
                                                            conv_param)

        mean, var = Cal_mean_var.forward(temp_Out[8], layer_no=8, save_txt=False, save_hex=False, phase=self.phase)

        Out[8], cache['8'] = Python_ConvB.forward(Out[7],
                                                self.params["W8"],
                                                self.params["b8"],
                                                conv_param
                                                )


        out = Out[8]
        # print('\n\nFwd Out', out.dtype, out[out != 0], '\n\n')
        if self.save_debug_data: Save_File("./Output_Sim_Python/Input_Image", X)
        if self.save_debug_data: Save_File("./Output_Sim_Python/Weight_Conv_0", self.params['W0'],)
        if self.save_debug_data: Save_File("./Output_Sim_Python/Output_Forward0", Out[0])
        
        Output_Image16 = out.to(torch.bfloat16)
        Output_Image16 = Output_Image16.to(torch.float32)
        if self.save_debug_data: 
           if self.save_16_data: 
            Save_File("./Output_Sim_Python/Output_Forward", Output_Image16)
           else:
            Save_File("./Output_Sim_Python/Output_Forward", out) 

        return out, cache, Out

    def loss(self, out, gt_boxes=None, gt_classes=None, num_boxes=None):

        # print('Calculating the loss and its gradients for python model.')
        out = torch.tensor(out, requires_grad=True)
        gt_boxes = gt_boxes.to("cuda")
        gt_classes = gt_classes.to("cuda")
        num_boxes = num_boxes
        scores = out.to("cuda")
        bsize, _, h, w = out.shape
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, 13 * 13 * 5, 5 + 20)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        output_variable = (delta_pred, conf_pred, class_score)
        output_data = [v.data for v in output_variable]
        gt_data = (gt_boxes, gt_classes, num_boxes)
        target_data = build_target(output_data, gt_data, h, w)

        target_variable = [v for v in target_data]

        box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)
        loss = box_loss + iou_loss + class_loss

        # print(f"\nLoss = {loss}\n")
        out = scores
        out.retain_grad()
        loss.backward(retain_graph=True)
        dout = out.grad.detach()
        # Loss_Gradient16 = dout.to(torch.bfloat16)
        # Loss_Gradient16 = Loss_Gradient16.to(torch.float32)
        # Save_File("./Output_Sim_Python/Loss_Gradient", Loss_Gradient16)
        return loss, dout

    def backward(self, dout, cache):
        grads = {}
        dOut = {}
        self.phase = 'Backwards'

        dOut[8], grads['W8'], grads['b8'] = Python_ConvB.backward(dout,
                                                                  cache['8'])

        
        dOut[7], grads['W7'], grads['gamma7'], grads['beta7'] = Python_Conv_BatchNorm_ReLU.backward(
            dOut[8],
            cache['7'],
            layer_no=7,
            save_txt=False,
            save_hex=False,
            phase=self.phase,
        )

    

        dOut[6], grads['W6'], grads['gamma6'], grads['beta6'] = Python_Conv_BatchNorm_ReLU.backward(
            dOut[7],
            cache['6'],
            layer_no=6,
            save_txt=False,
            save_hex=False,
            phase=self.phase,
        )

        dOut[5], grads['W5'], grads['gamma5'], grads['beta5'] = Python_Conv_BatchNorm_ReLU.backward(
            dOut[6],
            cache['5'],
            layer_no=5,
            save_txt=False,
            save_hex=False,
            phase=self.phase,
        )

        dOut[4], grads['W4'], grads['gamma4'], grads['beta4'] = Python_Conv_BatchNorm_ReLU_Pool.backward(
            dOut[5],
            cache['4'],
            layer_no=4,
            save_txt=False,
            save_hex=False,
            phase=self.phase,
        )

        dOut[3], grads['W3'], grads['gamma3'], grads['beta3'] = Python_Conv_BatchNorm_ReLU_Pool.backward(
            dOut[4],
            cache['3'],
            layer_no=3,
            save_txt=False,
            save_hex=False,
            phase=self.phase,
        )

        dOut[2], grads['W2'], grads['gamma2'], grads['beta2'] = Python_Conv_BatchNorm_ReLU_Pool.backward(
            dOut[3],
            cache['2'],
            layer_no=2,
            save_txt=False,
            save_hex=False,
            phase=self.phase,
        )

        dOut[1], grads['W1'], grads['gamma1'], grads['beta1'] = Python_Conv_BatchNorm_ReLU_Pool.backward(
            dOut[2],
            cache['1'],
            layer_no=1,
            save_txt=False,
            save_hex=False,
            phase=self.phase,
        )

        dOut[0], grads['W0'], grads['gamma0'], grads['beta0'] = Python_Conv_BatchNorm_ReLU_Pool.backward(
            dOut[1],
            cache['0'],
            layer_no=0,
            save_txt=False,
            save_hex=False,
            phase=self.phase,
        )
        
        if self.save_debug_data: 
            
            if self.save_16_data: 
                Input_Grad_Layer8_16 = dOut[8].to(torch.bfloat16)
                Input_Grad_Layer8_16 = Input_Grad_Layer8_16.to(torch.float32)
                Weight_Gradient_Layer8_16 = grads['W8'].to(torch.bfloat16)
                Weight_Gradient_Layer8_16 = Weight_Gradient_Layer8_16.to(torch.float32)
                Input_Grad_Layer7_16 = dOut[7].to(torch.bfloat16)
                Input_Grad_Layer7_16 = Input_Grad_Layer7_16.to(torch.float32)
                Weight_Gradient_Layer7_16 = grads['W7'].to(torch.bfloat16)
                Weight_Gradient_Layer7_16 = Weight_Gradient_Layer7_16.to(torch.float32)
                Input_Grad_Layer2_16 = dOut[2].to(torch.bfloat16)
                Input_Grad_Layer2_16 = Input_Grad_Layer2_16.to(torch.float32)
                Weight_Gradient_Layer2_16 = grads['W2'].to(torch.bfloat16)
                Weight_Gradient_Layer2_16 = Weight_Gradient_Layer2_16.to(torch.float32)
                
                Save_File("./Output_Sim_Python/Layer_8_Backward_Input_Gradient", Input_Grad_Layer8_16)
                Save_File("./Output_Sim_Python/Layer_8_Backward_Weight_Gradient", Weight_Gradient_Layer8_16)
                Save_File("./Output_Sim_Python/Layer_7_Backward_Input_Gradient", Input_Grad_Layer7_16)
                Save_File("./Output_Sim_Python/Layer_7_Backward_Weight_Gradient", Weight_Gradient_Layer7_16)
                Save_File("./Output_Sim_Python/Layer_2_Backward_Input_Gradient", Input_Grad_Layer2_16)
                Save_File("./Output_Sim_Python/Layer_2_Backward_Weight_Gradient", Weight_Gradient_Layer2_16)
            else:
                Save_File("./Output_Sim_Python/Layer_8_Backward_Input_Gradient", dOut[8])
                Save_File("./Output_Sim_Python/Layer_8_Backward_Weight_Gradient", grads['W8'])
                Save_File("./Output_Sim_Python/Layer_7_Backward_Input_Gradient", dOut[7])
                Save_File("./Output_Sim_Python/Layer_7_Backward_Weight_Gradient", grads['W7'])
                Save_File("./Output_Sim_Python/Layer_2_Backward_Input_Gradient", dOut[2])
                Save_File("./Output_Sim_Python/Layer_2_Backward_Weight_Gradient", grads['W2'])
                
            Save_File("./Output_Sim_Python/Bias_Grad", grads['b8'])
            Save_File("./Output_Sim_Python/Gamma_Gradient_Layer7", grads['gamma7'])    
            Save_File("./Output_Sim_Python/Beta_Gradient_Layer7", grads['beta7'])   
            Save_File("./Output_Sim_Python/Gamma_Gradient_Layer2",  grads['gamma2'])
            Save_File("./Output_Sim_Python/Beta_Gradient_Layer2", grads['beta2'])      
                         
        return dOut, grads


################################################################################
################################################################################
###############################  Functions Used  ###############################
################################################################################
################################################################################

class last_layer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv9 = nn.Conv2d(1024, 125, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        return self.conv9(x)


def build_target(output, gt_data, H, W):
    """
    Build the training target for output tensor

    Arguments:

    output_data -- tuple (delta_pred_batch, conf_pred_batch, class_pred_batch), output data of the yolo network
    gt_data -- tuple (gt_boxes_batch, gt_classes_batch, num_boxes_batch), ground truth data

    delta_pred_batch -- tensor of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred_batch -- tensor of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score_batch -- tensor of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2, ..)

    gt_boxes_batch -- tensor of shape (B, N, 4), ground truth boxes, normalized values
                       (x1, y1, x2, y2) range 0~1
    gt_classes_batch -- tensor of shape (B, N), ground truth classes (cls)
    num_obj_batch -- tensor of shape (B, 1). number of objects


    Returns:
    iou_target -- tensor of shape (B, H * W * num_anchors, 1)
    iou_mask -- tensor of shape (B, H * W * num_anchors, 1)
    box_target -- tensor of shape (B, H * W * num_anchors, 4)
    box_mask -- tensor of shape (B, H * W * num_anchors, 1)
    class_target -- tensor of shape (B, H * W * num_anchors, 1)
    class_mask -- tensor of shape (B, H * W * num_anchors, 1)

    """
    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    gt_boxes_batch = gt_data[0]
    gt_classes_batch = gt_data[1]
    num_boxes_batch = gt_data[2]

    bsize = delta_pred_batch.size(0)

    num_anchors = 5  # hard code for now

    # initial the output tensor
    # we use `tensor.new()` to make the created tensor has the same devices and data type as input tensor's
    # what tensor is used doesn't matter
    iou_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    iou_mask = delta_pred_batch.new_ones((bsize, H * W, num_anchors, 1)) * cfg.noobject_scale

    box_target = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 4))
    box_mask = delta_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    class_target = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))
    class_mask = conf_pred_batch.new_zeros((bsize, H * W, num_anchors, 1))

    # get all the anchors

    anchors = torch.FloatTensor(cfg.anchors)

    # note: the all anchors' xywh scale is normalized by the grid width and height, i.e. 13 x 13
    # this is very crucial because the predict output is normalized to 0~1, which is also
    # normalized by the grid width and height
    all_grid_xywh = generate_all_anchors(anchors, H, W)  # shape: (H * W * num_anchors, 4), format: (x, y, w, h)
    all_grid_xywh = delta_pred_batch.new(*all_grid_xywh.size()).copy_(all_grid_xywh)
    all_anchors_xywh = all_grid_xywh.clone()
    all_anchors_xywh[:, 0:2] += 0.5
    if cfg.debug:
        print('all grid: ', all_grid_xywh[:12, :])
        print('all anchor: ', all_anchors_xywh[:12, :])
    all_anchors_xxyy = xywh2xxyy(all_anchors_xywh)

    # process over batches
    for b in range(bsize):
        num_obj = num_boxes_batch[b].item()
        delta_pred = delta_pred_batch[b]
        gt_boxes = gt_boxes_batch[b][:num_obj, :]
        gt_classes = gt_classes_batch[b][:num_obj]

        # rescale ground truth boxes
        gt_boxes[:, 0::2] *= W
        gt_boxes[:, 1::2] *= H

        # step 1: process IoU target

        # apply delta_pred to pre-defined anchors
        all_anchors_xywh = all_anchors_xywh.view(-1, 4)
        box_pred = box_transform_inv(all_grid_xywh, delta_pred)
        box_pred = xywh2xxyy(box_pred)

        # for each anchor, its iou target is corresponded to the max iou with any gt boxes
        ious = box_ious(box_pred, gt_boxes)  # shape: (H * W * num_anchors, num_obj)
        ious = ious.view(-1, num_anchors, num_obj)
        max_iou, _ = torch.max(ious, dim=-1, keepdim=True)  # shape: (H * W, num_anchors, 1)
        if cfg.debug:
            print('ious', ious)

        # iou_target[b] = max_iou

        # we ignore the gradient of predicted boxes whose IoU with any gt box is greater than cfg.threshold
        iou_thresh_filter = max_iou.view(-1) > cfg.thresh
        n_pos = torch.nonzero(iou_thresh_filter).numel()

        if n_pos > 0:
            iou_mask[b][max_iou >= cfg.thresh] = 0

        # step 2: process box target and class target
        # calculate overlaps between anchors and gt boxes
        overlaps = box_ious(all_anchors_xxyy, gt_boxes).view(-1, num_anchors, num_obj)
        gt_boxes_xywh = xxyy2xywh(gt_boxes)

        # iterate over all objects

        for t in range(gt_boxes.size(0)):
            # compute the center of each gt box to determine which cell it falls on
            # assign it to a specific anchor by choosing max IoU

            gt_box_xywh = gt_boxes_xywh[t]
            gt_class = gt_classes[t]
            cell_idx_x, cell_idx_y = torch.floor(gt_box_xywh[:2])
            cell_idx = cell_idx_y * W + cell_idx_x
            cell_idx = cell_idx.long()

            # update box_target, box_mask
            overlaps_in_cell = overlaps[cell_idx, :, t]
            argmax_anchor_idx = torch.argmax(overlaps_in_cell)

            assigned_grid = all_grid_xywh.view(-1, num_anchors, 4)[cell_idx, argmax_anchor_idx, :].unsqueeze(0)
            gt_box = gt_box_xywh.unsqueeze(0)
            target_t = box_transform(assigned_grid, gt_box)
            if cfg.debug:
                print('assigned_grid, ', assigned_grid)
                print('gt: ', gt_box)
                print('target_t, ', target_t)
            box_target[b, cell_idx, argmax_anchor_idx, :] = target_t.unsqueeze(0)
            box_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update cls_target, cls_mask
            class_target[b, cell_idx, argmax_anchor_idx, :] = gt_class
            class_mask[b, cell_idx, argmax_anchor_idx, :] = 1

            # update iou target and iou mask
            iou_target[b, cell_idx, argmax_anchor_idx, :] = max_iou[cell_idx, argmax_anchor_idx, :]
            if cfg.debug:
                print(max_iou[cell_idx, argmax_anchor_idx, :])
            iou_mask[b, cell_idx, argmax_anchor_idx, :] = cfg.object_scale

    return iou_target.view(bsize, -1, 1), \
        iou_mask.view(bsize, -1, 1), \
        box_target.view(bsize, -1, 4), \
        box_mask.view(bsize, -1, 1), \
        class_target.view(bsize, -1, 1).long(), \
        class_mask.view(bsize, -1, 1)


def yolo_loss(output, target):
    """
    Build yolo loss

    Arguments:
    output -- tuple (delta_pred, conf_pred, class_score), output data of the yolo network
    target -- tuple (iou_target, iou_mask, box_target, box_mask, class_target, class_mask) target label data

    delta_pred -- Variable of shape (B, H * W * num_anchors, 4), predictions of delta σ(t_x), σ(t_y), σ(t_w), σ(t_h)
    conf_pred -- Variable of shape (B, H * W * num_anchors, 1), prediction of IoU score σ(t_c)
    class_score -- Variable of shape (B, H * W * num_anchors, num_classes), prediction of class scores (cls1, cls2 ..)

    iou_target -- Variable of shape (B, H * W * num_anchors, 1)
    iou_mask -- Variable of shape (B, H * W * num_anchors, 1)
    box_target -- Variable of shape (B, H * W * num_anchors, 4)
    box_mask -- Variable of shape (B, H * W * num_anchors, 1)
    class_target -- Variable of shape (B, H * W * num_anchors, 1)
    class_mask -- Variable of shape (B, H * W * num_anchors, 1)

    Return:
    loss -- yolo overall multi-task loss
    """

    delta_pred_batch = output[0]
    conf_pred_batch = output[1]
    class_score_batch = output[2]

    iou_target = target[0]
    iou_mask = target[1]
    box_target = target[2]
    box_mask = target[3]
    class_target = target[4]
    class_mask = target[5]

    b, _, num_classes = class_score_batch.size()
    class_score_batch = class_score_batch.view(-1, num_classes)
    class_target = class_target.view(-1)
    class_mask = class_mask.view(-1)

    # ignore the gradient of noobject's target
    class_keep = class_mask.nonzero().squeeze(1)
    class_score_batch_keep = class_score_batch[class_keep, :]
    class_target_keep = class_target[class_keep]

    # if cfg.debug:
    #     print(class_score_batch_keep)
    #     print(class_target_keep)

    # calculate the loss, normalized by batch size.
    box_loss = 1 / b * cfg.coord_scale * F.mse_loss(delta_pred_batch * box_mask, box_target * box_mask,
                                                    reduction='sum') / 2.0
    iou_loss = 1 / b * F.mse_loss(conf_pred_batch * iou_mask, iou_target * iou_mask, reduction='sum') / 2.0
    class_loss = 1 / b * cfg.class_scale * F.cross_entropy(class_score_batch_keep, class_target_keep, reduction='sum')

    return box_loss, iou_loss, class_loss




def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',

                        dtype=torch.float64):
    """
  Implement Kaiming initialization for linear and convolution layers.
  
  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions for
    this layer
  - K: If K is None, then initialize weights for a linear layer with Din input
    dimensions and Dout output dimensions. Otherwise if K is a nonnegative
    integer then initialize the weights for a convolution layer with Din input
    channels, Dout output channels, and a kernel size of KxK.
  - relu: If Python_ReLU=True, then initialize weights with a gain of 2 to account for
    a Python_ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
    with a gain of 1 (Xavier initialization).
  - device, dtype: The device and datatype for the output tensor.

  Returns:
  - weight: A torch Tensor giving initialized weights for this layer. For a
    linear layer it should have shape (Din, Dout); for a convolution layer it
    should have shape (Dout, Din, K, K).
  """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###########################################################################

        # The weight scale is sqrt(gain / fan_in),                                #
        # where gain is 2 if Python_ReLU is followed by the layer, or 1 if not,          #
        # and fan_in = num_in_channels (= Din).                                   #
        # The output should be a tensor in the designated size, dtype, and device.#
        ###########################################################################
        weight_scale = gain / (Din)
        weight = torch.zeros(Din, Dout, dtype=dtype, device=device)
        weight += weight_scale * torch.randn(Din, Dout, dtype=dtype, device=device)

    else:
        ###########################################################################
        # The weight scale is sqrt(gain / fan_in),                                #
        # where gain is 2 if Python_ReLU is followed by the layer, or 1 if not,          #
        # and fan_in = num_in_channels (= Din) * K * K                            #
        # The output should be a tensor in the designated size, dtype, and device.#
        ###########################################################################
        weight_scale = gain / (Din * K * K)
        weight = torch.zeros(Din, Dout, K, K, dtype=dtype, device=device)
        weight += weight_scale * torch.randn(Din, Dout, K, K, dtype=dtype, device=device)

    return weight


def svm_loss(x, y):
    """
  Computes the loss and gradient using for multiclass SVM classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
    N = x.shape[0]
    correct_class_scores = x[torch.arange(N), y]
    margins = (x - correct_class_scores[:, None] + 1.0).clamp(min=0.)
    margins[torch.arange(N), y] = 0.
    loss = margins.sum() / N
    num_pos = (margins > 0).sum(dim=1)
    dx = torch.zeros_like(x)
    dx[margins > 0] = 1.
    dx[torch.arange(N), y] -= num_pos.to(dx.dtype)
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
  Computes the loss and gradient for softmax classification.
  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
    class for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C
  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0 / N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx


def box_ious(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (K, 4), second set of boxes

    Returns:
    ious -- tensor of shape (N, K), ious between boxes
    """

    N = box1.size(0)
    K = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    ious = inter / union_area

    return ious


def xxyy2xywh(box):
    """
    Convert the box (x1, y1, x2, y2) encoding format to (c_x, c_y, w, h) format

    Arguments:
    box: tensor of shape (N, 4), boxes of (x1, y1, x2, y2) format

    Returns:
    xywh_box: tensor of shape (N, 4), boxes of (c_x, c_y, w, h) format
    """

    c_x = (box[:, 2] + box[:, 0]) / 2
    c_y = (box[:, 3] + box[:, 1]) / 2
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([c_x, c_y, w, h], dim=1)
    return xywh_box


def xywh2xxyy(box):
    """
    Convert the box encoding format form (c_x, c_y, w, h) to (x1, y1, x2, y2)

    Arguments:
    box -- tensor of shape (N, 4), box of (c_x, c_y, w, h) format

    Returns:
    xxyy_box -- tensor of shape (N, 4), box of (x1, y1, x2, y2) format
    """

    x1 = box[:, 0] - (box[:, 2]) / 2
    y1 = box[:, 1] - (box[:, 3]) / 2
    x2 = box[:, 0] + (box[:, 2]) / 2
    y2 = box[:, 1] + (box[:, 3]) / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box


def box_transform(box1, box2):
    """
    Calculate the delta values σ(t_x), σ(t_y), exp(t_w), exp(t_h) used for transforming box1 to  box2

    Arguments:
    box1 -- tensor of shape (N, 4) first set of boxes (c_x, c_y, w, h)
    box2 -- tensor of shape (N, 4) second set of boxes (c_x, c_y, w, h)

    Returns:
    deltas -- tensor of shape (N, 4) delta values (t_x, t_y, t_w, t_h)
                   used for transforming boxes to reference boxes
    """

    t_x = box2[:, 0] - box1[:, 0]
    t_y = box2[:, 1] - box1[:, 1]
    t_w = box2[:, 2] / box1[:, 2]
    t_h = box2[:, 3] / box1[:, 3]

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    # σ(t_x), σ(t_y), exp(t_w), exp(t_h)
    deltas = torch.cat([t_x, t_y, t_w, t_h], dim=1)
    return deltas


def box_transform_inv(box, deltas):
    """
    apply deltas to box to generate predicted boxes

    Arguments:
    box -- tensor of shape (N, 4), boxes, (c_x, c_y, w, h)
    deltas -- tensor of shape (N, 4), deltas, (σ(t_x), σ(t_y), exp(t_w), exp(t_h))

    Returns:
    pred_box -- tensor of shape (N, 4), predicted boxes, (c_x, c_y, w, h)
    """

    c_x = box[:, 0] + deltas[:, 0]
    c_y = box[:, 1] + deltas[:, 1]
    w = box[:, 2] * deltas[:, 2]
    h = box[:, 3] * deltas[:, 3]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    pred_box = torch.cat([c_x, c_y, w, h], dim=-1)
    return pred_box


def generate_all_anchors(anchors, H, W):
    """
    Generate dense anchors given grid defined by (H,W)

    Arguments:
    anchors -- tensor of shape (num_anchors, 2), pre-defined anchors (pw, ph) on each cell
    H -- int, grid height
    W -- int, grid width

    Returns:
    all_anchors -- tensor of shape (H * W * num_anchors, 4) dense grid anchors (c_x, c_y, w, h)
    """

    # number of anchors per cell
    A = anchors.size(0)

    # number of cells
    K = H * W

    shift_x, shift_y = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)])

    # transpose shift_x and shift_y because we want our anchors to be organized in H x W order
    shift_x = shift_x.t().contiguous()
    shift_y = shift_y.t().contiguous()

    # shift_x is a long tensor, c_x is a float tensor
    c_x = shift_x.float()
    c_y = shift_y.float()

    centers = torch.cat([c_x.view(-1, 1), c_y.view(-1, 1)], dim=-1)  # tensor of shape (h * w, 2), (cx, cy)

    # add anchors width and height to centers
    all_anchors = torch.cat([centers.view(K, 1, 2).expand(K, A, 2),
                             anchors.view(1, A, 2).expand(K, A, 2)], dim=-1)

    all_anchors = all_anchors.view(-1, 4)

    return all_anchors


def prepare_im_data(img):
    """
	Prepare image data that will be feed to network.

	Arguments:
	img -- PIL.Image object

	Returns:
	im_data -- tensor of shape (3, H, W).
	im_info -- dictionary {height, width}

	"""

    im_info = dict()
    im_info['width'], im_info['height'] = img.size

    # resize the image
    H, W = cfg.input_size
    im_data = img.resize((H, W))

    # to torch tensor
    im_data = torch.from_numpy(np.array(im_data)).float() / 255

    im_data = im_data.permute(2, 0, 1).unsqueeze(0)

    return im_data, im_info


def compressed_pickle(title, data):
    with bz2.BZ2File(title + ".pbz2", 'w') as f:
        pickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = pickle.load(data)
    return data


class WeightLoader(object):
    def __init__(self):
        super(WeightLoader, self).__init__()
        self.start = 0
        self.buf = None
        self.b = 'b'
        self.g = 'g'
        self.rm = 'rm'
        self.rv = 'rv'

    def load_conv_bn(self, conv_model, bn_model):

        # Make directories
        Path('./weight_parameter/bn_param/bias').mkdir(parents=True, exist_ok=True)
        Path('./weight_parameter/bn_param/gamma').mkdir(parents=True, exist_ok=True)
        Path('./weight_parameter/bn_param/running_mean').mkdir(parents=True, exist_ok=True)
        Path('./weight_parameter/bn_param/running_var').mkdir(parents=True, exist_ok=True)
        Path('./weight_parameter/conv_param/w').mkdir(parents=True, exist_ok=True)
        Path('./weight_parameter/bias/').mkdir(parents=True, exist_ok=True)

        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()

        bn_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.bias.data.shape == self.scratch.params['beta0'].shape:
            self.scratch.params['beta0'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == self.scratch.params['beta1'].shape:
            self.scratch.params['beta1'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == self.scratch.params['beta2'].shape:
            self.scratch.params['beta2'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == self.scratch.params['beta3'].shape:
            self.scratch.params['beta3'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == self.scratch.params['beta4'].shape:
            self.scratch.params['beta4'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif bn_model.bias.data.shape == self.scratch.params['beta5'].shape:
            self.scratch.params['beta5'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.bias.data))
        elif (bn_model.bias.data.shape == self.scratch.params['beta6'].shape) and self.b == "b":
            self.scratch.params['beta6'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.bias.data))
            self.b = 'bb'
        elif (self.scratch.params['beta7'].shape == bn_model.bias.data.shape) and self.b == "bb":
            self.scratch.params['beta7'] = bn_model.bias.data
            with open('./weight_parameter/bn_param/bias/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.bias.data))

        self.start = self.start + num_b

        bn_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.weight.data.shape == self.scratch.params['gamma0'].shape:
            self.scratch.params['gamma0'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == self.scratch.params['gamma1'].shape:
            self.scratch.params['gamma1'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == self.scratch.params['gamma2'].shape:
            self.scratch.params['gamma2'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == self.scratch.params['gamma3'].shape:
            self.scratch.params['gamma3'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == self.scratch.params['gamma4'].shape:
            self.scratch.params['gamma4'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif bn_model.weight.data.shape == self.scratch.params['gamma5'].shape:
            self.scratch.params['gamma5'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.weight.data))
        elif (bn_model.weight.shape == self.scratch.params['gamma6'].shape) and self.g == "g":
            self.scratch.params['gamma6'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.weight.data))
            self.g = 'gg'
        elif (self.scratch.params['gamma7'].shape == bn_model.weight.data.shape) and self.g == "gg":
            self.scratch.params['gamma7'] = bn_model.weight.data
            with open('./weight_parameter/bn_param/gamma/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.weight.data))

        self.start = self.start + num_b

        bn_model.running_mean.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.running_mean.data.shape == self.scratch.bn_params[0]['running_mean'].shape:
            self.scratch.bn_params[0]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[1]['running_mean'].shape:
            self.scratch.bn_params[1]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[2]['running_mean'].shape:
            self.scratch.bn_params[2]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[3]['running_mean'].shape:
            self.scratch.bn_params[3]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[4]['running_mean'].shape:
            self.scratch.bn_params[4]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[5]['running_mean'].shape:
            self.scratch.bn_params[5]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[6]['running_mean'].shape and self.rm == "rm":
            self.scratch.bn_params[6]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.running_mean.data))
            self.rm = "rmrm"
        elif bn_model.running_mean.data.shape == self.scratch.bn_params[7]['running_mean'].shape and self.rm == "rmrm":
            self.scratch.bn_params[7]['running_mean'] = bn_model.running_mean.data
            with open('./weight_parameter/bn_param/running_mean/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.running_mean.data))

        self.start = self.start + num_b

        bn_model.running_var.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))

        if bn_model.running_var.data.shape == self.scratch.bn_params[0]['running_var'].shape:
            self.scratch.bn_params[0]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(0), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == self.scratch.bn_params[1]['running_var'].shape:
            self.scratch.bn_params[1]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(1), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == self.scratch.bn_params[2]['running_var'].shape:
            self.scratch.bn_params[2]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(2), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == self.scratch.bn_params[3]['running_var'].shape:
            self.scratch.bn_params[3]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(3), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == self.scratch.bn_params[4]['running_var'].shape:
            self.scratch.bn_params[4]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(4), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == self.scratch.bn_params[5]['running_var'].shape:
            self.scratch.bn_params[5]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(5), mode='w') as f:
                f.write(str(bn_model.running_var.data))
        elif bn_model.running_var.data.shape == self.scratch.bn_params[6]['running_var'].shape and self.rv == "rv":
            self.scratch.bn_params[6]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(6), mode='w') as f:
                f.write(str(bn_model.running_var.data))
            self.rv = "rvrv"
        elif bn_model.running_var.data.shape == self.scratch.bn_params[7]['running_var'].shape and self.rv == "rvrv":
            self.scratch.bn_params[7]['running_var'] = bn_model.running_var.data
            with open('./weight_parameter/bn_param/running_var/{}'.format(7), mode='w') as f:
                f.write(str(bn_model.running_var.data))

        self.start = self.start + num_b

        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))

        if conv_model.weight.data.shape == (16, 3, 3, 3):
            self.scratch.params['W0'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(0), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (32, 16, 3, 3):
            self.scratch.params['W1'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(1), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (64, 32, 3, 3):
            self.scratch.params['W2'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(2), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (128, 64, 3, 3):
            self.scratch.params['W3'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(3), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (256, 128, 3, 3):
            self.scratch.params['W4'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(4), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (512, 256, 3, 3):
            self.scratch.params['W5'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(5), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (1024, 512, 3, 3):
            self.scratch.params['W6'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(6), mode='w') as f:
                f.write(str(conv_model.weight.data))
        elif conv_model.weight.data.shape == (1024, 1024, 3, 3):
            self.scratch.params['W7'] = conv_model.weight.data
            with open('./weight_parameter/conv_param/w/{}'.format(7), mode='w') as f:
                f.write(str(conv_model.weight.data))
        self.start = self.start + num_w

    def load_conv(self, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), conv_model.bias.size()))
        self.scratch.params['b8'] = conv_model.bias.data
        with open('./weight_parameter/bias/{}'.format(7), mode='w') as f:
            f.write(str(conv_model.bias.data))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.scratch.params['W8'] = conv_model.weight.data
        with open('./weight_parameter/conv_param/w/{}'.format(8), mode='w') as f:
            f.write(str(conv_model.weight.data))
        self.start = self.start + num_w

    def dfs(self, m):
        children = list(m.children())
        for i, c in enumerate(children):
            if isinstance(c, torch.nn.Sequential):
                self.dfs(c)
            elif isinstance(c, torch.nn.Conv2d):
                if c.bias is not None:
                    self.load_conv(c)
                else:
                    self.load_conv_bn(c, children[i + 1])

    def load(self, model_to_load_weights_to, model, weights_file):
        self.scratch = model_to_load_weights_to
        self.start = 0
        fp = open(weights_file, 'rb')
        header = np.fromfile(fp, count=4, dtype=np.int32)
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        size = self.buf.size
        self.dfs(model)
        # make sure the loaded weight is right
        assert size == self.start
        return self.scratch


class Yolov2(nn.Module):
    num_classes = 20
    num_anchors = 5

    def __init__(self, classes=None, weights_file=False):
        super(Yolov2, self).__init__()
        if classes:
            self.num_classes = len(classes)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.slowpool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(1024)

        self.conv8 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(1024)

        self.conv9 = nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1)

    def forward(self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False):
        """
    x: Variable
    gt_boxes, gt_classes, num_boxes: Tensor
    """

        x = self.maxpool(self.lrelu(self.bn1(self.conv1(x))))
        x = self.maxpool(self.lrelu(self.bn2(self.conv2(x))))
        x = self.maxpool(self.lrelu(self.bn3(self.conv3(x))))
        x = self.maxpool(self.lrelu(self.bn4(self.conv4(x))))
        x = self.maxpool(self.lrelu(self.bn5(self.conv5(x))))
        x = self.lrelu(self.bn6(self.conv6(x)))
        # x = F.pad(x, (0, 1, 0, 1))
        # x = self.slowpool(x)
        x = self.lrelu(self.bn7(self.conv7(x)))
        x = self.lrelu(self.bn8(self.conv8(x)))
        out = self.conv9(x)

        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * self.num_anchors, 5 + self.num_classes)

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [v for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss, class_loss

        return delta_pred, conf_pred, class_pred


def Floating2Binary(num, Exponent_Bit, Mantissa_Bit):
    sign = ('1' if num < 0 else '0')
    num = abs(num)
    bias = math.pow(2, (Exponent_Bit - 1)) - 1
    if num == 0:
        e = 0
    else:
        e = math.floor(math.log(num, 2) + bias)

    if e > (math.pow(2, Exponent_Bit) - 2):  # overflow
        exponent = '1' * Exponent_Bit
        mantissa = '0' * Mantissa_Bit
    else:
        if e > 0:
            s = num / math.pow(2, e - bias) - 1
            exponent = bin(e)[2:].zfill(Exponent_Bit)
        else:  # submoral
            s = num / math.pow(2, (-bias + 1))
            exponent = '0' * Exponent_Bit
        # Rounding Mode By Adding 0.5 (Half-Rounding or Banker's Rounding)
        # Number is smaller or equal 0.5 is rounding down
        # Number is larger 0.5 is rounding up
        mantissa = bin(int(s * (math.pow(2, Mantissa_Bit)) + 0.5))[2:].zfill(Mantissa_Bit)[:Mantissa_Bit]
        # Non-Rounding Mode
        # mantissa = bin(int(s * (math.pow(2, Mantissa_Bit)))[2:].zfill(Mantissa_Bit)[:Mantissa_Bit]
    Floating_Binary = sign + exponent + mantissa

    return Floating_Binary


def Binary2Floating(value, Exponent_Bit, Mantissa_Bit):
    sign = int(value[0], 2)
    if int(value[1:1 + Exponent_Bit], 2) != 0:
        exponent = int(value[1:1 + Exponent_Bit], 2) - int('1' * (Exponent_Bit - 1), 2)
        mantissa = int(value[1 + Exponent_Bit:], 2) * (math.pow(2, (-Mantissa_Bit))) + 1
    else:  # subnormal
        exponent = 1 - int('1' * (Exponent_Bit - 1), 2)
        # mantissa = int(value[1 + Exponent_Bit:], 2) * 2 ** (-Mantissa_Bit)
        mantissa = int(value[1 + Exponent_Bit:], 2) * math.pow(2, (-Mantissa_Bit))
    Floating_Decimal = (math.pow(-1, sign)) * (math.pow(2, exponent)) * mantissa
    return Floating_Decimal


def Truncating_Rounding(Truncated_Hexadecimal):
    # Consider only the Length of Truncated_Hexadecimal only in [0:5]
    if len(Truncated_Hexadecimal) >= 5:
        # If this Truncated_Hexadecimal[4] >= 5 => Rounding Up the First 16 Bits
        if int(Truncated_Hexadecimal[4], 16) >= 8:
            Rounding_Hexadecimal = hex(int(Truncated_Hexadecimal[:4], 16) + 1)[2:]
        else:
            Rounding_Hexadecimal = Truncated_Hexadecimal[:4]
    else:
        Rounding_Hexadecimal = Truncated_Hexadecimal

    Rounding_Hexadecimal_Capitalized = Rounding_Hexadecimal.upper()

    return Rounding_Hexadecimal_Capitalized


################################################################################
################################################################################
#################   Python Implementations and Sandwich Layers  #################
################################################################################
################################################################################


# Python_Convolution without Bias
class Python_Conv(object):

    @staticmethod
    def forward(x, w, conv_param):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        w = w.to(device)
        out = None
        pad = 1
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        exp_bits = 8
         
        pad = conv_param['pad']
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        out = torch.zeros((N, F, H_out, W_out), dtype=torch.float32, device="cuda")
        output_ptr = out.flatten().contiguous().data_ptr()
        x_ptr = x.flatten().contiguous().data_ptr()
        w_ptr = w.flatten().contiguous().data_ptr()

        libconv.conv2d(N, C, H, W, 
               F, HH, WW, 
               ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)),
               ctypes.cast(w_ptr, ctypes.POINTER(ctypes.c_float)),
               ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),        
               pad, stride, exp_bits)
            
        out = out.reshape(N, F, H_out, W_out)

        cache = (x, w, conv_param)
 
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, conv_param = cache
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dout = dout.to(device)
        x = x.to(device)
        w = w.to(device)
        pad = 1 
        stride = conv_param['stride']
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape

        dout_gpu = dout.contiguous()
        x_gpu = x.contiguous()
        w_gpu = w.contiguous()

        dx = torch.zeros_like(x_gpu)
        dw = torch.zeros_like(w_gpu)

        x_ptr = x_gpu.flatten().data_ptr()
        w_ptr = w_gpu.flatten().data_ptr()
        dout_ptr = dout_gpu.flatten().data_ptr()
        dw_ptr = dw.flatten().data_ptr()
        dx_ptr = dx.flatten().data_ptr()

        exp_bits = 8
        libconv.conv2d_backward_dw(
            N, C, H, W, ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), 
            F, HH, WW,ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)), 
            ctypes.cast(dw_ptr,ctypes.POINTER(ctypes.c_float)), 
            H, W, stride, pad, exp_bits)
           
        reshaped_w = w.permute(1, 0, 2, 3)
        w_flipped = torch.flip(reshaped_w, dims=(2, 3))
        FF, CC, HH, WW = w_flipped.shape
        w_transpose = w_flipped.contiguous()
        w_ptr_transpose = w_transpose.flatten().data_ptr()

        libconv.conv2d(N, F, H, W, 
                       FF, HH, WW, 
                       ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)),
                       ctypes.cast(w_ptr_transpose, ctypes.POINTER(ctypes.c_float)),
                       ctypes.cast(dx_ptr, ctypes.POINTER(ctypes.c_float)),
                       pad, stride, exp_bits)

        dx = dx.reshape(N, C, H, W)
     
        return dx, dw 

# Python_Convolution with Bias
class Python_ConvB(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        pad = conv_param['pad']
        stride = conv_param['stride']
        exp_bits = 6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        w = w.to(device)
        b = b.to(device)
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        H_out = int(1 + (H + 2 * pad - HH) / stride)
        W_out = int(1 + (W + 2 * pad - WW) / stride)
        
        out = torch.zeros((N, F, H_out, W_out), dtype=torch.float32, device="cuda")
        x_ptr = x.flatten().contiguous().data_ptr()
        w_ptr = w.flatten().contiguous().data_ptr()
        b_ptr = b.flatten().contiguous().data_ptr()
        output_ptr = out.flatten().contiguous().data_ptr()

        libconv.conv2d_WB(N, C, H, W, 
               ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)),
               F, HH, WW, 
               ctypes.cast(w_ptr, ctypes.POINTER(ctypes.c_float)),
               ctypes.cast(b_ptr, ctypes.POINTER(ctypes.c_float)),
               ctypes.cast(output_ptr, ctypes.POINTER(ctypes.c_float)),      
               pad, stride, exp_bits)
        out = out.reshape(N, F, H_out, W_out)

        cache = (x, w, b, conv_param)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, w, b, conv_param = cache
        w = w.to(device)
        b = b.to(device)

        pad = conv_param['pad']
        stride = conv_param['stride']
        stride = conv_param['stride']
        N, F, H_dout, W_dout = dout.shape
        F, C, HH, WW = w.shape
        
        db = torch.zeros_like(b, dtype=torch.float32, device=device)
        dw_bias = torch.zeros_like(w, dtype=torch.float32, device=device)
        A, B, X, Y = x.shape
        dx_bias = torch.zeros((A, B, X, Y), dtype=torch.float32, device=device)

        dout_ptr = dout.flatten().contiguous().data_ptr()
        db_ptr = db.flatten().contiguous().data_ptr()
        x_ptr = x.flatten().contiguous().data_ptr()
        dw_ptr_bias = dw_bias.flatten().contiguous().data_ptr()
        dx_ptr_bias = dx_bias.flatten().contiguous().data_ptr()
        exp_bits = 6
        
        libconv.conv2d_backward_db(
            N, F, H_dout, W_dout, 
            ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)), 
            ctypes.cast(db_ptr, ctypes.POINTER(ctypes.c_float)),
            exp_bits
        )
        
        libconv.conv2d_backward_dw(
            N, C, H_dout, W_dout, ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), 
            F, HH, WW,ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)), 
            ctypes.cast(dw_ptr_bias, ctypes.POINTER(ctypes.c_float)), 
            H_dout, W_dout, stride, pad,exp_bits)
        
        reshaped_w = w.permute(1, 0, 2, 3)
        w_flipped = torch.flip(reshaped_w, dims=(2, 3))
        FF, CC, HH, WW = w_flipped.shape
        w_transpose = w_flipped.contiguous()
        w_ptr_transpose = w_transpose.flatten().data_ptr()
        N, C, H, W = x.shape

        libconv.conv2d(N, F, H, W, 
                       FF, HH, WW,
                       ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)),
                       ctypes.cast(w_ptr_transpose, ctypes.POINTER(ctypes.c_float)),
                       ctypes.cast(dx_ptr_bias, ctypes.POINTER(ctypes.c_float)),
                       pad, stride, exp_bits)


        dx_bias = dx_bias.reshape(A,B,X,Y)
        

        return dx_bias, dw_bias, db   


class Python_MaxPool(object):

    @staticmethod
    def forward(x, pool_param, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        # Extract pooling parameters
        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']

        # Get input dimensions
        N, C, H, W = x.shape

        # Calculate output dimensions
        H_out = int(1 + (H - pool_height) / stride)
        W_out = int(1 + (W - pool_width) / stride)

        # Allocate memory for output and positions on GPU
    
        out = torch.zeros((N, C, H_out, W_out), dtype=x.dtype, device="cuda")
        positions = torch.zeros((N, C, H_out, W_out), dtype=torch.int32, device="cuda")
        # Ensure input tensor is on GPU and contiguous
        x_gpu = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        positions_gpu = positions.contiguous().cuda()if not positions.is_cuda else positions.contiguous()
        # Get pointers to the data
        x_ptr = x_gpu.flatten().data_ptr()
        out_ptr = out.flatten().data_ptr()
        pos_ptr = positions_gpu.flatten().data_ptr()
        _curr_time = time.time()
        # Launch the kernel
        libpool.max_pooling_forward(N, C, H, W, 
                                    ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), 
                                    ctypes.cast(out_ptr, ctypes.POINTER(ctypes.c_float)), 
                                    ctypes.cast(pos_ptr, ctypes.POINTER(ctypes.c_float)), 
                                    pool_height, pool_width, stride)

        # Ensure synchronization of CUDA operations
        
        out = out.reshape(N, C, H_out, W_out)
        cache = (x, positions_gpu, pool_param)
        _time= (time.time() - _curr_time)  
        # print("Time taken by Pooling Layer",layer_no,"is: ",_time)
                
        return out, cache

    @staticmethod
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        x, positions, pool_param = cache

        N, C, H, W = x.shape
        stride = pool_param['stride']
        pool_width = pool_param['pool_width']
        pool_height = pool_param['pool_height']

        # Create an output tensor dx
        dx = torch.zeros((N, C, H, W), dtype=x.dtype, device="cuda")

        # Convert tensors to contiguous if they are not already
        dout_ptr = dout.flatten().contiguous().data_ptr()
        positions_ptr = positions.flatten().contiguous().data_ptr()
        dx_ptr = dx.flatten().contiguous().data_ptr()
        
        # Call the CUDA function
        libpool.max_pooling_backward(N, C, H, W, ctypes.cast(dout_ptr, ctypes.POINTER(ctypes.c_float)),  ctypes.cast(dx_ptr, ctypes.POINTER(ctypes.c_float)),
                                      ctypes.cast(positions_ptr, ctypes.POINTER(ctypes.c_int32)),
                                    pool_height, pool_width, stride)

        dx = dx.reshape(N, C, H, W)

        return dx

class Python_BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_params, layer_no=[], save_txt=False, save_hex=False, phase=[]):


        mode = bn_params['mode']
        eps = bn_params.get('eps', 1e-5)
        momentum = bn_params.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_params.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
        running_var = bn_params.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))


        out, cache = None, None
        if mode == 'train':

            # step1: calculate mean
            mu = 1. / N * torch.sum(x, axis=0)
            running_mean = momentum * running_mean + (1 - momentum) * mu

            # step2: subtract mean vector of every trainings example
            xmu = x - mu

            # step3: following the lower branch - calculation denominator
            sq = xmu ** 2

            # step4: calculate variance
            var = 1. / N * torch.sum(sq, axis=0)
            running_var = momentum * running_var + (1 - momentum) * var
            
            # step5: add eps for numerical stability, then sqrt
            sqrtvar = torch.sqrt(var + eps)

            # step6: invert sqrtwar
            ivar = 1. / sqrtvar

            # step7: execute normalization
            xhat = xmu * ivar

            # step8: Nor the two transformation steps
            # print(gamma)

            gammax = gamma * xhat

            # step9
            out = gammax + beta

            cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

        elif mode == 'test':

            normolized = ((x - running_mean) / (running_var + eps) ** (1 / 2))
            out = normolized * gamma + beta

        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_params
        bn_params['running_mean'] = running_mean.detach()
        bn_params['running_var'] = running_var.detach()



        return out, cache

    @staticmethod
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
 
        dx, dgamma, dbeta = None, None, None

        xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

        N, D = dout.shape

        # step9
        dbeta = torch.sum(dout, axis=0)
        dgammax = dout  # not necessary, but more understandable

        # step8
        dgamma = torch.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * gamma

        # step7
        divar = torch.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar

        # step6
        dsqrtvar = -1. / (sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / torch.sqrt(var + eps) * dsqrtvar

        # step4
        dsq = 1. / N * torch.ones((N, D), device=dout.device) * dvar

        # step3
        dxmu2 = 2 * xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * torch.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * torch.ones((N, D), device=dout.device) * dmu

        # step0
        dx = dx1 + dx2

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
    
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
        dx, dgamma, dbeta = None, None, None

        xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache
        N, D = dout.shape
        # get the dimensions of the input/output
        dbeta = torch.sum(dout, dim=0)
        dgamma = torch.sum(xhat * dout, dim=0)
        dx = (gamma * ivar / N) * (N * dout - xhat * dgamma - dbeta)

        return dx, dgamma, dbeta

class Python_ReLU(object):

    @staticmethod
    def forward(x, alpha=0.1, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        out = None
        out = x.clone()
        out[out < 0] = out[out < 0] * alpha
        cache = x

        # Generating the Sign for Leaky ReLU
        sign = torch.zeros_like(x) 
        sign[x<0] = 1

        return out, cache

    @staticmethod
    def backward(dout, cache, alpha=0.1, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        dx, x = None, cache

        dl = torch.ones_like(x)
        dl[x < 0] = alpha
        dx = dout * dl


        return dx


class Python_Conv_ReLU(object):

    @staticmethod
    def forward(x, w, conv_param, layer_no=[], save_txt=False, save_hex=False, phase=[]):

        a, conv_cache = Python_Conv.forward(
            x,
            w,
            conv_param)
        out, relu_cache = Python_ReLU.forward(
            a,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )
        cache = (conv_cache, relu_cache)
        # print(f'{layer_no}', end=',')
        return out, cache

    @staticmethod
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        """
    Backwards pass for the conv-relu convenience layer.
    """
        conv_cache, relu_cache = cache
        da = Python_ReLU.backward(
            dout,
            relu_cache,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )
        dx, dw = Python_Conv.backward(
            da,
            conv_cache
        )
        # print(f'{layer_no}', end=',')
        return dx, dw

class Python_Conv_Pool(object):

    @staticmethod
    def forward(x, w, conv_param, pool_param, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        """
    A convenience layer that performs a convolution, a Python_ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
        a, conv_cache = Python_Conv.forward(
            x,
            w,
            conv_param
        )

        out, pool_cache = Python_MaxPool.forward(
            a,
            pool_param,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        cache = (conv_cache, pool_cache)
        # print(f'{layer_no}', end=',')
        return out, cache

    @staticmethod
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):

        conv_cache, pool_cache = cache
        ds = Python_MaxPool.backward(
            dout,
            pool_cache,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )



        dx, dw = Python_Conv.backward(
            ds,
            conv_cache
        )

        # print(f'{layer_no}', end=',')
        return dx, dw

class Python_Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, conv_param, pool_param, layer_no=[], save_txt=False, save_hex=False, phase=[]):

        a, conv_cache = Python_Conv.forward(
            x,
            w,
            conv_param
        )

        s, relu_cache = Python_ReLU.forward(
            a,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        out, pool_cache = Python_MaxPool.forward(
            s,
            pool_param,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        cache = (conv_cache, relu_cache, pool_cache)
        # print(f'{layer_no}', end=',')
        return out, cache

    @staticmethod
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):

        conv_cache, relu_cache, pool_cache = cache
        ds = Python_MaxPool.backward(
            dout,
            pool_cache,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        da = Python_ReLU.backward(
            ds,
            relu_cache,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        dx, dw = Python_Conv.backward(
            da,
            conv_cache
        )

        # print(f'{layer_no}', end=',')
        return dx, dw


class Python_Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, gamma, beta, conv_param, bn_params, mean, var, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        a, conv_cache = Python_Conv.forward(
            x,
            w,
            conv_param)

        an, bn_cache = Python_SpatialBatchNorm.forward(
            a,
            gamma,
            beta,
            bn_params,
            mean,
            var,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase)

        out, relu_cache = Python_ReLU.forward(
            an,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase)

        cache = (conv_cache, bn_cache, relu_cache)
        # print(f'{layer_no}', end=',')
        return out, cache

    @staticmethod
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        conv_cache, bn_cache, relu_cache = cache

        dan = Python_ReLU.backward(
            dout,
            relu_cache,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        da, dgamma, dbeta = Python_SpatialBatchNorm.backward(
            dan,
            bn_cache,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        dx, dw = Python_Conv.backward(
            da,
            conv_cache
        )

        # print(f'{layer_no}', end=',')
        return dx, dw, dgamma, dbeta


class Python_Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, gamma, beta, conv_param, bn_params, mean, var, pool_param, layer_no=[], save_txt=False, save_hex=False,
                phase=[]):
        a, conv_cache = Python_Conv.forward(
            x,
            w,
            conv_param
        )

        an, bn_cache = Python_SpatialBatchNorm.forward(
            a,
            gamma,
            beta,
            bn_params,
            mean,
            var,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        s, relu_cache = Python_ReLU.forward(
            an,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        out, pool_cache = Python_MaxPool.forward(
            s,
            pool_param,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        # print(f'{layer_no}', end=',')
        return out, cache

    @staticmethod
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        conv_cache, bn_cache, relu_cache, pool_cache = cache

        ds = Python_MaxPool.backward(
            dout,
            pool_cache,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        dan = Python_ReLU.backward(
            ds,
            relu_cache,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        da, dgamma, dbeta = Python_SpatialBatchNorm.backward(
            dan,
            bn_cache,
            layer_no=layer_no,
            save_txt=save_txt,
            save_hex=save_hex,
            phase=phase
        )

        dx, dw = Python_Conv.backward(
            da,
            conv_cache
        )

        # print(f'{layer_no}', end=',')
        return dx, dw, dgamma, dbeta

def origin_idx_calculator(idx, B, H, W, num_chunks):
    origin_idx = []
    if num_chunks < H*W//num_chunks:
        for i in range(len(idx)):
            for j in range(len(idx[0])):
                origin_idx.append([(j*num_chunks*B+int(idx[i][j]))//(H*W), i, 
                        ((j*num_chunks*B+int(idx[i][j]))%(H*W))//H, ((j*num_chunks*B+int(idx[i][j]))%(H*W))%H])
    else:
        for i in range(len(idx)):
            for j in range(len(idx[0])):
                origin_idx.append([(j*B*H*W//num_chunks+int(idx[i][j]))//(H*W), i,
                        ((j*B*H*W//num_chunks+int(idx[i][j]))%(H*W))//H, ((j*B*H*W//num_chunks+int(idx[i][j]))%(H*W))%H])
    return origin_idx

class Cal_mean_var(object):

    @staticmethod
    def forward(x, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        out, cache = None, None
        eps = 1e-5
        num_chunks = 8
        B, C, H, W = x.shape
        y = x.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, num_chunks, B * H * W // num_chunks)
        avg_max = y.max(-1)[0].mean(-1)  # C
        avg_min = y.min(-1)[0].mean(-1)  # C
        avg = y.view(C, -1).mean(-1)  # C
        max_index = origin_idx_calculator(y.max(-1)[1], B, H, W, num_chunks)
        min_index = origin_idx_calculator(y.min(-1)[1], B, H, W, num_chunks)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  

        avg = avg.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)

        cache = x
        return avg, scale
    
    @staticmethod
    def backward(dout, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        
        x = cache
        B, C, H, W = x.shape
        dL_davg = (dout).sum(dim=(0, 2, 3), keepdim=True)
        avg_pc = dL_davg / (B * H * W)
        
        
        return avg_pc
    

class Python_SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_params, mean, var, layer_no=[], save_txt=False, save_hex=False, phase=[]):  
        out, cache = None, None
        gamma = gamma.to(x.device)
        beta = beta.to(x.device)
        mean = mean.to(x.device)
        var = var.to(x.device)
        eps = 1e-5
        D = gamma.shape[0]
        num_chunks = 8
        running_mean = bn_params["running_mean"]
        running_var = bn_params["running_var"]
        running_mean = running_mean.to(x.device)
        running_var = running_var.to(x.device)
        B, C, H, W = x.shape
        y = x.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, num_chunks, B * H * W // num_chunks)
        avg_max = y.max(-1)[0].mean(-1)  # C
        avg_min = y.min(-1)[0].mean(-1)  # C
        avg = y.view(C, -1).mean(-1)  # C
        max_index = origin_idx_calculator(y.max(-1)[1], B, H, W, num_chunks)
        min_index = origin_idx_calculator(y.min(-1)[1], B, H, W, num_chunks)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  
        avg = avg.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)
        momentum = 0.1
        output = (x - mean) * var

        output = output * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
        
        running_mean = running_mean * momentum + (1 - momentum) * avg
        running_var = running_var * momentum + (1 - momentum) * scale
        
        cache = (x, gamma, beta, output, var, scale, mean, avg_max, avg_min, eps, num_chunks, max_index, min_index)
        return output, cache
    
    @staticmethod
    def backward(grad_output, cache, layer_no=[], save_txt=False, save_hex=False, phase=[]):
        X, gamma, beta, output, scale, scale_fix, avg, avg_max, avg_min, eps, num_chunks, max_index, min_index = cache
        B, C, H, W = X.shape
        dL_dxi_hat = grad_output * gamma.view(1, -1, 1, 1)
        
        # Compute dL_dvar
        dL_dvar = (dL_dxi_hat * (X - avg) * -0.5 * torch.sqrt(scale) * torch.sqrt(scale) * torch.sqrt(scale)).sum(dim=(0, 2, 3), keepdim=True)
        
        # Compute dL_dxmax_mean and dL_dxmin_mean
        dL_dxmax_mean = (dL_dvar / scale_fix).sum(dim=(0, 2, 3), keepdim=True)
        dL_dxmin_mean = (-1 * dL_dvar / scale_fix).sum(dim=(0, 2, 3), keepdim=True)
        
        # Compute dL_dxmax and dL_dxmin
        dL_dxmax = (dL_dxmax_mean / num_chunks).sum(dim=(0, 2, 3), keepdim=True)
        dL_dxmin = (dL_dxmin_mean / num_chunks).sum(dim=(0, 2, 3), keepdim=True)
        
        # Compute dL_dgamma and dL_dbeta
        dL_dgamma = (grad_output * output).sum(dim=(0, 2, 3), keepdim=True) # TO DO - Is it really required to keep dim
        dL_dbeta = grad_output.sum(dim=(0, 2, 3), keepdim=True)
        dL_davg = grad_output.sum(dim=(0, 2, 3), keepdim=True)

        # Average per channel
        avg_pc = (dL_dxi_hat * -1.0).sum(dim=(0, 2, 3), keepdim=True) / (B * H * W)
        dL_dxi_ = avg_pc + dL_dxi_hat
        
        # Backward coefficient
        backward_const = scale
        
        # Final output calculation
        dL_dxi = dL_dxi_ * backward_const

        return dL_dxi, dL_dgamma, dL_dbeta