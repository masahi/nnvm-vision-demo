import mxnet.ndarray as nd
import mxnet as mx
import numpy as np
import tvm
from skimage.io import imread

def block(data, num_filter, name):
    data2 = conv(data, num_filter, 1, name=name)
    data2 = mx.sym.Convolution(data=data2, num_filter=num_filter, kernel=(3,3), pad=(1,1), name='%s_conv1'%name)
    data2 = mx.sym.BatchNorm(data=data2, momentum=0.9, name='%s_bn1'%name)
    return mx.sym.Activation(data=data+data2, act_type='relu')

def conv(data, num_filter, stride, name):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), pad=(1,1), stride=(stride, stride), name='%s_conv'%name)
    data = mx.sym.BatchNorm(data=data, momentum=0.9, name='%s_conv'%name)
    data = mx.sym.Activation(data=data, act_type='relu')
    return data

def generator_symbol():
    data = mx.sym.Variable('data')
    data = mx.sym.Convolution(data=data, num_filter=32, kernel=(9,9), pad=(4,4), name='conv0')
    data = mx.sym.BatchNorm(data=data, name='bn0')
    data = mx.sym.Activation(data=data, act_type='relu')
    data = conv(data, 64, 2, name='downsample0')
    data = conv(data, 128, 2, name='downsample1')
    data = block(data, 128, name='block0')
    data = block(data, 128, name='block1')
    data = block(data, 128, name='block2')
    data = block(data, 128, name='block3')
    data = block(data, 128, name='block4')
    data = mx.sym.Deconvolution(data=data, kernel=(4,4), pad=(1,1), stride=(2,2), num_filter=64, name='deconv0')
    data = mx.sym.BatchNorm(data=data, momentum=0.9, name='dcbn0')
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Deconvolution(data=data, kernel=(4,4), pad=(1,1), stride=(2,2), num_filter=32, name='deconv1')
    data = mx.sym.BatchNorm(data=data, momentum=0.9, name='dcbn1')
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Convolution(data=data, num_filter=3, kernel=(9,9), pad=(4,4), name='lastconv')
    return data


arg = nd.load("data/style_model/the_scream_args.nd")
aux = nd.load("data/style_model/the_scream_auxs.nd")
sym_mx = generator_symbol()

x = imread('data/style_model/tubingen.jpg')
x = np.transpose(x, axes=(2, 0, 1)).astype(np.float32)
x[0,:] -= 123.68
x[1,:] -= 116.779
x[2,:] -= 103.939

x = np.expand_dims(x, axis=0)
print("input shape", x.shape)
arg["data"] = nd.array(x, ctx=mx.cpu())
                       
import nnvm
sym, params = nnvm.frontend.from_mxnet(sym_mx, arg, aux)
import nnvm.compiler
target = 'rocm'

shape_dict = {'data': x.shape}
with nnvm.compiler.build_config(opt_level=1):
    with tvm.build_config(auto_unroll_max_step=128,
                          unroll_explicit=True):
        graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

from tvm.contrib import graph_runtime
ctx = tvm.context(target, 0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
m.set_input('data', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
m.run()

output = tvm.nd.empty((x.shape), dtype)
tvm_output = m.get_output(0, output).asnumpy()
print("output shape", tvm_output.shape)

from skimage.io import imsave
output = tvm_output[0]
output[0,:] += 123.68
output[1,:] += 116.779
output[2,:] += 103.939
output = np.transpose(output, axes=(1, 2, 0))
output[output<0] = 0
output[output>255] = 255
imsave("style_transfer_output.png", output.astype(np.uint8))

ftimer = m.module.time_evaluator("run", ctx, 50)
print(ftimer().mean)
