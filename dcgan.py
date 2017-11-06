import mxnet as mx
from mxnet.gluon import nn
import nnvm
import tvm
import numpy as np

netG = nn.HybridSequential()
with netG.name_scope():
    ngf = 64
    nc = 3
    # input is Z, going into a convolution
    netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 4 x 4
    netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 8 x 8
    netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 16 x 16
    netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 32 x 32
    netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, use_bias=False))
    netG.add(nn.Activation('tanh'))
    # state size. (nc) x 64 x 64

ctx = mx.cpu()
batch_size = 1
latent_z_size = 100
latent_z = mx.nd.random_normal(0, 1, shape=(batch_size, latent_z_size, 1, 1), ctx=ctx)
netG.load_params("data/netG_epoch100.params", ctx)

sym, params = nnvm.frontend.from_mxnet(netG)

import nnvm.compiler
target = 'rocm' # or 'cuda'
shape_dict = {'data': latent_z.shape}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

from tvm.contrib import graph_runtime
tvm_ctx = tvm.context(target, 0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, tvm_ctx)
m.set_input(**params)

rows, cols = 8, 8
output_height = 64
output_width = 64
img_tile = np.zeros((output_height * rows, output_width * cols, 3))

for i in range(rows):
    for j in range(cols):
        latent_z = mx.nd.random_normal(0, 1, shape=(batch_size, latent_z_size, 1, 1), ctx=ctx) 
        noise = tvm.nd.array(latent_z.asnumpy().astype("float32"))
        m.set_input("data", noise)
        m.run()
        output_shape = (1, 3, output_height, output_width)
        tvm_output = m.get_output(0, tvm.nd.empty(output_shape, dtype)).asnumpy()
        img_tile[i * output_height:(i+1) * output_height,
                 j * output_width:(j+1) * output_width] = tvm_output[0].transpose(1,2,0)

img_tile = (img_tile + 1.0) * 127.5

from skimage.io import imsave
imsave("fake.png", img_tile.astype(np.uint8))
