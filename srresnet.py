"""
The model is from the paper
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
https://arxiv.org/abs/1609.04802

Pytorch model is exported from
https://github.com/twtygqyy/pytorch-SRResNet

Images are from the original paper. It is available at
https://twitter.app.box.com/s/lcue6vlrd01ljkdtdkhmfvk7vtjhetog

"""

import nnvm
import tvm
import onnx
import numpy as np
from PIL import Image
import nnvm.compiler
from tvm.contrib import graph_runtime

rescale_factor = 4
input_name = "butterfly"
im_path = 'data/srresnet/%s_LR.png' % input_name
bicubic_im_path = 'data/srresnet/%s_bicubic.png' % input_name

img = np.array(Image.open(im_path))
bicubic_im = np.array(Image.open(bicubic_im_path))
print(img.shape)
height = img.shape[0]
width = img.shape[1]
onnx_graph = onnx.load('data/srresnet/%dx%d.onnx' % (height, width))
sym, params = nnvm.frontend.from_onnx(onnx_graph)

input = img.transpose(2, 0, 1)
x = input[np.newaxis, :, :, :] / 255.

target = 'rocm'
shape_dict = {'input_0': x.shape}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

ctx = tvm.context(target, 0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
m.set_input('input_0', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
m.run()

height_rescaled = height * rescale_factor
width_rescaled = width * rescale_factor
output_shape = (1, 3, height_rescaled, width_rescaled)
tvm_output = m.get_output(0, tvm.nd.empty(output_shape, dtype)).asnumpy()
im_resized = tvm_output[0].transpose(1, 2, 0) * 255.
im_resized = np.clip(im_resized, 0, 255)

from skimage.io import imsave
canvas = np.full((height_rescaled, width_rescaled*3, 3), 255)
canvas[0:height, 0:width, :] = img
canvas[:, width_rescaled:width_rescaled*2, :] = bicubic_im
canvas[:, width_rescaled*2:, :] = im_resized
imsave("srresnet_%s_canvas.png" % input_name, canvas.astype(np.uint8))
imsave("srresnet_output.png", im_resized.astype(np.uint8))
