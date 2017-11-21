from skimage.io import imread
from skimage import measure, color
import matplotlib.pyplot as plt
from os.path import join
import glob
import numpy as np

import nnvm
import tvm
import onnx
import nnvm.compiler
from tvm.contrib import graph_runtime

def overlay_mask(img, mask, alpha=0.8):
    rows, cols = img.shape
    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [1, 0, 1]
    color_mask[mask == 2] = [0, 1, 0]
    color_mask[mask == 3] = [1, 1, 0]
    color_mask[mask == 4] = [0, 1, 1]        
    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    return color.hsv2rgb(img_hsv)

def rescale(img):
    mx = np.max(img)
    mn = np.min(img)
    return (img - mn) / (mx - mn)

img = np.load("data/unet/223.npy")
label = imread("data/unet/223.png")
height = img.shape[0]
width = img.shape[1]

onnx_graph = onnx.load('data/unet/unet_acdc.onnx')
sym, params = nnvm.frontend.from_onnx(onnx_graph)

x = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
print("x.shape", x.shape)
target = 'rocm'
shape_dict = {'input_0': x.shape}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

ctx = tvm.context(target, 0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
m.set_input('input_0', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
m.run()

n_class = 4
out_shape = (1, n_class, height, width)
tvm_output = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
seg = np.argmax(tvm_output[0], axis=0)

from skimage.io import imsave
plt.figure(figsize=(18,15))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap=plt.gray())
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(overlay_mask(rescale(img), seg))
plt.axis("off")
plt.title("Prediction")

plt.subplot(1, 3, 3)
plt.imshow(overlay_mask(rescale(img), label))
plt.axis("off")
plt.title("Ground Truth")

plt.savefig("unet_seg.png", bbox_inches='tight')

# ftimer = m.module.time_evaluator("run", ctx, 50)
# print(ftimer().mean)
