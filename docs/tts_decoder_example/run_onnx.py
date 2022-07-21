import numpy as np 
import onnx
import onnxruntime as rt

npz = np.load('conformer_decoder_in_fp32.npz')
input_data = npz['input']
mask_data = npz['mask']
mask_data = mask_data.astype(np.bool)

sess = rt.InferenceSession("conformer_decoder_kss_space_constant_v3.onnx")

# get output name
input_name = sess.get_inputs()[0].name
print("input name", input_name)
mask_name = sess.get_inputs()[1].name
print("mask name", mask_name)
output_name= sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
#forward model
res = sess.run([output_name], {input_name: input_data, mask_name: mask_data})
out = np.array(res)

np.savez('out_onnx.npz', output = out)