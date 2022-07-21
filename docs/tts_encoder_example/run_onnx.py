import numpy as np 
import onnx
import onnxruntime as rt

npz = np.load('conformer_encoder_in_fp32.npz')
input_data = npz['input']
input_data = input_data.astype(np.int64)
mask_data = npz['mask']
mask_data = mask_data.astype(np.bool)

sess = rt.InferenceSession("conformer_encoder_kss_space_constant_v3.onnx")

# get output name
input_name = sess.get_inputs()[0].name
print("input name", input_name)
mask_name = sess.get_inputs()[1].name
print("mask name", mask_name)
output_name= sess.get_outputs()[0].name
print("output name", output_name)
duration= sess.get_outputs()[1].name
print("output name", duration)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
#forward model
res_1, res_2 = sess.run([output_name, duration], {input_name: input_data, mask_name: mask_data})
out = np.array(res_1)
res_2 = np.array(res_2)

np.savez('out_onnx.npz', encoder_output = out, duration = res_2)