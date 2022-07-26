import numpy as np 
import onnx
import onnxruntime as rt

npz = np.load('resnet18_in_fp32.npz')
input_data = npz['input']

sess = rt.InferenceSession("resnet18.onnx")

# get output name
input_name = sess.get_inputs()[0].name
print("input name", input_name)
output_name= sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
#forward model
res = sess.run([output_name], {input_name: input_data})
out = np.array(res)

np.savez('out_onnx.npz', output = out)