#!/usr/bin/env python3

import numpy as np
from PyRuntime import ExecutionSession
import pandas as pd

# Load the model mnist.so compiled with onnx-mlir.
model = './conformer_decoder_kss_space_constant_v3.so'
session = ExecutionSession(model)
# Print the models input/output signature, for display.
# Signature functions for info only, commented out if they cause problems.
print("input signature in json", session.input_signature())
print("output signature in json",session.output_signature())

# input = np.random.random((1,3,224,224))
input_1 = np.load('conformer_decoder_in_fp32.npz')['input']
input_1 = input_1.astype(np.float32)
# image = input
#pd.DataFrame(image).to_csv("image.csv",header=None,index=None)
input_2 = np.load('conformer_decoder_in_fp32.npz')['mask']
input_2 = input_2.astype(np.bool)
out_1, out_2 = session.run([input_1, input_2])

np.savez('out_so.npz', encoder_output = out_1, duration = out_2)

print(outputs[0].shape)

prediction = outputs[0]
print(outputs[0][0,3])
digit = -1
prob = 0.0
for i in range(0,10):
    print("prediction ",i,"=",prediction[0,i])
    if prediction[0,i] > prob:
        digit = i
        prob = prediction[0,i]

