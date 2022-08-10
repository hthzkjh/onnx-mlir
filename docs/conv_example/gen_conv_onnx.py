import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np

def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor


model_input_name = "X"
X = onnx.helper.make_tensor_value_info(model_input_name,
                                        onnx.TensorProto.FLOAT,
                                        [None, 32, 22, 22])
model_output_name = "Y"
model_output_channels = 16
Y = onnx.helper.make_tensor_value_info(model_output_name,
                                        onnx.TensorProto.FLOAT,
                                        [None, model_output_channels, 18, 18])

conv1_output_node_name = model_output_name
# Dummy weights for conv.
conv1_in_channels = 32
conv1_out_channels = 16
conv1_kernel_shape = (5, 5)
conv1_pads = (0, 0, 0, 0)
conv1_W = np.ones(shape=(1, 4, 4, 2, 16, 
                          *conv1_kernel_shape),dtype=np.float32)
conv1_B = np.ones(shape=(conv1_out_channels)).astype(np.float32)
# Create the initializer tensor for the weights.
conv1_W_initializer_tensor_name = "Conv1_W"
conv1_W_initializer_tensor = create_initializer_tensor(
    name=conv1_W_initializer_tensor_name,
    tensor_array=conv1_W,
    data_type=onnx.TensorProto.FLOAT)
conv1_B_initializer_tensor_name = "Conv1_B"
conv1_B_initializer_tensor = create_initializer_tensor(
    name=conv1_B_initializer_tensor_name,
    tensor_array=conv1_B,
    data_type=onnx.TensorProto.FLOAT)

conv1_node = onnx.helper.make_node(
    name="Conv1",  # Name is optional.
    op_type="Conv",
    # Must follow the order of input and output definitions.
    # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
    inputs=[
        model_input_name, conv1_W_initializer_tensor_name,
        conv1_B_initializer_tensor_name
    ],
    outputs=[conv1_output_node_name],
    # The following arguments are attributes.
    kernel_shape=conv1_kernel_shape,
    # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
    pads=conv1_pads,
)


# Create the graph (GraphProto)
graph_def = helper.make_graph(
  [conv1_node],
  'test-model',
  [X],
  [Y],
  initializer=[
    conv1_W_initializer_tensor, conv1_B_initializer_tensor
  ],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example')

print('The model is:\n{}'.format(model_def))
# onnx.checker.check_model(model_def)
onnx.save(model_def, "conv2.onnx")
print('The model is checked!')