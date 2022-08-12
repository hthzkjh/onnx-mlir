module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func @main_graph(%arg0: tensor<?x16x6x6xf32>) -> tensor<?x16x4x4xf32> attributes {input_names = ["input"], output_names = ["output"]} {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x16x3x3xf32>} : () -> tensor<16x16x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_0", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<?x16x6x6xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>) -> tensor<*xf32>
    %3 = "onnx.Relu"(%2) {onnx_node_name = "Relu_1"} : (tensor<*xf32>) -> tensor<?x16x4x4xf32>
    return %3 : tensor<?x16x4x4xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
