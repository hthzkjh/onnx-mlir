module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func @main_graph(%arg0: tensor<?x16x6x6xf32>) -> tensor<?x?x?x?x?xf32> attributes {input_names = ["input"], output_names = ["output"]} {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x16x3x3xf32>} : () -> tensor<16x16x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %2 = "onnx.Constant"() {value = dense<[-1, 1, 16, 6, 6]> : tensor<5xi64>} : () -> tensor<5xi64>
    %3 = "onnx.Reshape"(%arg0, %2) : (tensor<?x16x6x6xf32>, tensor<5xi64>) -> tensor<?x1x16x6x6xf32>
    %4 = "onnx.Transpose"(%3) {perm = [0, 1, 3, 4, 2]} : (tensor<?x1x16x6x6xf32>) -> tensor<?x1x6x6x16xf32>
    %5 = "onnx.Conv4"(%4, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<?x1x6x6x16xf32>, tensor<16x16x3x3xf32>, tensor<16xf32>) -> tensor<?x1x4x4x16xf32>
    %6 = "onnx.Transpose"(%5) {perm = [0, 1, 4, 2, 3]} : (tensor<?x1x4x4x16xf32>) -> tensor<?x1x16x4x4xf32>
    %7 = "onnx.Constant"() {value = dense<[-1, 16, 4, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
    %8 = "onnx.Reshape"(%6, %7) : (tensor<?x1x16x4x4xf32>, tensor<4xi64>) -> tensor<?x16x4x4xf32>
    %9 = "onnx.Relu"(%8) {onnx_node_name = "Relu_1"} : (tensor<?x16x4x4xf32>) -> tensor<?x16x4x4xf32>
    %10 = "onnx.Shape"(%9) {onnx_node_name = "Shape_2"} : (tensor<?x16x4x4xf32>) -> tensor<4xi64>
    %11 = "onnx.Constant"() {onnx_node_name = "Constant_3", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %12 = "onnx.Gather"(%10, %11) {axis = 0 : si64, onnx_node_name = "Gather_4"} : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
    %13 = "onnx.UnsqueezeV11"(%12) {axes = [0], onnx_node_name = "Unsqueeze_5"} : (tensor<i64>) -> tensor<1xi64>
    %14 = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
    %15 = "onnx.Constant"() {value = dense<16> : tensor<1xi64>} : () -> tensor<1xi64>
    %16 = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
    %17 = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
    %18 = "onnx.Concat"(%13, %14, %15, %16, %17) {axis = 0 : si64, onnx_node_name = "Concat_6"} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5xi64>
    %19 = "onnx.Reshape"(%9, %18) {onnx_node_name = "Reshape_7"} : (tensor<?x16x4x4xf32>, tensor<5xi64>) -> tensor<?x?x?x?x?xf32>
    return %19 : tensor<?x?x?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
