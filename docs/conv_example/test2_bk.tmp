module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func @main_graph(%arg0: tensor<?x16x32x32xf32>) -> tensor<?x16x32x32xf32> attributes {input_names = ["input"], output_names = ["output"]} {
    %0 = "onnx.Constant"() {value = dense<[-1, 1, 16, 32, 32]> : tensor<5xi64>} : () -> tensor<5xi64>
    %1 = "onnx.Reshape"(%arg0, %0) : (tensor<?x16x32x32xf32>, tensor<5xi64>) -> tensor<?x1x16x32x32xf32>
    %2 = "onnx.Transpose"(%1) {perm = [2, 1, 3, 4, 0]} : (tensor<?x1x16x32x32xf32>) -> tensor<16x1x32x32x?xf32>
    %3 = "onnx.Constant"() {value = dense<[-1, 16, 32, 32]> : tensor<4xi64>} : () -> tensor<4xi64>
    %4 = "onnx.Reshape"(%2, %3) : (tensor<16x1x32x32x?xf32>, tensor<4xi64>) -> tensor<?x16x32x32xf32>
    %5 = "onnx.Relu"(%4) {onnx_node_name = "Relu_1"} : (tensor<?x16x32x32xf32>) -> tensor<?x16x32x32xf32>
    return %5 : tensor<?x16x32x32xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
