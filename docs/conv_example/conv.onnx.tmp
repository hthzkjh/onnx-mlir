module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func @main_graph(%arg0: tensor<?x32x32x32xf32>) -> tensor<?x16x28x28xf32> attributes {input_names = ["X"], output_names = ["Y"]} {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16x32x5x5xf32>} : () -> tensor<16x32x5x5xf32>
    %1 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {kernel_shape = [5, 5], onnx_node_name = "Conv1", pads = [0, 0, 0, 0]} : (tensor<?x32x32x32xf32>, tensor<16x32x5x5xf32>, tensor<16xf32>) -> tensor<?x16x28x28xf32>
    return %2 : tensor<?x16x28x28xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
