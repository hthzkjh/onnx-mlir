module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func @main_graph(%arg0: tensor<?x32x22x22xf32>) -> tensor<?x16x18x18xf32> attributes {input_names = ["X"], output_names = ["Y"]} {
    %0 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1x4x4x2x16x5x5xf32>} : () -> tensor<1x4x4x2x16x5x5xf32>
    %1 = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<16xf32>} : () -> tensor<16xf32>
    %2 = "onnx.Constant"() {value = dense<[-1, 2, 16, 22, 22]> : tensor<5xi64>} : () -> tensor<5xi64>
    %3 = "onnx.Reshape"(%arg0, %2) : (tensor<?x32x22x22xf32>, tensor<5xi64>) -> tensor<?x2x16x22x22xf32>
    %4 = "onnx.Transpose"(%3) {perm = [0, 1, 3, 4, 2]} : (tensor<?x2x16x22x22xf32>) -> tensor<?x2x22x22x16xf32>
    %5 = "onnx.Conv4"(%4, %0, %1) {kernel_shape = [5, 5], pads = [0, 0, 0, 0]} : (tensor<?x2x22x22x16xf32>, tensor<1x4x4x2x16x5x5xf32>, tensor<16xf32>) -> tensor<?x0x18x18x16xf32>
    %6 = "onnx.Transpose"(%5) {perm = [0, 1, 4, 2, 3]} : (tensor<?x0x18x18x16xf32>) -> tensor<?x0x16x18x18xf32>
    %7 = "onnx.Constant"() {value = dense<[-1, 16, 18, 18]> : tensor<4xi64>} : () -> tensor<4xi64>
    %8 = "onnx.Reshape"(%6, %7) : (tensor<?x0x16x18x18xf32>, tensor<4xi64>) -> tensor<?x16x18x18xf32>
    return %8 : tensor<?x16x18x18xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
