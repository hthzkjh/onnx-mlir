/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Conv.cpp - Shape Inference for Conv Op ---------------===//
//
// This file implements shape inference for the ONNX Conv Operator.
//
//===----------------------------------------------------------------------===//

// #include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
// #include "src/Dialect/ONNX/ShapeInference/Conv.cpp"

using namespace mlir;

namespace onnx_mlir {

ONNXConv4OpShapeHelper::ONNXConv4OpShapeHelper(
    ONNXConv4Op *newOp, IndexExprScope *inScope)
    : ONNXGenericPoolShapeHelper<ONNXConv4Op, ONNXConv4OpAdaptor>(
          newOp, true /*hasFilter*/, false /*hasCeil*/, inScope) {}

ONNXConv4OpShapeHelper::ONNXConv4OpShapeHelper(ONNXConv4Op *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXGenericPoolShapeHelper<ONNXConv4Op, ONNXConv4OpAdaptor>(newOp,
          true /*hasFilter*/, false /*hasCeil*/, rewriter, fGetDenseVal,
          fLoadVal, inScope) {}

LogicalResult ONNXConv4OpShapeHelper::computeShape(
    ONNXConv4OpAdaptor operandAdaptor) {
  
    mlir::LogicalResult outputFlag = ONNXGenericPoolShapeHelper<ONNXConv4Op,
      ONNXConv4OpAdaptor>::computeShape(operandAdaptor, operandAdaptor.W(),
      op->kernel_shape(), op->pads(), op->strides(), op->dilations());
     if (failed(outputFlag)) {
        return op->emitError("Failed to scan Conv4 parameters successfully");
     }
     
    
    // after this function, ONNXGenericPoolShapeHelper<ONNXConv4Op,ONNXConv4OpAdaptor>.dimsForOutput() == self dimsForOutput()
    // SmallVector<int64_t, 4> outputDims_conv;
    // IndexExpr::getShape(dimsForOutput(), outputDims_conv);
    DimsExpr outputDims = dimsForOutput();
    // convert from SmallVector<int64_t, 4> to using DimsExpr = llvm::SmallVector<IndexExpr, 4>;

    
//     auto elementType = operandAdaptor.X().getType().cast<ShapedType>().getElementType();
//   return shapeHelperInferShapes<ONNXConv4OpShapeHelper, ONNXConv4Op,ONNXConv4OpAdaptor>(*this, elementType);
//   in shapeHelperInferShapes
//   if (failed(shapeHelper.computeShape(operandAdaptor)))
//     return op.emitError("Failed to scan " + OP::getOperationName() +
//                         " parameters successfully");

// // using DimsExpr = llvm::SmallVector<IndexExpr, 4>;
//   SmallVector<int64_t, 4> outputDims;
//   IndexExpr::getShape(shapeHelper.dimsForOutput(), outputDims);
//   op.getResult().setType(RankedTensorType::get(outputDims, elementType));

    

    // outputDims.resize(inputRank);
    // MemRefBoundsIndexCapture inputBounds(input);
    DimIndexExpr N(outputDims[0]);
    DimIndexExpr C(outputDims[1]);
    DimIndexExpr H(outputDims[2]);
    DimIndexExpr W(outputDims[3]);
    outputDims.resize(5);
    auto cc = C - C;
    llvm::SmallVector<IndexExpr, 5> output({N, C.ceilDiv(16), cc+16, H, W});
    // output.emplace_back(N);
    // output.emplace_back(C.ceilDiv(16));
    // output.emplace_back(cc + 16);
    // output.emplace_back(H);
    // output.emplace_back(W);

    // int64_t cc = 16;
    // int w = W.getValue().getImpl()->getKind();
    // MathBuilder createMath(scope->getRewriter(), op->getLoc());
    // Value indexVal = createMath.castToIndex(cc);
    // auto cc = C - C;
    outputDims[0] = N;
    outputDims[1] = C.ceilDiv(16);
    outputDims[2] = H;
    outputDims[3] = W;
    outputDims[4] = cc + 16;  // resize 没用，这里用5也不报错
    //  error: ambiguous overload for 'operator=' (operand types are 'onnx_mlir::IndexExpr' and 'int64_t' {aka 'long int'})
    // watch /Dialect/ONNX/MLIR/IndexExpre.cpp, need a IndexExpr to interaction with int

  // Save the final result.
    dimsForOutput() = outputDims;
    // dimsForOutput() = output;
    // std::cout << outputDims.shape() << std::endl;
    return success();
}



} // namespace onnx_mlir
