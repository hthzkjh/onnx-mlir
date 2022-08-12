/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Conv.cpp - Shape Inference for Conv Op ---------------===//
//
// This file implements shape inference for the ONNX Conv Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

LogicalResult ONNXConv4OpShapeHelper::computeShape(
    ONNXConv4OpAdaptor operandAdaptor) {
  
  DimsExpr outputDims;

  Value filterValue = operandAdaptor.W();
  Optional<ArrayAttr> kernelShapeOpt = op->kernel_shape();
  Optional<ArrayAttr> padOpt = op->pads();
  Optional<ArrayAttr> strideOpt = op->strides();
  Optional<ArrayAttr> dilationOpt = op->dilations();

  bool ceilMode = false;
  bool hasFilter = true;
  llvm::SmallVector<IndexExpr, 2> kernelShape;
  llvm::SmallVector<IndexExpr, 4> pads;
  llvm::SmallVector<int64_t, 2> strides;
  llvm::SmallVector<int64_t, 2> dilations;

  Value xValue = (Value)operandAdaptor.X();
  int64_t rank = xValue.getType().cast<ShapedType>().getRank() - 1;
  int64_t spatialOffset = 2;
  int64_t spatialRank = rank - spatialOffset;

  MemRefBoundsIndexCapture XBounds(operandAdaptor.X());
  MemRefBoundsIndexCapture WBounds(filterValue);

  // Fill the stride, dilation, kernel.
  for (int i = 0; i < spatialRank; ++i) {
    // Strides, default 1.
    strides.emplace_back(
        strideOpt.hasValue() ? ArrayAttrIntVal(strideOpt, i) : 1);
    // Dilations, default 1.
    dilations.emplace_back(
        dilationOpt.hasValue() ? ArrayAttrIntVal(dilationOpt, i) : 1);
    // Kernel shape from attribute, default from Weight's spatial dims.
    if (kernelShapeOpt.hasValue()) {
      kernelShape.emplace_back(
          LiteralIndexExpr(ArrayAttrIntVal(kernelShapeOpt, i)));
    } else if (hasFilter) {
      int ii = i + spatialOffset; // W shape    // if input W got dims = 7, this += 2 alone is enough for change computeShape!
      // int ii = i + spatialOffset + 2;
      kernelShape.emplace_back(WBounds.getSymbol(ii)); // what is symbol? shape?
    } else {
      llvm_unreachable("should have tested the availability of kernel shape");
    }
  }
  // Pads, at this stage a given compile-time literal or default 0.
  for (int i = 0; i < 2 * spatialRank; ++i) {
    int64_t p = padOpt.hasValue() ? ArrayAttrIntVal(padOpt, i) : 0;
    pads.emplace_back(LiteralIndexExpr(p));
  }

  // Handle output size: start by inserting batch size and output channels.
  
  outputDims.emplace_back(XBounds.getDim(0));
  if (hasFilter)
    // outputDims.emplace_back(WBounds.getDim(0) * 16); // CO may be different from CI.
    outputDims.emplace_back(WBounds.getDim(0));
  // else
  //   outputDims.emplace_back(XBounds.getDim(1)); // CO is CI. 

  // Insert dimensions for the spatial axes. From MaxPool:
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool
  //
  // NOSET:
  //  * O[i] = floor((I[i] + P[i] - ((K[i] - 1) * d[i] + 1)) / s[i] + 1)
  // VALID:
  // * O[i] = floor((I[i] - {(K[i] - 1) * d[i] + 1} + 1) / s[i])
  // * P = 0
  // SAME_LOWER or SAME_UPPER:
  // * O[i] = ceil(I[i] / s[i])
  // * p' = (O[i] - 1) * s[i] + ((K[i] - 1) * d[i] + 1) - I[i]
  // * P[i] = p' / 2, if odd, first or second are increased by one.
  auto autoPad = op->auto_pad();
  auto autoFlag = true;
  LiteralIndexExpr zeroIE(0);
  LiteralIndexExpr oneIE(1);
  for (int64_t i = 0; i < spatialRank; ++i) {
    int64_t ii = i + spatialOffset;
    IndexExpr I = XBounds.getDim(ii);
    IndexExpr K = kernelShape[i];
    LiteralIndexExpr d(dilations[i]);
    LiteralIndexExpr s(strides[i]);
    IndexExpr t1 = K - oneIE;
    IndexExpr kdTerm = t1 * d + oneIE; // (k - 1) * d + 1
    if (autoPad == "NOTSET") {
      IndexExpr p = pads[i] + pads[i + spatialRank]; // Sum both pads.
      IndexExpr t1 = I + p; // Compute floor/ceil((I + p - kdTerm) / s) + 1.
      IndexExpr t2 = t1 - kdTerm;
      IndexExpr O;
      if (ceilMode)
        O = t2.ceilDiv(s);
      else
        O = t2.floorDiv(s);
      O = O + oneIE;
      // Set output dim, and pads already set, nothing more to do.
      outputDims.emplace_back(O);
    } 
    
    else if (autoPad == "VALID") {
      IndexExpr t1 = I - kdTerm; // Compute ceil((I - kdTerm +1)/s).
      IndexExpr t2 = t1 + oneIE;
      IndexExpr O = t2.ceilDiv(s);
      // Set output dim, and pads already set to zero, nothing more to do.
      outputDims.emplace_back(O);
    } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
      // Compute output as O = ceil(I/s).
      IndexExpr O = I.ceilDiv(s);
      outputDims.emplace_back(O);
      // Compute sum of pads padSum = (O -1)*s + kdTerm - I.
      IndexExpr t1 = O - oneIE;
      IndexExpr t2 = t1 * s + kdTerm;
      IndexExpr t3 = t2 - I;
      IndexExpr padSum = IndexExpr::max(t3, zeroIE);
      // Single pad value is padSump / 2.
      IndexExpr p = padSum.floorDiv(2);
      // Increment is 1 when pp % 2 != 0
      IndexExpr test = (padSum % 2) != zeroIE;
      IndexExpr inc = IndexExpr::select(test, oneIE, zeroIE);
      // Increment 1st value for SAME_LOWER and 2nd for SAME_UPPER.
      if (autoPad == "SAME_UPPER") {
        pads[i] = p;
        pads[i + spatialRank] = p + inc;
      } else { // SAME_LOWER.
        pads[i] = p + inc;
        pads[i + spatialRank] = p;
      }
    }
    
  }

#if DEBUG
  if (true) {
  // if (outputDims.size() == 4) {
    cerr << "2d conv const params";
    cerr << " rank: " << outputDims.size() ;
    if (outputDims[0].isLiteral())
      cerr << ", N " << outputDims[0].getLiteral();
    if (outputDims[1].isLiteral())
      cerr << ", CO " << outputDims[1].getLiteral();
    if (outputDims[2].isLiteral())
      cerr << ", WO " << outputDims[2].getLiteral();
    if (outputDims[3].isLiteral())
      cerr << ", HO " << outputDims[3].getLiteral();
    if (pads[0].isLiteral())
      cerr << ", ph begin " << pads[0].getLiteral();
    if (pads[2].isLiteral())
      cerr << ", ph end " << pads[2].getLiteral();
    if (pads[1].isLiteral())
      cerr << ", pw begin " << pads[1].getLiteral();
    if (pads[3].isLiteral())
      cerr << ", pw end " << pads[3].getLiteral();
    cerr << endl;
  }
#endif
  outputDims.resize(outputDims.size() + 1 );
  auto c = outputDims[1].floorDiv(16);
  outputDims[1] = c;
  auto c_zero = c -c;
  onnx_mlir::IndexExpr c_16(c_zero + 16);
  outputDims[4] = c_16;
  // outputDims[0] = c_zero;
  // outputDims[1] = c_zero;
  // Set type for the first output.
  dimsForOutput() = outputDims;
  return success();

}



} // namespace onnx_mlir
