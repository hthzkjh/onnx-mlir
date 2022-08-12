/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- ONNXDecompose.cpp - ONNX High Level Rewriting ------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters to decompose an ONNX operation into
// composition of other ONNX operations.
//
// This pass is applied before any other pass so that there is no need to
// implement shape inference for the decomposed operation. Hence, it is expected
// that there is no knowledge about tensor shape at this point.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

// Create an DenseElementsAttr of ArrayAttr.
// This function is used to get Value Type of an EXISTING ArrayAttr for Scaler
// function.
// PatternRewriter
DenseElementsAttr createDenseArrayAttr(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  assert(origAttrs && "handle EXISTING ArrayAttr only");

  if (origAttrs.getValue()[0].dyn_cast<FloatAttr>()) {
    Type elementType = rewriter.getF32Type();
    int nElements = origAttrs.getValue().size();
    SmallVector<float, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i)
      wrapper[i] = origAttrs.getValue()[i].cast<FloatAttr>().getValueAsDouble();

    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }

  if (origAttrs.getValue()[0].dyn_cast<IntegerAttr>()) {
    Type elementType = rewriter.getIntegerType(64);
    int nElements = origAttrs.getValue().size();
    SmallVector<int64_t, 4> wrapper(nElements, 0);
    for (int i = 0; i < nElements; ++i)
      wrapper[i] = origAttrs.getValue()[i].cast<IntegerAttr>().getInt();

    return DenseElementsAttr::get(
        RankedTensorType::get(wrapper.size(), elementType),
        llvm::makeArrayRef(wrapper));
  }

  llvm_unreachable("unexpected attribute type");
}

/// Create an Scalar DenseElementsAttr from FloatAttr or IntergerAttr.
/// This is used to create an ONNXConstant of rank 0, e.g. tensor<f32>.
DenseElementsAttr createScalarDenseAttr(
    PatternRewriter &rewriter, Attribute attr) {
  if (attr.dyn_cast<FloatAttr>()) {
    Type elementType = rewriter.getF32Type();
    SmallVector<float, 1> wrapper;
    wrapper.emplace_back(attr.cast<FloatAttr>().getValueAsDouble());
    return DenseElementsAttr::get(
        RankedTensorType::get({}, elementType), llvm::makeArrayRef(wrapper));
  }

  if (attr.dyn_cast<IntegerAttr>()) {
    Type elementType = rewriter.getIntegerType(64);
    SmallVector<int64_t, 1> wrapper;
    wrapper.emplace_back(attr.cast<IntegerAttr>().getInt());
    return DenseElementsAttr::get(
        RankedTensorType::get({}, elementType), llvm::makeArrayRef(wrapper));
  }

  llvm_unreachable("unexpected attribute type");
}

Value createUnitConstant(PatternRewriter &rewriter, Location loc) {
  return rewriter.create<ONNXNoneOp>(loc);
}

// Create an DenseElementsAttr of ArrayAttr.
// When ArrayAttr is Null, an empty Integer DenseElementAttr is returned
DenseElementsAttr createDenseArrayAttrOrEmpty(
    PatternRewriter &rewriter, ArrayAttr origAttrs) {
  if (origAttrs)
    return createDenseArrayAttr(rewriter, origAttrs);

  Type elementType = rewriter.getIntegerType(64);
  int nElements = 0;
  SmallVector<int64_t, 4> wrapper(nElements, 0);
  for (int i = 0; i < nElements; ++i)
    wrapper[i] = i;

  return DenseElementsAttr::get(
      RankedTensorType::get(wrapper.size(), elementType),
      llvm::makeArrayRef(wrapper));
}

Value createSequenceConstructOp(
    PatternRewriter &rewriter, mlir::Value seq, mlir::OperandRange inputs) {
  Type resType = seq.getType();
  Location loc = seq.getLoc();
  Value position = rewriter.create<ONNXNoneOp>(loc);

  for (auto input : inputs)
    seq = rewriter.create<ONNXSequenceInsertOp>(
        loc, resType, seq, input, position);

  return seq;
}

} // namespace onnx_mlir

namespace onnx_mlir {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Transform/ONNX/ONNXDecompose.inc"

RankedTensorType createResultType(
    Type outputType, int64_t axisValue, bool keepDims) {
  RankedTensorType outputShapeType = outputType.dyn_cast<RankedTensorType>();
  llvm::ArrayRef<int64_t> shapeVector = outputShapeType.getShape();
  int64_t rank = outputShapeType.getRank();
  if (axisValue < 0)
    axisValue += rank;
  SmallVector<int64_t, 4> reducedShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (i != axisValue)
      reducedShape.push_back(shapeVector[i]);
    else if (keepDims)
      reducedShape.push_back(1);
  }
  Type elementType = outputShapeType.getElementType();
  RankedTensorType resultType =
      RankedTensorType::get(reducedShape, elementType);
  return resultType;
}

// 
//   IndexExpr::getShape(shapeHelper.dimsForOutput(), outputDims);
//   op.getResult().setType(RankedTensorType::get(outputDims, elementType));
//   (A.getType().cast<ShapedType>().getRank() == 2) 
// ArrayRef<int64_t> aShape = A.getType().cast<RankedTensorType>().getShape();
// Type resultType = RankedTensorType::get(input.getType().cast<ShapedType>().getShape(), to.getValue());
// 

// TransposeOp Type wrong, should give right size in Type
struct DecomposeConvPattern : public ConversionPattern {
  DecomposeConvPattern(MLIRContext *context)
      : ConversionPattern(ONNXConvOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op0, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Variables for capturing values and attributes used while creating ops
    StringAttr auto_pad;
    ArrayAttr dilation;
    IntegerAttr group;
    ArrayAttr kernel_shape;
    ArrayAttr pads;
    ArrayAttr stride;

    // Match
    ONNXConvOp convOp = ::llvm::dyn_cast<ONNXConvOp>(op0);
    Value x = convOp.X();
    Type xType = x.getType();
    // Shape xShape = x.getShape();
  //   llvm::ArrayRef<int64_t> xShape = x.getShape();
  //   SmallVector<int64_t, 4> reducedShape;
  // for (int64_t i = 0; i < rank; ++i) {
  //   if (i != axisValue)
  //     reducedShape.push_back(shapeVector[i]);
      
    onnx_mlir::MemRefBoundsIndexCapture XBounds(x);
    Value w = convOp.W();
    Value b = convOp.B();

    auto_pad = op0->getAttrOfType<StringAttr>("auto_pad");
    dilation = op0->getAttrOfType<ArrayAttr>("dilations");
    group = op0->getAttrOfType<IntegerAttr>("group");
    kernel_shape = op0->getAttrOfType<ArrayAttr>("kernel_shape");
    pads = op0->getAttrOfType<ArrayAttr>("pads");
    stride = op0->getAttrOfType<ArrayAttr>("strides");

    // if (!axis)
    //   axis = rewriter.getIntegerAttr(
    //       rewriter.getIntegerType(64, /*isSigned=*/true), -1);
    // int64_t axisValue = axis.getSInt();
    // Rewrite
    Location odsLoc = rewriter.getFusedLoc({op0->getLoc()});

    // onnx_mlir::IndexExpr N(XBounds.getDim(0));
    // onnx_mlir::IndexExpr C(XBounds.getDim(1));
    // onnx_mlir::IndexExpr H(XBounds.getDim(2));
    // onnx_mlir::IndexExpr W(XBounds.getDim(3));
    // onnx_mlir::IndexExpr C_(C.ceilDiv(16));
    // onnx_mlir::IndexExpr c_zero(C - C);
    // onnx_mlir::IndexExpr c_16(c_zero + 16);
    llvm::ArrayRef<int64_t> shape_1 = x.getType().cast<ShapedType>().getShape();
    // using DimsExpr = SmallVector<onnx_mlir::IndexExpr, 4>;
    // DimsExpr shape1;
    // shape1.emplace_back(shape_1[0]);
    // shape1.emplace_back(shape_1[1] / 16);
    // shape1.emplace_back(16);
    // shape1.emplace_back(shape_1[2]);
    // shape1.emplace_back(shape_1[3]);
    // SmallVector<int64_t,5> shape_11({shape_1[0], shape_1[1] / 16, shape_1[2], shape_1[3]}.);
    // ArrayRef<NamedAttribute> shapeArrayW11("shape",shape_11);
    // ArrayAttr shapeAttrW11 = rewriter.getI64ArrayAttr(shapeArrayW11);
    // Value value1 = rewriter.create<ONNXReshapeOp>(odsLoc, xType, x, nullptr); 
    Type type_1 = x.getType().cast<ShapedType>().getElementType();
    Type reshape1Type = RankedTensorType::get({shape_1[0], shape_1[1] / 16, 16, shape_1[2], shape_1[3]}, type_1);
    // RankedTensorType shape111 = RankedTensorType::get({N.getLiteral(),C_.getLiteral(),16,H.getLiteral() ,W.getLiteral()}, );/
    Value shape111 = rewriter.create<ONNXConstantOp>(odsLoc, nullptr, rewriter.getI64TensorAttr({shape_1[0], shape_1[1] / 16, 16, shape_1[2], shape_1[3]})).getResult();
    Value value1 = rewriter.create<ONNXReshapeOp>(odsLoc, reshape1Type, x, shape111); 
    //  note:   no known conversion for argument 5 from 'ArrayRef<long int>' to 'ArrayRef<mlir::NamedAttribute>'
    // ror: no matching function for call to 'mlir::RankedTensorType::RankedTensorType(mlir::detail::ValueImpl* const&)'


  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transposed, ::mlir::Value data, /*optional*/::mlir::ArrayAttr perm);
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, /*optional*/::mlir::ArrayAttr perm);
  // static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});

  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type reshaped, ::mlir::Value data, ::mlir::Value shape);
  // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value shape);
  // static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});


    // ArrayAttr perm1 = [0,1,3,4,2];
    // ArrayAttr per2 = [0,1,4,2,3];
    // SmallVector<int64_t, 4> perm1;
    // perm1.emplace_back(0);
    // perm1.emplace_back(1);
    // perm1.emplace_back(3);
    // perm1.emplace_back(4);
    // perm1.emplace_back(2);
    SmallVector<int64_t, 5> perm1({0,1,3,4,2});
    ArrayRef<int64_t> permArrayW1(perm1);
    ArrayAttr permAttrW1 = rewriter.getI64ArrayAttr(permArrayW1);
    RankedTensorType transpose1Type = RankedTensorType::get(perm1, type_1);
    // Value transposeOp1 = rewriter.create<ONNXTransposeOp>(odsLoc, xType, value1, permArrayW1);
    Value transposeOp1 = rewriter.create<ONNXTransposeOp>(odsLoc, transpose1Type, value1, permAttrW1);

    Type conv4Type = RankedTensorType::get({shape_1[0], shape_1[1] / 16, shape_1[2], shape_1[3], 16}, type_1);
    Value value3 = rewriter.create<ONNXConv4Op>(odsLoc, conv4Type, transposeOp1,
      convOp.W(), convOp.B(), auto_pad, dilation, group, kernel_shape, pads, stride);

    SmallVector<int64_t, 5> perm2({0,1,4,2,3});
    ArrayRef<int64_t> permArrayW2(perm2);
    ArrayAttr permAttrW2 = rewriter.getI64ArrayAttr(permArrayW2);
    RankedTensorType transpose2Type = RankedTensorType::get(perm2, type_1);
    Value value4 = rewriter.create<ONNXTransposeOp>(odsLoc, transpose2Type, value3, permAttrW2);        
    
    // get value4 shape [N,c,16,h,w] shape2=[N,c*16,h,w]
    // onnx_mlir::IndexExpr a = value4.get
    
    
    // RankedTensorType value4ShapeType = value4.dyn_cast<onnx_mlir::ONNXTransposeOP>().getResult().getShape().dyn_cast<RankedTensorType>();
    ArrayRef<int64_t> value4Vector = value4.getType().cast<ShapedType>().getShape();
    int16_t rank1 = value1.getType().cast<ShapedType>().getRank();
    // auto reshapeOp1 = llvm::dyn_cast<mlir::memref::ReshapeOp>(value1);
    // auto rank11 = reshapeOp1.getResult().cast<ShapedType>().getRank();
    int16_t rank2 = transposeOp1.getType().cast<ShapedType>().getRank();
    int16_t rank3 = value3.getType().cast<ShapedType>().getRank();
    int16_t rank4 = value4.getType().cast<ShapedType>().getRank();
    // std:: cout << type_1.dump() << std::endl;
    // type_1.print(std::cout);
    std:: cout << rank1 << " " << rank2 << " " << rank3 << " " <<rank4 << " " << std::endl;
    // ONNXReshapeOpAdaptor operandAdaptor(operands, op->getAttrDictionary()); // Conversion/ONNXToMhlo/Tensor/Reshape.cpp:36:
    /*
    // dyn_cast error
    llvm::ArrayRef<int64_t> value4Vector = value4ShapeType.getShape();
    SmallVector<int64_t, 4> shape2;
    shape2.push_back(value4Vector[0]);
    shape2.push_back(value4Vector[1] * value4Vector[2]);
    shape2.push_back(value4Vector[3]);
    shape2.push_back(value4Vector[4]);
    ArrayRef<int64_t> shapeArrayW22(shape2);


    // Value value5 = rewriter.create<ONNXReshapeOp>(odsLoc, xType, value4, shapeArrayW22);
    */
    // Value value5 = rewriter.create<ONNXReshapeOp>(odsLoc, xType, value4, nullptr); // pass ok
    Value shape222 = rewriter.create<ONNXConstantOp>(odsLoc, nullptr, rewriter.getI64TensorAttr({value4Vector[0], value4Vector[1] * value4Vector[2], value4Vector[3], value4Vector[4]})).getResult();
    Type reshape2Type = RankedTensorType::get({value4Vector[0], value4Vector[1] * value4Vector[2], value4Vector[3], value4Vector[4]}, type_1); // 编译通过，执行不通过(原因在于上面不是5维是4维啦)
    Value value5 = rewriter.create<ONNXReshapeOp>(odsLoc, reshape2Type, value4, shape222); 
    int16_t rank5 = value5.getType().cast<ShapedType>().getRank();
    std::cout << rank5 << std::endl;

    
    
    rewriter.replaceOp(op0, value5);
    // rewriter.replaceOp(op0,value4);
    return success();
  }
};







struct DecomposeONNXToONNXPass
    : public PassWrapper<DecomposeONNXToONNXPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeONNXToONNXPass)

  StringRef getArgument() const override { return "decompose-onnx"; }

  StringRef getDescription() const override {
    return "Decompose ONNX operations into composition of other ONNX "
           "operations.";
  }

  void runOnOperation() final;
};

void DecomposeONNXToONNXPass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXDialect, arith::ArithmeticDialect,
      func::FuncDialect>();

  // These ops will be decomposed into other ONNX ops. Hence, they will not be
  // available after this pass.
  target.addIllegalOp<ONNXClipV6Op>();
  target.addIllegalOp<ONNXClipV11Op>();
  target.addIllegalOp<ONNXClipV12Op>();
  target.addIllegalOp<ONNXLogSoftmaxOp>();
  target.addIllegalOp<ONNXPadV2Op>();
  target.addIllegalOp<ONNXPadV11Op>();
  target.addIllegalOp<ONNXReduceL1Op>();
  target.addIllegalOp<ONNXReduceL2Op>();
  target.addIllegalOp<ONNXReduceLogSumOp>();
  target.addIllegalOp<ONNXReduceLogSumExpOp>();
  target.addIllegalOp<ONNXReduceSumSquareOp>();
  target.addIllegalOp<ONNXResizeV11Op>();
  target.addIllegalOp<ONNXResizeV10Op>();
  target.addIllegalOp<ONNXScalerOp>();
  target.addIllegalOp<ONNXScatterOp>();
  target.addIllegalOp<ONNXSequenceConstructOp>();
  target.addIllegalOp<ONNXUpsampleOp>();
  target.addIllegalOp<ONNXUpsampleV9Op>();
  target.addIllegalOp<ONNXUpsampleV7Op>();
  target.addIllegalOp<ONNXConvOp>();    // this is ok

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);
  patterns.insert<onnx_mlir::DecomposeConvPattern>(&getContext());

  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> createDecomposeONNXToONNXPass() {
  return std::make_unique<DecomposeONNXToONNXPass>();
}

} // namespace onnx_mlir
