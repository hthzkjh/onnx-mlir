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



namespace  {

struct TransposeWPattern : public ConversionPattern {
  TransposeWPattern(MLIRContext *context)
      : ConversionPattern(ONNXConstantOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op0, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    // Match
    ONNXConstantOp constantOp = ::llvm::dyn_cast<ONNXConstantOp>(op0);

    Location odsLoc = rewriter.getFusedLoc({op0->getLoc()});

    Value output = constantOp.output();
    auto attr1 = constantOp.valueAttr();  // *** //编译生成，会生成这个函数
    llvm::ArrayRef<int64_t> shape_1 = output.getType().cast<ShapedType>().getShape();
    /*
    ## 看 ShapedType 里面有clone函数
    auto clone(::llvm::ArrayRef<int64_t> shape) {
      return (*this).cloneWith(shape, (*this).getElementType());
    }
    */


    auto elementType = output.getType().cast<ShapedType>().getElementType();
    auto rank = output.getType().cast<ShapedType>().getRank();

    auto result = op0->getResults()[0];
    auto *operand = result.getUsers()[0];  // *** 参考文件里的源码写，但是有错
    if (operand->getName == "ONNXConv4Op" && rank == 4) {

      DenseIntElementsAttr shape2 = rewriter.getI64TensorAttr({shape_1[0] / 16, 4, 4, shape_1[1] / 16, 16, shape_1[2], shape_1[2]});
      // DenseIntElementsAttr 的初始化参数
      /// An attribute that represents a reference to a dense integer vector or tensor
/// object.

      // llvm::ArrayRef<int64_t> shape22(shape2, 7);
      Type tentorType = RankedTensorType::get({shape_1[0] / 16, 4, 4, shape_1[1] / 16, 16, shape_1[2], shape_1[2]}, elementType);
      auto constantDenseAttribute = DenseElementsAttr::get(tentorType, llvm::makeArrayRef(attr1));  // *** makeArrayRef param smallVector
      //  static DenseElementsAttr get(ShapedType type, ArrayRef<Attribute> values); //builder/FrontendDialectTransformer
      //
      
      Value constOp_2 = rewriter.create<ONNXConstantOp>(odsLoc, nullptr, constantDenseAttribute);
      rewriter.replaceOp(op0, constOp_2);
      return success();


/*
      
       //Attributes are usually passed by value.
      //  Instances of the Attribute class are references to immortal key-value pairs with immutable, uniqued keys owned by MLIRContext.
      // FrontendDialectTransform.cpp
      llvm::ArrayRef<int64_t> tensorDims(dims.data(), dims.size());   //ref.data()
      auto tensorType = RankedTensorType::get(tensorDims, elementType);
      auto constantDenseAttribute =
          DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
      auto constantOp = builder_.create<ONNXConstantOp>(
          UnknownLoc(), Attribute(), constantDenseAttribute);

ArrayAttr Builder::getF32ArrayAttr(ArrayRef<float> values) {
  auto attrs = llvm::to_vector<8>(llvm::map_range(
      values, [this](float v) -> Attribute { return getF32FloatAttr(v); }));
  return getArrayAttr(attrs);
}
    SmallVector<int64_t, 5> perm1({0,1,3,4,2});
    ArrayRef<int64_t> permArrayW1(perm1);
    ArrayAttr permAttrW1 = rewriter.getI64ArrayAttr(permArrayW1);

*/
      
    }
    // return failure();
    return success();
  }
};


void populateTransposeWONNXBeforePatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<TransposeWPattern>(ctx);
}



struct TransposeWToConv4Pass
    : public PassWrapper<TransposeWToConv4Pass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposeWToConv4Pass)

  StringRef getArgument() const override { return "TransposeWToConv4Pass-onnx"; }

  StringRef getDescription() const override {
    return "make the input W of conv4 to 7dims.";
  }

  void runOnOperation() final;
};

void TransposeWToConv4Pass::runOnOperation() {
  func::FuncOp function = getOperation();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXDialect, arith::ArithmeticDialect,
      func::FuncDialect>();

  RewritePatternSet patterns(context);
  populateWithGenerated(patterns);
  populateTransposeWONNXBeforePatterns(patterns, context);


  if (failed(applyPartialConversion(function, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

namespace onnx_mlir {

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> createTransposeWToConv4Pass() {
  return std::make_unique<TransposeWToConv4Pass>();
}

} // namespace onnx_mlir
