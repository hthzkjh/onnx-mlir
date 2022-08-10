def HaveSameLastDim: Constraint<
  CPred<"onnx_mlir::hasShapeAndRank($0) && "
        "$0.getType().cast<RankedTensorType>().getShape()[" # dim # "] != -1 && "
        "($0.getType().cast<RankedTensorType>().getShape()[" # dim # "] =="
        "3">,
  "ConstantOp W is the input of Conv4Op and has rank = 4">;

def IsFromONNXConstantOp: Constraint<
    CPred<"llvm::dyn_cast_or_null<ONNXConstantOp>($0.getDefiningOp())">,
    "Is a value from ONNXConstantOp">;

def IsFromONNXConstantOpWithDenseElementsAttr: Constraint<
    And<[CPred<" isa<ONNXConstantOp>($_self.getDefiningOp()) ">,
         CPred<" onnx_mlir::getONNXConstantOp($_self).valueAttr().isa<DenseElementsAttr>() ">
        ]>, "Value is not a ONNXConstantOp with a DenseElementsAttr">;
        
def TransposeVariadicInput: NativeCodeCall<
  "onnx_mlir::transposeVariadicInput($_builder, $_loc, $0, $1)">; // function in rewrite.cpp 

// Do transpose on concat's inputs instead of output in order to propagate
// transpose operations together, which allows more chance for transpose fusion.
// Do this only when all inputs are produced by transpose operations.
def SwapTransposeConcatPattern: Pat<
  (ONNXTransposeOp:$res (ONNXConcatOp $inputs, $axis), $perm),
  (ONNXConcatOp (TransposeVariadicInput $inputs, $perm),
                (GetIndexOfAxisInPerm $perm, $axis)),
  [(ProducedByTransposeOp:$inputs)]
>;
// rewrite.cpp
// Transpose a variadic input using a permutation array.
SmallVector<Value, 4> transposeVariadicInput(PatternRewriter &rewriter,
    Location loc, ValueRange inputs, ArrayAttr permAttr) {
  SmallVector<Value, 4> transposedInputs;
  for (Value inp : inputs) {
    ShapedType inpType = inp.getType().cast<ShapedType>();
    assert(inpType && "Type is not ShapedType");
    ONNXTransposeOp transposeOp = rewriter.create<ONNXTransposeOp>(
        loc, UnrankedTensorType::get(inpType.getElementType()), inp, permAttr);
    (void)transposeOp.inferShapes([](Region &region) {});   // do shape inference function (infer result shape)
    transposedInputs.emplace_back(transposeOp.getResult());
  }
  return transposedInputs;
}
// Check if all values are produced by ONNXTransposeOp.
bool areProducedByTransposeOp(ValueRange values) {
  return llvm::all_of(values, [](Value v) {
    if (v.isa<BlockArgument>())
      return false;
    return isa<ONNXTransposeOp>(v.getDefiningOp());
  });
}

// Check if all values are produced by ONNXConstantOp.
bool areProducedByConstantOp(ValueRange values) {
  return llvm::all_of(values, [](Value v) {
    if (v.isa<BlockArgument>())
      return false;
    return isa<ONNXConstantOp>(v.getDefiningOp());
  });
}
// 我需要替换constantOp。如果从Conv4的角度出发，没法替换conv4Op之前的Op

