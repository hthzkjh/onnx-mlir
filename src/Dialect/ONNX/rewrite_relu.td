//===----------------------------------------------------------------------===//
// Canonicalization for ONNXReluOp when meet cvon4Op
//===----------------------------------------------------------------------===//

def FuseConv4ReluPattern:  Pat<
   (ONNXReluOp (ONNXTransposeOp (ONNXReshapeOp Conv4($v, $w, $b), $s1), $p1)),
   (ONNXReluOp Conv4($v, $w, $b))>;