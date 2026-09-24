include "mlir/Pass/PassBase.td"

def LowerToTensor : Pass<"LowerToTensor"> {
    let summary = "Convert Tac To Tensor/Linalg/Arith";
    let constructor = "tac::createLowerToTensorPass()";
    let options = [
        Option<"useTranspose", "use-transpose", "bool", "false", "ijk --> ikj">
    ];
}

def LowerToLLVM : Pass<"LowerToLLVM"> {
    let summary = "LowerToLLVM";
    let constructor = "tac::createLowerToLLVMPass()";
}