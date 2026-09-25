include "mlir/Pass/PassBase.td"

def LowerToTensor : Pass<"LowerToTensor"> {
    let summary = "Convert Tac To Tensor/Linalg/Arith";
    let constructor = "tac::createLowerToTensorPass()";
    let options = [
        // 代码里的名字 命令行的名字 类型    默认值   帮助
        Option<"useTranspose", "use-transpose", "bool", "false", "KN --> NK">,
        Option<"useExchange", "loop-nterchange", "bool", "false", "ijk --> ikj">
    ];
}

def LowerToLLVM : Pass<"LowerToLLVM"> {
    let summary = "LowerToLLVM";
    let constructor = "tac::createLowerToLLVMPass()";
}

def LinalgTiling : Pass<"LinalgTiling"> {
    let summary = "LinalgTiling";
    let constructor = "tac::createLinalgTilingPass()";
    let options = [
        // 代码里的名字 命令行的名字 类型    默认值   帮助
        Option<"tileSize", "tile-size", "std::string", "\"64,64,64\"", "tileSize">
    ];
}