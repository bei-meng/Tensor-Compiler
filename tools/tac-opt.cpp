#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
// 导入MLIR自带的Pass
#include "mlir/Transforms/Passes.h"
// 导入Function dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 导入dialect
#include "tac/TacDialect.h" 
#include "tac/TacPasses.h"


// ===== 方言头文件 =====
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"


using namespace mlir;
using namespace llvm;

int main(int argc,char **argv){
    DialectRegistry registry;

    // 注册dialect
        registry.insert<
        tac::TacDialect,
        func::FuncDialect,
        linalg::LinalgDialect,
        tensor::TensorDialect,
        affine::AffineDialect,  // 新增：注册 Affine 方言
        scf::SCFDialect,
        cf::ControlFlowDialect,
        memref::MemRefDialect,
        arith::ArithDialect,
        LLVM::LLVMDialect
    >();
    
    registerCSEPass();
    registerCanonicalizerPass();

    // 自定义pass
    tac::registerPasses();
    return asMainReturnCode(MlirOptMain(argc,argv,"tac-opt",registry));
}
