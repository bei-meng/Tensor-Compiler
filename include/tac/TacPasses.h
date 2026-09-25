
#ifndef TAC_PASSES_H
#define TAC_PASSES_H
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>


#include "tac/TacDialect.h"
#include "tac/TacOps.h"

namespace mlir {
    class Pass;
}   // namespace mlir

namespace tac { 
    #define GEN_PASS_DECL
    #include "TacPasses.h.inc"
    std::unique_ptr<mlir::Pass> createLowerToTensorPass(LowerToTensorOptions options={});
    std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
    std::unique_ptr<mlir::Pass> createLinalgTilingPass();

    #define GEN_PASS_REGISTRATION
    #include "TacPasses.h.inc"
}   // namespace tac

#endif // TAC_PASSES_H