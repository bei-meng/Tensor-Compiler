add_mlir_tool(tac-opt tac-opt.cpp)
target_link_libraries(
    tac-opt
    PRIVATE 
    # ===== 自定义方言 =====
    MLIRTacDialect

    # ===== 方言库 =====
    MLIROptLib              # 对应 #include "mlir/Tools/mlir-opt/MlirOptMain.h"
    MLIRFuncDialect     # 对应 #include "mlir/Dialect/Func/IR/FuncOps.h"
    MLIRTransforms          # 对应 #include "mlir/Transforms/Passes.h"
    MLIRSupport
    MLIRIR
    MLIRParser
)