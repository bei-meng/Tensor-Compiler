# 生成方言框架代码
set(LLVM_TARGET_DEFINITIONS TacDialect.td)
mlir_tablegen(TacDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TacDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRTacDialectIncGen)

# 操作
set(LLVM_TARGET_DEFINITIONS TacOps.td)
mlir_tablegen(TacOps.h.inc -gen-op-decls)
mlir_tablegen(TacOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTacOpsIncGen)


# 编译方言静态库
add_mlir_dialect_library(MLIRTacDialect
    tac.cpp
    DEPENDS
    MLIRTacDialectIncGen
    MLIRTacOpsIncGen
)

# 将生成文件目录加入公开头文件搜索路径，编译时生效
target_include_directories(MLIRTacDialect
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
