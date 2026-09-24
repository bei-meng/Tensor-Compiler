# 生成 Pass 声明代码
set(LLVM_TARGET_DEFINITIONS TacPasses.td)
mlir_tablegen(TacPasses.h.inc -gen-pass-decls)

# Pass 模块公开聚合目标
add_public_tablegen_target(MLIRTacTransformsIncGen)

# 编译变换静态库
add_mlir_library(MLIRTacTransforms
  LowerToTensor.cpp
  LowerToLLVM.cpp

  DEPENDS
  MLIRTacTransformsIncGen


  LINK_LIBS PUBLIC
  MLIRTacDialect

  MLIRArithDialect 
  MLIRTensorDialect
  MLIRLinalgDialect
)

target_include_directories(MLIRTacTransforms
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

