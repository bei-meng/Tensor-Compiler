find_package(Protobuf REQUIRED)

# 构建 ONNX 解析与打印库
add_library(tac_onnx_parser STATIC
    OnnxParser.cpp
    OnnxDumping.cpp
    OnnxModelToMlir.cpp
    ${CMAKE_SOURCE_DIR}/include/onnx/onnx-ml.pb.cc
)

# 头文件搜索路径
target_include_directories(tac_onnx_parser
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    PRIVATE
    ${PROTOBUF_INCLUDE_DIRS}
)

# 链接依赖
target_link_libraries(tac_onnx_parser
    PUBLIC
    LLVMSupport
    ${PROTOBUF_LIBRARIES}

    MLIRTacDialect # 依赖库放在被依赖库的后面,MLIRInferTypeOpInterface必须放在后面
    # ===== 新增：MLIR相关依赖 =====
    MLIRIR
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRInferTypeOpInterface
)
