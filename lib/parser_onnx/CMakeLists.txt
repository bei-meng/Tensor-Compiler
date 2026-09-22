find_package(Protobuf REQUIRED)

# 构建 ONNX 解析与打印库
add_library(tac_onnx_parser STATIC
    OnnxParser.cpp
    OnnxDumping.cpp
    ${CMAKE_SOURCE_DIR}/include/onnx/onnx-ml.pb.cc
)

# 头文件搜索路径
target_include_directories(tac_onnx_parser
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${PROTOBUF_INCLUDE_DIRS}
)

# 链接依赖：静态库场景下，protobuf 改成 PUBLIC，保证上层可执行文件能继承到
target_link_libraries(tac_onnx_parser
    PUBLIC
        LLVMSupport
        ${PROTOBUF_LIBRARIES}
)
