cmake_minimum_required(VERSION 3.13.4)

project(tensor-compiler VERSION 0.0.0)

# 防止 cmake 生成一堆 warning
cmake_policy(SET CMP0116 NEW)

# 生成 compile_commands.json 便于代码高亮
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_compile_options(-g)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR})

# include_directories用于c++找h文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# 构建顺序：先库后工具
add_subdirectory(lib)
add_subdirectory(tools)


# ==== onnx解析测试 ====
add_executable(onnx_parse_test
    onnxParseTest.cpp
)

target_link_libraries(onnx_parse_test
    PRIVATE
        tac_onnx_parser
)