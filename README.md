# Tensor-Compiler
基于 MLIR 框架实现自定义**TAC 张量方言**（Tensor Accelerator Compiler），支持部分 ONNX 算子（Constant、Add、ReLU、MatMul），实现了从 ONNX 计算图经 MLIR 到 LLVM IR 的逐层降级流水线，并通过 JIT 编译执行。

## 1.环境配置
### 1.1 MLIR库安装
平台：Ubuntu 22.04.5 LTS \
mlir项目工程模版：
```
mlir-onnx
├── install             # Install Prefix，把 MLIR 编译后安装到这里
├── llvm-project        # MLIR 项目
└── tensor-compiler     # 自己的 MLIR 工程
```
首先，按照 MLIR [getting started](https://mlir.llvm.org/getting_started/) 的方法，安装 MLIR。
注意，安装的时候要设置 PREFIX 为 install 目录，如下面所示，和 getting start 上的略有区别：
```bash
# 使用当前MLIR稳定分支版本 22.1.8，只拉取对应标签的代码，国内镜像更快
# 4核8G编译时经常内存爆掉，使用动态库替代默认静态库，以及单核编译
git clone --depth 1 --branch llvmorg-22.1.8 https://mirrors.bfsu.edu.cn/git/llvm-project.git
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
  -DCMAKE_INSTALL_PREFIX=/mlir-onnx/install \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DLLVM_USE_LINKER=gold
```
在build完成之后，安装到prefix，耗时比较长，而且占用内存非常大
```bash
ninja -j1 install
nohup ninja -j1 install > install.log 2>&1 & # 后台安装，2核
```
现在 mlir 把所有二进制文件、库文件都装到了 install 目录下。
可以通过 `export PATH=/mlir-onnx/install/bin:$PATH` 将 MLIR 工具链加入环境变量，直接调用 `mlir-opt` `mlir-translate` 等二进制程序。
若需要永久生效，可把该环境变量配置追加到 `~/.bashrc` 文件末尾，执行 `source ~/.bashrc` 加载配置。
```bash
echo 'export PATH=/home/ubuntu/mlir-onnx/install/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```
验证:
```bash
which mlir-opt
```

### 1.2 配置clangd插件
有时候，mlir 的编译选项与 clangd 冲突，在 tensor-compiler 目录下建立 .clangd 文件，去掉相关的选项：
```
CompileFlags:
  Remove:
    - -fno-lifetime-dse
```

### 1.3 完整目录环境
```
├── build                   # 编译文件夹 
├── include                 
│   └── tac                 # tac方言的头文件
├── lib
│   ├── dialect_tac         # tac方言定义的td文件和实现cpp
│   └── pass_tac            # tac方言的转换和优化pass的td文件和实现
├── tools                   # tac-opt实现，方便测试
├── README.md
└── LICENSE
```

## 2. tac方言
### 2.1 tac方言定义
在 lib/dialect_tac 文件夹创建 TacDialect.td \
在 include/tac 文件夹创建 TacDialect.h \
其中 TacDialect.h.inc 为mlir按照TacDialect.td自动生成的文件
```mlir
// ============TacDialect.td============
#ifndef TAC_DIALECT_TD
#define TAC_DIALECT_TD

include "mlir/IR/DialectBase.td"
def TacDialect : Dialect{
    let name = "tac";
    let cppNamespace = "tac";
    let summary = "Tac dialect - by bei meng";
}
#endif

// ============TacDialect.h============
#pragma once
# include "mlir/IR/Dialect.h"
# include "mlir/IR/BuiltinDialect.h"
# include "TacDialect.h.inc"

```

### 2.2 tac op定义
在 lib/dialect_tac 文件夹创建 TacOps.td \
在 include/tac 文件夹创建 TacOps.h \
这里创建了一个constant的op，用于将字面值转换为一个SSA值，数据作为属性挂载在算子上。 \
AddOp、ReluOp、MatMulOp见完整文件内容 \
注意事项：td里面include的td头文件，在h文件里面也有导入对应的h文件
```mlir
// ============TacOps.td============
#ifndef TAC_OPS_TD
#define TAC_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "TacDialect.td"

class TacOp<string mnemonic,list<Trait> traits = []>:
    Op<TacDialect,mnemonic,traits>;

def ConstantOp : TacOp<"constant",[Pure]>{
    let summary = "constant";
    let description = [{
        常量算子将字面值转换为一个SSA值，数据作为属性挂载在算子上。
    }];
    let arguments = (ins ElementsAttr:$value);
    let results = (outs F32Tensor:$result);
    let hasVerifier = true;
    let builders = [
        OpBuilder<(ins "::mlir::DenseElementsAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>
    ];

    let assemblyFormat = "$value attr-dict `:` type($result)";
}
#endif

// ============TacOps.h============
#pragma once
// 基础IR
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
// td里include的
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

// td里面include,这里也要include对应的h文件
#include "TacDialect.h"         // 1. 先包含方言核心头文件（同目录下直接写文件名）
#define GET_OP_CLASSES          // 2. 控制宏：告诉.inc文件输出所有Op的完整类声明
#include "TacOps.h.inc"         // 3. TableGen生成的Op声明文件（构建目录下，通过搜索路径找到）
```

### 2.3 tac方言实现
在 lib/dialect_tac 文件夹创建 tac.cpp
```mlir
#include "tac/TacDialect.h"
#include "tac/TacOps.h"

// 对应具体实现
#define GET_DIALECT_DEFS
#include "TacDialect.cpp.inc"
#define GET_OP_CLASSES
#include "TacOps.cpp.inc"

using namespace tac;
using namespace mlir;
void TacDialect::initialize(){
    addOperations<
        #define GET_OP_LIST
        #include "TacOps.cpp.inc"
    >();
}

// 对应ConstantOp的hasVerifier= true
LogicalResult ConstantOp::verify() {
    auto tensorAttr = llvm::dyn_cast<DenseElementsAttr>(getValue());
    if (!tensorAttr) {return emitOpError("requires a dense elements attribute");}
    auto tensorType = llvm::cast<RankedTensorType>(getResult().getType());
    if (tensorAttr.getNumElements() != tensorType.getNumElements()) {
        return emitOpError() << "number of elements in 'value' attribute ("
                                << tensorAttr.getNumElements()
                                << ") does not match the result type ("
                                << tensorType.getNumElements() << ")";
    }
    return success();
}
```

### 2.4 tac-opt工具
tac-opt工具能方便在命令行测试 \
在 tools 文件夹创建 tac-opt.cpp
```mlir
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "tac/TacDialect.h"

using namespace mlir;
using namespace llvm;

int main(int argc,char **argv){
    DialectRegistry registry;
    // 注册dialect
    registry.insert<tac::TacDialect,func::FuncDialect>();
    // 注册pass
    registerCSEPass();
    registerCanonicalizerPass();
    return asMainReturnCode(MlirOptMain(argc,argv,"tac-opt",registry));
}
```

### 2.5 编译和测试
在build目录下面使用ninja工具进行编译
```bash
cd build
cmake .. -GNinja
ninja
ninja tac-opt
```
创建一个constant.mlir文件，用于测试方言和算子是否正确注册
```mlir
// constant.mlir
func.func @test_const() -> tensor<2x3xf32> {
  %0 = "tac.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}
```
build目录下使用tac-opt工具能正确读取和输出，说明方言和算子创建成功
```bash
./tac-opt constant.mlir

# 输出：
# module {
#   func.func @test_const() -> tensor<2x3xf32> {
#     %0 = tac.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32> : tensor<2x3xf32>
#     return %0 : tensor<2x3xf32>
#   }
# }
```
