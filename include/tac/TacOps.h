#ifndef TAC_OPS_H
#define TAC_OPS_H
// 基础IR
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
// td里include的
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

// td里面include,这里也要include对应的h文件
#include "TacDialect.h"       // 1. 先包含方言核心头文件（同目录下直接写文件名）
#define GET_OP_CLASSES       // 2. 控制宏：告诉.inc文件输出所有Op的完整类声明
#include "TacOps.h.inc"     // 3. TableGen生成的Op声明文件（构建目录下，通过搜索路径找到）

#endif