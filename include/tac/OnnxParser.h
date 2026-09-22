#ifndef ONNX_PARSER_H
#define ONNX_PARSER_H
#include "OnnxModelInfo.h"

// 编译器只需要知道：ModelProto是一个类即可，不需要看到完整类定义
// 避免引入大的头文件

namespace onnx {
    class ModelProto;
}

namespace tac {
    ModelInfo parseModelProto(const onnx::ModelProto &model);
}
#endif //ONNX_PARSER_H