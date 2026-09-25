#ifndef ONNX_DUMPING_H
#define ONNX_DUMPING_H

#include "tac/OnnxModelInfo.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace onnx {
    class ModelProto;
}

namespace tac {
std::unique_ptr<onnx::ModelProto> loadONNXModel(llvm::StringRef filename);
void dumpPROTO(const tac::ModelInfo &modelInfo);
void dumpTensor(const tac::TensorInfo &t);
void dumpValueInfo(const tac::ValueInfo &v);
}   // namespace tac

#endif // ONNX_DUMPING_H