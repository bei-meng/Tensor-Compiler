#ifndef ONNX_MODEL_TO_MLIR_H
#define ONNX_MODEL_TO_MLIR_H

#include <memory.h>

// Forward declare
namespace mlir {
    class MLIRContext;
    template <typename OpTy>
    class OwningOpRef;
    class ModuleOp;
}// namespace mlir

namespace tac {
    class ModelInfo;
    mlir::OwningOpRef<mlir::ModuleOp> onnxModelToMlir(
        mlir::MLIRContext &context,ModelInfo &model
    );
}

#endif // ONNX_MODEL_TO_MLIR_H