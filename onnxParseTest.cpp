#include "tac/OnnxParser.h"
#include "tac/OnnxModelInfo.h"
#include "tac/OnnxDumping.h"
#include "tac/OnnxModelToMlir.h"
#include "tac/TacDialect.h"
#include "tac/DriverUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include "onnx/onnx-ml.pb.h"

int main(int argc, char** argv) {
    bool enable_dump_proto = false;
    bool enable_print_mlir = false;

    // 参数至少要有模型路径
    if (argc < 2) {
        llvm::errs() << "Usage: " << argv[0] << " <onnx_model_path> [--dump] [--mlir]\n";
        return 1;
    }

    // 解析可选flag
    for (int i = 2; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--dump") {
            enable_dump_proto = true;
        } else if (arg == "--mlir") {
            enable_print_mlir = true;
        } else {
            llvm::errs() << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    const char* onnx_path = argv[1];

    // 1. 加载 ONNX 模型文件
    std::unique_ptr<onnx::ModelProto> modelProto = tac::loadONNXModel(onnx_path);
    if (!modelProto) {
        llvm::errs() << "Load model failed, exit.\n";
        return 1;
    }

    // 2. 解析proto到ModelInfo
    tac::ModelInfo modelInfo = tac::parseModelProto(*modelProto);

    // 按需打印onnx proto dump
    if (enable_dump_proto) {
        tac::dumpPROTO(modelInfo);
    }

    // 3. 转MLIR
    mlir::MLIRContext context;
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<tac::TacDialect>();
    mlir::OwningOpRef<mlir::ModuleOp> module = tac::onnxModelToMlir(context, modelInfo);
    if (!module) {
        llvm::errs() << "Generate MLIR module failed!\n";
        return 1;
    }

    // 按需打印MLIR IR
    if (enable_print_mlir) {
        module->print(llvm::outs());
    }

    return 0;
}
