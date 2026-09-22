#include "tac/OnnxParser.h"
#include "tac/OnnxModelInfo.h"
#include "tac/OnnxDumping.h"
#include "llvm/Support/raw_ostream.h" // llvm::errs()
#include <memory>
#include "onnx/onnx-ml.pb.h"

int main(int argc, char** argv) {
    // 参数校验
    if (argc != 2) {
        llvm::errs() << "Usage: " << argv[0] << " <onnx_model_path>\n";
        return 1;
    }

    // 1. 加载 ONNX 模型文件
    std::unique_ptr<onnx::ModelProto> model = tac::loadONNXModel(argv[1]);
    if (!model) {
        llvm::errs() << "Load model failed, exit.\n";
        return 1;
    }

    // 2. 解析并打印完整模型结构
    // dumpPROTO 内部会自动调用 parseModelProto 转成自定义结构再输出
    tac::dumpPROTO(*model);

    return 0;
}
