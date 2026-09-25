#include "tac/DriverUtils.h"

#include <fstream>
#include <random>
#include <algorithm>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

// ========== 命令行参数定义 ==========
cl::list<float> inputDataList(
    "input-data",
    cl::desc("Specify input data as a space-separated list of floats"),
    cl::ZeroOrMore,
    cl::CommaSeparated
);

cl::list<std::string> inputFiles(
    "input-files",
    cl::desc("Specify input binary files as a comma-separated list of paths"),
    cl::ZeroOrMore,
    cl::CommaSeparated
);

cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input onnx file>"),
    cl::Required,
    cl::value_desc("filename")
);

cl::opt<bool> runTiling(
    "tiling",
    cl::desc("Enable tiling optimization"),
    cl::init(false)
);

cl::opt<bool> runTranspose(
    "transpose",
    cl::desc("Enable B-transpose optimization"),
    cl::init(false)
);

cl::opt<bool> runOpFusion(
    "op-fusion",
    cl::desc("enable op fusion"),
    cl::init(false)
);

cl::opt<enum Action> emitAction(
    "emit",
    cl::desc("Select the kind of output desired"),
    cl::values(
        clEnumValN(DumpPROTO, "proto", "output the ONNX protobuf graph"),
        clEnumValN(DumpMLIR, "mlir", "output the generated MLIR module"),
        clEnumValN(DumpMLIRTensor, "mlir-tensor", "output the MLIR dump after tensor lowering"),
        clEnumValN(DumpMLIRMemRef, "mlir-memref", "output the MLIR dump after memref lowering"),
        clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output the MLIR-LLVM dump after llvm lowering"),
        clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump"),
        clEnumValN(RunJIT, "jit", "JIT the code and run it")
    ),
    cl::init(None)
);

// ========== 工具函数实现 ==========
std::vector<std::vector<float>> prepareInputData(
    const std::vector<std::vector<int64_t>>& inShapes,
    const llvm::cl::list<float>& cliData,
    const llvm::cl::list<std::string>& cliFiles
){
    // 计算模型期望的总元素数
    size_t totalExpected = 0;
    for (const auto& shape : inShapes) {
        int64_t prod = 1;
        for (int64_t d : shape) prod *= d;
        totalExpected += prod;
    }

    if (!cliData.empty() && cliData.size() != totalExpected) {
        llvm::errs() << "Warning: provided " << cliData.size()
                     << " elements, but the model expects " << totalExpected << ".\n";
    }

    // 状态与随机数初始化
    std::vector<std::vector<float>> inputDataStore;
    size_t fileIdx = 0;
    size_t cliOffset = 0;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // 逐个张量填充数据
    for (const auto& shape : inShapes) {
        int64_t numElements = 1;
        for (int64_t dim : shape) numElements *= dim;
        std::vector<float> data(numElements);

        // 从二进制文件读取
        if (fileIdx < cliFiles.size()) {
            std::ifstream ifs(cliFiles[fileIdx++], std::ios::binary);
            if (!ifs) {
                llvm::errs() << "Error: Could not open input file" << cliFiles[fileIdx-1] << "\n";
                exit(1);
            }
            ifs.read(reinterpret_cast<char*>(data.data()), numElements * sizeof(float));
        // 从命令行参数读取
        } else if (cliOffset < cliData.size()) {
            for (int64_t j = 0; j < numElements; ++j) {
                if (cliOffset < cliData.size()) {
                    data[j] = cliData[cliOffset++];
                } else {
                    data[j] = 0.0f;
                }
            }
        // 随机生成
        } else {
            for (int64_t j = 0; j < numElements; ++j) {
                data[j] = dist(gen);
            }
        }

        inputDataStore.push_back(std::move(data));
    }
    return inputDataStore;
}
