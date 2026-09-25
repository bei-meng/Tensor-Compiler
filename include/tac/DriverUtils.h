#ifndef CLI_UTILS_H
#define CLI_UTILS_H

#include "llvm/Support/CommandLine.h"
#include <vector>
#include <string>
#include <cstdint>

namespace cl = llvm::cl;

// 输出动作枚举（从原匿名命名空间移到全局，供主文件使用）
enum Action { 
    None,
    DumpPROTO,
    DumpMLIR,
    DumpMLIRTensor,
    DumpMLIRMemRef,
    DumpMLIRLLVM,
    DumpLLVMIR,
    RunJIT
};

// ========== 命令行参数声明（extern，定义在 cpp 中） ==========
extern cl::list<float> inputDataList;
extern cl::list<std::string> inputFiles;
extern cl::opt<std::string> inputFilename;
extern cl::opt<bool> runTiling;
extern cl::opt<bool> runTranspose;
extern cl::opt<bool> runOpFusion;
extern cl::opt<enum Action> emitAction;

// ========== 工具函数声明 ==========
std::vector<std::vector<float>> prepareInputData(
    const std::vector<std::vector<int64_t>>& inShapes,
    const llvm::cl::list<float>& cliData,
    const llvm::cl::list<std::string>& cliFiles);

#endif // CLI_UTILS_H
