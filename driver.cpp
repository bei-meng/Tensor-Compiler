// ========== 项目自定义 TAC 模块 ==========
#include "tac/DriverUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tac/OnnxModelInfo.h"
#include "tac/OnnxDumping.h"
#include "tac/OnnxParser.h"
#include "tac/OnnxModelToMlir.h"
#include "tac/TacDialect.h"
#include "tac/TacPasses.h"
#include "tac/DriverUtils.h"
#include "onnx/onnx-ml.pb.h"

// ========== LLVM 基础工具 ==========
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Support/raw_ostream.h"

// ========== MLIR 基础设施 ==========
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassRegistry.h"

// ========== 方言定义 ==========
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// ========== Linalg/SCF 官方 Pass ==========
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

// ========== 缓冲化（Bufferization）完整依赖 ==========
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
// 各方言缓冲化接口实现
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
// ValueBounds 接口实现（缓冲化分析依赖）
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"

// ========== 后端降级与转换 ==========
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

// ========== LLVM IR 导出与 JIT 执行 ==========
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"

// ========== C++ 标准库 ==========
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <ostream>
#include <chrono>
#include <random>
#include <fstream>
#include <vector>



namespace{

    static int processMLIR(mlir::MLIRContext *context_ptr, mlir::ModuleOp module) {
        mlir::PassManager pm(context_ptr);
        if(emitAction >= DumpMLIRTensor){
            tac::LowerToTensorOptions opt;
            opt.useTranspose = runTranspose;  
            pm.addPass(tac::createLowerToTensorPass(opt));
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
        }

        if(emitAction >= DumpMLIRMemRef){
            // Lower Tensor/linalg -> MemRef
            // 缓冲化前置处理,赋予内存分配语义
            pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
            mlir::bufferization::OneShotBufferizePassOptions passOptions;

            // 禁止循环内部分配的内存作为返回值逃逸到循环外，避免生命周期不匹配导致的内存泄漏或越界。
            passOptions.allowReturnAllocsFromLoops = false;
            // 连函数边界一起缓冲化 —— 函数的参数、返回值也从 Tensor 类型转成 MemRef 类型，
            // 整个函数接口完全变成内存语义，而不是只处理函数内部
            passOptions.bufferizeFunctionBoundaries = true;
            // 一次性把 IR 中所有值语义的 Tensor 算子，完整转换成内存语义的 MemRef 算子
            pm.addPass(mlir::bufferization::createOneShotBufferizePass(passOptions));
            // 内存分配简化,合并连续分配、消除不必要的 “释放 - 再分配”、优化生命周期重叠的内存，降低运行时内存开销
            pm.addPass(mlir::bufferization::createBufferDeallocationSimplificationPass());

            pm.addPass(mlir::createCanonicalizerPass()); 
            pm.addPass(mlir::createCSEPass());

            // Linalg 算子转循环Affine.for
            // 把 Linalg 层级的高层计算算子（比如 `linalg.matmul`、`linalg.conv2d`）
            // 降级成**嵌套循环 + 逐元素内存读写**的形式。
            // 这是从 “声明式计算算子” 到 “命令式控制流”
            pm.addPass(mlir::createConvertLinalgToLoopsPass());
            pm.addPass(mlir::createCanonicalizerPass());
            pm.addPass(mlir::createCSEPass());
        }

        if(emitAction >= DumpMLIRLLVM || emitAction == DumpLLVMIR){
            // MemRef 元数据展开
            pm.addPass(mlir::memref::createExpandReallocPass());
            // Affine 循环降级
            pm.addPass(mlir::createLowerAffinePass());
            // LLVM 降级
            pm.addPass(tac::createLowerToLLVMPass());
            // 无效转换清理
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        }

        if (mlir::failed(pm.run(module))) {
            llvm::errs() << "Lowering pipeline failed\n";
            return 1;
        }

        // Print MLIR module    
        if (emitAction <= DumpMLIRLLVM) {
            mlir::OpPrintingFlags flags;
            flags.enableDebugInfo();
            module->print(llvm::outs(), flags);
            llvm::outs() << "\n";
        }
        return 0;
    }

    // 把 MLIR 体系内的 LLVM 方言（MLIR LLVM Dialect）
    // 转换成 LLVM 原生的中间表示（LLVM IR）
    static int dumpLLVMIR(mlir::ModuleOp module) {
        // 内置类型翻译 + LLVM 方言算子翻译
        mlir::registerBuiltinDialectTranslation(*module->getContext());
        mlir::registerLLVMDialectTranslation(*module->getContext());

        // 将MLIR的module转换为LLVM IR的module
        llvm::LLVMContext llvmContext;
        auto llvmModule = 
            mlir::translateModuleToLLVMIR(module, llvmContext);
        
        if (!llvmModule) {
            llvm::errs() << "Failed to emit LLVM IR\n";
            return -1;
        }

        // 初始化目标环境：绑定硬件架构（指令集、数据布局）
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        // 配置目标信息：写入架构与数据布局
        // 自动检测当前主机的 CPU 架构、操作系统、指令集，生成目标三元组
        auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (!tmBuilderOrError) {
            llvm::errs() << "Could not create JITTargetMachineBuilder\n";
            return -1;
        }
        auto tmOrError = tmBuilderOrError->createTargetMachine();
        if (!tmOrError) {
            llvm::errs() << "Could not create TargetMachine\n";
            return -1;
        }
        mlir::ExecutionEngine::setupTargetTripleAndDataLayout(
            llvmModule.get(), tmOrError.get().get()
        );

        llvm::outs() << *llvmModule << "\n";
        return 0;
    }

    /*
        MLIR 的 memref（内存引用）不是只传一个数据指针。
        它是带元数据的内存视图，包含数据地址、对齐、偏移、形状、步长。
        底层调用时必须按固定格式打包成「描述符数组」，
        这是 MLIR 的标准二进制接口约定。
        0           allocated pointer   整块内存的起始地址（释放内存时用）
        1           aligned pointer     对齐后的实际访问起始地址（CPU 访问对齐内存更快）
        2           offset              从对齐地址到第一个有效元素的偏移（元素个数）
        3-3+rank-1  sizes               每个维度的大小
        3+rank ~ 3+2*rank-1     strides 每个维度的步长（跳多少元素到下一维）
    */
    void prepareInputMemRef(
        std::vector<std::vector<int64_t>> &inputDescriptors,    // MemRef描述器
        const std::vector<int64_t>& inRanks,                    // 输入的rank
        const std::vector<std::vector<int64_t>>& inShapes,      // 输入形状
        std::vector<std::vector<float>>& inputDataStore){       // 输入数据

        inputDescriptors.reserve(inRanks.size());
        // 动态处理输入数据的形状，i表示第i个输入数据
        for(size_t i=0;i<inRanks.size();++i){
            int64_t rank = inRanks[i];
            const auto& shape = inShapes[i];

            // 构建MemRef描述符
            size_t slots = 3 + (2*rank);        // 数据的起始地址
            inputDescriptors.emplace_back(slots,0);
            auto &desc = inputDescriptors.back();

            // allocated pointer字段
            desc[0] = reinterpret_cast<int64_t>(inputDataStore[i].data());
            // aligned pointer字段
            desc[1] = desc[0];
            // offset字段
            desc[2] = 0;

            // 填充sizes和strides字段
            for(int64_t j=0;j<rank;++j){
                // sizes字段
                desc[3+j] = shape[j];
                // 计算strides
                int64_t stride = 1;
                for(int64_t k=j+1;k<rank;++k){
                    stride *= shape[k];
                }
                // strides字段
                desc[3+rank+j] = stride;
            }
        }
    }

    void prepareOutputMemRef(std::vector<int64_t> &outDesc,
            int64_t outRank,const std::vector<int64_t>& outShape){
        int64_t totalOutElements = 1;
        for (int64_t dim : outShape) totalOutElements *= dim;
        float* outData = (float*)malloc(totalOutElements * sizeof(float));
        
        // 准备输出的MemRef描述符
        size_t outSlots = 3 + (2 * outRank);
        outDesc.assign(outSlots, 0);

        outDesc[0] = reinterpret_cast<int64_t>(outData);    // Allocated
        outDesc[1] = outDesc[0];                            // Aligned
        outDesc[2] = 0;

        for (int64_t i = 0; i < outRank; ++i) {
            outDesc[3 + i] = outShape[i];

            int64_t stride = 1;
            for (int64_t k = i + 1; k < outRank; ++k) {
                stride *= outShape[k];
            }
            outDesc[3 + outRank + i] = stride;
        }
    }

    void printResult(std::vector<int64_t> &outDesc,int64_t outRank,
                        int64_t totalOutElements){
        int64_t offset = outDesc[2];
        float *aligned = reinterpret_cast<float *>(outDesc[1]);

        // Helper to print nested dimensions
        auto printRef = [&](auto self, int64_t dim, int64_t currentOffset) -> void {
            if (dim == outRank) {
                llvm::outs() << llvm::format("%.7f", aligned[currentOffset]);
                return;
            }

            llvm::outs() << "[";
            int64_t size = outDesc[3 + dim];
            int64_t stride = outDesc[3 + outRank + dim];

            for (int64_t i = 0; i < size; ++i) {
                self(self, dim + 1, currentOffset + i * stride);
                if (i < size - 1) llvm::outs() << ", ";
            }
            llvm::outs() << "]";
        };

        llvm::outs() << "Result Shape: [ ";
        for (int64_t i = 0; i < outRank; ++i) {
            int64_t s = outDesc[3 + i];
            llvm::outs() << s << " ";
        }
        llvm::outs() << "]\n";

        // Only print data if less than 100 elements
        if (totalOutElements <= 100) {
            llvm::outs() << "Data: ";
            printRef(printRef, 0, offset);
            llvm::outs() << "\n";
        } else {
            llvm::outs() << "Data: [Large Tensor - Printing skipped for performance]\n";
            llvm::outs() << "First Element: " << aligned[offset] << "\n";
        }
    }

    static int RunFunc(mlir::ModuleOp module,
                    int64_t outRank,
                    const std::vector<int64_t>& inRanks,
                    const std::vector<int64_t>& outShape,
                    const std::vector<std::vector<int64_t>>& inShapes,
                    std::vector<std::vector<float>>& inputDataStore) {
        
        // 注册硬件架构后端,加载对应的指令集生成器、汇编器
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        // 注册方言翻译规则,MLIR的Tensor、MemRef、SCF 等各种高层方言转成底层的 LLVM IR
        mlir::registerBuiltinDialectTranslation(*module->getContext());
        mlir::registerLLVMDialectTranslation(*module->getContext());

        // 创建一个MLIR执行引擎
        // 设置优化级别3，让LLVM进行优化
        auto optPipeline = mlir::makeOptimizingTransformer(
            3,0, nullptr);
        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.transformer = optPipeline;
        // 代码生成阶段也开激进优化
        engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
        engineOptions.sharedLibPaths = {
            "/home/ubuntu/mlir-onnx/llvm-project/build/lib/libmlir_runner_utils.so",
            "/home/ubuntu/mlir-onnx/llvm-project/build/lib/libmlir_c_runner_utils.so"
        };

        // 创建执行引擎，这一步就会把 MLIR 编译成机器码
        auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
        if (!maybeEngine) {
            llvm::errs() << "Failed to construct execution engine\n";
            return -1;
        }
        auto &engine = maybeEngine.get();

        // 准备输入：构建 MemRef 描述符
        std::vector<std::vector<int64_t>> inputDescriptors;
        prepareInputMemRef(inputDescriptors, inRanks, inShapes, inputDataStore);
        
        // 准备输出：分配内存 + 构建 MemRef 描述符
        std::vector<int64_t> outDesc;
        prepareOutputMemRef(outDesc, outRank, outShape);

        /*
            1. **MemRef 的底层调用约定**：MemRef 是按「指针传递」的，目标函数的每个形参本身就是 `MemRefDescriptor*`（描述符指针），函数通过这个指针读取描述符信息。
            2. **invokePacked 的通用规则**：它是一个通用的打包调用接口，不管形参是什么类型，统一要求 `args` 里的每个元素都是「指向实参的指针」—— 相当于所有参数都按引用语义传递，函数拿到后解引用一次就能取到实参值。
            packed 约定要求每个参数都必须传「指向实参的指针」
            这里的实参为MemRef的指针，指向实参的指针就是，指向MemRef指针的指针
        */
        // 打包参数数组，结果在前，后面为输入
        std::vector<void*> args;
        void* outDescPtr = outDesc.data();
        args.push_back(&outDescPtr);

        std::vector<int64_t*> inputPtrs;
        for (auto& desc : inputDescriptors) {
            inputPtrs.push_back(desc.data());
        }

        for (size_t i = 0; i < inputPtrs.size(); ++i) {
            args.push_back(&inputPtrs[i]);
        }

        // MLIR 自动生成的**C 兼容包装函数**。`ciface` = C Interface，MLIR 
        // 原生函数的调用约定和 C++ 不兼容，所以会生成一层包装，统一用打包参数的方式调用

        // 执行
        auto start_exec = std::chrono::high_resolution_clock::now();
        
        if (engine->invokePacked("_mlir_ciface_main", args)) {
            llvm::errs() << "JIT execution failed\n";
            return -1;
        }

        auto end_exec = std::chrono::high_resolution_clock::now();
        
        // 读结果并输出
        auto start_io = std::chrono::high_resolution_clock::now();

        int64_t totalOutElements = 1;
        for (int64_t dim : outShape) totalOutElements *= dim;
        float *allocated = reinterpret_cast<float *>(outDesc[0]);
        float *aligned = reinterpret_cast<float *>(outDesc[1]);
        std::ofstream ofs("output.bin", std::ios::binary);
        if (ofs) {
            ofs.write(reinterpret_cast<char*>(aligned), totalOutElements * sizeof(float));
            ofs.close();
            llvm::outs() << "Result saved to output.bin\n";
        }

        auto end_io = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> exec_dur = end_exec - start_exec;
        std::chrono::duration<double, std::milli> io_dur = end_io - start_io;

        llvm::outs() << "实际计算时间: " << exec_dur.count() << " ms\n";
        llvm::outs() << "写磁盘时间: " << io_dur.count() << " ms\n";

        printResult(outDesc,outRank,totalOutElements);

        llvm::outs().flush();
        if (allocated) free(allocated);
        return 0;
    }

    static int runJit(mlir::MLIRContext *context_ptr,mlir::ModuleOp module){
        auto mainFunc = 
            module.lookupSymbol<mlir::func::FuncOp>("main");
        if (!mainFunc) {
            llvm::errs() << "Error: main function not found after mlirGen\n";
            return 1;
        }

        // 获取输入的维度和形状
        std::vector<int64_t> inRanks;
        std::vector<std::vector<int64_t>> inShapes;
        for(auto type:mainFunc.getArgumentTypes()){
            auto tensorType = llvm::cast<mlir::RankedTensorType>(type);
            inRanks.push_back(tensorType.getRank());
            inShapes.push_back(tensorType.getShape().vec());
        }

        // 获取输出的维度和形状
        auto outTensorType = llvm::cast<mlir::RankedTensorType>(
            mainFunc.getResultTypes()[0]);
        int64_t outRank = outTensorType.getRank();
        std::vector<int64_t> outShape = outTensorType.getShape().vec();

        // 准备输入数据
        auto inputDataStore = 
            prepareInputData(inShapes, inputDataList, inputFiles);
    
            
        // 降级模型到LLVM IR
        if (processMLIR(context_ptr, module)) return 1;
        return RunFunc(module, outRank, inRanks, outShape, inShapes, inputDataStore);
    }
}
int main(int argc,char **argv){
    // 注册所有命令行参数
    mlir::registerAsmPrinterCLOptions();        // 注册 MLIR IR 汇编打印器的命令行选项
    mlir::registerMLIRContextCLOptions();       // 注册 MLIR 全局上下文的命令行选项
    mlir::registerPassManagerCLOptions();       // 注册 Pass 管理器的命令行选项

    llvm::cl::ParseCommandLineOptions(
        argc, argv, "deep learning compiler\n");

    // 方言注册器
    mlir::DialectRegistry registry;

    // 注册所需方言
    registry.insert<
        mlir::arith::ArithDialect,
        mlir::tensor::TensorDialect,
        mlir::linalg::LinalgDialect,
        mlir::memref::MemRefDialect,
        mlir::func::FuncDialect,
        mlir::scf::SCFDialect,
        mlir::cf::ControlFlowDialect,
        tac::TacDialect
    >();

    // BufferizableOpInterface 注册
    // 定义了「一个算子怎么从 `tensor` 值语义转换成 `memref` 内存语义」。
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

    // ValueBoundsOpInterface 注册  
    // 静态边界推导的接口：给定一个算子，能推导出它的输出值 / 维度的上下界
    mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
    mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);


    mlir::MLIRContext context(registry);
    context.loadDialect<
        tac::TacDialect,
        mlir::cf::ControlFlowDialect,
        mlir::arith::ArithDialect,
        mlir::func::FuncDialect,
        mlir::scf::SCFDialect
    >();

    // 模型文件读取与解析
    tac::ModelInfo modelInfo;
    mlir::OwningOpRef<mlir::ModuleOp> mlirModule;
    llvm::StringRef input(inputFilename);

    if(input.ends_with(".onnx")){
        std::unique_ptr<onnx::ModelProto> model_ptr = tac::loadONNXModel(inputFilename);
        if(model_ptr){
            modelInfo = tac::parseModelProto(*model_ptr);
            mlirModule = tac::onnxModelToMlir(context,modelInfo); 
        }
    }else{
        mlirModule = mlir::parseSourceFile<mlir::ModuleOp>(
            inputFilename,
            &context
        );
    }


    if(!mlirModule){
        llvm::errs() << "输入文件加载/解析失败: " << inputFilename << "\n";
        return 1;
    }

    // 执行对应pass和jit
    switch(emitAction){
        case DumpPROTO:
            tac::dumpPROTO(modelInfo);
            return 0;
        case DumpMLIR:
        case DumpMLIRTensor:
        case DumpMLIRMemRef:
        case DumpMLIRLLVM:
            return processMLIR(&context,*mlirModule);
        case DumpLLVMIR:
            if (processMLIR(&context, *mlirModule)) return 1;
            return dumpLLVMIR(*mlirModule);
        case RunJIT:
            return runJit(&context,*mlirModule);
        default:
            llvm::errs()
                << "No action specified, use -emit=proto\n";
            return 1;
    }
    return 0;
}
