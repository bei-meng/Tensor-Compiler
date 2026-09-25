// Pass 基础设施
#include "mlir/Pass/Pass.h"
// 核心 IR 与内置算子
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
// 方言转换框架
#include "mlir/Transforms/DialectConversion.h"
// LLVM 转换通用组件
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// 各子转换的规则注入
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
// 涉及的方言定义
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 自定义命名空间
#include "tac/TacPasses.h"


using namespace mlir;

namespace {
// 声明这是一个以整个模块ModuleOp为单位执行的Pass
// 每次处理一个完整的 MLIR 模块
struct LowerToLLVMPass:mlir::PassWrapper<LowerToLLVMPass,
    OperationPass<ModuleOp>>{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerToLLVMPass)

    // 命令行调用的开关名，对应 --lower-to-llvm
    StringRef getArgument() const final { return "LowerToLLVM"; }
    // Pass 的可读名称，用于日志和调试
    StringRef getName() const final { return "LowerToLLVMPass"; }
    // Pass 功能描述，--help 时会展示
    StringRef getDescription() const final { 
        return "Lower SCF + Arith + MemRef + Func dialects to LLVM dialect"; 
    }
    
    void getDependentDialects(DialectRegistry &registry)const override{
        registry.insert<
            LLVM::LLVMDialect,
            scf::SCFDialect,
            cf::ControlFlowDialect,
            memref::MemRefDialect,
            func::FuncDialect
        >();
    }

    void runOnOperation() final {
        // 定义转换目标
        LLVMConversionTarget target(getContext());
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalOp<mlir::ModuleOp>();

        // 定义类型转换器
        LLVMTypeConverter typeConverter(&getContext());

        // 收集所有的patterns
        RewritePatternSet patterns(&getContext());

        // 标准方言的转换pattern
        // SCF --> CF
        populateSCFToControlFlowConversionPatterns(patterns);

        // Arith --> LLVM
        arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

        // MemRef --> LLVM
        populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

        // CF --> LLVM
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        
        // Func --> LLVM
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);

        auto module = getOperation();
        if(failed(applyFullConversion(
            module,
            target,
            std::move(patterns)
        ))){
            signalPassFailure();
        }
    }
};

}

std::unique_ptr<mlir::Pass> tac::createLowerToLLVMPass(){
    return std::make_unique<LowerToLLVMPass>();
}