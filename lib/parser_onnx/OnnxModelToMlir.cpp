#include "tac/OnnxModelToMlir.h"
#include "tac/TacDialect.h"
#include "tac/TacOps.h"
#include "tac/OnnxParser.h"



#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <cstddef>

// using namespace mlir;
// using namespace ::tac;
namespace tac {
    // 将ONNX解析得到的ENUM类型转换为MLIR的Type
    static mlir::Type getMlirType(mlir::OpBuilder &builder,
        TensorInfo::DataType type){
        switch (type) {
            case TensorInfo::DataType::FLOAT :return builder.getF32Type();
            case TensorInfo::DataType::INT32 :return builder.getI32Type();
            case TensorInfo::DataType::INT64 :return builder.getI64Type();
            default :return builder.getNoneType();
        }
    }

    // 获取位置(node的名字)
    static mlir::Location getLoc(const NodeInfo &node,mlir::OpBuilder &builder){
        std::string identifier = node.outputs.empty() ? node.op_type : node.outputs[0];
        return mlir::NameLoc::get(builder.getStringAttr(identifier));
    }


    /*
    ONNX 和 MLIR 引用张量的方式完全不同：
    - **ONNX 侧**：通过**字符串名字**引用张量。比如节点的 `inputs` 是 `["x", "stem.1.weight"]`，都是纯字符串，通过名字匹配找到对应的张量。
    - **MLIR 侧**：通过**Value 对象**引用张量。算子之间直接传递值句柄，不使用字符串名字。
    `valueMap` 就是两者之间的适配层，把 ONNX 的「按名引用」转换成 MLIR 的「按值引用」
    */
    // 将ONNX侧的字符串名字引用张量-->MLIR侧Value对象引用张量
    // 为了弥合ModelInfo基于名称的数据流与MLIR基于Value系统之间的差距，
    // 初始化了一个valueMap, 将每个输入名映射到其对应的块参数。


    // 这里只考虑了单输出的结果！！！！！！！！！
    static bool onnxNameToMlirValue(mlir::ImplicitLocOpBuilder &builder,
        ModelInfo &model,llvm::StringMap<mlir::Value> &valueMap){
        // 解析onnx的时候已经top有序
        for(const NodeInfo &node :model.graph.nodes){
            mlir::Location loc = getLoc(node,builder);
            builder.setLoc(loc);

            llvm::SmallVector<mlir::Value, 2> operands;
            for(const std::string &inName : node.inputs){
                if(valueMap.count(inName)){
                    operands.push_back(valueMap[inName]);
                }else if(model.graph.initializers.count(inName)){
                    const TensorInfo &tensor = model.graph.initializers[inName];
                    auto type = mlir::RankedTensorType::get(
                        tensor.shape,getMlirType(builder,tensor.elementType));

                    auto denseAttr = mlir::DenseElementsAttr::getFromRawBuffer(
                        type, llvm::ArrayRef<char>(tensor.rawData.data(), tensor.rawData.size())
                    );
                    
                    // 输入参数顺序：`结果类型` + `所有属性/操作数参数`
                    auto constOp = 
                        builder.create<tac::ConstantOp>(type,denseAttr);
                    
                    // 只考虑单输出
                    valueMap[inName] = constOp.getResult();
                    operands.push_back(constOp.getResult());
                }
            }

            if(node.op_type=="Add"){
                if (operands.size() < 2) {
                    llvm::errs() << "Error: Add op requires 2 operands\n";
                    return false;
                }
                // 返回的op会隐式转换为结果Value
                valueMap[node.outputs[0]] = builder.create<AddOp>(operands[0], operands[1]);
            } else if (node.op_type == "Relu") {
                if (operands.empty()) {
                    llvm::errs() << "Error: Relu op requires 1 operand\n";
                    return false;
                }
                valueMap[node.outputs[0]] = builder.create<ReluOp>(operands[0]);
            } else if (node.op_type == "MatMul") {
                if (operands.size() < 2) {
                    llvm::errs() << "Error: MatMul op requires 2 operands\n";
                    return false;
                }
                valueMap[node.outputs[0]] = builder.create<MatMulOp>(operands[0], operands[1]);
            }
        }
        return true;
    }

    mlir::OwningOpRef<mlir::ModuleOp> onnxModelToMlir(mlir::MLIRContext &context,
        ModelInfo &model){
        // 注册dialect
        context.getOrLoadDialect<TacDialect>();
        context.getOrLoadDialect<mlir::func::FuncDialect>();

        // builder
        mlir::OpBuilder builder(&context);

        llvm::SmallVector<mlir::Type, 4> argTypes;
        for(const auto &inputInfo :model.graph.inputs){
            auto elementType = getMlirType(builder, inputInfo.elementType);
            argTypes.push_back(mlir::RankedTensorType::get(inputInfo.shape,elementType));
        }

        // 创建module
        auto module = mlir::ModuleOp::create(
            ::mlir::UnknownLoc::get(&context));

        // 创建主函数
        auto func = mlir::func::FuncOp::create(
            builder.getUnknownLoc(),"main",
            builder.getFunctionType(argTypes,{})
        );

        // 解决 **MLIR 函数默认调用约定与标准 C ABI 不兼容**的问题，
        // 让外部 C/C++ 编写的推理运行时、宿主程序可以直接调用生成的模型函数
        
        func->setAttr(mlir::LLVM::LLVMDialect::getEmitCWrapperAttrName(),
            builder.getUnitAttr());

        mlir::Block *entry = func.addEntryBlock();
        builder.setInsertionPointToStart(entry);
        module.push_back(func);

        // 避免builder每次创建op都需要传位置
        mlir::ImplicitLocOpBuilder Ibuilder(builder.getUnknownLoc(),builder);

        // 用于映射ONNX基于字符串的数据流到MLIR基于Value的数据流
        llvm::StringMap<mlir::Value> valueMap;

        // 输入映射到块参数
        size_t n = model.graph.inputs.size();
        for (size_t i = 0; i < n; ++i) {
            const auto &inputInfo = model.graph.inputs[i];
            valueMap[inputInfo.name] = entry->getArgument(i);
        }

        // 处理nodes
        if(!onnxNameToMlirValue(Ibuilder,model,valueMap)){
            return nullptr;
        }

        llvm::SmallVector<mlir::Value, 4> returnValues;
        llvm::SmallVector<mlir::Type, 4> returnTypes;

        for(const auto &outputInfo :model.graph.outputs){
            if(valueMap.count(outputInfo.name)){
                mlir::Value val = valueMap[outputInfo.name];
                returnValues.push_back(val);
                returnTypes.push_back(val.getType());
            }else{
                llvm::errs() << "Error: Graph output '" << outputInfo.name << "' not found!\n";
                return nullptr;
            }
        }

        Ibuilder.create<mlir::func::ReturnOp>(returnValues);
        func.setType(builder.getFunctionType(argTypes, returnTypes));
        // 验证一下
        if (failed(mlir::verify(module))) {
            llvm::errs() << "MLIR verification failed\n";
            return nullptr;
        }
        return module;
    }
}// tac namespace

