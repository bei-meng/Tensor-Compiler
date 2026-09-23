#ifndef TAC_OPS_TD
#define TAC_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
// 继承方言
include "TacDialect.td"


// Tac 方言算子的基类。
// 该算子继承自 OpBase.td 中定义的基础 'Op' 类。
class TacOp<string mnemonic,list<Trait> traits = []>:
    Op<TacDialect,mnemonic,traits>;


//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
def ConstantOp : TacOp<"constant",[Pure]>{
    let summary = "constant";
    let description = [{
        常量算子将字面值转换为一个SSA值。数据作为属性挂载在算子上
        ```mlir
            %0 = tac.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                                : tensor<2x3xf32>
        ```
    }];
    let arguments = (ins ElementsAttr:$value);
    let results = (outs F32Tensor:$result);
    let hasVerifier = true;
    let builders = [
        // 先结果类型，然后按 ODS `arguments` 声明顺序，依次放操作数、属性
        OpBuilder<(ins "::mlir::DenseElementsAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>
    ];

    let assemblyFormat = "$value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//
def AddOp : TacOp<"add"> {
    let summary = "addition operation";
    let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
    let results = (outs F32Tensor);
    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
            build($_builder, $_state, lhs.getType(), lhs, rhs);
        }]>
    ];
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//
def ReluOp : TacOp<"relu", [Pure, SameOperandsAndResultType]> {
    let summary = "rectified linear unit activation";
    let description = [{
        Element-wise Rectified Linear Unit operation.
        f(x) = max(0, x)
    }];
    let arguments = (ins F32Tensor:$input);
    let results = (outs F32Tensor:$output);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//
def MatMulOp : TacOp<"matmul"> {
    let summary = "matrix multiplication";
    let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
    let results = (outs F32Tensor);
    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs)>
    ];
}


#endif


