#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h" 


#include "tac/TacDialect.h"
#include "tac/TacOps.h"
// 对应具体实现
#define GET_DIALECT_DEFS
#include "TacDialect.cpp.inc"
#define GET_OP_CLASSES
#include "TacOps.cpp.inc"

using namespace tac;
using namespace mlir;


void TacDialect::initialize(){
    addOperations<
        #define GET_OP_LIST
        #include "TacOps.cpp.inc"
    >();
}


//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
    // 获取属性和结果类型
    auto tensorAttr = 
        llvm::dyn_cast<DenseElementsAttr>(getValue());
    if (!tensorAttr) {
        return emitOpError("requires a dense elements attribute");
    }
    auto tensorType = 
        llvm::cast<RankedTensorType>(getResult().getType());
    // 检查元素数量是否匹配
    if (tensorAttr.getNumElements() != tensorType.getNumElements()) {
        return emitOpError() << "number of elements in 'value' attribute ("
                                << tensorAttr.getNumElements()
                                << ") does not match the result type ("
                                << tensorType.getNumElements() << ")";
    }
    return success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify(){
    // 获取操作数和结果的类型,有秩张量类型
    auto lhsType = 
        llvm::cast<RankedTensorType>(getLhs().getType());
    auto rhsType = 
        llvm::cast<RankedTensorType>(getRhs().getType());
    auto resType = 
        llvm::cast<RankedTensorType>(getResult().getType());
    
    // 检查lhs和rhs的形状是否匹配
    if(lhsType.getShape() != rhsType.getShape()){
        return emitOpError() << "requires LHS and RHS to have the same shape, but got "
                        << lhsType << " and " << rhsType;
    }

    // 检查输出结果形状是否匹配
    if (resType.getShape() != lhsType.getShape()) {
        return emitOpError() << "requires result shape to match input shape";
    }

    return success();
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

void MatMulOp::build(OpBuilder &builder,OperationState &state,
    Value lhs,Value rhs){
    auto lhsType = 
        llvm::cast<RankedTensorType>(lhs.getType());
    auto rhsType = 
        llvm::cast<RankedTensorType>(rhs.getType());
    auto elemType = lhsType.getElementType();


    unsigned lhsRank = lhsType.getRank();
    unsigned rhsRank = rhsType.getRank();
    
    // 只支持 1 维向量、2 维矩阵
    assert(lhsRank >= 1 && lhsRank <= 2 && rhsRank >= 1 && rhsRank <= 2 
            && "only supports 1D vector and 2D matrix inputs");

    // 矩阵乘法形状变化: (M x K) * (K x N) -> (M x N)
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();

    bool lhsIsVector = lhsRank == 1;
    bool rhsIsVector = rhsRank == 1;

    int64_t lhsK = lhsShape.back();
    int64_t rhsK = rhsShape.front();

    assert(lhsK == rhsK && "matmul inner dimensions must match");

    RankedTensorType resultType;
    if(lhsIsVector && !rhsIsVector){
        // [K] @ [K, N] -> [N]
        resultType = RankedTensorType::get(
            {rhsShape[1]},elemType
        );
    }else if(!lhsIsVector && rhsIsVector){
        // [M, K] @ [K] -> [M]
        resultType = RankedTensorType::get(
            {lhsShape[0]},elemType
        );
    } else if (!lhsIsVector && !rhsIsVector) {
        // [M, K] @ [K, N] -> [M, N]
        resultType = RankedTensorType::get({lhsShape[0], rhsShape[1]}, elemType);
    } else {
        // [K] @ [K] -> scalar
        resultType = RankedTensorType::get({}, elemType);
    }

    // 将计算结果的类型和输入操作数放入states
    state.addTypes(resultType);
    state.addOperands({lhs,rhs});
}


LogicalResult MatMulOp::verify(){
    auto lhsShape = llvm::cast<RankedTensorType>(getLhs().getType()).getShape();
    auto rhsShape = llvm::cast<RankedTensorType>(getRhs().getType()).getShape();
    auto resShape = llvm::cast<RankedTensorType>(getResult().getType()).getShape();

    // Input Rank check
    if (lhsShape.size() < 1 || lhsShape.size() > 2 ||
        rhsShape.size() < 1 || rhsShape.size() > 2) {
        return emitOpError("LHS and RHS must be either 1D or 2D tensors");
    }

    // Inner Dimension (K) check
    int64_t lhsK = lhsShape.back();
    int64_t rhsK = rhsShape[0];
    if (lhsK != rhsK) {
        return emitOpError("inner dimensions (K) must match: ")
                << lhsK << " (LHS) != " << rhsK << " (RHS)";
    }


    int64_t lhsRank = lhsShape.size();
    int64_t rhsRank = rhsShape.size();
    int64_t resRank = resShape.size();
    // vector @ matrix -> vector [N]
    if (lhsRank == 1 && rhsRank == 2) {
        if (resRank != 1 || resShape[0] != rhsShape[1]) {
            return emitOpError("expected result shape [N] for vector x matrix");
        }
    }
    // matrix @ vector -> vector [M]
    else if (lhsRank == 2 && rhsRank == 1) {
        if (resRank != 1 || resShape[0] != lhsShape[0]) {
            return emitOpError("expected result shape [M] for matrix x vector");
        }
    }
    // matrix @ matrix -> matrix [M, N]
    else if (lhsRank == 2 && rhsRank == 2) {
        if (resRank != 2 ||
            resShape[0] != lhsShape[0] ||
            resShape[1] != rhsShape[1]) {
                return emitOpError("expected result shape [M, N] for matrix x matrix");
            }
    }
    else {
        return emitOpError("vector x vector matmul not supported");
    }
    return success();
}