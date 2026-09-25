#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "tac/TacDialect.h"
#include "tac/TacOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <cstddef>
#include <cstdint>
#include <utility>
#define GEN_PASS_DEF_LOWERTOTENSOR
#include "tac/TacPasses.h"


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


using namespace mlir;
using namespace tac;
namespace {
//===----------------------------------------------------------------------===//
// Pattern: tac.constant -> arith.constant
//===----------------------------------------------------------------------===//
struct ConstantOpLowering : OpConversionPattern<tac::ConstantOp>{
    using OpConversionPattern<tac::ConstantOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tac::ConstantOp op,
            OpAdaptor adaptor,ConversionPatternRewriter &rewriter) const override{
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
            op, op.getValue());
        return success();
    }
};


//===----------------------------------------------------------------------===//
// Pattern: tac.add -> linalg.add
//===----------------------------------------------------------------------===//
struct AddOpLowering : OpConversionPattern<tac::AddOp>{
    using OpConversionPattern<tac::AddOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tac::AddOp op,OpAdaptor adaptor,
            ConversionPatternRewriter &rewriter) const override{
        // 语法糖 等价于 op.getResult().getType();
        auto resultType = 
            cast<RankedTensorType>(op.getType());
        // 结果预分配张量
        auto dest = tensor::EmptyOp::create(
            rewriter,
            op.getLoc(),
            resultType.getShape(),
            resultType.getElementType()
        );
        
        // linalg为Destination-Passing（目标传递）风格算子
        // 需要给预分配的张量目标
        rewriter.replaceOpWithNewOp<linalg::AddOp>(
            op,TypeRange{resultType},                       // Result Type
            ValueRange{adaptor.getLhs(),adaptor.getRhs()},  // ins
            Value{dest});                                   // outs
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pattern: tac.relu -> linalg.max
//===----------------------------------------------------------------------===//
struct ReluOpLowering : OpConversionPattern<tac::ReluOp>{
    using OpConversionPattern<tac::ReluOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tac::ReluOp op,OpAdaptor adaptor,
            ConversionPatternRewriter &rewriter)const override{
        auto loc = op.getLoc();
        auto type = cast<RankedTensorType>(op.getType());
        Value zero = arith::ConstantOp::create(
            rewriter,
            loc,
            rewriter.getZeroAttr(type.getElementType())
        );
        
        auto dest = tensor::EmptyOp::create(
            rewriter,
            loc,
            type.getShape(),
            type.getElementType()
        );
        
        rewriter.replaceOpWithNewOp<linalg::MaxOp>(op,
            op.getType(),                       // 输出类别
            ValueRange{adaptor.getInput(),zero},     // ins
            ValueRange{dest}                    // outs
        );
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pattern: tac.relu -> linalg.generic
//===----------------------------------------------------------------------===//
// struct ReluOpLowering : public OpConversionPattern<tac::ReluOp> {
//     using OpConversionPattern<tac::ReluOp>::OpConversionPattern;
//     LogicalResult matchAndRewrite(tac::ReluOp op,OpAdaptor adaptor,
//             ConversionPatternRewriter &rewriter)const override{
//         auto loc = op.getLoc();
//         auto type = cast<RankedTensorType>(op.getType());

//         // 索引映射（Indexing Map），定义「循环迭代维度」和「张量维度」
//         auto indexingMap = rewriter.getMultiDimIdentityMap(type.getRank());
//         // 先所有输入的映射，再所有输出的映射，主元素一一对应
//         llvm::SmallVector<AffineMap> maps(2,indexingMap);
//         llvm::SmallVector<utils::IteratorType> iterators(
//             type.getRank(),utils::IteratorType::parallel);

//         auto dest = rewriter.create<tensor::EmptyOp>(loc,
//             type.getShape(),type.getElementType());

//         rewriter.replaceOpWithNewOp<linalg::GenericOp>(op,
//             type,                   // 结果类型
//             adaptor.getInput(),     // 输入，单输入直接用
//             ValueRange{dest},       // 输出
//             maps,
//             iterators,
//             [](OpBuilder &nestedBuilder,Location nestedLoc,ValueRange args){
//                 Value input = args[0];
//                 Value zero = nestedBuilder.create<arith::ConstantOp>(
//                     nestedLoc,nestedBuilder.getFloatAttr(input.getType(),0.0));
//                 Value max = nestedBuilder.create<arith::MaxNumFOp>(
//                     nestedLoc,input,zero);
//                 nestedBuilder.create<linalg::YieldOp>(nestedLoc,max);
//             }
//         );
        
//         return success();
//     }
// };


//===----------------------------------------------------------------------===//
// Pattern: tac.matmul -> linalg.matmul/linalg.generic
//===----------------------------------------------------------------------===//
struct MatMulOpLowering : OpConversionPattern<tac::MatMulOp>{
    bool useTransposeB;
    MatMulOpLowering(MLIRContext *context, bool useTransposeB):
        OpConversionPattern<tac::MatMulOp>(context),useTransposeB(useTransposeB){}

    // 张量升维度
    Value rank1ToRank2(Value val,const Location loc,
        ConversionPatternRewriter &rewriter,size_t dim)const{
        assert((dim == 0 || dim == 1) && "dim must be 0 or 1");
        auto val_type = llvm::cast<RankedTensorType>(val.getType());
        if(val_type.getRank()==1){
            RankedTensorType newType;
            // 扩展第0维度
            if(dim==0){
                newType = RankedTensorType::get(
                    {1,val_type.getShape()[0]},
                    val_type.getElementType()
                );
            }else if(dim==1){
                newType = RankedTensorType::get(
                    {val_type.getShape()[0],1},
                    val_type.getElementType()
                );
            }

            // 原来第0维度扩展为01两维
            SmallVector<ReassociationIndices> reassoc = {{0, 1}};
            val = tensor::ExpandShapeOp::create(
                rewriter,
                loc,
                newType,
                val,
                reassoc
            );
        }
        return val;
    }

    // 填充对应形状的张量Value
    Value getFill(ConversionPatternRewriter &rewriter,
        const Location loc,RankedTensorType resultType)const{
        auto outs = tensor::EmptyOp::create(
            rewriter,
            loc,
            resultType.getShape(),
            resultType.getElementType()
        );

        auto zeroConstant = arith::ConstantOp::create(
            rewriter,
            loc,
            rewriter.getZeroAttr(resultType.getElementType())
        );

        auto fill = linalg::FillOp::create(
            rewriter,
            loc,
            ValueRange{zeroConstant.getResult()},
            ValueRange{outs.getResult()}
        );

        return fill.getResult(0);
    }

    LogicalResult matchAndRewrite(tac::MatMulOp op,OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter)const override{
        auto loc = op.getLoc();
        Value lhs = adaptor.getLhs();
        Value rhs = adaptor.getRhs();

        // 将Rank-1扩展到Rank-2,[K]->[1,K]
        lhs = rank1ToRank2(lhs,loc,rewriter,0);
        // 将Rank-1扩展到Rank-2,[K]->[K,1]
        rhs = rank1ToRank2(rhs,loc,rewriter,1);

        auto lhsType = llvm::cast<RankedTensorType>(lhs.getType());
        auto rhsType = llvm::cast<RankedTensorType>(rhs.getType());

        int64_t M = lhsType.getShape()[0];
        int64_t K = lhsType.getShape()[1];
        int64_t N = rhsType.getShape()[1];

        auto resultType = RankedTensorType::get(
            {M,N},
            lhsType.getElementType()
        );

        // 构造linalg.matmul所需要的0结果矩阵
        Value fill_results = getFill(rewriter,loc,resultType);
        Value matmul_results;

        if(useTransposeB){
            // 定义索引映射：A[M,K] * B[K,N] -> C[M,N]
            // Map 0 (A):       (m,k,n) -> (m,k)
            // Map 1 (B):       (m,k,n) -> (k,n)
            // Map 2 (C):       (m,k,n) -> (m,n)
            // 保证创建的仿射映射和当前 IR 图在同一个上下文中
            // results定义输出的每个索引，分别对应输入的哪个维度表达式
            auto mapA = AffineMap::get(3,0,
                {rewriter.getAffineDimExpr(0),rewriter.getAffineDimExpr(1)},
                rewriter.getContext()
            );
            // [K,N]
            auto mapB = AffineMap::get(3,0,
                {rewriter.getAffineDimExpr(1),rewriter.getAffineDimExpr(2)},
                rewriter.getContext()
            );
            auto mapC = AffineMap::get(3,0,
                {rewriter.getAffineDimExpr(0),rewriter.getAffineDimExpr(2)},
                rewriter.getContext()
            );
            SmallVector<AffineMap> maps = {mapA, mapB, mapC};

            // - `parallel`：并行维度，循环间无依赖，可并行执行。输出矩阵的行 `m`、列 `n` 每个点的计算互相独立，因此标记为并行。
            // - `reduction`：规约维度，循环内有累加依赖，不能直接并行。`k` 是矩阵乘的乘加累加维度，因此标记为规约
            SmallVector<utils::IteratorType>iterTypes = {
                utils::IteratorType::parallel,  // m
                utils::IteratorType::reduction, // k 
                utils::IteratorType::parallel,  // n 
            };

            SmallVector<Value> genericIputs = {lhs,rhs};

            auto genericOp = linalg::GenericOp::create(
                rewriter,
                loc,
                resultType,
                ValueRange{genericIputs},
                ValueRange{fill_results},
                maps,
                iterTypes,
                [](OpBuilder &b,Location l,ValueRange args){
                    Value mul = arith::MulFOp::create(b,l,args[0],args[1]);
                    Value add = arith::AddFOp::create(b,l,args[2],mul);
                    linalg::YieldOp::create(b,l,add);
                }
            );

            matmul_results = genericOp.getResult(0);
        }else{
            auto matmulop = linalg::MatmulOp::create(
                rewriter,
                loc,
                resultType,
                ValueRange{lhs,rhs},
                ValueRange{fill_results}
            );

            matmul_results = matmulop.getResult(0);
        }

        if(op.getType().getRank()==1){
            // Collapse [1,N]->[N],降维度
            SmallVector<ReassociationIndices> reassoc = {{0,1}};
            matmul_results = tensor::CollapseShapeOp::create(
                rewriter,
                loc,
                op.getType(),
                matmul_results,
                reassoc
            );
        }

        rewriter.replaceOp(op,matmul_results);
        return success();
    }
};


//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//
// CRPT奇异递归模版，把自身作为模版参数传给基类
struct LowerToTensorPass : 
    tac::impl::LowerToTensorBase<LowerToTensorPass> {
    // 使用父类构造函数
    using tac::impl::LowerToTensorBase<LowerToTensorPass>::LowerToTensorBase;
    
    // 设置依赖dialect
    void getDependentDialects(DialectRegistry &registry) const override{
        registry.insert<arith::ArithDialect,linalg::LinalgDialect,
                    tensor::TensorDialect,func::FuncDialect>();
    }

    void runOnOperation() final{
        ConversionTarget tgt(getContext());

        // 定义合法dialect
        tgt.addLegalDialect<arith::ArithDialect,linalg::LinalgDialect,
            tensor::TensorDialect,func::FuncDialect>();
        
        tgt.addIllegalDialect<TacDialect>();

        RewritePatternSet patterns(&getContext());
        patterns.add<
            ConstantOpLowering,
            AddOpLowering,
            ReluOpLowering
        >(&getContext());

        patterns.add<MatMulOpLowering>(&getContext(), useTranspose);
        if(failed(applyPartialConversion(
            getOperation(),
            tgt,
            std::move(patterns)
        ))){
            signalPassFailure();
        }
    }
};


} // tac namespace

std::unique_ptr<mlir::Pass> tac::createLowerToTensorPass(LowerToTensorOptions options) {
    return std::make_unique<LowerToTensorPass>(options);
}
