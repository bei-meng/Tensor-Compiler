#include "tac/TacDialect.h"
#include "tac/TacOps.h"
#include <cstdint>

#define GEN_PASS_DEF_LINALGTILING
#include "tac/TacPasses.h"

// ===== Pass 基础设施 =====
#include "mlir/Pass/Pass.h"

// ===== IR 与重写器 =====
#include "mlir/IR/PatternMatch.h"

// ===== 接口与方言定义 =====
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

// ===== 分块实现 =====
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"

#include <cstdio>
#include <tuple>
// #include <string>

using namespace mlir;
namespace {

static std::tuple<int64_t, int64_t, int64_t> parseTileSize(const std::string &str) {
    int m, n, k;
    // 解析失败直接用默认值
    if (sscanf(str.data(), "%d,%d,%d", &m, &n, &k) != 3)
        return {64, 64, 64};
    return {m, n, k};
}

struct LinalgTilingPass: tac::impl::LinalgTilingBase<LinalgTilingPass> {
    // 使用父类构造函数
    using tac::impl::LinalgTilingBase<LinalgTilingPass>::LinalgTilingBase;

    void getDependentDialects(DialectRegistry &registry) const override {
        // Register dialects so the pass knows how to create loops/slices
        registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                        tensor::TensorDialect, arith::ArithDialect>();

        linalg::registerTilingInterfaceExternalModels(registry);
    }
    void runOnOperation() final{
        auto module = getOperation();
        IRRewriter rewriter(&getContext());

        // 目标乘法
        SmallVector<TilingInterface> targets;
        module->walk([&](TilingInterface op){
            if(auto genericOp = dyn_cast<linalg::GenericOp>(op.getOperation())){
                if(!genericOp)return;
                if(genericOp.getNumLoops() == 3 && genericOp.getNumReductionLoops() == 1){
                    targets.push_back(op);
                }
            }
        });

        auto [M, N, K] = parseTileSize(tileSize);

        for(auto target:targets){
            scf::SCFTilingOptions options;
            options.setTileSizes({
                rewriter.getIndexAttr(M),
                rewriter.getIndexAttr(N),
                rewriter.getIndexAttr(K)
            });

            rewriter.setInsertionPoint(target);

            auto result = scf::tileUsingSCF(
                rewriter,
                target,
                options
            );

            if(failed(result))continue;

            rewriter.replaceOp(
                target.getOperation(),
                result->replacements
            );
        }
    }
};
}


std::unique_ptr<mlir::Pass> tac::createLinalgTilingPass(LinalgTilingOptions options){
    return std::make_unique<LinalgTilingPass>(options);
}