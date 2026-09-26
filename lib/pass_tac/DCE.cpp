#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "tac/TacOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#define GEN_PASS_DEF_DCE
#include "tac/TacPasses.h"
using namespace mlir;
using namespace llvm;
// using namespace tac;

namespace {
} // namespace

struct DCEPass : tac::impl::DCEBase<DCEPass>{
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::func::FuncDialect, mlir::scf::SCFDialect>();
  }

  // 反向传播遍历，标记活的op【不需要改动】
  void propagateLiveOpsBackward(llvm::DenseSet<Operation*> &liveOps,llvm::SmallVector<Operation*, 64> &worklist){
    while(!worklist.empty()){
      Operation* op = worklist.pop_back_val();
      for(Value operand:op->getOperands()){
        // 块参数不用处理
        if(isa<BlockArgument>(operand)){
          continue;
        }
        Operation *defOp = operand.getDefiningOp();
        if(!defOp){
          continue;
        }
        // 首次标记为存活继续遍历
        if(liveOps.insert(defOp).second){
          worklist.push_back(defOp);
        }
      }
    }
  }

  void deleteDeadOps(mlir::func::FuncOp &op_,llvm::DenseSet<Operation*> &liveOps){
    SmallVector<Operation*, 64> opsToErase;

    // walk 递归遍历所有region，收集所有待删除的死op
    op_.walk([&](Operation *op) {
        // 遇到FuncOp不递归进入，跳过
        if(isa<func::FuncOp>(op)) {
            return WalkResult::skip(); // skip：不遍历该op的子Region
        }
        // 只有不带子Region的普通op，才允许加入待删除列表
        if (op->getNumRegions() == 0) {
            if (!liveOps.count(op)) {
            opsToErase.push_back(op);
            }
        }
        return WalkResult::advance();
    });


    // 逆序删除，保证先删子op，再删父op
    for (Operation* op:reverse(opsToErase)){
      op->erase();
    }
    // 分析结果保留标记：DCE 不改变控制流结构，支配树仍然有效
    if (opsToErase.empty()) {
      markAllAnalysesPreserved();
    }
    // 有删除操作：什么都不写，默认所有分析失效
  }

  void runOnOperation() final{
    mlir::func::FuncOp func = getOperation(); // 可以隐式转换为指针
    llvm::DenseSet<Operation*> liveOps;     // 存活操作标记集合
    SmallVector<Operation*, 64> worklist; // 迭代工作队列，替代递归

    // walk递归遍历op下所有嵌套Region，遍历每个Block中的Operation
    // 用于初始化标记live terminator / side-effect ops
    // walk不光遍历 FuncOp，还遍历 scf.if、scf.for 等任意 Operation。
    func.walk([&](Operation *op) {
      bool mustLive = false;
      // 控制流的终止符必须保留
      // 包括 return、分支、跳转等所有块结尾指令
      // 有副作用的必须保留
      if(op->hasTrait<OpTrait::IsTerminator>()){
        mustLive = true;
      }else if(!isMemoryEffectFree(op)){
        mustLive = true;
      }
      if(mustLive){
          // 判断原理是否插入过，已经插入过就是 second 为 false，不会再加入工作列表，避免重复遍历死循环
          if(liveOps.insert(op).second){
            worklist.push_back(op);
          }
      }
      // walk自动递归深入这个op的子region（scf.if, scf.for都带region）
    });

    propagateLiveOpsBackward(liveOps,worklist);
    deleteDeadOps(func, liveOps);
  }
};

// **CRTP（奇异递归模板模式）**：把自身作为模板参数传给基类
std::unique_ptr<mlir::Pass> tac::createDCEPass() {
  return std::make_unique<DCEPass>();
}
