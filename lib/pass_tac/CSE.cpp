
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/RecyclingAllocator.h"
#include <cassert>
#include <deque>
#include <memory>
#include <utility>
#include <vector>
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"


#define GEN_PASS_DEF_CSE
#include "tac/TacPasses.h"
using namespace mlir;
using namespace tac;

namespace {
/// 操作等价性的哈希与比较规则（供 ScopedHashTable 使用）
struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {
    static unsigned getHashValue(const Operation *opC) {
        return OperationEquivalence::computeHash(
            const_cast<Operation *>(opC),
            /*hashOperands=*/OperationEquivalence::directHashValue,
            /*hashResults=*/OperationEquivalence::ignoreHashValue,
            OperationEquivalence::IgnoreLocations);
    }
    static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
        auto *lhs = const_cast<Operation *>(lhsC);
        auto *rhs = const_cast<Operation *>(rhsC);
        if (lhs == rhs){
            return true;
        }
        if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
            rhs == getTombstoneKey() || rhs == getEmptyKey()){
            return false;
        }
        return OperationEquivalence::isEquivalentTo(
            const_cast<Operation *>(lhsC), 
            const_cast<Operation *>(rhsC),
            OperationEquivalence::IgnoreLocations
        );
    }
};
} // namespace

namespace{
class TacCSEDriver{
public:
    TacCSEDriver(RewriterBase &rewriter,DominanceInfo *domInfo)
        :rewriter(rewriter),domInfo(domInfo){}

    // 简化所给op中的operations
    void simplify(Operation *op,bool *changed = nullptr);
private:
    /*
    这是一个**专门给哈希表节点优化的高性能内存分配器**，是两个组件的组合：
    - `llvm::BumpPtrAllocator`：**线性分配器**，一次向系统申请一大块内存，然后像切蛋糕一样一点点分配给小对象。速度极快，但不能单独释放某个对象，只能整体销毁。
    - `llvm::RecyclingAllocator`：**回收器包装**，在 Bump 分配器基础上加了节点回收能力。销毁作用域时不直接释放内存，而是把空节点缓存起来，下次创建作用域时直接复用，避免频繁申请 / 释放内存。
    - `llvm::ScopedHashTableVal<Operation*, Operation*>`：哈希表节点的值类型，固定为「键是操作指针，值也是操作指针」
    */
    // 给哈希表提速的内存分配器
    using AllocatorTy = llvm::RecyclingAllocator<
        llvm::BumpPtrAllocator,
        llvm::ScopedHashTableVal<Operation *, Operation *>>;

    // 存储「已处理的等价操作」的带作用域哈希表。
    // 分层记录已计算表达式的核心哈希表
    using ScopedMapTy = llvm::ScopedHashTable<
        Operation *, Operation *,
        SimpleOperationInfo, AllocatorTy>;

    /*
    用于缓存两个操作之间的内存副作用信息。第一个操作作为键存储，
    第二个操作存在值的 pair 中；pair 同时保存两个操作之间的内存副作用。
    如果 `MemoryEffects` 是空指针，我们就认为两个操作之间没有写入类型的内存副作用。
    */
    // 这是一个**性能优化用的缓存表**，专门解决「两个相同的读操作能不能合并」的问题
    // 加速内存副作用检查的缓存
    using MemEffectsCache =DenseMap<Operation *, 
        std::pair<Operation *, MemoryEffects::Effect *>>;


    // 迭代遍历支配树的栈帧，防止栈溢出
    struct CFGStackNode {
        CFGStackNode(ScopedMapTy &knownValues, DominanceInfoNode *node)
            : scope(knownValues), node(node), childIterator(node->begin()) {}

        // 已经知道的公共表达式的值，范围领域
        ScopedMapTy::ScopeTy scope;

        DominanceInfoNode *node;
        DominanceInfoNode::const_iterator childIterator;

        // 这个节点是否全部被处理完成了
        bool processed = false;
    };

    // 尝试消除冗余操作，如果操作被标记删除，返回success，否则返回false
    LogicalResult simplifyOperation(ScopedMapTy &knownValues, Operation *op,
                                    bool hasSSADominance);
    void simplifyBlock(ScopedMapTy &knownValues, Block *bb, bool hasSSADominance);
    void simplifyRegion(ScopedMapTy &knownValues, Region &region);

    void replaceUsesAndDelete(ScopedMapTy &knownValues, Operation *op,
                                Operation *existing, bool hasSSADominance);

    // 检查两个操作之间是否有副作用
    bool hasOtherSideEffectingOpInBetween(Operation *fromOp, Operation *toOp);

    // 用于改写IR的rewriter
    RewriterBase &rewriter;
    // 标记为dead将要被删除的op
    std::vector<Operation *> opsToErase;
    DominanceInfo *domInfo = nullptr;
    MemEffectsCache memEffectsCache;

    // 统计数据
    int64_t numCSE = 0;
    int64_t numDCE = 0;
};
} // namespace

void TacCSEDriver::replaceUsesAndDelete(ScopedMapTy &knownValues,
    Operation *op,Operation *existing,bool hasSSADominance){
    // knownValues  带作用域的哈希表，存着所有已经处理过、可以复用的算子
    // hasSSADominance当前区域是不是**严格 SSA 支配**的（决定了能不能放心全替换）

    // 如果找到等价算子，就把当前算子的所有使用都替换成已存在的算子，然后标记为待删除。
    // 注意：只有当操作数的使用点还没被访问过的时候，我们才能替换它。
    if(hasSSADominance){
        // 如果区域有 SSA 支配，我们能保证：当前算子的所有使用点都还没被访问过
        if(auto *rewriteListener = 
            llvm::dyn_cast_if_present<RewriterBase::Listener>(rewriter.getListener())){
            rewriteListener->notifyOperationReplaced(op,existing);
        }
        // 替换所有使用，但暂时不删除算子。这里不通知监听器，因为原算子还没被擦除
        rewriter.replaceAllUsesWith(op->getResults(),existing->getResults());
        opsToErase.push_back(op);
        // 现在还在遍历算子，当场删除会破坏遍历器
        // 替换引用的时候，还需要 `op` 本身的结果信息
    }else{
        // 区域没有 SSA 支配时，替换前必须检查：这个使用点是不是已经被访问过了
        auto wasVisited = [&](OpOperand &operand) {
            // 处理过返回false，没有处理返回true
            return !knownValues.count(operand.getOwner());
        };
        if(auto *rewriteListener = 
            llvm::dyn_cast_if_present<RewriterBase::Listener>(rewriter.getListener())){
            for (Value v : op->getResults())
            if (all_of(v.getUses(), wasVisited))
            rewriteListener->notifyOperationReplaced(op, existing);

        }
        // 只替换满足条件的使用点
        rewriter.replaceUsesWithIf(op->getResults(),existing->getResults(),wasVisited);
        if(op->use_empty()){
            opsToErase.push_back(op);
        }
    }
    // 如果已有的算子位置未知，而当前算子有位置，就把当前的位置同步给已有的算子
    if (isa<UnknownLoc>(existing->getLoc()) && !isa<UnknownLoc>(op->getLoc()))
        existing->setLoc(op->getLoc());

    ++numCSE;
}
bool TacCSEDriver::hasOtherSideEffectingOpInBetween(Operation *fromOp,
    Operation *toOp){
    // 只有同一块内的代码才是严格按顺序执行的，才能这样逐个往后遍历检查。
    // 跨分支、跨块的话执行顺序不确定，不能用这种线性遍历的方式判断。
    assert(fromOp->getBlock() == toOp->getBlock());
    // 必须实现了内存副作用接口，同时是内存读操作
    assert(
        isa<MemoryEffectOpInterface>(fromOp) &&
        cast<MemoryEffectOpInterface>(fromOp).hasEffect<MemoryEffects::Read>() &&
        isa<MemoryEffectOpInterface>(toOp) &&
        cast<MemoryEffectOpInterface>(toOp).hasEffect<MemoryEffects::Read>()
    );

    Operation *nextOp = fromOp->getNextNode();
    // 键是起始操作，值是 `<检查终点, 有没有写>`
    /*
        `try_emplace`：尝试往缓存里插入一条记录。
        - 如果键已经存在 → 不插入，`result.second = false`
        - 如果键不存在 → 插入成功，`result.second = true`
        - `result.first`：永远指向缓存里对应这个键的那条记录
    */
    auto result = memEffectsCache.try_emplace(fromOp,
        std::make_pair(nextOp, nullptr));
    // 原来不存在的
    if(result.second){
        auto memEffectsCachePair = result.first->second;
        if(memEffectsCachePair.second == nullptr){
            // 缓存里到目前位置还没发现写，从缓存的终点继续往后查
            nextOp = memEffectsCachePair.first;
        }else{
            // 有副作用
            return true;
        }
    }
    // 要么原来存在，要么原来不存在，都有nextop
    while(nextOp && nextOp != toOp){
        // 递归收集一个 Op 以及它内部所有嵌套 Region 里全部 Op 的内存副作用
        std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
            getEffectsRecursively(nextOp);

        if(!effects){
            // 这个操作不实现副作用接口，保守认为它会写
            result.first->second = std::make_pair(nextOp,
                MemoryEffects::Write::get());
            return true;
        }

        for(const MemoryEffects::EffectInstance &effect:*effects){
            // isa<目标类型>(待检查指针)
            if(isa<MemoryEffects::Write>(effect.getEffect())){
                result.first->second = {nextOp,MemoryEffects::Write::get()};
                return true;
            }
        }

        nextOp = nextOp->getNextNode();
    }

    // 迭代遍历，发现fromOp到toOp中间没有写副作用的操作
    result.first->second = std::make_pair(toOp,nullptr);
    return false;
}   

// 尝试消除冗余操作，包含多个block的region，或者嵌套regions这种op不做简化替换处理
LogicalResult TacCSEDriver::simplifyOperation(ScopedMapTy &knownValues,
    Operation *op,bool hasSSADominance) {
    // 不要简化终止符操作
    if(op->hasTrait<OpTrait::IsTerminator>()){
        return failure();
    }
    // 如果这个操作本来就是明显没用的死代码，直接加到删除列表里
    if(isOpTriviallyDead(op)){
        opsToErase.push_back(op);
        ++numDCE;
        return success();
    }

    // 不要简化带有「包含多个基本块的 Region」的操作。
    // 还需要更多测试来验证能正确处理这种 IR。
    if(!llvm::all_of(op->getRegions(), [](Region &r){
        return r.getBlocks().empty() || llvm::hasSingleElement(r.getBlocks());
    })){
        return failure();
    }
    // 这里处理一些简单的内存副作用场景。无副作用的操作在后面处理。
    // 目前只能处理纯读副作用的场景，更复杂的模式还需要更多工作
    // 有副作用
    if(!isMemoryEffectFree(op)){
        auto memEffects = dyn_cast<MemoryEffectOpInterface>(op);
        // 没有副作用接口，或者不只是只有read副作用
        if(!memEffects || !memEffects.onlyHasEffect<MemoryEffects::Read>()){
            return failure();
        }
        // 查找已经存在的op定义
        if(auto *existing = knownValues.lookup(op)){
            // 找到的op必须处于同一个block，还有中间必须没有其他副作用
            // 跨块的话执行顺序不确定，中间可能夹着写操作，不能随便合并
            if(existing->getBlock() == op->getBlock() &&
                !hasOtherSideEffectingOpInBetween(existing, op)){
                replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
                return success();
            }
        }
        // 没有找到op定义
        knownValues.insert(op, op);
        return failure();
    }

    // 没有副作用
    if(auto *existing = knownValues.lookup(op)){
        replaceUsesAndDelete(knownValues,op, existing, hasSSADominance);
        return success();
    }

    // 不存在
    knownValues.insert(op, op);
    return failure();
}

// 这里针对多个region的会进行区域进行迭代，处理op的内部，但是op单独替换
void TacCSEDriver::simplifyBlock(ScopedMapTy &knownValues, Block *bb,
                              bool hasSSADominance) {
    for(auto &op : bb->getOperations()){
        // 大多数操作都不包含嵌套区域，所以对无区域的情况走快速路径
        if(op.getNumRegions()!=0){
            /*
            `IsIsolatedFromAbove`（与上层隔离）是 MLIR 的一个操作特质，
            标记了这个特质的操作，**内部区域不能隐式使用外层的任何值**，
            所有外层值必须显式传参捕获。

            - 如果直接用外层的 `knownValues` 做 CSE，
            内层操作就会直接复用外层的计算结果，相当于「隐式捕获了外层的值」，破坏了隔离语义。
            - 所以官方的做法是：新建一个**完全空的哈希表 `nestedKnownValues`**，
            用全新的作用域处理内部区域，不继承任何外层的已知值，保证隔离性。
            */
            if(op.mightHaveTrait<OpTrait::IsIsolatedFromAbove>()){
                ScopedMapTy nestedKnownValues;
                for(auto &region:op.getRegions()){
                    simplifyRegion(nestedKnownValues, region);
                }
            }else{
                for(auto &region:op.getRegions()){
                    simplifyRegion(knownValues, region);
                }
            }
        }
        if(succeeded(simplifyOperation(knownValues, &op, hasSSADominance))){
            continue;
        }
    }
    // 清空内存副作用缓存，因为它只在单个基本块内有效
    memEffectsCache.clear();
}

void TacCSEDriver::simplifyRegion(ScopedMapTy &knownValues, Region &region) {
    if (region.empty()){
        return;
    }

    bool hasSSADominance = domInfo->hasSSADominance(&region);
    if(region.hasOneBlock()){
        // 带作用域的哈希表
        ScopedMapTy::ScopeTy scope(knownValues);
        simplifyBlock(knownValues, &region.front(), hasSSADominance);
        return;
    }
    // 非 SSA 支配的多块区域：直接跳过不处理
    // 没有严格 SSA 支配的区域，控制流不规范，
    // 没法保证跨块复用的正确性，保守起见直接跳过，不做多块 CSE
    if(!hasSSADominance){
        return;
    }

    /*
    这里使用 deque（双端队列），是因为在容器变得非常大时，
    针对这种特定的访问模式，deque 的性能比 vector 有显著提升。
    */
    std::deque<std::unique_ptr<CFGStackNode>> stack;

    // 处理这个reigon支配树根节点节点
    stack.emplace_back(std::make_unique<CFGStackNode>(knownValues,
        domInfo->getRootNode(&region)));

    while(!stack.empty()){
        auto &currentNode = stack.back();
        // 第一步：当前节点没处理过，就先处理当前块
        if(!currentNode->processed){
            currentNode->processed=true;
            simplifyBlock(knownValues, currentNode->node->getBlock(), hasSSADominance);
        }
        // 第二步：还有子节点没处理，就压入下一个子节点
        if(currentNode->childIterator!=currentNode->node->end()){
            auto *childNode = *(currentNode->childIterator++);
            stack.emplace_back(std::make_unique<CFGStackNode>(knownValues,childNode));
        }else{
            // 第三步：所有子节点都处理完了，弹出当前节点
            stack.pop_back();
        }
    }
}

void TacCSEDriver::simplify(Operation *op, bool *changed) {
    // 简化region
    ScopedMapTy knownValues;
    for (auto &region : op->getRegions())
        simplifyRegion(knownValues, region);

    // 擦除所有在简化过程中标记为死亡的操作
    for (auto *op : opsToErase)
        rewriter.eraseOp(op);
    if (changed)
        *changed = !opsToErase.empty();

    // CSE 目前不会删除带区域的操作，所以支配树信息不需要失效
}

/// Tac CSE Pass 实现
struct TacCSEPass : public tac::impl::CSEBase<TacCSEPass> {
    void runOnOperation() override {
        IRRewriter rewriter(&getContext());
        TacCSEDriver driver(rewriter, &getAnalysis<DominanceInfo>());
        bool changed = false;
        driver.simplify(getOperation(), &changed);

        if (!changed)
            return markAllAnalysesPreserved();
        markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
    };
};

// 构造函数实现
std::unique_ptr<Pass> tac::createCSEPass() {
  return std::make_unique<TacCSEPass>();
}
