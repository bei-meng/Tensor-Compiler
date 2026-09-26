// RUN: /home/ubuntu/mlir-tutorial/mlir-tac/build/tac-opt --tac-dce %s | tee test3_mixed_all_opt.mlir | /home/ubuntu/mlir-tutorial/llvm-project/build/bin/FileCheck %s --dump-input=fail

func.func @mixed_test(%cond:i1, %lb:i32, %ub:i32, %step:i32, %mem:memref<i32>) -> i32 {
  %dead_val = arith.constant 123 : i32 // 入口块死代码
  %side_val = arith.constant 500 : i32
  memref.store %side_val, %mem[] : memref<i32>
  // i32 转 index，供给 scf.for 使用
  %lb_idx = arith.index_cast %lb : i32 to index
  %ub_idx = arith.index_cast %ub : i32 to index
  %step_idx = arith.index_cast %step : i32 to index
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %bb1_dead = arith.constant 444 : i32 // bb1内部死代码
  %sum_init = arith.constant 0 : i32
  %loop_res = scf.for %i = %lb_idx to %ub_idx step %step_idx iter_args(%s = %sum_init)->(i32){
    %loop_inner_dead = arith.constant 9999 : i32 // 循环体内死代码
    %i_i32 = arith.index_cast %i : index to i32 // index转i32
    %tmp = arith.addi %s, %i_i32 : i32
    scf.yield %tmp : i32
  }
  cf.br ^bb3(%loop_res : i32)
^bb2:
  %bb2_dead = arith.constant 789 : i32 // bb2内部死代码
  %c = arith.constant 42 : i32
  cf.br ^bb3(%c : i32)
^bb3(%final : i32):
  %bb3_dead = arith.constant 666 : i32 // bb3汇合块死代码
  func.return %final : i32
}

// ===== FileCheck 预期输出（严格按输出从上到下顺序） =====
// CHECK-LABEL: func.func @mixed_test

// 入口块存活指令
// CHECK: arith.constant 500 : i32
// CHECK: memref.store {{.*}}, {{.*}}[] : memref<i32>
// CHECK: arith.index_cast {{.*}} : i32 to index
// CHECK: arith.index_cast {{.*}} : i32 to index
// CHECK: arith.index_cast {{.*}} : i32 to index
// CHECK: cf.cond_br {{.*}}, ^bb1, ^bb2

// bb1 分支
// CHECK: ^bb1:
// CHECK: arith.constant 0 : i32
// CHECK: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}} = {{.*}}) -> (i32)
// 循环体内
// CHECK: arith.index_cast {{.*}} : index to i32
// CHECK: arith.addi {{.*}}, {{.*}} : i32
// CHECK: scf.yield {{.*}}: i32
// CHECK: }
// CHECK: cf.br ^bb3({{.*}}: i32)

// bb2 分支
// CHECK: ^bb2:
// CHECK: arith.constant 42 : i32
// CHECK: cf.br ^bb3({{.*}}: i32)

// bb3 汇合块
// CHECK: ^bb3({{.*}}: i32):
// CHECK: return {{.*}}: i32

// 所有死常量必须被删除
// CHECK-NOT: arith.constant 123
// CHECK-NOT: arith.constant 444
// CHECK-NOT: arith.constant 9999
// CHECK-NOT: arith.constant 789
// CHECK-NOT: arith.constant 666
