// RUN: /home/ubuntu/mlir-tutorial/mlir-tac/build/tac-opt --tac-dce %s | tee test2_nested_region_opt.mlir | /home/ubuntu/mlir-tutorial/llvm-project/build/bin/FileCheck %s --dump-input=fail

func.func @dce_nested_region(%lb:i32, %ub:i32, %step:i32, %mem : memref<i32>) -> i32 {
  %dead_c = arith.constant 999 : i32
  %sum = arith.constant 0 : i32
  %global_store_val = arith.constant 777 : i32
  memref.store %global_store_val, %mem[] : memref<i32>
  // i32 -> index
  %lb_idx = arith.index_cast %lb : i32 to index
  %ub_idx = arith.index_cast %ub : i32 to index
  %step_idx = arith.index_cast %step : i32 to index
  %res = scf.for %i = %lb_idx to %ub_idx step %step_idx iter_args(%s = %sum) -> (i32) {
    %inside_dead = arith.constant 111 : i32
    %i_i32 = arith.index_cast %i : index to i32
    %new_sum = arith.addi %s, %i_i32 : i32
    memref.store %new_sum, %mem[] : memref<i32>
    scf.yield %new_sum : i32
  }
  %after_loop_dead = arith.constant 555 : i32
  func.return %res : i32
}

// ===== FileCheck 预期输出（严格按输出从上到下的顺序） =====
// CHECK-LABEL: func.func @dce_nested_region

// 入口块存活指令：按实际输出顺序排列
// CHECK: arith.constant 0 : i32
// CHECK: arith.constant 777 : i32
// CHECK: memref.store {{.*}}, {{.*}}[] : memref<i32>
// CHECK: arith.index_cast {{.*}} : i32 to index
// CHECK: arith.index_cast {{.*}} : i32 to index
// CHECK: arith.index_cast {{.*}} : i32 to index

// scf.for 循环头
// CHECK: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}} = {{.*}}) -> (i32)

// 循环体内存活指令
// CHECK: arith.index_cast {{.*}} : index to i32
// CHECK: arith.addi {{.*}}, {{.*}} : i32
// CHECK: memref.store {{.*}}, {{.*}}[] : memref<i32>
// CHECK: scf.yield {{.*}}: i32

// 循环后返回
// CHECK: return {{.*}}: i32

// 死常量必须被删除
// CHECK-NOT: arith.constant 999
// CHECK-NOT: arith.constant 111
// CHECK-NOT: arith.constant 555
