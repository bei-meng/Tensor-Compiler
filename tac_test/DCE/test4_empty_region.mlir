// RUN: /home/ubuntu/mlir-tutorial/mlir-tac/build/tac-opt --tac-dce %s | tee test4_empty_region_opt.mlir | /home/ubuntu/mlir-tutorial/llvm-project/build/bin/FileCheck %s --dump-input=fail

func.func @empty_region_test(%cond:i1, %mem:memref<i32>) -> i32 {
  %outer_dead = arith.constant 1001 : i32
  %zero = arith.constant 0 : i32
  %ret_val = scf.if %cond -> (i32) {
    %dead1 = arith.constant 10 : i32
    %dead2 = arith.constant 20 : i32
    %dead_add = arith.addi %dead1, %dead2 : i32
    scf.yield %zero : i32
  } else {
    %dead3 = arith.constant 30 : i32
    %dead4 = arith.constant 40 : i32
    %dead_mul = arith.muli %dead3, %dead4 : i32
    scf.yield %zero : i32
  }
  %after_if_dead = arith.constant 2002 : i32
  memref.store %ret_val, %mem[] : memref<i32>
  func.return %ret_val : i32
}

// ===== FileCheck 预期输出（严格按输出从上到下顺序） =====
// CHECK-LABEL: func.func @empty_region_test

// 入口块存活指令
// CHECK: arith.constant 0 : i32

// scf.if 整体结构存活
// CHECK: scf.if {{.*}} -> (i32)
// CHECK: scf.yield {{.*}}: i32
// CHECK: } else {
// CHECK: scf.yield {{.*}}: i32
// CHECK: }

// if 后存活指令
// CHECK: memref.store {{.*}}, {{.*}}[] : memref<i32>
// CHECK: return {{.*}}: i32

// 所有死常量与死运算必须被删除
// CHECK-NOT: arith.constant 1001
// CHECK-NOT: arith.constant 10
// CHECK-NOT: arith.constant 20
// CHECK-NOT: arith.addi
// CHECK-NOT: arith.constant 30
// CHECK-NOT: arith.constant 40
// CHECK-NOT: arith.muli
// CHECK-NOT: arith.constant 2002
