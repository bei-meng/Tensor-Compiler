// RUN: /build/tac-opt --tac-cse %s | tee test_cse_comprehensive_opt.mlir | /home/ubuntu/mlir-onnx/llvm-project/build/bin/FileCheck %s --dump-input=fail

// ================================================================
// 测试1：同一块内的重复常量、重复二元运算 → CSE合并为1个
// ================================================================
// CHECK-LABEL: func.func @test_same_block_cse
func.func @test_same_block_cse(%a : i32, %b : i32) -> i32 {
  %c1 = arith.constant 42 : i32
  %c2 = arith.constant 42 : i32
  %add1 = arith.addi %a, %b : i32
  %add2 = arith.addi %a, %b : i32
  %res = arith.muli %add1, %c1 : i32
  func.return %res : i32
}
// 验证：常量42只1个；addi只1个；muli保留；return保留
// CHECK:      arith.constant 42 : i32
// CHECK:      arith.addi
// CHECK:      arith.muli
// CHECK:      return {{.*}}: i32
// CHECK-NOT:  arith.constant 42
// CHECK-NOT:  arith.addi

// ================================================================
// 测试2：有副作用的memref.store → 不能CSE，全部保留
// ================================================================
// CHECK-LABEL: func.func @test_side_effect_preserve
func.func @test_side_effect_preserve(%mem : memref<i32>, %val : i32) {
  memref.store %val, %mem[] : memref<i32>
  memref.store %val, %mem[] : memref<i32>
  func.return
}
// 验证：两个store全部保留
// CHECK-COUNT-2: memref.store

// ================================================================
// 测试3：兄弟分支的相同计算 → 不支配，不能CSE，全部保留
// ================================================================
// CHECK-LABEL: func.func @test_sibling_no_cse
func.func @test_sibling_no_cse(%cond : i1, %a : i32, %b : i32) -> i32 {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %mul1 = arith.muli %a, %b : i32
  cf.br ^exit(%mul1 : i32)
^bb2:
  %mul2 = arith.muli %a, %b : i32
  cf.br ^exit(%mul2 : i32)
^exit(%res : i32):
  func.return %res : i32
}
// 验证：两个muli都保留
// CHECK-COUNT-2: arith.muli

// ================================================================
// 测试4：嵌套scf.for循环内的同块重复 → 合并
// ================================================================
// CHECK-LABEL: func.func @test_nested_for_cse
func.func @test_nested_for_cse(%lb : index, %ub : index, %step : index, %a : i32, %b : i32) -> i32 {
  %add_outer = arith.addi %a, %b : i32
  %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %add_outer) -> (i32) {
    %cast1 = arith.index_cast %i : index to i32
    %cast2 = arith.index_cast %i : index to i32
    %next = arith.addi %acc, %cast1 : i32
    scf.yield %next : i32
  }
  func.return %res : i32
}
// 验证：外部1个addi；循环内index_cast只1个
// CHECK:      arith.addi
// CHECK:      scf.for
// CHECK:      arith.index_cast
// CHECK-NOT:  arith.index_cast
// CHECK:      scf.yield

// ================================================================
// 测试5：嵌套scf.if两分支 → 兄弟分支不合并，各自同块重复合并
// ================================================================
// CHECK-LABEL: func.func @test_nested_if_cse
func.func @test_nested_if_cse(%cond : i1, %a : i32, %b : i32) -> i32 {
  %res = scf.if %cond -> (i32) {
    %mul1 = arith.muli %a, %b : i32
    %mul2 = arith.muli %a, %b : i32
    scf.yield %mul1 : i32
  } else {
    %mul_else = arith.muli %a, %b : i32
    scf.yield %mul_else : i32
  }
  func.return %res : i32
}
// 验证：then分支1个muli；else分支1个muli
// CHECK:      scf.if
// CHECK:      arith.muli
// CHECK-NOT:  arith.muli
// CHECK:      } else {
// CHECK:      arith.muli
// CHECK:      }

// ================================================================
// 测试6：多分支+循环嵌套综合场景
// ================================================================
// CHECK-LABEL: func.func @test_mixed_comprehensive
func.func @test_mixed_comprehensive(%cond:i1, %lb:i32, %ub:i32, %step:i32, %mem:memref<i32>) -> i32 {
  %c1 = arith.constant 100 : i32
  %c2 = arith.constant 100 : i32
  %lb_idx = arith.index_cast %lb : i32 to index
  %ub_idx = arith.index_cast %ub : i32 to index
  %step_idx = arith.index_cast %step : i32 to index
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  %add1 = arith.addi %c1, %c1 : i32
  %add2 = arith.addi %c1, %c1 : i32
  %loop = scf.for %i = %lb_idx to %ub_idx step %step_idx iter_args(%s = %add1) -> (i32) {
    %tmp1 = arith.muli %s, %c1 : i32
    %tmp2 = arith.muli %s, %c1 : i32
    memref.store %tmp1, %mem[] : memref<i32>
    scf.yield %tmp1 : i32
  }
  cf.br ^bb3(%loop : i32)

^bb2:
  %mul_bb2 = arith.muli %c1, %c1 : i32
  cf.br ^bb3(%mul_bb2 : i32)

^bb3(%final : i32):
  func.return %final : i32
}
// 验证：常量100只1个；bb1内addi只1个；循环内muli只1个；bb2的muli保留
// CHECK-COUNT-1: arith.constant 100 : i32
// CHECK: ^bb1:
// CHECK-COUNT-1: arith.addi
// CHECK: scf.for
// CHECK-COUNT-1: arith.muli
// CHECK: ^bb2:
// CHECK: arith.muli
