// RUN: /home/ubuntu/mlir-tutorial/mlir-tac/build/tac-opt --tac-dce %s | tee test1_multi_block_opt.mlir | /home/ubuntu/mlir-tutorial/llvm-project/build/bin/FileCheck %s --dump-input=fail

func.func @multi_block_dce(%cond : i1, %a : i32, %b : i32, %mem : memref<i32>) -> i32 {
  %dead_const = arith.constant 100 : i32 // 入口块死代码
  %side_const = arith.constant 200 : i32
  memref.store %side_const, %mem[] : memref<i32> // store有副作用，保留
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %block1_dead = arith.constant 111 : i32 // bb1内部死代码
  %add1 = arith.addi %a, %b : i32
  cf.br ^bb3(%add1 : i32)
^bb2:
  %block2_dead = arith.constant 222 : i32 // bb2内部死代码
  %add2 = arith.muli %a, %b : i32
  cf.br ^bb3(%add2 : i32)
^bb3(%val : i32):
  %block3_dead = arith.constant 333 : i32 // bb3汇合块死代码
  func.return %val : i32
}

// ===== FileCheck 预期输出 =====
// CHECK-LABEL: func.func @multi_block_dce
// CHECK: arith.constant 200 : i32
// CHECK: memref.store {{.*}}, {{.*}}[] : memref<i32>
// CHECK: cf.cond_br {{.*}}, ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK: arith.addi {{.*}}, {{.*}} : i32
// CHECK: cf.br ^bb3({{.*}}: i32)
// CHECK: ^bb2:
// CHECK: arith.muli {{.*}}, {{.*}} : i32
// CHECK: cf.br ^bb3({{.*}}: i32)
// CHECK: ^bb3({{.*}}: i32):
// CHECK: return {{.*}}: i32

// 这些常量DCE后要被删掉
// CHECK-NOT: arith.constant 100
// CHECK-NOT: arith.constant 111
// CHECK-NOT: arith.constant 222
// CHECK-NOT: arith.constant 333
