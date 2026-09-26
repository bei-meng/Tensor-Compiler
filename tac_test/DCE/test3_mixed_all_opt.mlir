module {
  func.func @mixed_test(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<i32>) -> i32 {
    %c500_i32 = arith.constant 500 : i32
    memref.store %c500_i32, %arg4[] : memref<i32>
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg3 : i32 to index
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %c0_i32 = arith.constant 0 : i32
    %3 = scf.for %arg5 = %0 to %1 step %2 iter_args(%arg6 = %c0_i32) -> (i32) {
      %5 = arith.index_cast %arg5 : index to i32
      %6 = arith.addi %arg6, %5 : i32
      scf.yield %6 : i32
    }
    cf.br ^bb3(%3 : i32)
  ^bb2:  // pred: ^bb0
    %c42_i32 = arith.constant 42 : i32
    cf.br ^bb3(%c42_i32 : i32)
  ^bb3(%4: i32):  // 2 preds: ^bb1, ^bb2
    return %4 : i32
  }
}

