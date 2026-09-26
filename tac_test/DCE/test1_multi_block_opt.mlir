module {
  func.func @multi_block_dce(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: memref<i32>) -> i32 {
    %c200_i32 = arith.constant 200 : i32
    memref.store %c200_i32, %arg3[] : memref<i32>
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %0 = arith.addi %arg1, %arg2 : i32
    cf.br ^bb3(%0 : i32)
  ^bb2:  // pred: ^bb0
    %1 = arith.muli %arg1, %arg2 : i32
    cf.br ^bb3(%1 : i32)
  ^bb3(%2: i32):  // 2 preds: ^bb1, ^bb2
    return %2 : i32
  }
}

