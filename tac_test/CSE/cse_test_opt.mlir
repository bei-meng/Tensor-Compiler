module {
  func.func @test_same_block_cse(%arg0: i32, %arg1: i32) -> i32 {
    %c42_i32 = arith.constant 42 : i32
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.muli %0, %c42_i32 : i32
    return %1 : i32
  }
  func.func @test_side_effect_preserve(%arg0: memref<i32>, %arg1: i32) {
    memref.store %arg1, %arg0[] : memref<i32>
    memref.store %arg1, %arg0[] : memref<i32>
    return
  }
  func.func @test_sibling_no_cse(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %0 = arith.muli %arg1, %arg2 : i32
    cf.br ^bb3(%0 : i32)
  ^bb2:  // pred: ^bb0
    %1 = arith.muli %arg1, %arg2 : i32
    cf.br ^bb3(%1 : i32)
  ^bb3(%2: i32):  // 2 preds: ^bb1, ^bb2
    return %2 : i32
  }
  func.func @test_nested_for_cse(%arg0: index, %arg1: index, %arg2: index, %arg3: i32, %arg4: i32) -> i32 {
    %0 = arith.addi %arg3, %arg4 : i32
    %1 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %0) -> (i32) {
      %2 = arith.index_cast %arg5 : index to i32
      %3 = arith.addi %arg6, %2 : i32
      scf.yield %3 : i32
    }
    return %1 : i32
  }
  func.func @test_nested_if_cse(%arg0: i1, %arg1: i32, %arg2: i32) -> i32 {
    %0 = scf.if %arg0 -> (i32) {
      %1 = arith.muli %arg1, %arg2 : i32
      scf.yield %1 : i32
    } else {
      %1 = arith.muli %arg1, %arg2 : i32
      scf.yield %1 : i32
    }
    return %0 : i32
  }
  func.func @test_mixed_comprehensive(%arg0: i1, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<i32>) -> i32 {
    %c100_i32 = arith.constant 100 : i32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg2 : i32 to index
    %2 = arith.index_cast %arg3 : i32 to index
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %3 = arith.addi %c100_i32, %c100_i32 : i32
    %4 = scf.for %arg5 = %0 to %1 step %2 iter_args(%arg6 = %3) -> (i32) {
      %7 = arith.muli %arg6, %c100_i32 : i32
      memref.store %7, %arg4[] : memref<i32>
      scf.yield %7 : i32
    }
    cf.br ^bb3(%4 : i32)
  ^bb2:  // pred: ^bb0
    %5 = arith.muli %c100_i32, %c100_i32 : i32
    cf.br ^bb3(%5 : i32)
  ^bb3(%6: i32):  // 2 preds: ^bb1, ^bb2
    return %6 : i32
  }
}

