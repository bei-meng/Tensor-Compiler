module {
  func.func @dce_nested_region(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<i32>) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c777_i32 = arith.constant 777 : i32
    memref.store %c777_i32, %arg3[] : memref<i32>
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = scf.for %arg4 = %0 to %1 step %2 iter_args(%arg5 = %c0_i32) -> (i32) {
      %4 = arith.index_cast %arg4 : index to i32
      %5 = arith.addi %arg5, %4 : i32
      memref.store %5, %arg3[] : memref<i32>
      scf.yield %5 : i32
    }
    return %3 : i32
  }
}

