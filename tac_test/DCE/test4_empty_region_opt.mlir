module {
  func.func @empty_region_test(%arg0: i1, %arg1: memref<i32>) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.if %arg0 -> (i32) {
      scf.yield %c0_i32 : i32
    } else {
      scf.yield %c0_i32 : i32
    }
    memref.store %0, %arg1[] : memref<i32>
    return %0 : i32
  }
}

