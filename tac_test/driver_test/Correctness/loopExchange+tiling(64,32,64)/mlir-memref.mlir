#loc = loc(unknown)
module {
  func.func @main(%arg0: memref<2048x1024xf32, strided<[?, ?], offset: ?>> loc(unknown), %arg1: memref<1024x2048xf32, strided<[?, ?], offset: ?>> loc(unknown)) -> memref<2048x2048xf32> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index loc(#loc)
    %c32 = arith.constant 32 : index loc(#loc)
    %c64 = arith.constant 64 : index loc(#loc)
    %c1024 = arith.constant 1024 : index loc(#loc1)
    %c2048 = arith.constant 2048 : index loc(#loc)
    %c0 = arith.constant 0 : index loc(#loc)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2048x2048xf32> loc(#loc1)
    scf.for %arg2 = %c0 to %c2048 step %c1 {
      scf.for %arg3 = %c0 to %c2048 step %c1 {
        memref.store %cst, %alloc[%arg2, %arg3] : memref<2048x2048xf32> loc(#loc1)
      } loc(#loc1)
    } loc(#loc1)
    scf.for %arg2 = %c0 to %c2048 step %c64 {
      scf.for %arg3 = %c0 to %c1024 step %c32 {
        scf.for %arg4 = %c0 to %c2048 step %c64 {
          %subview = memref.subview %arg0[%arg2, %arg3] [64, 32] [1, 1] : memref<2048x1024xf32, strided<[?, ?], offset: ?>> to memref<64x32xf32, strided<[?, ?], offset: ?>> loc(#loc1)
          %subview_0 = memref.subview %arg1[%arg3, %arg4] [32, 64] [1, 1] : memref<1024x2048xf32, strided<[?, ?], offset: ?>> to memref<32x64xf32, strided<[?, ?], offset: ?>> loc(#loc1)
          %subview_1 = memref.subview %alloc[%arg2, %arg4] [64, 64] [1, 1] : memref<2048x2048xf32> to memref<64x64xf32, strided<[2048, 1], offset: ?>> loc(#loc1)
          scf.for %arg5 = %c0 to %c64 step %c1 {
            scf.for %arg6 = %c0 to %c32 step %c1 {
              scf.for %arg7 = %c0 to %c64 step %c1 {
                %0 = memref.load %subview[%arg5, %arg6] : memref<64x32xf32, strided<[?, ?], offset: ?>> loc(#loc1)
                %1 = memref.load %subview_0[%arg6, %arg7] : memref<32x64xf32, strided<[?, ?], offset: ?>> loc(#loc1)
                %2 = memref.load %subview_1[%arg5, %arg7] : memref<64x64xf32, strided<[2048, 1], offset: ?>> loc(#loc1)
                %3 = arith.mulf %0, %1 : f32 loc(#loc1)
                %4 = arith.addf %2, %3 : f32 loc(#loc1)
                memref.store %4, %subview_1[%arg5, %arg7] : memref<64x64xf32, strided<[2048, 1], offset: ?>> loc(#loc1)
              } loc(#loc1)
            } loc(#loc1)
          } loc(#loc1)
        } loc(#loc1)
      } loc(#loc1)
    } loc(#loc1)
    return %alloc : memref<2048x2048xf32> loc(#loc1)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("Z")

