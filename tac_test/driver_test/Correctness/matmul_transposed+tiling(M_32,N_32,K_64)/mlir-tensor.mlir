#loc = loc(unknown)
#loc1 = loc("Z")
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<2048x2048xf32> loc(unknown), %arg1: tensor<2048x2048xf32> loc(unknown)) -> tensor<2048x2048xf32> attributes {llvm.emit_c_interface} {
    %c64 = arith.constant 64 : index loc(#loc)
    %c2048 = arith.constant 2048 : index loc(#loc)
    %c0 = arith.constant 0 : index loc(#loc)
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %0 = tensor.empty() : tensor<2048x2048xf32> loc(#loc1)
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2048x2048xf32>) -> tensor<2048x2048xf32> loc(#loc1)
    %transposed = linalg.transpose ins(%arg1 : tensor<2048x2048xf32>) outs(%0 : tensor<2048x2048xf32>) permutation = [1, 0]  loc(#loc1)
    %2 = scf.for %arg2 = %c0 to %c2048 step %c64 iter_args(%arg3 = %1) -> (tensor<2048x2048xf32>) {
      %3 = scf.for %arg4 = %c0 to %c2048 step %c64 iter_args(%arg5 = %arg3) -> (tensor<2048x2048xf32>) {
        %4 = scf.for %arg6 = %c0 to %c2048 step %c64 iter_args(%arg7 = %arg5) -> (tensor<2048x2048xf32>) {
          %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg6] [64, 64] [1, 1] : tensor<2048x2048xf32> to tensor<64x64xf32> loc(#loc1)
          %extracted_slice_0 = tensor.extract_slice %transposed[%arg4, %arg6] [64, 64] [1, 1] : tensor<2048x2048xf32> to tensor<64x64xf32> loc(#loc1)
          %extracted_slice_1 = tensor.extract_slice %arg7[%arg2, %arg4] [64, 64] [1, 1] : tensor<2048x2048xf32> to tensor<64x64xf32> loc(#loc1)
          %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%extracted_slice_1 : tensor<64x64xf32>) {
          ^bb0(%in: f32 loc(unknown), %in_2: f32 loc("Z"), %out: f32 loc("Z")):
            %6 = arith.mulf %in, %in_2 : f32 loc(#loc1)
            %7 = arith.addf %out, %6 : f32 loc(#loc1)
            linalg.yield %7 : f32 loc(#loc1)
          } -> tensor<64x64xf32> loc(#loc1)
          %inserted_slice = tensor.insert_slice %5 into %arg7[%arg2, %arg4] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<2048x2048xf32> loc(#loc1)
          scf.yield %inserted_slice : tensor<2048x2048xf32> loc(#loc1)
        } loc(#loc1)
        scf.yield %4 : tensor<2048x2048xf32> loc(#loc1)
      } loc(#loc1)
      scf.yield %3 : tensor<2048x2048xf32> loc(#loc1)
    } loc(#loc1)
    return %2 : tensor<2048x2048xf32> loc(#loc1)
  } loc(#loc)
} loc(#loc)

