#loc = loc(unknown)
module {
  func.func @main(%arg0: tensor<2048x2048xf32> loc(unknown), %arg1: tensor<2048x2048xf32> loc(unknown), %arg2: tensor<2048x2048xf32> loc(unknown)) -> tensor<2048x2048xf32> attributes {llvm.emit_c_interface} {
    %0 = "tac.matmul"(%arg0, %arg1) : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32> loc(#loc1)
    %1 = "tac.add"(%0, %arg2) : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32> loc(#loc2)
    %2 = tac.relu %1 : tensor<2048x2048xf32> loc(#loc3)
    return %2 : tensor<2048x2048xf32> loc(#loc3)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("matmul_out")
#loc2 = loc("add_out")
#loc3 = loc("Output")

