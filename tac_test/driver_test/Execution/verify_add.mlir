
// RUN: python3 verify_add.py
func.func @main() -> tensor<2xf32> attributes {llvm.emit_c_interface} {
  %0 = arith.constant dense<[0.27841705083847046, 0.5915667414665222]> : tensor<2xf32>
  %1 = arith.constant dense<[0.013476683758199215, 0.5922263860702515]> : tensor<2xf32>
  %2 = "tac.add"(%0, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  return %2 : tensor<2xf32>
}
