func.func @test_all_ops() -> tensor<2x2xf32> {
  // 1. 常量1：2x3 f32
  %c1 = "tac.constant"() {value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>

  // 2. 常量2：2x3 f32（与c1同形状，用于add）
  %c2 = "tac.constant"() {value = dense<[[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>

  // 3. 加法：逐元素相加
  %add = "tac.add"(%c1, %c2) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>

  // 4. Relu激活：逐元素max(0,x)
  %relu = "tac.relu"(%add) : (tensor<2x3xf32>) -> tensor<2x3xf32>

  // 5. 常量3：3x2 f32（用于矩阵乘法，维度匹配2x3 * 3x2 = 2x2）
  %c3 = "tac.constant"() {value = dense<[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>

  // 6. 矩阵乘法
  %matmul = "tac.matmul"(%relu, %c3) : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>

  return %matmul : tensor<2x2xf32>
}
