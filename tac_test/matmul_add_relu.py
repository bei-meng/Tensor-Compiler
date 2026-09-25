import onnx
from onnx import helper, TensorProto
import numpy as np
import os
os.makedirs("onnx_files",exist_ok=True)
os.makedirs("proto_files",exist_ok=True)
def create_fused_matmul(M,N,K):
    # Define Value infos (Inputs and Outputs)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [K, N])
    Bias = helper.make_tensor_value_info('Bias', TensorProto.FLOAT, [M, N])
    Final_Output = helper.make_tensor_value_info('Output', TensorProto.FLOAT, [M, N])

    # Node 1: MatMul (X * Y -> matmul_out)
    matmul_node = helper.make_node(
        'MatMul',
        ['X', 'Y'],
        ['matmul_out']
    )

    # Node 2: Add (matmul_out + bias -> add_out)
    add_node = helper.make_node(
        'Add',
        ['matmul_out', 'Bias'],
        ['add_out']
    )

    # Node 3: ReLu (add_out -> output)
    relu_node = helper.make_node(
        'Relu',
        ['add_out'],
        ['Output']
    )

    graph = helper.make_graph(
        [matmul_node, add_node, relu_node],
        'fused_matmul_graph',
        [X, Y, Bias],
        [Final_Output]
    )

    # Save model
    opset_version = helper.make_opsetid("", 11)
    model = helper.make_model(graph, ir_version=8, opset_imports=[opset_version])
    onnx.checker.check_model(model)
    filename = f'onnx_files/matmul_add_relu_M{M}_N{N}_K{K}.onnx'
    onnx.save(model, filename)
    print(f"生成 onnx_files/matmul_add_relu_M{M}_N{N}_K{K}.onnx")
    with open(f"proto_files/matmul_add_relu_M{M}_N{N}_K{K}.onnx", "w") as f:
        f.write(str(model))

create_fused_matmul(128,128,128)
create_fused_matmul(256,256,256)
create_fused_matmul(512,512,256)
create_fused_matmul(1024,1024,512)
create_fused_matmul(2048,2048,1024)
create_fused_matmul(4096,4096,2048)

create_fused_matmul(512,512,512)
create_fused_matmul(1024,1024,1024)
create_fused_matmul(2048,2048,2048)
create_fused_matmul(4096,4096,4096)