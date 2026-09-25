import onnx
from onnx import helper, TensorProto
import os
os.makedirs("onnx_files",exist_ok=True)
os.makedirs("proto_files",exist_ok=True)
def create_large_matmul(M,N,K):
    # inputs: A[dim, dim], B[dim, dim]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [K, N])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [M, N])

    node = helper.make_node('MatMul', ['X', 'Y'], ['Z'])
    graph = helper.make_graph([node], 'large_matmul_graph', [X, Y], [Z])
    opset_version = helper.make_opsetid("", 11)
    model = helper.make_model(graph, producer_name='bench_gen', ir_version=8, opset_imports=[opset_version])
    onnx.checker.check_model(model)
    onnx.save(model, f'onnx_files/matmul_M{M}_N{N}_K{K}.onnx')
    print(f"生成 matmul_M{M}_N{N}_K{K}.onnx")
    with open(f"proto_files/matmul_M{M}_N{N}_K{K}.onnx", "w") as f:
        f.write(str(model))

create_large_matmul(128,128,128)
create_large_matmul(256,256,256)
create_large_matmul(512,512,256)
create_large_matmul(1024,1024,512)
create_large_matmul(2048,2048,1024)
create_large_matmul(4096,4096,2048)

create_large_matmul(512,512,512)
create_large_matmul(1024,1024,1024)
create_large_matmul(2048,2048,2048)
create_large_matmul(4096,4096,4096)
