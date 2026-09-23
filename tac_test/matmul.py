import onnx
from onnx import helper, TensorProto
import os
os.makedirs("onnx_files",exist_ok=True)
os.makedirs("proto_files",exist_ok=True)
def create_large_matmul(dim):
    # inputs: A[dim, dim], B[dim, dim]
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [dim, dim])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [dim, dim])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [dim, dim])

    node = helper.make_node('MatMul', ['X', 'Y'], ['Z'])
    graph = helper.make_graph([node], 'large_matmul_graph', [X, Y], [Z])
    opset_version = helper.make_opsetid("", 11)
    model = helper.make_model(graph, producer_name='bench_gen', ir_version=8, opset_imports=[opset_version])
    onnx.checker.check_model(model)
    onnx.save(model, f'onnx_files/matmul_{dim}.onnx')
    print(f"生成 matmul_{dim}.onnx")
    with open(f"proto_files/matmul_{dim}.onnx", "w") as f:
        f.write(str(model))

create_large_matmul(512)
create_large_matmul(1024)
create_large_matmul(2048)
create_large_matmul(4096)
