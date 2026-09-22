import onnx
from onnx import helper, TensorProto
import numpy as np
import os
import onnx.checker


os.makedirs("onnx_files",exist_ok=True)
os.makedirs("proto_files",exist_ok=True)

# INITIALIZER: 初始化权重
weight_val = np.array([10.0,20.0],dtype=np.float32)
initializer_tensor = helper.make_tensor(
    name = "stem.1.weight",
    data_type = TensorProto.FLOAT,
    dims=[2],
    vals=weight_val.tobytes(),
    raw=True
)

# 常量Nodes，内部硬编码
val1 = np.array([1.0, 1.0], dtype=np.float32)
tensor1 = helper.make_tensor(
    name="const_tensor_1",
    data_type=TensorProto.FLOAT,
    dims=[2],
    vals=val1.tobytes(),
    raw=True
)

# Constant Op
const_node = helper.make_node(
    op_type="Constant",
    inputs=[],
    outputs=["const_out"],
    value = tensor1,
    name="my_constant_op"
)

# Add
add_node1 = helper.make_node(
    "Add",
    inputs=["x","stem.1.weight"],
    outputs=["sum1"],
    name="add_weights"
)

add_node2 = helper.make_node(
    "Add",
    inputs=["sum1","const_out"],
    outputs=["final_out"],
    name="add_const"
)

# graph
graph = helper.make_graph(
    nodes=[const_node,add_node1,add_node2],
    name="ToyCompilerGraph",
    inputs=[
        # Function argument 'x'
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    ],
    outputs=[
        helper.make_tensor_value_info("final_out", TensorProto.FLOAT, [2])
    ],
    initializer=[initializer_tensor] # 权重值
)

# model
opset_version = helper.make_opsetid("", 11)
model = helper.make_model(graph,
                        producer_name="tac-compiler",
                        ir_version=8,
                        opset_imports=[opset_version])

import onnx.checker
print("CHECK",onnx.checker.check_model(model))
# 保存模型
filename = "add_constant"
onnx_path = f"onnx_files/{filename}.onnx"
onnx.save(model, onnx_path)


with open(f"proto_files/{filename}.prototxt", "w") as f:
    f.write(str(model))

print(f"Generated {onnx_path} and {filename}.prototxt")