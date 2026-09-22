#ifndef ONNX_MODEL_INFO_H
#define ONNX_MODEL_INFO_H

#include<string>
#include<vector>
#include<cstdint>
#include<map>

namespace tac{
// TensorProto 带数据，还有形状和类型
struct TensorInfo{
    std::string name;
    enum class DataType {FLOAT, INT64, INT32, BOOL, UINT8, UNKNOWN};
    std::vector<int64_t> shape;
    DataType elementType;
    std::vector<char> rawData;

    bool isScalar() const{
        return shape.empty() || (shape.size()==1 && shape[0]==1);
    }
};


// ValueInfoProto
// 只有张量的形状、类型信息，**没有实际数据**，用来定义输入输出的维度信息
struct ValueInfo{
    std::string name;
    std::vector<int64_t> shape;
    TensorInfo::DataType elementType;
};

// AttributeProto
// 定义了OP中的具体参数，比如Conv中的stride和[kernel_size]等
// 把tensor当做一个op的属性挂载
struct AttributeInfo{
    std::string name;
    enum class Type { TENSOR, INT, FLOAT, STRING, INTS, FLOATS, UNKNOWN };
    Type type;

    // union-like storage for values
    TensorInfo tensor;
    int64_t i;
    float f;
    std::string s;
    std::vector<int64_t> ints;
    std::vector<float> floats;
};

// 定义了每个OP的具体操作，计算算子，比如 Add、MatMul、Relu
// 属性包括tensor这种，如constant的属性是tensor
struct NodeInfo{
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<AttributeInfo> attributes;
};

// 图，包括nodes，inputs，output，initializers
struct GraphInfo{
    std::string name;
    std::vector<NodeInfo> nodes;
    std::vector<ValueInfo> inputs;
    std::vector<ValueInfo> outputs;
    std::map<std::string,TensorInfo> initializers;
};


struct ModelInfo{
    int64_t ir_version; // ONNX 本身的 IR（中间表示）规范版本号
    std::string producer_name;
    GraphInfo graph;
};
}// namespace tac
#endif // ONNX_MODEL_INFO_H

