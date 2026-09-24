#include "tac/OnnxParser.h"
#include "tac/OnnxModelInfo.h"

#include "onnx/onnx-ml.pb.h"
#include <cstddef>
#include <cstring>
#include <string>
#include <sys/types.h>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace tac{
static TensorInfo parseTensor(const onnx::TensorProto &t){
    TensorInfo info;
    if(!t.name().empty()){
        info.name = t.name();
    }

    // 提取形状
    for(int64_t d:t.dims()){
        info.shape.push_back(d);
    }

    // 提取元素类型
    switch(t.data_type()){
        case onnx::TensorProto::FLOAT :
            info.elementType = TensorInfo::DataType::FLOAT;break;
        case onnx::TensorProto::INT32 :
            info.elementType = TensorInfo::DataType::INT32;break;
        case onnx::TensorProto::INT64 :
            info.elementType = TensorInfo::DataType::INT64;break;
        default:
            info.elementType = TensorInfo::DataType::UNKNOWN;
            std::cerr << "Warning: Unsupported ONNX data type: " 
                << t.data_type() << std::endl;
            break;
    }

    // 提取值
    // onnx的raw_data规范强制小端
    // Protobuf协议中float存储编码时强制小端序，解析到内存时，还原为本机float数组，平台为小端序
    if(t.has_raw_data()){
        // 直接复制二进制值
        const std::string &bytes = t.raw_data();
        info.rawData.assign(bytes.begin(),bytes.end());
    }else if(t.float_data_size()>0){
        // 如果 ONNX 使用了 repeated float 字段，则将其打包为字节数据
        size_t byteSize = t.float_data_size() * sizeof(float);
        info.rawData.resize(byteSize);
        std::memcpy(info.rawData.data(),t.float_data().data(),byteSize);
    }

    return info;
}

static ValueInfo parseValue(const onnx::ValueInfoProto &v){
    // ONNX 的 TypeProto 是多类型联合（oneof）它的类型不只有张量一种
    ValueInfo info;
    info.name = v.name();
    if(v.has_type() && v.type().has_tensor_type()){
        const auto &tensorType = 
            v.type().tensor_type();
        // 提取元素类型
        switch(tensorType.elem_type()){
            case onnx::TensorProto::FLOAT :
                info.elementType = TensorInfo::DataType::FLOAT;break;
            case onnx::TensorProto::INT32 :
                info.elementType = TensorInfo::DataType::INT32;break;
            case onnx::TensorProto::INT64 :
                info.elementType = TensorInfo::DataType::INT64;break;
            default:
                info.elementType = TensorInfo::DataType::UNKNOWN;
                std::cerr << "Warning: Unsupported ONNX data type: " 
                    << tensorType.elem_type() << std::endl;
                break;
        }
        // 提取形状
        for (const auto &dim : tensorType.shape().dim()) {
            if (dim.has_dim_value()) {
                info.shape.push_back(dim.dim_value());
            } else {
                // 处理动态形状
                info.shape.push_back(-1);
            }
        }
    }
    return info;
}

static AttributeInfo parseAttribute(const onnx::AttributeProto &attr){
    AttributeInfo info;
    info.name = attr.name();
    switch (attr.type()) {
        case onnx::AttributeProto::TENSOR:
            info.type = AttributeInfo::Type::TENSOR;
            info.tensor = parseTensor(attr.t());
            break;

        case onnx::AttributeProto::INT:
            info.type = AttributeInfo::Type::INT;
            info.i = attr.i();
            break;

        case onnx::AttributeProto::FLOAT:
            info.type = AttributeInfo::Type::FLOAT;
            info.f = attr.f();
            break;

        case onnx::AttributeProto::STRING:
            info.type = AttributeInfo::Type::STRING;
            info.s = attr.s();
            break;

        case onnx::AttributeProto::INTS:
            info.type = AttributeInfo::Type::INTS;
            info.ints.assign(attr.ints().begin(), attr.ints().end());
            break;

        case onnx::AttributeProto::FLOATS:
            info.type = AttributeInfo::Type::FLOATS;
            info.floats.assign(attr.floats().begin(), attr.floats().end());
            break;
        default:
            std::cerr << "Warning: Unsupported ONNX attr: " << attr.type() << std::endl;
            break;
    }

    return info;
}

static NodeInfo parseNode(const onnx::NodeProto &node){
    NodeInfo info;
    info.op_type = node.op_type();
    for(const auto &in:node.input()){
        info.inputs.push_back(in);
    }
    for(const auto &out:node.output()){
        info.outputs.push_back(out);
    }
    for(const auto &attr:node.attribute()){
        info.attributes.push_back(parseAttribute(attr));
    }
    return info;
}

// static void NodeTopology(GraphInfo &info){
//     size_t nodeNum = info.nodes.size();
//     if(nodeNum > 1){
//         // 构建节点的输入名字集合
//         std::unordered_set<std::string> externalTensorNames;
//         for(const auto &in :info.inputs){
//             externalTensorNames.insert(in.name);
//         }
//         // 初始化集合
//         for(const auto &init:info.initializers){
//             externalTensorNames.insert(init.first);
//         }

//         // 构建输出张量到节点索引的映射
//         std::unordered_map<std::string, size_t> outputToNodeIdx;
//         for(size_t i=0;i<nodeNum;++i){
//             const NodeInfo &node = info.nodes[i];
//             // 遍历该节点的所有输出，全部映射到同一个节点索引
//             for (const std::string &outputName : node.outputs) {
//                 outputToNodeIdx[outputName] = i;
//             }
//         }

//         // 计算每个节点的入度，同时构建邻接表
//         std::vector<size_t> inDegree(nodeNum,0);
//         std::vector<std::vector<size_t>> adj(nodeNum);


//         for(size_t i=0;i<nodeNum;++i){
//             const NodeInfo &node = info.nodes[i];
//             for(const std::string &inName :node.inputs){
//                 if(externalTensorNames.count(inName)){
//                     continue;
//                 }
//                 // 输入来自其他节点
//                 auto it = outputToNodeIdx.find(inName);
//                 if(it == outputToNodeIdx.end()){
//                     // 输入既不是外部资源也不是其他节点输出 → 非法图
//                     std::cerr << "Error: tensor '" << inName 
//                             << "' has no source in graph" << std::endl;
//                     assert(false && "Invalid ONNX graph");
//                 }
//                 size_t preIdx = it->second;
//                 // 输入指向输出
//                 adj[preIdx].push_back(i);
//                 inDegree[i]++;
//             }
//         }

//         std::queue<size_t> q;
//         for(size_t i=0;i<nodeNum;++i){
//             if(inDegree[i]==0){
//                 q.push(i);
//             }
//         }

//         std::vector<NodeInfo> sortedNodes;
//         sortedNodes.reserve(nodeNum);
//         while(!q.empty()){
//             size_t curr = q.front();
//             q.pop();
//             sortedNodes.push_back(std::move(info.nodes[curr]));

//             // 遍历后继节点
//             for(size_t v:adj[curr]){
//                 --inDegree[v];
//                 if(inDegree[v]==0){
//                     q.push(v);
//                 }
//             }
//         }

//         // 成环校验：排序后节点数必须和原数一致，否则图中存在环，不是合法DAG
//         if (sortedNodes.size() != nodeNum) {
//             std::cerr << "Error: ONNX graph has cycle, not a valid DAG" << std::endl;
//             assert(false && "Graph has cycle");
//         }

//         // 替换为拓扑有序的节点列表
//         info.nodes = std::move(sortedNodes);
//     }
// }

static GraphInfo parseGraph(const onnx::GraphProto &graph){
    GraphInfo info;
    info.name = graph.name();
    // 解析initalizer
    for(const auto &init:graph.initializer()){
        info.initializers[init.name()] = parseTensor(init);
    }
    // 解析node
    for(const auto &node :graph.node()){
        if(node.op_type()=="Constant"){
            for(const auto&attr:node.attribute()){
                // 对应value属性，且为张量proto
                if(attr.name() == "value" && attr.has_t()){
                    TensorInfo t = parseTensor(attr.t());
                    info.initializers[node.output(0)] = t;
                }
            }
        }else{
            info.nodes.push_back(parseNode(node));
        }
    }

    // 解析input和output
    for(const auto &in:graph.input()){
        info.inputs.push_back(parseValue(in));
    }
    for(const auto &out:graph.output()){
        info.outputs.push_back(parseValue(out));
    }

    // 对节点进行拓扑排序，在生成onnx模型的时候通过check检查，一定保证拓扑有序
    // NodeTopology(info);
    return info;
}

ModelInfo parseModelProto(const onnx::ModelProto &model){
    ModelInfo info;
    info.ir_version = model.ir_version();
    info.producer_name = model.producer_name();
    info.graph = parseGraph(model.graph());
    return info;
}
} // namespace tac