#!/bin/bash
# 输出目录，不存在则创建
mkdir -p ./tac_test/onnx_to_mlir
mkdir -p ./tac_test/onnx_dump

for onnxfile in ./tac_test/onnx_files/*.onnx; do
    if [ -f "$onnxfile" ]; then
        echo "====================================="
        echo "test: $onnxfile"

        # 获取不带路径和后缀的纯文件名
        basename_no_ext=$(basename "$onnxfile" .onnx)
        mlir_out="./tac_test/onnx_to_mlir/${basename_no_ext}.mlir"

        onnx_dump="./tac_test/onnx_dump/${basename_no_ext}_dump.txt"


        # 执行转换程序，--print-mlir 输出MLIR，重定向到mlir文件
        ./build/onnx_parse_test "$onnxfile" --mlir > "$mlir_out"

        ./build/onnx_parse_test "$onnxfile" --dump > "$onnx_dump"

        ./build/tac-opt "$mlir_out"
        echo ""
    fi
done
echo "====================================="
echo "All scripts done."
