#!/bin/bash
# 遍历当前目录下 tac_test 里面所有 .py 文件
for pyfile in ./tac_test/*.py; do
    # 判断是不是普通文件（防止目录匹配）
    if [ -f "$pyfile" ]; then
        echo "====================================="
        echo "Running: $pyfile"

        # 获取py脚本所在目录
        py_dir=$(dirname "$pyfile")
        # 进入py脚本所在目录执行python，这样生成的 onnx_files proto_files 就落在 tac_test/ 下面
        pushd "$py_dir" > /dev/null
        python3 "$(basename "$pyfile")"
        ret=$?
        popd > /dev/null

        # 获取上一条命令退出码
        if [ $ret -ne 0 ]; then
            echo "ERROR: $pyfile failed, exit code $ret"
        else
            echo "SUCCESS: $pyfile finished"
        fi
        echo ""
    fi
done
echo "====================================="
echo "All scripts done."
