#!/bin/bash
# 遍历当前目录下所有 .py 文件
for pyfile in ./*.py; do
    # 判断是不是普通文件（防止目录匹配）
    if [ -f "$pyfile" ]; then
        echo "====================================="
        echo "Running: $pyfile"
        python3 "$pyfile"
        # 获取上一条命令退出码
        ret=$?
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
