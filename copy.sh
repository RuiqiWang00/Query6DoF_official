#!/bin/bash

# 检查是否提供了足够的参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_directory> <destination_directory>"
    exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="$2"

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# 创建目标目录（如果不存在）
mkdir -p "$DEST_DIR"

COUNT=0
TOTAL_DIRS=0

# 计算总共有多少个子目录
for SUB_DIR in "$SOURCE_DIR"/*/; do
    TOTAL_DIRS=$((TOTAL_DIRS + 1))
done

# 遍历源目录下的所有子目录
for SUB_DIR in "$SOURCE_DIR"/*/; do
    # 获取子目录名
    COUNT=$((COUNT + 1))
    DIR_NAME=$(basename "$SUB_DIR")
    
    # 定义目标路径
    DEST_SUB_DIR="$DEST_DIR/$DIR_NAME"
    
    # 使用rsync复制并显示进度
    # rsync -av --progress "$SUB_DIR" "$DEST_SUB_DIR"
    echo "Copying directory $COUNT of $TOTAL_DIRS: $SUB_DIR  $DEST_SUB_DIR"
    cp -r "$SUB_DIR" "$DEST_SUB_DIR"
done