#!/bin/bash

# GalSim CPP版本数据收集脚本
# 作者：GalSim CUDA测试团队
# 日期：2025-11-28

set -e  # 遇到错误立即退出

echo "=========================================="
echo "GalSim CPP版本数据收集"
echo "=========================================="
echo "开始时间: $(date)"
echo ""

# 进入构建目录
cd "$(dirname "$0")/../build"
echo "当前目录: $(pwd)"

# 清理之前的构建
echo "清理之前的构建文件..."
rm -f CMakeCache.txt 2>/dev/null || true
rm -f *.log 2>/dev/null || true

echo ""
echo "步骤1: 构建CPP版本"
echo "配置CMake (禁用CUDA)..."
cmake .. -DENABLE_CUDA=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "编译项目..."
make -j$(nproc)

echo "安装项目..."
make install

echo ""
echo "步骤2: 准备Python环境"
source /home/wnk/miniconda3/etc/profile.d/conda.sh
conda activate galsim

echo "当前Python版本: $(python --version)"
echo "当前工作目录: $(pwd)"

# 确保结果目录存在
mkdir -p ../results
echo "结果目录: $(pwd)/../results"

echo ""
echo "步骤3: 运行CPP版本测试"
echo "执行数据收集..."

# 执行Python数据收集脚本
cd ..
PYTHONPATH=$(pwd) python scripts/collect_version_data.py CPP

echo ""
echo "步骤4: 验证数据收集结果"
echo "检查生成的文件..."

# 更新验证逻辑以适应新的测试规模
if [ -f "results/cpp_photons_10000.pkl" ] && \
   [ -f "results/cpp_photons_100000.pkl" ] && \
   [ -f "results/cpp_photons_1000000.pkl" ]; then

    echo "✅ CPP版本数据收集完成!"
    echo "生成的文件:"
    ls -la results/cpp_*.pkl

    # 显示文件大小
    echo ""
    echo "文件详情:"
    for file in results/cpp_*.pkl; do
        if [ -f "$file" ]; then
            size=$(du -h "$file" | cut -f1)
            echo "- $(basename $file): $size"
        fi
    done

else
    echo "❌ 数据收集失败，部分文件缺失"
    ls -la results/cpp_*.pkl 2>/dev/null || echo "未找到CPP数据文件"
    exit 1
fi

echo ""
echo "=========================================="
echo "CPP版本数据收集完成"
echo "完成时间: $(date)"
echo "=========================================="