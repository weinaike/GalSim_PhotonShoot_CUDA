#!/bin/bash

# GalSim CUDA/CPP版本数据比较脚本
# 作者：GalSim CUDA测试团队
# 日期：2025-11-28

set -e  # 遇到错误立即退出

echo "=========================================="
echo "GalSim CUDA/CPP版本数据比较分析"
echo "=========================================="
echo "开始时间: $(date)"
echo ""

# 进入项目根目录
cd "$(dirname "$0")/.."
echo "当前目录: $(pwd)"

echo ""
echo "步骤1: 检查数据文件完整性"
echo "验证所需的数据文件是否存在..."

required_files=(
    "results/cpp_photons_10000.pkl"
    "results/cpp_photons_100000.pkl"
    "results/cpp_photons_1000000.pkl"
    "results/cuda_photons_10000.pkl"
    "results/cuda_photons_100000.pkl"
    "results/cuda_photons_1000000.pkl"
)

missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ 缺少以下数据文件:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "请先运行以下脚本收集数据:"
    echo "  - scripts/1_collect_cpp_data.sh"
    echo "  - scripts/2_collect_cuda_data.sh"
    exit 1
else
    echo "✅ 所有必需的数据文件都存在"
fi

echo ""
echo "步骤2: 准备Python环境"
source /home/wnk/miniconda3/etc/profile.d/conda.sh
conda activate galsim

echo "当前Python版本: $(python --version)"

# 检查依赖包
echo "检查Python依赖包..."
python -c "import scipy, numpy, matplotlib, sklearn" 2>/dev/null && echo "✅ 依赖包检查通过" || echo "⚠️ 部分依赖包缺失，可能影响分析结果"

echo ""
echo "步骤3: 执行版本比较分析"

# 确保results目录存在
mkdir -p results

echo "运行一致性分析脚本..."
python scripts/consistency_analysis.py > results/consistency_analysis.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 一致性分析完成"
    echo "分析结果已保存到: results/consistency_analysis.log"
else
    echo "❌ 一致性分析失败"
    echo "查看错误信息: results/consistency_analysis.log"
    exit 1
fi

echo ""
echo "步骤4: 生成详细比较报告"
echo "创建Markdown格式的比较报告..."

# 生成比较报告
python scripts/generate_comparison_report.py > results/comparison_report.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 比较报告生成完成"
    echo "报告已保存到: report/comparison_report.md"
else
    echo "⚠️ 比较报告生成有问题"
    echo "查看日志: results/comparison_report.log"
fi

echo ""
echo "步骤5: 显示分析结果摘要"
echo "=========================================="
echo "数据文件统计:"
echo "- CPP版本文件: $(ls -1 results/cpp_*.pkl | wc -l) 个"
echo "- CUDA版本文件: $(ls -1 results/cuda_*.pkl | wc -l) 个"

echo ""
echo "生成的报告文件:"
echo "- 一致性分析日志: results/consistency_analysis.log"
echo "- 比较报告日志: results/comparison_report.log"
if [ -f "report/comparison_report.md" ]; then
    echo "- 比较报告: report/comparison_report.md"
fi

echo ""
echo "文件大小统计:"
for file in results/*.pkl; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        name=$(basename "$file" .pkl)
        echo "- $name: $size"
    fi
done

echo ""
echo "=========================================="
echo "版本数据比较分析完成"
echo "完成时间: $(date)"
echo "=========================================="

echo ""
echo "📊 下一步操作建议:"
echo "1. 查看 report/comparison_report.md 获取详细比较结果"
echo "2. 查看 results/consistency_analysis.log 了解分析过程"
echo "3. 如需重新分析，可删除results目录后重新运行此脚本"