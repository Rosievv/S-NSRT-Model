#!/bin/bash
# 激活ARM64虚拟环境的便捷脚本

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
source "$REPO_ROOT/venv_arm64/bin/activate"

echo "✓ 虚拟环境已激活: venv_arm64"
echo ""
echo "已安装的主要包:"
pip list | grep -E "numpy|pandas|scikit-learn|xgboost|lightgbm"
echo ""
echo "使用方法:"
echo "  - 训练模型: python scripts/training/train_baseline_models.py"
echo "  - 退出环境: deactivate"
echo ""

# 如果有参数，则执行命令
if [ $# -gt 0 ]; then
    exec "$@"
else
    # 启动交互式shell
    exec $SHELL
fi
