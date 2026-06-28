#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG_FILE="logs/test_collection_retry.log"

echo "====================================="
echo "测试数据采集监控"
echo "====================================="
echo "开始时间: $(date '+%H:%M:%S')"
echo ""

for i in {1..25}; do
    sleep 30
    
    # 检查进程
    if ! pgrep -f "python3 main.py.*test" > /dev/null; then
        echo ""
        echo "====================================="
        echo "进程已结束！"
        echo "====================================="
        echo ""
        echo "最后日志:"
        tail -30 "$LOG_FILE"
        echo ""
        echo "数据文件:"
        ls -lh data/raw/us_census*.parquet 2>/dev/null | tail -3
        break
    fi
    
    # 显示进度
    PROGRESS=$(tail -50 "$LOG_FILE" | grep -E "Progress:|Processing HS code:" | tail -1)
    TIME=$(date '+%H:%M:%S')
    MINS=$((i/2))
    echo "[$TIME] (${MINS}min) $PROGRESS"
    
    # 每5次显示分隔线
    if (( i % 5 == 0 )); then
        echo "-------------------------------------"
    fi
done

echo ""
echo "监控结束"
