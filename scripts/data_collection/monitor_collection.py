#!/usr/bin/env python3
"""
监控数据采集进度
"""
import time
import subprocess
import os
import sys

def check_process_running():
    """检查采集进程是否在运行"""
    result = subprocess.run(['pgrep', '-f', 'python3 main.py.*test'], 
                          capture_output=True, text=True)
    return result.returncode == 0

def get_latest_progress(log_file):
    """从日志获取最新进度"""
    try:
        result = subprocess.run(['tail', '-100', log_file], 
                              capture_output=True, text=True)
        
        lines = result.stdout.split('\n')
        progress_lines = [line for line in lines 
                         if 'Progress:' in line or 'Processing HS code:' in line or 'records collected' in line]
        
        if progress_lines:
            return progress_lines[-1]
    except Exception as e:
        return f"Error reading log: {e}"
    
    return "No progress info yet..."

def main():
    log_file = 'logs/test_collection_retry.log'
    check_interval = 30  # 每30秒检查一次
    max_runtime = 20 * 60  # 最多运行20分钟
    
    print("="*70)
    print("🔍 测试数据采集监控中...")
    print("="*70)
    print(f"日志文件: {log_file}")
    print(f"检查间隔: {check_interval}秒")
    print(f"预计时间: 12-15分钟")
    print("="*70)
    
    start_time = time.time()
    check_count = 0
    
    while (time.time() - start_time) < max_runtime:
        check_count += 1
        
        # 检查进程
        if not check_process_running():
            print("\n" + "="*70)
            print("✅ 采集进程已完成!")
            print("="*70)
            
            # 显示最后结果
            print("\n最后50行日志:")
            subprocess.run(['tail', '-50', log_file])
            
            # 检查数据文件
            print("\n" + "="*70)
            print("📁 生成的数据文件:")
            subprocess.run(['ls', '-lh', 'data/raw/us_census*.parquet'])
            
            break
        
        # 获取进度
        progress = get_latest_progress(log_file)
        elapsed = (time.time() - start_time) / 60
        timestamp = time.strftime('%H:%M:%S')
        
        print(f"[{timestamp}] ({elapsed:.1f}分钟) {progress.split(' - ')[-1] if ' - ' in progress else progress}")
        
        # 每5次检查显示分隔线
        if check_count % 5 == 0:
            print("-" * 70)
        
        time.sleep(check_interval)
    
    if check_process_running():
        print("\n⚠️  监控超时，但进程仍在运行")
        print("可手动检查: tail -f logs/test_collection.log")
    
    print("\n监控结束")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  监控已手动停止")
        print("采集进程仍在后台运行")
        sys.exit(0)
