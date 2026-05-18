#!/usr/bin/env python
"""
一键启动智能旅游规划系统
包含：Redis 服务、FastAPI 后端、Streamlit 前端
"""
import subprocess
import sys
import time
import os

REDIS_PATH = r"E:\Redis\redis-server.exe"
PROJECT_DIR = r"E:\Python\agent_test0\src"

def start_redis():
    """启动 Redis 服务"""
    print("[1/3] 启动 Redis 服务...")
    try:
        proc = subprocess.Popen([REDIS_PATH],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        time.sleep(2)
        # 检查进程是否还在运行
        if proc.poll() is None:
            print("    [OK] Redis 已启动 (端口 6379)")
            return proc
        else:
            print("    [ERROR] Redis 启动失败")
            return None
    except Exception as e:
        print(f"    [ERROR] 启动 Redis 失败: {e}")
        return None

def start_backend():
    """启动 FastAPI 后端"""
    print("[2/3] 启动 FastAPI 后端服务...")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "agent_test0.main"],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)
        print("    [OK] FastAPI 后端已启动 (端口 8000)")
        print("    访问: http://localhost:8000")
        return proc
    except Exception as e:
        print(f"    [ERROR] 启动后端失败: {e}")
        return None

def start_frontend():
    """启动 Streamlit 前端"""
    print("[3/3] 启动 Streamlit 前端界面...")
    try:
        proc = subprocess.Popen(
            ["streamlit", "run", "agent_test0/app_ui.py"],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)
        print("    [OK] Streamlit 前端已启动 (端口 8501)")
        print("    访问: http://localhost:8501")
        return proc
    except Exception as e:
        print(f"    [ERROR] 启动前端失败: {e}")
        return None

def main():
    print("=" * 50)
    print("智能旅游规划系统 - 一键启动")
    print("=" * 50)

    redis_proc = start_redis()
    time.sleep(1)

    backend_proc = start_backend()
    time.sleep(1)

    frontend_proc = start_frontend()

    print("=" * 50)
    print("所有服务已启动！")
    print(f"  Redis:    localhost:6379")
    print(f"  Backend:  http://localhost:8000")
    print(f"  Frontend: http://localhost:8501")
    print("=" * 50)

    try:
        # 等待所有进程
        if redis_proc:
            redis_proc.wait()
        if backend_proc:
            backend_proc.wait()
        if frontend_proc:
            frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
        if redis_proc:
            redis_proc.terminate()
        if backend_proc:
            backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()

if __name__ == "__main__":
    main()
