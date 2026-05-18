# start_all.py (部分代码替换)
import subprocess
import sys
import time
import os

def start_services():
    processes = []
    print("🚀 正在启动 AI 行程助手全栈服务...")

    # 确保子进程能识别 src 目录中的模块，并强制颜色和无缓冲输出
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src")
    env["PYTHONUNBUFFERED"] = "1"  # 强制实时输出
    env["FORCE_COLOR"] = "1"       # 强制保留 CrewAI 的彩色排版

    try:
        # 1. 启动 Redis (这里可以保持静默，因为不需要看 Redis 的日志)
        print("🟡 [1/3] 正在启动 Redis 服务器...")
        redis_process = subprocess.Popen(["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(redis_process)
        time.sleep(1)

        # 2. 启动 FastAPI 后端 (⚠️ 关键修改：取消写入文件，直接输出到当前屏幕)
        print("🟡 [2/3] 正在启动 FastAPI 后端 (端口:8000)...")
        print("         👉 后端日志将直接在下方【实时滚动】，请在这里观察 Agent 思考！")
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "agent_test0.main:app", "--host", "0.0.0.0", "--port", "8000"],
            env=env
            # 删除了 stdout=backend_log，让它默认走控制台
        )
        processes.append(backend_process)
        time.sleep(3) 

        # 3. 启动 Streamlit 前端 (保持不变)
        print("🟡 [3/3] 正在启动 Streamlit 前端UI...")
        frontend_process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "src/agent_test0/app_ui.py"],
            env=env
        )
        processes.append(frontend_process)
        
        # 保持主进程运行
        for p in processes:
            p.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 接收到停止信号，正在关闭所有服务...")
    finally:
        for p in processes:
            if p.poll() is None: # 如果进程还在运行
                p.terminate()
        print("👋 服务已全部安全关闭。")

if __name__ == "__main__":
    start_services()