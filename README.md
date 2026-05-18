# AgentTest0 - 智能旅游规划系统

基于 CrewAI 的多智能体旅游规划系统，使用 ReAct 模式的 Flow 作为调度核心。

## 功能特性

- **多智能体协作** - PlannerCrew, TravelExpertCrew, ValidatorCrew 三层协同
- **ReAct 循环** - 决策 → 执行 → 质检的闭环收敛机制
- **智能记忆系统** - 四级存储（Redis + SQLite），支持长期偏好学习
- **天气查询集成** - 和风天气 API，带缓存机制
- **流式 API** - SSE 实时返回进度和结果

## 安装

确保已安装 Python >=3.10 <3.14。本项目使用 [UV](https://docs.astral.sh/uv/) 管理依赖。

```bash
# 安装 uv（如未安装）
pip install uv

# 同步依赖
uv sync
```

### 环境配置

在项目根目录创建 `.env` 文件：

```env
# GLM 智谱 AI
GLM_API_KEY=your_api_key
GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4/
GLM_MODEL_NAME=glm-4-flash

# 和风天气
QWEATHER_API_KEY=your_key
QWEATHER_API_HOST=geoapi.qweather.com

# Redis（可选，不配置则回退到内存存储）
REDIS_HOST=localhost
REDIS_PORT=6379
```

## 运行项目

### 启动 API 服务

```bash
uv run python src/agent_test0/main.py
```

API 服务将在 `http://0.0.0.0:8000` 启动，提供 `/api/chat_stream` 端点。

### 启动 Streamlit 前端

```bash
uv run streamlit run src/agent_test0/app_ui.py
```

### 测试 API

```bash
uv run python src/client_test.py
```

### CrewAI CLI 测试

```bash
crewai test -n 5
```

## 架构说明

### 三层架构

1. **前端** - Streamlit 界面，渲染聊天界面
2. **API 网关** - FastAPI 服务，负责意图路由和记忆管理
3. **智能体调度** - CrewAI Flow 编排多智能体协作

### Flow 执行链路

```
用户消息 → plan_steps (PlannerCrew: 复杂度判定 + 偏好提取)
         → execute_step (TravelExpertCrew: 调研 + 起草)
         → validate_router (ValidatorCrew: 终审关卡)
         → 质检不通过则回环修正，通过则 END
```

### 记忆系统

| 层级 | 存储介质 | 内容 |
|------|----------|------|
| 情节记忆 | Redis List | 原始对话轮次 |
| 工作记忆 | Redis Hash | 当前行程的临时约束 |
| 工具缓存 | Redis KV | 全局工具结果缓存 |
| 语义记忆 | SQLite | 用户长期偏好 |

## 配置文件

- `src/agent_test0/config/agent.yaml` - Agent 定义
- `src/agent_test0/config/tasks.yaml` - PlannerCrew 任务
- `src/agent_test0/config/research_tasks.yaml` - TravelExpertCrew 任务
- `src/agent_test0/config/logic_validator_tasks.yaml` - ValidatorCrew 任务

## 关键工具

- `WeatherTool` - 和风天气查询（带 Redis 缓存）
- `ReadMemoryTool` - 读取用户长期偏好
- `SaveMemoryTool` - 保存用户长期偏好
- `ToolCacheManager` - 三层匹配工具缓存（L1/L2/L3）

## 支持

- [CrewAI 文档](https://docs.crewai.com)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
- [CrewAI Discord](https://discord.com/invite/X4JWnZnxPb)
