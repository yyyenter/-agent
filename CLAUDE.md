# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 常用命令

```bash
# 安装依赖
uv sync

# 启动 FastAPI 服务（API 监听 8000 端口）
uv run python src/agent_test0/main.py

# 启动 Streamlit 前端界面（需要另开一个终端）
uv run streamlit run src/agent_test0/app_ui.py

# 用预设测试用例测试 API
uv run python src/client_test.py

# 通过 CrewAI CLI 测试
crewai test -n 5

# 训练
crewai train -n 5 -f training.json

# 记忆管理
crewai reset-memories -a          # 重置所有记忆
crewai log-tasks-outputs          # 查看最近的任务输出
```

始终使用 `uv` 管理依赖。项目要求 Python >=3.10, <3.14。

## 架构概览

这是一个基于 CrewAI 的**多智能体旅游规划系统**，使用 ReAct 模式的 Flow 作为调度核心。

### 三层架构

1. **前端** — Streamlit 界面 (`app_ui.py`)，渲染聊天界面，通过 SSE 流式请求调用 FastAPI 后端
2. **API 网关** — FastAPI 服务 (`main.py`)，提供 `/api/chat_stream` 端点。负责意图路由（旅游 vs. 闲聊）、记忆生命周期管理，在线程池中调用 CrewAI Flow 并通过 SSE 回传进度
3. **智能体调度** — CrewAI Flow（`crew.py` 中的 `TravelWorkflow`）按 决策 → 执行 → 质检 的 ReAct 循环编排多智能体协作

### Flow 执行链路（ReAct 闭环）

```
用户消息 → plan_steps (PlannerCrew: 复杂度判定 + 偏好提取)
         → execute_step (TravelExpertCrew: 分层级团队执行调研 + 起草)
         → validate_router (ValidatorCrew: 终审关卡)
         → 质检不通过则回环修正，通过则 END
```

- **PlannerCrew** (`config/tasks.yaml`): 判定任务复杂度，提取目的地/意图，通过 `ReadMemoryTool`/`SaveMemoryTool` 读写用户长期偏好
- **TravelExpertCrew** (`config/research_tasks.yaml`): 分层级流程 — 情报搜集 agent → 行程规划 agent → 内部质检 agent
- **ValidatorCrew** (`config/logic_validator_tasks.yaml`): 最终逻辑审核（通勤时间是否合理、天气是否支持户外活动、预算是否超支）

### 记忆系统 (`harness.py`)

四级存储架构，基于 Redis（不可用时自动回退到内存）+ SQLite：

| 记忆层 | 存储介质 | 内容 |
|--------|----------|------|
| 情节记忆 (桶3) | Redis List | 原始对话轮次 |
| 工作记忆 (桶5) | Redis Hash | 当前行程的临时约束 |
| 工具缓存 (桶4) | Redis KV | 全局工具结果缓存（L1/L2/L3 三层匹配） |
| 语义记忆 (桶6) | SQLite `user_memory` 表 | 用户长期偏好（动态 KV 结构） |

记忆流转链路：
- **Episodic → Working**: LLM 从原始对话中提取当前行程的临时约束
- **Working → Semantic**: LLM 从短期摘要中蒸馏长期特征，持久化到 SQLite

### LLM 与环境配置

- 使用 **GLM（智谱 AI）** 作为 LLM，通过 OpenAI 兼容 API 接入，配置在 `.env` 中
- `main.py` 在启动时将 `GLM_API_KEY`/`GLM_API_BASE`/`GLM_MODEL_NAME` 映射为 `OPENAI_API_KEY`/`OPENAI_API_BASE`/`OPENAI_MODEL_NAME`，使 CrewAI 的 LiteLLM 路由能透明接入智谱
- 每个 Crew 和 Flow 各自通过 `crewai.LLM` 实例化模型连接

### 配置文件

- `src/agent_test0/config/agent.yaml` — 4 个 agent 定义：planner_agent, info_search_agent, itinerary_planner_agent, logic_validator_agent
- `src/agent_test0/config/tasks.yaml` — planning_task（PlannerCrew 用）
- `src/agent_test0/config/research_tasks.yaml` — research_task, drafting_task, validation_task（TravelExpertCrew 用）
- `src/agent_test0/config/logic_validator_tasks.yaml` — validation_task（ValidatorCrew 用）

### 关键自定义工具 (`tools/custom_tool.py`)

- `WeatherTool` — 调用和风天气 API，带 Redis 缓存（5 小时 TTL），支持 L1/L2/L3 三层匹配
- `ReadMemoryTool` / `SaveMemoryTool` — 基于 SQLite 的 KV 读写，用于用户画像的持久化（按 user_id + memory_key 做 upsert）
- `ToolCacheManager` — 全局工具缓存管理器，支持精确匹配(L1)、归一匹配(L2)、语义匹配(L3)

### 环境变量 (`.env`)

```env
# GLM 智谱 AI
GLM_API_KEY=your_api_key
GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4/
GLM_MODEL_NAME=glm-4-flash

# 和风天气
QWEATHER_API_KEY=your_key
QWEATHER_API_HOST=geoapi.qweather.com

# Redis (可选，不配置则回退到内存存储)
REDIS_HOST=localhost
REDIS_PORT=6379

# MySQL (用于长期记忆存储)
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=agent_test0
```

## 注意事项

- `.env` 中包含真实 API Key，绝对不能提交到 Git。`.gitignore` 中已包含 `.env`。
- Agent 配置文件位于 `src/agent_test0/config/agent.yaml`（单数，而非复数 `agents.yaml`）。
- Redis 预期运行在 `localhost:6379`。当 Redis 不可用时，`harness.py` 会自动回退到内存存储（仅会话有效，重启后丢失）。
- `client_test.py` 使用的是 `/api/chat_stream`（SSE 流式）端点，这是正确的。测试脚本可以直接运行。
- Flow 使用 `@start()`, `@listen()`, `@router()` 装饰器定义事件驱动的工作流。
- 记忆系统有四级存储，分别对应不同的生命周期和存储介质。
- 舆情判断依赖本地 Ollama 服务（`http://localhost:11434`）运行的 `nomic-embed-text` 模型进行语义路由。

## 故障排查

### 常见问题

1. **无法连接 Redis**
   - 错误信息: `Connection refused`
   - 解决方案: 启动 Redis 服务 `redis-server`，或等待自动回退到内存存储

2. **天气查询失败**
   - 检查 `QWEATHER_API_KEY` 和 `QWEATHER_API_HOST` 是否在 `.env` 中正确配置
   - 确认网络可以访问和风天气 API

3. **意图路由总是返回 chitchat**
   - 检查 Ollama 是否运行: `curl http://localhost:11434/api/tags`
   - 如需要，重新加载语义路由模型

4. **MySQL 连接失败**
   - 确认 MySQL 服务运行中
   - 检查用户权限和数据库是否存在

5. **Flow 状态丢失**
   - 确认 Redis 正常运行（Flow 状态持久化到 Redis）
   - 或检查 Redis key: `session:<session_id>:flow_state`

## 开发流程

### 添加新功能

1. **修改 Agent 配置**: 编辑 `src/agent_test0/config/agent.yaml`
2. **添加新工具**: 在 `src/agent_test0/tools/custom_tool.py` 中创建新类，继承 `BaseTool`
3. **修改 Workflow**: 在 `src/agent_test0/crew.py` 的 `TravelWorkflow` 中添加新的 `@listen()` 或 `@router()` 方法
4. **更新记忆 schema**: 如需新增记忆类型，修改 `harness.py` 中的 `MemoryManager` 类

### 测试流程

1. 本地测试: `uv run python src/client_test.py`
2. API 测试: 启动服务后访问 `http://localhost:8000/docs` 查看自动生成的 Swagger 文档
3. 前端测试: `uv run streamlit run src/agent_test0/app_ui.py`

### 飞书集成 (`src/agent_test0/feishu/`)

使用 WebSocket 长连接方式，按飞书官方示例实现。

```bash
# 安装依赖
uv add lark-oapi

# 启动飞书机器人（长连接，持续运行）
uv run python -m agent_test0.feishu.long_conn_bot
```

**核心文件**：
- `long_conn_bot.py` — 唯一入口，包含全部逻辑
- `CONNECTION_GUIDE.md` — 🔗 Agent 连接关系可视化说明

**数据流**：飞书消息 → WebSocket → `do_p2_im_message_receive_v1()` → `_call_agent()` → `TravelWorkflow.kickoff()` → `_send_reply()` → 飞书用户
