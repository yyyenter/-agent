# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 常用命令

```bash
# 安装依赖
uv sync

# 启动 FastAPI 服务（API 监听 8000 端口）
uv run python src/agent_test0/api/server.py
# 旧路径仍兼容：
# uv run python src/agent_test0/main.py

# 启动 Streamlit 前端界面（需要另开一个终端）
uv run streamlit run src/agent_test0/ui/streamlit_app.py

# 同步本地调试（不启 HTTP，命令行交互）
uv run python src/agent_test0/api/debug_cli.py

# 用预设测试用例测试 API（需先启动 server.py）
uv run python tests/integration/client_test.py

# 单元测试 / 集成测试
uv run python tests/test_two_turns.py     # 两轮追问→假设的端到端测试
uv run python tests/test_agent.py         # Workflow 端到端
uv run python tests/test_redis.py         # Redis + 记忆系统冒烟测试

# 飞书长连接 bot（持续运行）
uv run python -m agent_test0.connectors.feishu.bot
# 旧路径仍兼容：
# uv run python -m agent_test0.feishu.long_conn_bot
```

始终使用 `uv` 管理依赖。项目要求 Python >=3.10, <3.14。

## 目录结构

```
src/agent_test0/
├── workflow/              # 状态机（CrewAI Flow）
│   ├── state.py          # Pydantic 状态模型（StepPlan/TravelState 等）
│   ├── llm.py            # 共享 LLM 实例（zhipu_llm + search_tool）
│   ├── crews.py          # 7 个 @CrewBase 类
│   ├── callbacks.py      # Agent 步骤回调与日志
│   ├── parsing.py        # 统一 JSON 解析
│   ├── ask_user.py       # AskUser 中断机制（AskUserInterrupt + asked_fields）
│   ├── nodes.py          # 6 个状态节点的业务逻辑
│   └── flow.py           # TravelWorkflow Flow 主体 + run_for_user 外部入口
├── memory/                # 记忆系统
│   └── manager.py        # MemoryManager（原 harness.py）
├── api/                   # HTTP 入口
│   ├── server.py         # FastAPI + SSE（原 main.py）
│   └── debug_cli.py      # 本地调试 CLI（原 debug_main.py）
├── ui/                    # 前端
│   └── streamlit_app.py  # Streamlit 聊天界面（原 app_ui.py）
├── connectors/            # 第三方对接
│   └── feishu/           # 飞书长连接 bot
│       ├── bot.py        # WebSocket 主循环（原 long_conn_bot.py）
│       ├── README.md
│       └── CONNECTION_GUIDE.md
├── tools/                 # 自定义工具
│   └── custom_tool.py    # WeatherTool / ToolCacheManager 等
├── config/                # YAML 配置
│   ├── agent.yaml        # agent 定义
│   └── *_tasks.yaml      # 各 Crew 的 task 定义
├── eval/                  # 评估系统（独立子项目）
├── crew.py                # 兼容入口（薄壳，re-export workflow/）
├── harness.py             # 兼容薄壳（re-export memory.manager）
├── main.py                # 兼容薄壳（re-export api.server）
├── debug_main.py          # 兼容薄壳（re-export api.debug_cli）
└── feishu/                # 兼容薄壳目录（re-export connectors.feishu）

tests/                     # 测试脚本
├── test_two_turns.py     # 两轮对话端到端
├── test_agent.py         # Workflow 端到端
├── test_workflow.py      # Workflow 单元测试
├── test_redis.py         # Redis + 记忆冒烟测试
└── integration/
    ├── test_streaming.py
    └── client_test.py    # API 客户端测试
```

## 架构概览

基于 CrewAI 的**多智能体旅游规划系统**，核心是 `workflow/flow.py` 中的 `TravelWorkflow` 状态机。

### 三层架构

1. **前端** — Streamlit 界面 (`ui/streamlit_app.py`)，通过 SSE 调用 FastAPI
2. **API 网关** — FastAPI (`api/server.py`)，提供 `/api/chat_stream` 端点。意图路由（旅游 vs. 闲聊）+ 记忆生命周期 + 在线程池中跑 Flow
3. **智能体调度** — `TravelWorkflow` 6 状态机（`workflow/`）

### 状态机执行链路

```
用户消息
  ↓
[1] Planner          (复杂度判定 + 步骤生成 + 缺失信息追问)
  ↓
[2] StepPreparer     (为粗粒度步骤生成工具调用计划)
  ↓
[3] StepExecutor     (执行工具调用，写入 step.result)
  ↓
[4] StepVerifier     (单步骤审核：pass / retry / fail)
  ├── pass    → 下一步骤 / FinalVerifier
  ├── retry   → 重试当前步骤（最多 3 次）
  └── fail    → [5] PartialReplanner (局部重规划)
  ↓
[6] FinalVerifier    (整体审核 → 合成最终报告)
  ↓
final_report → 用户
```

### AskUser 中断机制

任何状态节点发现关键信息缺失，可立即中断本轮 Flow 并向用户提问：

```python
from agent_test0.workflow.ask_user import ask_user_and_exit, has_already_asked

# 一行调用：写状态字段 + 抛 AskUserInterrupt → Flow 顶层捕获 → 立即停止
if not has_already_asked(flow, "trip_days"):
    ask_user_and_exit(flow, "请问想去几天？", blocking_field="trip_days")
else:
    # 已经问过一次，做合理假设
    flow.state.assumptions.append("默认 3 天行程")
```

`asked_fields` 是 state 字段，**跨轮持久化到 Redis**，保证用户回答后下一轮不会重复问。

### 记忆系统 (`memory/manager.py`)

四级存储架构，基于 Redis（不可用时自动回退到内存）+ MySQL：

| 记忆层 | 存储介质 | 内容 |
|--------|----------|------|
| 情节记忆 (桶3) | Redis List | 原始对话轮次 |
| 工作记忆 (桶5) | Redis Hash | 当前行程的临时约束 |
| 工具缓存 (桶4) | Redis KV | 全局工具结果缓存（L1/L2/L3 三层匹配） |
| 语义记忆 (桶6) | MySQL `user_memory` 表 | 用户长期偏好（动态 KV 结构） |

记忆流转链路（`workflow/flow.py:run_for_user` 中确定性管理，**不再作为 Agent tool**）：
- **Episodic → Working**: LLM 从原始对话中提取当前行程的临时约束
- **Working → Semantic**: LLM 从短期摘要中蒸馏长期特征，持久化到 MySQL

### LLM 与环境配置

- 使用 **GLM（智谱 AI）** 作为 LLM，通过 OpenAI 兼容 API 接入
- LLM 实例在 `workflow/llm.py` 中单例化，所有 Crew 共享
- `api/server.py` 在启动时把 `GLM_API_KEY`/`GLM_API_BASE`/`GLM_MODEL_NAME` 映射为 `OPENAI_*` 环境变量，供 LiteLLM 路由

### 配置文件 (`config/`)

- `agent.yaml` — 4 个 agent 定义：planner_agent, info_search_agent, itinerary_planner_agent, logic_validator_agent
- `tasks.yaml` — Planner 任务
- `step_preparer_tasks.yaml` / `executor_tasks.yaml` — StepPreparer / StepExecutor 任务
- `step_validator_tasks.yaml` / `final_validator_tasks.yaml` — StepVerifier / FinalVerifier 任务（liberal pass 策略）
- `replan_tasks.yaml` — PartialReplanner 任务
- `logic_validator_tasks.yaml` — 旧 ValidatorCrew 用，新主路径不再调用

### 关键自定义工具 (`tools/custom_tool.py`)

- `WeatherTool` — 调用和风天气 API，带 Redis 缓存（5 小时 TTL）
- `ToolCacheManager` — 全局工具缓存管理器，支持精确匹配(L1)、归一匹配(L2)、语义匹配(L3)
- ~~`ReadMemoryTool` / `SaveMemoryTool`~~ — 已废弃，记忆读写改为 Flow 确定性管理

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

- `.env` 包含真实 API Key，绝对不能提交。`.gitignore` 中已包含 `.env`
- Agent 配置文件是 `config/agent.yaml`（单数）
- Redis 预期运行在 `localhost:6379`；不可用时 `memory/manager.py` 自动回退到内存
- Flow 只用 `@start()` 作为入口（`plan_steps`），方法体调 `nodes.run_state_machine(self)`；6 状态机由一个显式 `while` 循环驱动（支持 retry/replan 循环语义，且避免 `@listen` 自动传播 + 手动调用并存导致的双触发）。节点函数 `run_xxx(flow)` 是纯函数，只读写 `flow.state`、返回 verdict，不再手动调下游
- 飞书长连接 bot 走 WebSocket 接口，**不**经过 `/api/chat_stream`；FastAPI SSE 仅供 Streamlit 前端
- 意图路由依赖本地 Ollama (`http://localhost:11434`) 运行 `nomic-embed-text`；不可用时降级为关键词匹配

## 故障排查

1. **无法连接 Redis** — 启动 `redis-server`，或等自动回退到内存
2. **天气查询失败** — 检查 `QWEATHER_API_KEY` / `QWEATHER_API_HOST`
3. **意图路由总是 chitchat** — 检查 Ollama: `curl http://localhost:11434/api/tags`
4. **MySQL 连接失败** — 确认 MySQL 服务 + 用户权限 + 数据库存在
5. **Flow 进入死循环** — 检查 `state.total_steps_counted` 计数；新代码用 `MAX_STEP_ITERATIONS` 上限保护

## 开发流程

### 添加新功能

1. **修改 Agent 配置** — `config/agent.yaml`
2. **添加新工具** — `tools/custom_tool.py`，继承 `BaseTool`
3. **添加新 Crew** — `workflow/crews.py` 加 `@CrewBase` 类
4. **修改 Workflow 节点** — `workflow/nodes.py` 改对应 `run_xxx()` 函数
5. **添加新状态字段** — 改 `workflow/state.py`，记得给默认值（旧 session 反序列化要兼容）
6. **更新记忆 schema** — `memory/manager.py` 改 `MemoryManager` 类

### 测试流程

1. 单元/集成测试: `uv run python tests/test_two_turns.py`
2. API 文档: 启 server.py 后访问 `http://localhost:8000/docs`
3. 前端测试: `uv run streamlit run src/agent_test0/ui/streamlit_app.py`

### 飞书集成 (`connectors/feishu/`)

WebSocket 长连接，按飞书官方示例实现。

```bash
uv add lark-oapi
uv run python -m agent_test0.connectors.feishu.bot
```

**核心文件**：
- `connectors/feishu/bot.py` — 主入口（`do_p2_im_message_receive_v1` + `_call_agent` + `_send_reply`）
- `connectors/feishu/CONNECTION_GUIDE.md` — Agent 连接关系图

**数据流**：飞书消息 → WebSocket → `do_p2_im_message_receive_v1()` → `_call_agent()` → `TravelWorkflow.run_for_user()` → `_send_reply()` → 飞书用户
