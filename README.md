# AgentTest0 — 智能旅游规划系统

基于 [CrewAI](https://docs.crewai.com) 的多智能体旅游规划系统。核心是一个 6 状态机（Planner → StepPreparer → StepExecutor → StepVerifier → PartialReplanner → FinalVerifier），用显式循环驱动，能在规划过程中发现信息不足时立即向用户追问。支持飞书长连接 bot、FastAPI + SSE、Streamlit 三种前端入口。

## 功能特性

- **6 状态机 + 有界循环** — 规划→准备→执行→审核→(局部重规划)→终审，retry / replan 受计数上限保护，不会死循环
- **AskUser 中断机制** — 任何节点发现关键信息缺失可立即中断本轮并向用户提问，`asked_fields` 跨轮持久化防重复问
- **四级记忆系统** — 情节记忆 / 工作记忆 / 工具缓存（Redis）+ 语义记忆（MySQL），Redis 不可用时自动回退内存
- **天气查询** — 和风天气 API，带 Redis 缓存（5 小时 TTL）
- **多入口** — 飞书长连接 bot、FastAPI + SSE、Streamlit、本地 debug CLI

## 环境要求

- Python >=3.10, <3.14
- [uv](https://docs.astral.sh/uv/) 依赖管理
- 可选外部服务：Redis（记忆/缓存）、MySQL（长期偏好）、Ollama（意图路由，不可用则降级关键词匹配）

## 安装

```bash
# 安装 uv（如未安装）
pip install uv

# 同步依赖
uv sync
```

## 环境配置

在项目根目录创建 `.env` 文件（**切勿提交真实 key**，`.gitignore` 已包含 `.env`）：

```env
# GLM 智谱 AI（必填，LLM）
GLM_API_KEY=your_api_key
GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4/
GLM_MODEL_NAME=glm-4-flash

# 和风天气（天气查询用）
QWEATHER_API_KEY=your_key
QWEATHER_API_HOST=geoapi.qweather.com

# Redis（可选，不配置则回退到内存存储）
REDIS_HOST=localhost
REDIS_PORT=6379

# MySQL（长期偏好存储，可选）
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=agent_test0

# 飞书长连接 bot（仅飞书入口需要）
FEISHU_APP_ID=cli_xxxxxxxxxxxxxxxx
FEISHU_APP_SECRET=your_feishu_app_secret_here
```

## 运行

### 1. 飞书长连接 bot（推荐，持续运行）

```bash
uv run python -m agent_test0.connectors.feishu.bot
```

启动后会打印 `App ID: ...`，确认非 `None` 即配置加载成功。用户在飞书给 bot 发消息 → WebSocket → 状态机 → 回复。

### 2. FastAPI + SSE（供 Streamlit 前端）

```bash
uv run python src/agent_test0/api/server.py
```

API 监听 `http://localhost:8000`，端点 `/api/chat_stream`，文档 `http://localhost:8000/docs`。

### 3. Streamlit 前端（需先启动 server.py）

```bash
uv run streamlit run src/agent_test0/ui/streamlit_app.py
```

### 4. 本地 debug CLI（不启 HTTP，命令行交互）

```bash
uv run python src/agent_test0/api/debug_cli.py
```

## 测试

```bash
uv run python tests/test_two_turns.py     # 两轮追问→假设的端到端测试
uv run python tests/test_agent.py         # Workflow 端到端
uv run python tests/test_redis.py         # Redis + 记忆系统冒烟测试
uv run python tests/integration/client_test.py  # API 客户端测试（需先启 server.py）
```

## 目录结构

```
src/agent_test0/
├── workflow/        # 状态机（CrewAI Flow）：state / llm / crews / nodes / flow / ask_user / parsing / callbacks
├── memory/          # 记忆系统：MemoryManager（Redis + MySQL）
├── api/             # FastAPI + SSE（server.py）、本地 debug CLI
├── ui/              # Streamlit 前端
├── connectors/feishu/  # 飞书长连接 bot
├── tools/           # 自定义工具（WeatherTool / ToolCacheManager）
├── config/          # YAML 配置（agent.yaml + *_tasks.yaml）
└── eval/            # 评估系统（独立子项目）
```

详细的架构、状态机执行链路、开发流程见 [CLAUDE.md](./CLAUDE.md)。

## 架构概览

```
用户消息
  ↓
[1] Planner          复杂度判定 + 步骤生成 + 缺失信息追问
  ↓
[2] StepPreparer     为粗粒度步骤生成工具调用计划
  ↓
[3] StepExecutor     执行工具调用，写入 step.result
  ↓
[4] StepVerifier     单步骤审核：pass / retry / fail
  ├── pass    → 下一步骤 / FinalVerifier
  ├── retry   → 重试当前步骤（最多 3 次）
  └── fail    → [5] PartialReplanner（局部重规划）
  ↓
[6] FinalVerifier    整体审核 → 合成最终报告
  ↓
final_report → 用户
```

## 常见故障排查

1. **飞书 bot 无回复 / `App ID: None`** — `.env` 未加载。从项目根目录启动 bot，或确认 `FEISHU_APP_ID` / `FEISHU_APP_SECRET` 已配置。
2. **无法连接 Redis** — 启动 `redis-server`，或等自动回退到内存（记忆不持久）。
3. **MySQL 连接失败** — 确认 MySQL 服务运行 + 用户权限 + 数据库 `agent_test0` 已创建（语义记忆会静默降级，不影响主流程）。
4. **意图路由总是 chitchat** — 检查 Ollama：`curl http://localhost:11434/api/tags`，不可用时自动降级关键词匹配。
5. **天气查询失败** — 检查 `QWEATHER_API_KEY` / `QWEATHER_API_HOST`。
6. **回复很慢** — 单轮会跑多段 LLM（记忆蒸馏 + 每步状态机 + 报告合成），属正常；若卡死检查 `total_steps_counted` 是否触发 `MAX_STEP_ITERATIONS` 上限保护。

## 支持

- [CrewAI 文档](https://docs.crewai.com)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)
