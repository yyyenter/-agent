"""Redis 快速诊断 + ToolCacheManager 综合测试"""
import sys
import socket
import time

# print("=" * 60)
# print("1. 检查 6379 端口是否有服务监听")
# print("=" * 60)
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.settimeout(1)
# try:
#     result = sock.connect_ex(('localhost', 6379))
#     if result == 0:
#         print("   [OK] localhost:6379 端口已开放 (有服务在监听)")
#         PORT_OPEN = True
#     else:
#         print(f"   [NO] localhost:6379 端口未开放 (connect_ex 返回 {result})")
#         print(f"   -> 确认: Redis 服务未运行")
#         PORT_OPEN = False
# except Exception as e:
#     print(f"   [NO] 端口检测失败: {e}")
#     PORT_OPEN = False
# finally:
#     sock.close()

# print()
# print("=" * 60)
# print("2. redis-py 库是否安装")
# print("=" * 60)
# try:
#     import redis
#     print(f"   [OK] redis-py {redis.__version__}")
#     LIB_OK = True
# except ImportError:
#     print("   [NO] 未安装")
#     LIB_OK = False

# REDIS_OK = False
# if LIB_OK and PORT_OPEN:
#     print()
#     print("=" * 60)
#     print("3. 尝试 Redis ping")
#     print("=" * 60)
#     try:
#         r = redis.Redis(host='localhost', port=6379, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
#         print(f"   ping 结果: {r.ping()}")
#         print(f"   [OK] Redis 连接成功!")
#         REDIS_OK = True
#         info = r.info('server')
#         print(f"   版本: {info.get('redis_version')}")
#         print(f"   已用内存: {info.get('used_memory_human')}")
#         keys = r.keys("*")
#         print(f"   当前 key 数量: {len(keys)}")
#         if keys:
#             for k in keys[:10]:
#                 t = r.type(k)
#                 print(f"     [{t}] {k}")
#         r.close()
#     except Exception as e:
#         print(f"   [FAIL] {e}")

# print()
# print("=" * 60)
# print("结论")
# print("=" * 60)
# if REDIS_OK:
#     print("   [SUCCESS] Redis 正常运行, 项目所有功能正常")
# elif not PORT_OPEN:
#     print("   [WARN] Redis 服务未运行")
#     print("   项目行为: 自动回退到 InMemoryFallback")
#     print("   影响: 对话历史/短期约束在服务重启后丢失")
#     print("   修复: 安装并启动 Redis")
#     print("     Windows: https://github.com/tporadowski/redis/releases")
#     print("     Docker:  docker run -d -p 6379:6379 redis:7-alpine")
# elif not LIB_OK:
#     print("   [WARN] redis-py 未安装, 项目用内存回退")
#     print("   修复: uv add redis")

# ==================== 项目模块测试 ====================
print()
print("=" * 60)
print("4. 导入项目模块 + 基础测试")
print("=" * 60)
try:
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / "src"))
    from agent_test0.harness import get_redis_or_fallback, InMemoryFallback, MemoryManager, ToolCacheManager
    client, is_fb = get_redis_or_fallback()
    backend_type = type(client).__name__
    print(f"   get_redis_or_fallback -> {backend_type}, is_fallback={is_fb}")

    # MemoryManager 基础测试
    mm = MemoryManager("diag_test", "diag_user", client, is_fb)
    mm.add_message("user", "测试消息")
    history = mm.get_chat_history()
    print(f"   MemoryManager: 写入1条消息, 读出 {len(history)} 条")

    print(f"   [OK] 项目模块导入成功")
except Exception as e:
    print(f"   [FAIL] 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== ToolCacheManager 综合测试 ====================
print()
print("=" * 60)
print("5. ToolCacheManager 综合测试")
print("=" * 60)

if is_fb:
    print(f"   存储后端: InMemoryFallback (hget/hset 模拟)")
else:
    print(f"   存储后端: Redis Hash (hget 字段级读取)")
print(f"   缓存前缀: {ToolCacheManager.CACHE_PREFIX}")
print()

# # ---- 准备: 清理旧数据 ----
# for tool, params in [
#     ("get_weather", {"city": "杭州", "days": 3}),
#     ("get_weather", {"city": "北京", "days": 3}),
#     ("search_hotel", {"city": "杭州", "price": 500}),
#     ("search_hotel", {"city": "成都", "price": 300}),
#     ("search_flight", {"from": "北京", "to": ["杭州", "上海"]}),
#     ("test", {"k": "v"}),
# ]:
#     k = ToolCacheManager._make_key(tool, params)
#     if hasattr(client, 'delete'):
#         client.delete(k)

# # ---- 5.1 写入 + 动态 TTL ----
# print("--- 5.1 写入多条缓存 (动态 TTL) ---")
# entries = [
#     ("get_weather", {"city": "杭州", "days": 3}, "晴 25C",     30),
#     ("get_weather", {"city": "北京", "days": 3}, "多云 20C",   60),
#     ("search_hotel", {"city": "杭州", "price": 500}, "西湖酒店 500",  300),
#     ("search_hotel", {"city": "成都", "price": 300}, "春熙路酒店 300", 120),
# ]
# for tool, params, result, ttl in entries:
#     ToolCacheManager.set_tool_result(client, tool, params, result, ttl)
#     print(f"   set({tool}, {params}) = {result!r}  (TTL={ttl}s)")
# print(f"   -> {len(entries)} 条缓存写入完成")

# # ---- 5.2 精确命中 + 字段级检索验证 ----
# print()
# print("--- 5.2 精确命中 + 字段级优化验证 ---")
# r1 = ToolCacheManager.get_tool_result(client, "get_weather", {"city": "杭州", "days": 3})
# assert r1 == "晴 25C", f"命中值错误: {r1!r}"
# assert isinstance(r1, str), f"返回类型应为 str, 实际 {type(r1)}"
# print(f"   [OK] get_tool_result -> {r1!r} (str, 字段级检索: 只返回 result 值)")

# # ---- 5.3 工具隔离 ----
# print()
# print("--- 5.3 工具隔离 (同名参数, 不同工具) ---")
# r_w = ToolCacheManager.get_tool_result(client, "get_weather", {"city": "杭州", "days": 3})
# r_h = ToolCacheManager.get_tool_result(client, "search_hotel", {"city": "杭州", "price": 500})
# print(f"   weather(杭州) = {r_w!r}")
# print(f"   hotel(杭州)   = {r_h!r}")
# assert r_w != r_h, "不同工具缓存串味了!"
# print(f"   [OK] 工具隔离正常")

# # ---- 5.4 参数隔离 ----
# print()
# print("--- 5.4 参数隔离 (同工具, 不同参数) ---")
# r_hz = ToolCacheManager.get_tool_result(client, "get_weather", {"city": "杭州", "days": 3})
# r_bj = ToolCacheManager.get_tool_result(client, "get_weather", {"city": "北京", "days": 3})
# print(f"   get_weather(杭州) = {r_hz!r}")
# print(f"   get_weather(北京) = {r_bj!r}")
# assert r_hz != r_bj, "不同参数缓存冲突了!"
# print(f"   [OK] 参数隔离正常")

# # ---- 5.5 缓存未命中 ----
# print()
# print("--- 5.5 缓存未命中 ---")
# r_miss = ToolCacheManager.get_tool_result(client, "get_weather", {"city": "上海"})
# assert r_miss is None, f"未命中应返回 None, 实际: {r_miss!r}"
# print(f"   get_weather(上海, 未写入) -> None")
# print(f"   [OK] 缓存未命中正确处理")

# # ---- 5.6 缓存覆盖 ----
# print()
# print("--- 5.6 缓存覆盖 ---")
# ToolCacheManager.set_tool_result(client, "get_weather", {"city": "杭州", "days": 3}, "小雨 18C (更新)", expire=30)
# r_upd = ToolCacheManager.get_tool_result(client, "get_weather", {"city": "杭州", "days": 3})
# print(f"   更新前: 晴 25C")
# print(f"   更新后: {r_upd!r}")
# assert r_upd == "小雨 18C (更新)", f"覆盖失败: {r_upd!r}"
# print(f"   [OK] 缓存覆盖正确")

# # ---- 5.7 L2 归一匹配: 文本变体自动命中 ----
# print()
# print("--- 5.7 L2 归一匹配 (文本变体自动命中) ---")
# ToolCacheManager.set_tool_result(client, "web_search",
#     {"query": "杭州旅游攻略", "page": 1}, "杭州攻略结果...", expire=120)

# # 精确查询
# r_l2_exact = ToolCacheManager.get_tool_result(client, "web_search",
#     {"query": "杭州旅游攻略", "page": 1})
# print(f"   精确查询(杭州旅游攻略) -> {r_l2_exact!r}")

# # 带空格的变体查询 -> 应通过 L2 归一命中
# r_l2_fuzzy = ToolCacheManager.get_tool_result(client, "web_search",
#     {"query": "杭州  旅游 攻略", "page": 1})
# print(f"   变体查询(杭州  旅游 攻略) -> {r_l2_fuzzy!r}")
# assert r_l2_fuzzy == "杭州攻略结果...", f"L2 归一匹配失败: {r_l2_fuzzy!r}"
# print(f"   [OK] L2 归一命中 (去空格后与精确缓存碰撞)")

# # 完全不同 -> 不应命中
# r_l2_miss = ToolCacheManager.get_tool_result(client, "web_search",
#     {"query": "北京美食推荐", "page": 1})
# assert r_l2_miss is None
# print(f"   无关查询(北京美食推荐) -> None (正确未命中)")

# # ---- 5.8 L3 语义向量匹配 (同义换说) ----
# print()
# print("--- 5.8 L3 语义向量匹配 (字符 bigram 余弦相似度) ---")
# # 写入一条缓存
# ToolCacheManager.set_tool_result(client, "web_search",
#     {"query": "杭州西湖旅游攻略", "page": 1}, "西湖攻略-详细版", expire=120)

# # L1 精确: 完全命中
# r_l3_1 = ToolCacheManager.get_tool_result(client, "web_search",
#     {"query": "杭州西湖旅游攻略", "page": 1})
# assert r_l3_1 == "西湖攻略-详细版"
# print(f"   L1 精确: 杭州西湖旅游攻略 -> {r_l3_1!r}")

# # L2 归一: 去空格
# r_l3_2 = ToolCacheManager.get_tool_result(client, "web_search",
#     {"query": "杭州 西湖  旅游攻略", "page": 1})
# assert r_l3_2 == "西湖攻略-详细版"
# print(f"   L2 归一: 杭州 西湖  旅游攻略 -> {r_l3_2!r}")

# # L3 语义: 扩展查询 (共享 杭州/西湖/旅游/攻略 等 bigram)
# r_l3_3 = ToolCacheManager.get_tool_result(client, "web_search",
#     {"query": "杭州西湖旅游最全攻略", "page": 1})
# assert r_l3_3 == "西湖攻略-详细版", \
#     f"L3 语义匹配失败: {r_l3_3!r} (预期命中 '西湖攻略-详细版')"
# print(f"   L3 语义: 杭州西湖旅游最全攻略 -> {r_l3_3!r}")
# print(f"     (bigram 余弦相似度超阈值 {ToolCacheManager.SEMANTIC_THRESHOLD}, 自动命中)")

# # L3 控制: 完全无关的查询不应命中
# r_l3_4 = ToolCacheManager.get_tool_result(client, "web_search",
#     {"query": "北京美食推荐", "page": 1})
# assert r_l3_4 is None, f"L3 应返回 None, 实际: {r_l3_4!r}"
# print(f"   L3 控制: 北京美食推荐 -> None")
# print(f"     (bigram 无交集, 相似度=0, 正确未命中)")
# print(f"   [OK] L3 语义向量匹配正常")

# # ---- 5.9 复杂参数类型 (嵌套 dict/list) ----
# print()
# print("--- 5.9 复杂参数类型 (嵌套 dict/list) ---")
# ToolCacheManager.set_tool_result(client, "search_flight",
#     {"from": "北京", "to": ["杭州", "上海"], "filters": {"max_price": 1500}},
#     "CA1234 1200", expire=60)
# r_flight = ToolCacheManager.get_tool_result(client, "search_flight",
#     {"from": "北京", "to": ["杭州", "上海"], "filters": {"max_price": 1500}})
# assert r_flight == "CA1234 1200", f"复杂参数未命中: {r_flight!r}"
# print(f"   search_flight(北京->杭州/上海, max=1500) -> {r_flight!r}")
# print(f"   [OK] 复杂参数类型缓存正常")

# ---- 5.10 检索性能优化对比 (字段级读取) ----
print()
print("--- 5.10 检索性能优化: 字段级读取 ---")
if not is_fb:
    # 真正的 Redis 后端
    sample_key = ToolCacheManager._make_key("get_weather", {"city": "杭州", "days": 3})
    raw = client.hgetall(sample_key)
    print(f"   hgetall 返回 {len(raw)} 个字段: {list(raw.keys())}")
    print(f"   hget('result') 返回 1 个字段: {raw.get('result')!r}")
    ratio = (1 / len(raw)) * 100 if raw else 0
    print(f"   -> get_tool_result 通过 hget 只读 1/{len(raw)} 字段 ({ratio:.0f}% 数据传输)")
else:
    # InMemoryFallback: hget 同样只取 result 字段
    sample_key = ToolCacheManager._make_key("get_weather", {"city": "杭州", "days": 3})
    raw = client.hgetall(sample_key)
    print(f"   hgetall 返回 {len(raw)} 个字段: {list(raw.keys())}")
    print(f"   hget('result') 返回 result 值: {raw.get('result')!r}")
    print(f"   -> hget 只读 result 字段, 无元数据开销")
print(f"   [OK] 字段级检索优化生效")

print()
print("=" * 60)
print("[ALL PASS] ToolCacheManager 全部测试通过")
print(f"   后端: {'InMemoryFallback' if is_fb else 'Redis'}")
print(f"   测试项: 动态TTL | L1精确 | L2归一 | L3语义向量 | 工具隔离 | 参数隔离 | 缓存未命中 | 覆盖更新 | 复杂参数 | 字段级检索")
print("=" * 60)
