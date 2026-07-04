# agent_test0/workflow/ask_user.py
"""
AskUser 中断机制 + 分层候选池 + 混合检测.

【三大能力】

  1. request_user_input(state, field, question)  ← 三层护栏, 追问入口
     - 拦截 ①: 候选池归一化 (LLM 造 key 漂移 → unknown)
     - 拦截 ②: asked_fields 去重
     - 拦截 ③: max_asks 硬上限
     任一层拦截返回 False, 走 assumption 兜底; 全过才真正中断.

  2. 分层候选池 (BASE + 7 个 DOMAIN + UNKNOWN)
     - BASE: 永远暴露给 LLM (8 个必答字段)
     - DOMAIN: 按用户消息触发暴露 (5-6 个/领域, 共 31 个)
     - UNKNOWN: 兜底 (LLM 说不清时归一化落点)

  3. 混合领域检测 (关键词 + LLM 兜底)
     - 第 ①层 关键词硬匹配: 免费高速, 覆盖 70% 直白表达
     - 第 ②层 LLM 语义分类: 关键词漏检时才调, 覆盖 "陪爸妈" "全家几代人" 等语义变体
     - 短消息 / 已识别多领域时不调 LLM (省成本)

【为什么这么设计】
  - LLM 造 key 漂移: 归一化到候选池, 落不到就 → unknown
  - 关键词收不全中文: 分类器 LLM 兜底
  - 候选池爆炸: 分层 + 按触发暴露, 每次 prompt 平均只暴露 ~14 个 key
  - 无重复保证: 同义词表启动时自检, 冲突立即崩
"""
from __future__ import annotations

import re


# ============================================================
# 分层候选池: BASE (必答) + DOMAIN (触发) + UNKNOWN (兜底)
# ============================================================

# 基础字段: 任何行程都可能需要, 永远暴露给 LLM
BASE_FIELDS: dict[str, str] = {
    "destination":     "目的地城市/国家",
    "trip_days":       "行程天数",
    "budget":          "预算金额",
    "group_size":      "出行人数",
    "departure_date":  "出发日期",
    "focus":           "核心关注 (美食/风景/购物/文化 ...)",
    "pace":            "行程节奏 (紧凑/悠闲)",
    "meal_pref":       "饮食偏好或禁忌",
}

# 领域字段: 用户消息触发对应领域时才暴露给 LLM
DOMAIN_FIELDS: dict[str, dict[str, str]] = {
    "companion": {
        "has_elderly":    "是否带老人",
        "elderly_age":    "老人年龄段",
        "elderly_health": "老人健康/行动便利",
        "has_child":      "是否带儿童",
        "child_age":      "儿童年龄",
    },
    "pet": {
        "has_pet":        "是否带宠物",
        "pet_type_size":  "宠物类型和体型",
        "pet_carrier":    "是否有航空箱/宠物托运准备",
    },
    "activity": {
        "dive_cert_level":     "潜水证书等级",
        "ski_level":           "滑雪水平",
        "hiking_difficulty":   "徒步/登山难度偏好",
        "altitude_ok":         "高原适应能力",
        "has_gear":            "是否自带装备",
        "physical_condition":  "整体体能状况",
    },
    "medical": {
        "pregnancy_stage":     "孕期阶段",
        "chronic_illness":     "慢性病类型",
        "medication_needed":   "是否需要冷藏药物",
        "wheelchair_needed":   "是否需要无障碍设施",
    },
    "international": {
        "passport_status":       "护照持有情况",
        "visa_status":           "签证办理情况",
        "foreign_language_ok":   "外语能力",
        "roaming_needed":        "是否需要国际漫游",
    },
    "logistics": {
        "accommodation_pref":     "住宿档次",
        "accommodation_type":     "住宿类型 (酒店/民宿/青旅)",
        "transport_mode":         "交通方式偏好",
        "car_rental_needed":      "是否需要租车",
        "checkin_flexibility":    "入住时间是否灵活",
    },
    "occasion": {
        "special_occasion":  "场合类型 (蜜月/生日/求婚/毕业/团建)",
        "occasion_mood":     "偏好氛围 (浪漫/热闹/安静)",
        "surprise_element":  "是否需要惊喜安排",
        "budget_flexible":   "预算是否可为特殊场合上浮",
    },
}

# 兜底
UNKNOWN_FIELD = "unknown"

# 全集: 供 _normalize_field 查找
_ALL_FIELDS: set[str] = set(BASE_FIELDS.keys()) | {
    k for d in DOMAIN_FIELDS.values() for k in d.keys()
} | {UNKNOWN_FIELD}


# ============================================================
# 同义词表 + 概念到领域映射
# ============================================================

# 概念 → 同义词列表. 每个词只允许在一个概念里出现 (启动时自检).
# 加词: 在对应概念下加一行; 加概念: 加一条 + 更新 CONCEPT_TO_DOMAIN.
CONCEPT_SYNONYMS: dict[str, list[str]] = {
    # ─── 同行者 ─────────────────────────────
    "老人":       ["老人", "长辈", "父母", "爸妈", "老年人", "爷爷", "奶奶",
                   "外公", "外婆", "老爷子", "老妈", "老爸", "退休"],
    "儿童":       ["儿童", "小孩", "孩子", "娃", "小朋友", "宝宝", "婴儿",
                   "幼儿", "带娃"],
    "宠物":       ["宠物", "狗", "猫", "小狗", "小猫", "汪星人", "喵星人"],

    # ─── 高专业度活动 ────────────────────────
    "潜水":       ["潜水", "浮潜", "深潜", "水肺"],
    "滑雪":       ["滑雪", "单板滑雪", "双板滑雪", "雪场"],
    "登山":       ["登山", "徒步", "爬山", "户外徒步", "trekking"],
    "高原":       ["高原", "藏区", "西藏", "青海", "高反", "高山反应"],

    # ─── 医疗健康 ──────────────────────────
    "怀孕":       ["怀孕", "孕妇", "孕期", "孕早期", "孕中期", "孕晚期"],
    "慢性病":     ["慢性病", "高血压", "糖尿病", "心脏病", "心血管", "哮喘"],
    "过敏":       ["过敏", "食物过敏", "海鲜过敏"],
    "残障":       ["残疾", "无障碍", "轮椅", "行动不便"],

    # ─── 出境游 ────────────────────────────
    "出境":       ["出境", "境外", "国外", "出国", "护照", "签证"],
    "国外目的地": ["日本", "韩国", "泰国", "越南", "新加坡", "马来西亚",
                   "欧洲", "美国", "澳大利亚", "东南亚"],

    # ─── 交通住宿 ──────────────────────────
    "住宿":       ["酒店", "民宿", "青旅", "客栈", "度假村"],
    "自驾":       ["自驾", "自驾游", "开车", "租车"],

    # ─── 特殊场合 ──────────────────────────
    "蜜月":       ["蜜月", "度蜜月", "新婚"],
    "生日":       ["生日", "庆生"],
    "求婚":       ["求婚", "订婚"],
    "毕业":       ["毕业", "毕业游", "毕业旅行"],
    "团建":       ["团建", "公司团建", "员工团建"],
}

# 概念 → 领域 (映射到 DOMAIN_FIELDS 的 key)
CONCEPT_TO_DOMAIN: dict[str, str] = {
    "老人": "companion",   "儿童": "companion",   "宠物": "pet",
    "潜水": "activity",    "滑雪": "activity",    "登山": "activity",   "高原": "activity",
    "怀孕": "medical",     "慢性病": "medical",   "过敏": "medical",     "残障": "medical",
    "出境": "international",  "国外目的地": "international",
    "住宿": "logistics",   "自驾": "logistics",
    "蜜月": "occasion",    "生日": "occasion",    "求婚": "occasion",
    "毕业": "occasion",    "团建": "occasion",
}


def _validate_no_duplicate_synonyms() -> None:
    """启动时校验: 同一个词不能出现在多个概念里, 否则报错拒绝启动."""
    seen: dict[str, str] = {}   # word → 首次出现的 concept
    for concept, syns in CONCEPT_SYNONYMS.items():
        for word in syns:
            if word in seen:
                raise ValueError(
                    f"[ask_user] 同义词冲突: '{word}' 同时在概念 "
                    f"'{seen[word]}' 和 '{concept}' 里, 请修正 CONCEPT_SYNONYMS."
                )
            seen[word] = concept
    # 顺便校验 CONCEPT_TO_DOMAIN 覆盖率
    missing = set(CONCEPT_SYNONYMS.keys()) - set(CONCEPT_TO_DOMAIN.keys())
    if missing:
        raise ValueError(
            f"[ask_user] CONCEPT_TO_DOMAIN 未覆盖概念: {missing}"
        )
    orphan = set(CONCEPT_TO_DOMAIN.values()) - set(DOMAIN_FIELDS.keys())
    if orphan:
        raise ValueError(
            f"[ask_user] CONCEPT_TO_DOMAIN 映射到不存在的领域: {orphan}"
        )


_validate_no_duplicate_synonyms()   # 模块加载时立即跑


# ============================================================
# 领域检测: 关键词 + LLM 混合
# ============================================================

def _detect_concepts_by_keyword(message: str) -> set[str]:
    """第 ①层: 扫消息, 关键词匹配返回被激活的概念集合."""
    active: set[str] = set()
    if not message:
        return active
    for concept, syns in CONCEPT_SYNONYMS.items():
        for word in syns:
            if word in message:   # 简单子串匹配
                active.add(concept)
                break             # 命中一个就够, 跳到下一概念
    return active


def _detect_concepts_by_llm(message: str, keyword_domains: set[str]) -> set[str]:
    """第 ②层: LLM 语义分类 (只在关键词层不足时调用).

    Returns: LLM 补充的**领域**集合 (不是概念, 直接是领域名).
             失败时返回空 set, 由调用方自然降级.
    """
    # 延迟 import 避免循环依赖
    try:
        from agent_test0.workflow.structured import call_structured, load_task_prompt
        from agent_test0.workflow.state import DomainClassifierOutput
    except ImportError as e:
        print(f"[DomainClassifier] import 失败, 降级到关键词: {e}")
        return set()

    try:
        slots = {
            "message": message,
            "precomputed_domains": ", ".join(sorted(keyword_domains)) or "(无)",
        }
        prompt = load_task_prompt("domain_classifier_task", slots)
        out: DomainClassifierOutput = call_structured(
            prompt, model_cls=DomainClassifierOutput, temperature=0.1
        )
        llm_domains = {d for d in (out.active_domains or []) if d in DOMAIN_FIELDS}
        new_domains = llm_domains - keyword_domains
        if new_domains:
            print(f"[DomainClassifier] LLM 补充激活: {new_domains} (reason={out.reasoning[:80]})")
        return llm_domains
    except Exception as e:
        print(f"[DomainClassifier] LLM 调用失败, 只用关键词: {e}")
        return set()


def _active_domains(message: str, use_llm: bool = True) -> set[str]:
    """混合检测: 关键词优先, LLM 兜底.

    Args:
        message:  用户消息
        use_llm:  是否允许调 LLM 兜底 (测试时可关闭)

    Returns: 被激活的领域名集合 (DOMAIN_FIELDS 的 key)
    """
    if not message or len(message.strip()) < 3:
        return set()   # 超短消息不做领域推断

    # ─── 第 ①层: 关键词 ───
    concepts = _detect_concepts_by_keyword(message)
    domains = {CONCEPT_TO_DOMAIN[c] for c in concepts if c in CONCEPT_TO_DOMAIN}

    # 短消息 (< 10 字) 或已识别多领域 → 不调 LLM
    if not use_llm:
        return domains
    if len(message.strip()) < 10:
        return domains
    if len(domains) >= 3:
        return domains   # 已经识别足够, 不再耗 LLM

    # ─── 第 ②层: LLM 兜底 (只在关键词漏检时) ───
    llm_domains = _detect_concepts_by_llm(message, domains)
    return domains | llm_domains


# ============================================================
# 候选池组装: 给 planner prompt 用
# ============================================================

def build_field_pool(state) -> str:
    """根据用户消息动态构造给 LLM 的候选池说明.

    格式:
      【基础字段 (永远暴露)】
        - key: 说明

      【<领域> 相关 (用户消息触发)】
        - key: 说明

      【兜底】
        - unknown: 说不清具体缺什么时用
    """
    lines: list[str] = ["【基础字段 (任何行程都可能需要)】"]
    for k, desc in BASE_FIELDS.items():
        lines.append(f"  - {k}: {desc}")

    active = _active_domains(state.message or "")
    # ★ dismissed 域: 用户在 scoping 菜单里没选, 就别在 prompt 里骚扰 LLM 追问.
    #   但如果用户本轮消息 **重新明确提到** 该领域, active 里会包含它 → 尊重用户意图, 让它复活.
    #   实现: 只剔除"没被本轮 active 命中"的 dismissed, 换句话说 active 优先.
    dismissed = set(getattr(state, "dismissed_domains", []) or [])
    still_dismissed = dismissed - active   # 本轮没提到, 保持消默
    active = active - still_dismissed      # 从可暴露池里去掉这些
    if active:
        for domain in sorted(active):
            fields = DOMAIN_FIELDS.get(domain, {})
            if not fields:
                continue
            lines.append(f"\n【{domain} 相关 (用户消息触发)】")
            for k, desc in fields.items():
                lines.append(f"  - {k}: {desc}")

    lines.append("\n【兜底】")
    lines.append(f"  - {UNKNOWN_FIELD}: 说不清具体缺什么, 但确实要问用户")

    # 追加"已问过"提示
    if getattr(state, "asked_fields", None):
        lines.append(f"\n【本会话已问过 (不要再问)】: {', '.join(state.asked_fields)}")

    return "\n".join(lines)


# ============================================================
# 主动 scoping: 领域菜单 (Clarification-First + Suggested Dimensions)
# ============================================================

# 每个领域用 1 个短标题 + 1-2 个触发词提示. LLM/关键词层不需要, 这只是给用户看的.
# 第 1 位 (domain_key) 必须是 DOMAIN_FIELDS 的 key, 便于按 domain 名过滤.
_DOMAIN_MENU: list[tuple[str, str, str, str]] = [
    # (domain_key,     emoji, title,           user_hint)
    ("companion",      "🧓", "带老人 / 儿童",  "回复 '带娃' / '带老人' / '带全家'"),
    ("pet",            "🐕", "带宠物",         "回复 '带狗' / '带猫'"),
    ("occasion",       "🎂", "特殊场合",       "回复 '蜜月' / '生日' / '求婚' / '毕业' / '团建'"),
    ("activity",       "🏔",  "高强度活动",     "回复 '潜水' / '滑雪' / '徒步' / '高原'"),
    ("medical",        "♿", "特殊需求 (健康/无障碍)", "回复 '孕妇' / '轮椅' / '慢性病' / '过敏'"),
    ("international",  "🌏", "出境游",         "回复 '出境' / '国外'"),
    ("logistics",      "🏨", "住宿 / 交通偏好", "回复 '民宿' / '自驾' / '青旅'"),
]


def build_scoping_menu(
    destination: str = "",
    show_domains: set[str] | None = None,
    already_triggered: set[str] | None = None,
) -> str:
    """生成给用户的领域菜单 (主动 scoping).

    Args:
        destination:        用户目的地 (用于菜单开头个性化文本)
        show_domains:       只列出这些领域. None = 全列.
        already_triggered:  用户消息里已触发的领域, 会在菜单头部以"已识别"提示,
                            不再列在选项里 (系统会通过 request_user_input 走细化追问).

    参考: OpenAI Deep Research clarification / Perplexity Focus / v0 Plan Preview.
    """
    dest_hint = f"{destination}行程收到, " if destination else "收到, "
    header_parts = [
        f"{dest_hint}我可以按常规方案 (2 位成人 / 中等预算 / 无特殊需求) 直接开工."
    ]

    # 已识别领域提示: 让用户知道系统已捕获, 不会漏
    if already_triggered:
        triggered_titles = [
            f"{emoji} {title}"
            for key, emoji, title, _ in _DOMAIN_MENU if key in already_triggered
        ]
        if triggered_titles:
            header_parts.append(
                f"**已识别到**: {' / '.join(triggered_titles)} — 稍后我会针对性追问细节."
            )

    header_parts.append("下面这些您也可以补充 (完全可选, 回复关键词即可):")

    NL = "\n"
    lines: list[str] = [NL.join(header_parts)]
    for key, emoji, title, hint in _DOMAIN_MENU:
        if show_domains is not None and key not in show_domains:
            continue
        if already_triggered and key in already_triggered:
            continue   # 已识别的不再列在选项里
        lines.append(f"  {emoji} **{title}** — {hint}")
    lines.append(NL + "如无特别需求, 直接回复 **'开工'** 或 **'看你安排'**, 我按常规继续.")
    return NL.join(lines)


def offered_domains_in_menu(
    show_domains: set[str] | None = None,
    already_triggered: set[str] | None = None,
) -> set[str]:
    """返回本次菜单实际展示给用户的领域集合 (即"未选择即消默"的候选集).

    用法: Planner 推完菜单后, 把这个集合塞进 state.pending_scoping_options,
    下一轮用户消息用 consume_scoping_reply 消费 — 用户提到的复活为触发,
    没提到的加入 state.dismissed_domains, 后续不再打扰.
    """
    offered = set()
    for key, _, _, _ in _DOMAIN_MENU:
        if show_domains is not None and key not in show_domains:
            continue
        if already_triggered and key in already_triggered:
            continue
        offered.add(key)
    return offered


def consume_scoping_reply(state) -> tuple[set[str], set[str]]:
    """消费上一轮 scoping 菜单的候选集 (基于本轮 state.message 判定).

    对 state.pending_scoping_options 里的每个 domain:
      - 用户消息 (关键词层) 命中该 domain → "选中", 从 pending 里出, 走细化追问
      - 用户消息未命中 → "未选择", 加入 state.dismissed_domains, 后续不再打扰

    副作用: 清空 state.pending_scoping_options, 更新 state.dismissed_domains.

    Returns:
        (selected: set[str], dismissed_this_round: set[str])
        - selected: 用户本轮主动选中/提到的领域
        - dismissed_this_round: 用户本轮未选择而被消默的领域

    调用位置: planner_node 开头, 早于其他 scoping 逻辑.
    """
    pending = set(state.pending_scoping_options or [])
    if not pending:
        return set(), set()

    triggered_concepts = _detect_concepts_by_keyword(state.message or "")
    triggered_domains = {
        CONCEPT_TO_DOMAIN[c] for c in triggered_concepts if c in CONCEPT_TO_DOMAIN
    }

    selected = pending & triggered_domains
    dismissed = pending - triggered_domains

    # 副作用: 消费掉 pending, 累积 dismissed
    state.pending_scoping_options = []
    existing_dismissed = set(state.dismissed_domains or [])
    state.dismissed_domains = list(existing_dismissed | dismissed)

    if selected or dismissed:
        print(
            f"[Scoping] 消费上一轮菜单: 选中={selected}, 消默={dismissed}, "
            f"累计 dismissed={state.dismissed_domains}"
        )
    return selected, dismissed



# ============================================================
# 归一化: 把 LLM 造的 key 收进候选池并集, 否则 → unknown
# ============================================================

def _normalize_field(field: str) -> str:
    """把任意字符串归一化到候选池 (BASE ∪ DOMAIN ∪ {unknown}).

    - 空串/None → "unknown"
    - 命中候选池 → 原样
    - 未命中 → "unknown" (LLM 造词漂移的兜底)
    """
    if not field:
        return UNKNOWN_FIELD
    field = field.strip()
    if not field:
        return UNKNOWN_FIELD
    # 大小写和空格不敏感 (LLM 可能大写)
    key = field.lower()
    return key if key in _ALL_FIELDS else UNKNOWN_FIELD


# 向后兼容: 老代码可能 import ASKABLE_FIELDS
ASKABLE_FIELDS: set[str] = _ALL_FIELDS


# ============================================================
# 异常: 硬中断路径 (老飞书 bot 顶层用)
# ============================================================

class AskUserInterrupt(Exception):
    """节点抛出后 Flow 顶层捕获, 立即终止本轮."""

    def __init__(self, question: str, blocking_field: str = ""):
        self.question = question
        self.blocking_field = blocking_field
        super().__init__(question)


# ============================================================
# 主入口: 三层护栏
# ============================================================

def request_user_input(state, field: str, question: str) -> bool:
    """任何节点想追问用户, 都走这个函数.

    Args:
        state:    TravelState (Pydantic, 就地修改)
        field:    缺失字段的结构化 key (会被归一化)
        question: 给用户看的中文问题文本

    Returns:
        True  —— 真正中断了本轮, 调用方应立即 return _dirty(state)
        False —— 三层护栏之一拦截了, 调用方继续 (assumptions 已写入)

    副作用:
        真正中断时:
          - state.needs_user_input = True
          - state.user_question / final_report = question
          - state.asked_fields 追加 normalized field
        被拦截时:
          - state.assumptions 追加 "未获取 X, 采用默认"
    """
    norm = _normalize_field(field)

    # 拦截 ①/②: 已问过同一 field → 走假设
    if norm in state.asked_fields:
        state.assumptions.append(f"用户未提供 {norm}, 已问过一次, 采用合理默认")
        print(f"[AskUser] 🛡️ 拦截: {norm} 已问过, 走假设")
        return False

    # 拦截 ③: 追问总次数超上限 → 走假设
    if len(state.asked_fields) >= state.max_asks:
        state.assumptions.append(
            f"用户未提供 {norm}, 已达追问上限 ({state.max_asks} 次), 采用合理默认"
        )
        print(f"[AskUser] 🛡️ 拦截: 已达追问上限 {state.max_asks}, 走假设")
        return False

    # 三层全过 → 真正中断
    state.needs_user_input = True
    state.user_question = question
    state.final_report = question
    state.asked_fields.append(norm)
    print(f"[AskUser] ✋ 中断: field={norm}, asked_fields={state.asked_fields}")
    return True


# ============================================================
# 旧 API (向后兼容, 不推荐在新代码用)
# ============================================================

def ask_user_and_exit(flow, question: str, blocking_field: str = "") -> None:
    """老式硬中断: 写 state + raise AskUserInterrupt (飞书 bot 顶层用)."""
    flow.state.needs_user_input = True
    flow.state.user_question = question
    flow.state.final_report = question

    norm = _normalize_field(blocking_field)
    if norm not in flow.state.asked_fields:
        flow.state.asked_fields.append(norm)

    flow.notify(f"🙋 [AskUser] {question}")
    print(f"[AskUser] blocking_field={norm!r}, asked_fields={flow.state.asked_fields}")

    raise AskUserInterrupt(question, norm)


def has_already_asked(flow, field: str) -> bool:
    """兼容老 API. 新代码用 request_user_input, 它内部去重."""
    return _normalize_field(field) in flow.state.asked_fields


def check_ask_user_hook(flow) -> bool:
    """老式 hook: 检查 state.needs_user_input, 调用方负责自己 return."""
    if flow.state.needs_user_input:
        flow.notify(f"🙋 [AskUser] {flow.state.user_question}")
        flow.state.final_report = flow.state.user_question
        return True
    return False


def set_ask_user_question(flow, question: str = None) -> None:
    """老式: 设置中断标记 (不抛异常). 新代码用 request_user_input."""
    if question is None:
        question = "信息不足，无法继续规划。"
    flow.state.needs_user_input = True
    flow.state.user_question = question
    flow.state.final_report = question

    print(f"\n{'='*60}")
    print(f"[AskUser] 信息不足，需要向用户提问...")
    print(f"{'='*60}")
    flow.notify(f"🙋 [AskUser] {question}")
