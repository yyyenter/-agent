import streamlit as st
import requests
import uuid
from datetime import date
import json

# ==================== 1. 页面基本配置与 CSS 美化 ====================
st.set_page_config(page_title="AI行程助手", page_icon="🌍", layout="centered")

# 自定义 CSS 还原截图中的“圆角卡片”和“紫色渐变 AI 规划按钮”
st.markdown("""
<style>
    /* 全局背景和卡片样式 */
    .main-card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* 紫色渐变主按钮样式 */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #5c72ff 0%, #8c52ff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 12px 0px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(140, 82, 255, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(140, 82, 255, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 AI行程助手")
st.caption("个性化多智能体旅游规划系统 (FastAPI + CrewAI)")

# ==================== 2. 初始化 Session State (状态保持) ====================
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []

# 用于绑定表单输入框状态，方便点击快捷卡片时自动填充
if "departure" not in st.session_state:
    st.session_state.departure = "杭州"
if "destination" not in st.session_state:
    st.session_state.destination = ""

# ==================== 3. 表单配置区域 ====================
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # 3.1 出发地
    st.session_state.departure = st.text_input(
        "📍 出发地", 
        value=st.session_state.departure, 
        placeholder="从哪里出发？"
    )

    # 3.2 目的地输入
    st.session_state.destination = st.text_input(
        "🔍 目的地", 
        value=st.session_state.destination,
        placeholder="你想去哪里玩？"
    )

    # 3.3 热门目的地卡片推荐 (点击自动填入目的地输入框)
    st.caption("🔥 热门目的地快捷选择")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://images.unsplash.com/photo-1599814420042-30f576e27bd3?auto=format&fit=crop&w=300&q=80", use_container_width=True)
        if st.button("🏞️ 杭州", use_container_width=True):
            st.session_state.destination = "杭州"
            st.rerun()

    with col2:
        st.image("https://images.unsplash.com/photo-1525625293386-3f8f99389edd?auto=format&fit=crop&w=300&q=80", use_container_width=True)
        if st.button("🗼 徐州", use_container_width=True):
            st.session_state.destination = "徐州"
            st.rerun()

    with col3:
        st.image("https://images.unsplash.com/photo-1542856391-010fb87dcfed?auto=format&fit=crop&w=300&q=80", use_container_width=True)
        if st.button("🌊 温州", use_container_width=True):
            st.session_state.destination = "温州"
            st.rerun()

    # 3.4 日期/时间选择
    st.markdown("---")
    date_range = st.date_input(
        "📅 出行日期/时间",
        value=(date.today(), date.today()),
        min_value=date.today(),
    )
    
    # 计算旅行天数
    travel_days = 3
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        travel_days = (end_date - start_date).days + 1

    # 3.5 旅行偏好
    st.markdown("---")
    preferences = st.multiselect(
        "❤️ 旅行偏好",
        ["地道美食", "人文历史", "自然风光", "特种兵行程", "亲子休闲", "预算极简", "小众打卡"],
        default=["地道美食", "自然风光"]
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== 4. 辅助发送函数（核心复用逻辑） ====================
def send_message_to_backend(user_prompt, metadata_payload=None):
    """向后端发送请求，解析 SSE 进度流并渲染"""
    # 立即把用户的提问存入前端历史
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    # 强制重新渲染，让用户的发言立刻气泡化显示在屏幕上
    st.rerun()

# ==================== 5. 渲染对话历史 ====================
# 将历史记录提到了按钮和输入框之上，确保对话自上而下流畅阅读
if st.session_state.messages:
    st.markdown("### 💬 规划详情")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ==================== 6. 交互触发触发源 ====================

# 触发源 A：点击表单中的“AI 规划旅程”紫色大按钮
trigger_ai_plan = st.button("✨ AI规划旅程", use_container_width=True)

# 触发源 B：使用页面最底部的【唯一】输入框手动提问或微调
user_typed_prompt = st.chat_input("你想去哪里玩？或者在这里输入对已有行程的修改意见...")

# --- 统一的请求发送执行块 ---
active_prompt = None
active_metadata = None

if trigger_ai_plan:
    if not st.session_state.destination:
        st.warning("⚠️ 请先输入或选择一个目的地！")
    else:
        pref_str = "、".join(preferences) if preferences else "无特定偏好"
        active_prompt = (
            f"请帮我规划从 【{st.session_state.departure}】 出发，"
            f"去 【{st.session_state.destination}】 玩 【{travel_days}】 天的行程。 "
            f"我的旅行偏好是：【{pref_str}】。"
        )
        active_metadata = {
            "departure": st.session_state.departure,
            "destination": st.session_state.destination,
            "days": travel_days,
            "preferences": preferences
        }

elif user_typed_prompt:
    active_prompt = user_typed_prompt
    active_metadata = {
        "departure": st.session_state.departure,
        "destination": st.session_state.destination or "未知"
    }

# 如果有任何一个触发源被激活，开始调用流式 API
if active_prompt:
    # 立即展示用户气泡并记录
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    
    with st.chat_message("assistant"):
        status_placeholder = st.empty()  # 进度提示占位符
        report_placeholder = st.empty()  # 报告内容占位符
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/api/chat_stream",  # 对应流式接口
                json={
                    "user_id": "test_user_001",
                    "session_id": st.session_state.session_id,
                    "message": active_prompt,
                    "metadata": active_metadata
                },
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            
            final_reply = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        data_str = decoded_line[5:].strip()
                        data = json.loads(data_str)
                        
                        # 实时的步骤状态广播
                        if data['type'] == 'status':
                            status_placeholder.info(data['content'])
                        # 最终生成的完美报告
                        elif data['type'] == 'finish':
                            final_reply = data['content']
                            status_placeholder.empty()  # 清理掉蓝色的“思考中”卡片
                            report_placeholder.markdown(final_reply)
                            
            if final_reply:
                st.session_state.messages.append({"role": "assistant", "content": final_reply})
                st.rerun()  # 重新运行刷新状态
                
        except Exception as e:
            st.error(f"连接后端失败: {str(e)}")