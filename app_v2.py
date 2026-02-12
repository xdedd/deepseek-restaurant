import streamlit as st
import pandas as pd
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional

# --- 1. 页面配置 (换成火辣的图标) ---
st.set_page_config(page_title="蜀香园 - 智能点餐", page_icon="🌶️", layout="wide")

# --- 2. 定义【四川菜菜单】 ---
# 这里我们换成了川菜，并加入了一些限制条件（比如微辣/特辣）
MENU_CONTEXT = """
【蜀香园 - 今日菜单】
1. 麻婆豆腐 (28元) - 经典川菜，麻辣鲜香。
2. 水煮牛肉 (68元) - 招牌菜，肉质滑嫩，油而不腻。
3. 宫保鸡丁 (38元) - 荔枝味型，甜酸微辣，不吃辣的朋友可选。
4. 鱼香肉丝 (32元) - 酸甜口，下饭神器。
5. 担担面 (18元) - 这里的特色主食。
6. 开水白菜 (888元) - 国宴菜，需提前预定。

注意规则：
- 水煮牛肉和麻婆豆腐默认是【中辣】，无法做【不辣】。
- 宫保鸡丁和鱼香肉丝可以做【免辣】。
- 其他任何不在菜单上的菜（如火锅、烤鱼）都做不了。
"""

# --- 3. 定义管理员密码 ---

ADMIN_PASSWORD = "Secrets"

# --- 4. 数据结构 (保持不变) ---
class Order(BaseModel):
    is_available: bool = Field(description="用户点的菜是否在菜单里？如果在设为True，不在设为False")
    item_name: str = Field(description="标准菜品名称")
    quantity: int = Field(description="数量")
    address: str = Field(description="地址")
    phone: Optional[str] = Field(description="电话")
    notes: Optional[str] = Field(description="备注。如果用户点了菜单没有的菜，请在这里说明'本店没有xx'")

# --- 5. 初始化模型 ---
@st.cache_resource
def get_chain():
    api_key = st.secrets.get("DEEPSEEK_API_KEY")

    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0
    )
    
    parser = PydanticOutputParser(pydantic_object=Order)
    
    template = """
    你是一个【蜀香园】的川菜接单员。
    请根据下面的【菜单】来判断用户的需求。
    
    {menu_context}
    
    如果用户点了菜单里没有的菜，或者提出了无法满足的口味要求（比如要求水煮牛肉不辣），请将 is_available 设为 False，并在 notes 里解释原因。
    请严格按照 JSON 格式输出。
    
    {format_instructions}
    
    用户输入：{query}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    final_prompt = prompt.partial(
        menu_context=MENU_CONTEXT,
        format_instructions=parser.get_format_instructions()
    )
    
    return final_prompt | llm | parser

# --- 6. 数据保存函数 ---
CSV_FILE = "sichuan_orders.csv"

def save_order(order_data):
    data = order_data.dict()
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df_new.to_csv(CSV_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        df_new.to_csv(CSV_FILE, mode='w', header=True, index=False, encoding='utf-8-sig')

# --- 7. 界面布局 ---
st.title("🌶️ 蜀香园 - 智能点餐系统")

tab1, tab2 = st.tabs(["🥢 我要点菜", "🔒 后台管理 (仅老板可见)"])

# === Tab 1: 点餐界面 ===
with tab1:
    with st.expander("📜 查看今日菜单", expanded=True):
        st.markdown(MENU_CONTEXT)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "欢迎光临蜀香园！想吃点什么辣的？"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("例如：一份水煮牛肉，要特辣，送到春熙路..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        chain = get_chain()
        
        with st.chat_message("assistant"):
            with st.status("正在咨询后厨..."):
                try:
                    result = chain.invoke({"query": prompt})
                    
                    if result.is_available:
                        save_order(result)
                        st.success(f"✅ 下单成功！🔥 {result.item_name}")
                        st.session_state.messages.append({"role": "assistant", "content": f"✅ 下单成功：{result.item_name} * {result.quantity} ({result.notes if result.notes else '正常做法'})"})
                    else:
                        st.error("🚫 无法下单")
                        st.warning(f"原因: {result.notes}")
                        st.session_state.messages.append({"role": "assistant", "content": f"🚫 抱歉：{result.notes}"})
                        
                except Exception as e:
                    st.error(f"系统繁忙: {e}")

# === Tab 2: 后台管理 (加锁逻辑) ===
with tab2:
    st.header("🔐 订单数据看板")
    
    # 1. 密码验证框
    password = st.text_input("请输入管理员密码查看数据", type="password")
    
    # 2. 只有密码正确才显示数据
    if password == ADMIN_PASSWORD:
        st.success("身份验证通过 ✅")
        if st.button("刷新最新数据"):
            st.rerun()
            
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            st.dataframe(df, use_container_width=True)
            
            # 统计区域
            col1, col2 = st.columns(2)
            col1.metric("总销售单数", len(df))
            if not df.empty:
                revenue = df['item_name'].apply(lambda x: 888 if "开水白菜" in x else (68 if "水煮牛肉" in x else (28 if "麻婆豆腐" in x else 30))).sum() # 简单估算
                col2.metric("预估流水 (元)", f"¥{revenue}")
            
            with open(CSV_FILE, "rb") as file:
                st.download_button(
                    label="📥 导出 Excel 报表",
                    data=file,
                    file_name="sichuan_orders.csv",
                    mime="text/csv"
                )
        else:
            st.info("暂无订单数据")
    elif password:
        st.error("密码错误 ❌")
    else:
        st.info("请输入密码以解锁数据。")