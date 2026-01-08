import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import re
from datetime import datetime, date, timedelta
import numpy as np
import glob
from scipy.stats import linregress
import time

# ============================================
# 0. 系統設定 & CSS (全域 24px 大字體優化 + 隱藏帳號)
# ============================================
st.set_page_config(
    page_title="Phoenix V75 帝王匿名版",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    /* 1. 強制放大所有文字元件 */
    html, body, [class*="css"], .stMarkdown, .stDataFrame, .stTable, p, div, input, label, span, button, .stSelectbox, .stRadio {
        font-family: 'Microsoft JhengHei', 'Arial', sans-serif !important;
        font-size: 24px !important;
        line-height: 1.6 !important;
    }
    
    /* 2. 標題特大化 */
    h1 { font-size: 48px !important; font-weight: 900 !important; color: #000; }
    h2 { font-size: 36px !important; font-weight: bold; color: #333; }
    h3 { font-size: 30px !important; font-weight: bold; color: #444; }

    /* 3. 版面間距調整 */
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    
    /* 4. 戰術指導區塊 */
    .tactical-guide {
        background-color: #e3f2fd;
        border-left: 8px solid #2196F3;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        font-size: 22px;
        color: #0d47a1;
        line-height: 1.6;
    }
    
    /* 5. 隱藏干擾元素 & 隱藏右下角帳號資訊 [V75 關鍵] */
    thead tr th:first-child { display:none }
    tbody th { display:none }
    .modebar { display: none !important; }
    
    /* 隱藏 Streamlit 的 Footer 和 Viewer Badge (右下角頭像) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none !important;}
    
    /* 6. 自訂大字體數據卡片 */
    .big-metric-box {
        background-color: #f8f9fa;
        border-left: 10px solid #DC3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-label { font-size: 24px; color: #555; font-weight: bold; margin-bottom: 5px; display: block; }
    .metric-value { font-size: 40px; color: #000; font-weight: 900; display: block; }
    
    /* 7. 表格框線 */
    div[data-testid="stDataFrame"] { border: 2px solid #CCC; }
    </style>
    """, unsafe_allow_html=True)

# 檔案路徑定義
CSV_FILE = "phoenix_history.csv"
PARQUET_FILE = "phoenix_history.parquet"
DAILY_SNAPSHOT = "daily_snapshot.csv" # 存放今日明細，供首頁與雷達讀取

# ============================================
# 1. 核心資料清洗與 I/O 邏輯
# ============================================

def clean_broker_name(name):
    if pd.isna(name): return "未知"
    name = str(name)
    cleaned = re.sub(r'^[A-Za-z0-9]+\s*', '', name)
    cleaned = re.sub(r'^\d+', '', cleaned)
    return cleaned.strip()

def scrub_history_file():
    """手動觸發清洗"""
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            if 'Broker' in df.columns:
                sample = df['Broker'].astype(str).iloc[0] if not df.empty else ""
                if re.match(r'^[A-Za-z0-9]', sample):
                    df['Broker'] = df['Broker'].apply(clean_broker_name)
                    df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
        except: pass

# 註解掉自動清洗以加速
# scrub_history_file()

@st.cache_data(ttl=600)
def load_db():
    """讀取歷史總表"""
    df = pd.DataFrame()
    if os.path.exists(PARQUET_FILE):
        try:
            df = pd.read_parquet(PARQUET_FILE)
            if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']).dt.date
            if 'Broker' in df.columns: df['Broker'] = df['Broker'].apply(clean_broker_name)
            return df
        except: pass
    
    if df.empty and os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            if 'Broker' in df.columns: df['Broker'] = df['Broker'].apply(clean_broker_name)
            cols = ['BuyCost', 'SellCost', 'TotalVol', 'BigHand', 'SmallHand', 'TxCount', 'BuyBrokers', 'SellBrokers']
            for c in cols:
                if c not in df.columns: df[c] = 0
            return df
        except: return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=600)
def load_daily_snapshot():
    """讀取今日快照 (給首頁、獵殺雷達用)"""
    if os.path.exists(DAILY_SNAPSHOT):
        try:
            df = pd.read_csv(DAILY_SNAPSHOT)
            if 'Broker' in df.columns: df['Broker'] = df['Broker'].apply(clean_broker_name)
            return df
        except: pass
    return pd.DataFrame()

def save_to_db(new_data_df, detail_df=None):
    """
    存檔邏輯：
    1. 更新歷史總表 (Aggregated)
    2. 如果有 detail_df，更新今日快照 (Snapshot)
    """
    if new_data_df is None or new_data_df.empty: return
    
    new_data_df['Broker'] = new_data_df['Broker'].apply(clean_broker_name)
    
    cols = ['Date', 'Broker', 'Buy', 'Sell', 'Net', 'BuyAvg', 'SellAvg', 'BuyCost', 'SellCost', 'DayClose', 'TotalVol', 'BigHand', 'SmallHand', 'TxCount', 'BuyBrokers', 'SellBrokers']
    for c in cols: 
        if c not in new_data_df.columns: new_data_df[c] = 0
    new_data_df = new_data_df[cols]

    old_db = load_db()
    new_data_df['Date'] = pd.to_datetime(new_data_df['Date']).dt.date
    if not old_db.empty:
        old_db['Date'] = pd.to_datetime(old_db['Date']).dt.date
        new_dates = new_data_df['Date'].unique()
        old_db = old_db[~old_db['Date'].isin(new_dates)]
        final_db = pd.concat([old_db, new_data_df], ignore_index=True)
    else:
        final_db = new_data_df

    final_db = final_db.sort_values(by=['Date', 'Net'], ascending=[True, False])
    final_db.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    try: final_db.to_parquet(PARQUET_FILE, index=False)
    except: pass
    
    # 儲存快照
    if detail_df is not None and not detail_df.empty:
        detail_df['Broker'] = detail_df['Broker'].apply(clean_broker_name)
        detail_df.to_csv(DAILY_SNAPSHOT, index=False, encoding='utf-8-sig')
    
    st.cache_data.clear()

def process_csv_content(df_raw, date_obj):
    try:
        df_L = df_raw.iloc[:, [1, 2, 3, 4]].copy()
        df_L.columns = ['Broker', 'Price', 'Buy', 'Sell']
        df_R = df_raw.iloc[:, [7, 8, 9, 10]].copy()
        df_R.columns = ['Broker', 'Price', 'Buy', 'Sell']
        df_detail = pd.concat([df_L, df_R], ignore_index=True)
        
        df_detail.dropna(subset=['Broker'], inplace=True)
        df_detail['Broker'] = df_detail['Broker'].apply(clean_broker_name)
        for col in ['Price', 'Buy', 'Sell']: df_detail[col] = pd.to_numeric(df_detail[col], errors='coerce').fillna(0)
        
        day_close = 0 # 移除收盤價
        total_vol = df_detail['Buy'].sum()
        tx_count = len(df_detail)
        
        df_detail['Net'] = df_detail['Buy'] - df_detail['Sell']
        big_hand_net = df_detail[df_detail['Buy'] >= 30000]['Buy'].sum() - df_detail[df_detail['Sell'] >= 30000]['Sell'].sum()
        small_hand_net = df_detail[df_detail['Buy'] <= 5000]['Buy'].sum() - df_detail[df_detail['Sell'] <= 5000]['Sell'].sum()

        df_detail['BuyCost'] = df_detail['Price'] * df_detail['Buy']
        df_detail['SellCost'] = df_detail['Price'] * df_detail['Sell']
        
        agg = df_detail.groupby('Broker')[['Buy', 'Sell', 'BuyCost', 'SellCost']].sum().reset_index()
        agg['Net'] = agg['Buy'] - agg['Sell']
        agg['BuyAvg'] = np.where(agg['Buy']>0, agg['BuyCost']/agg['Buy'], 0)
        agg['SellAvg'] = np.where(agg['Sell']>0, agg['SellCost']/agg['Sell'], 0)
        
        agg['Date'] = date_obj
        agg['DayClose'] = day_close
        agg['TotalVol'] = total_vol
        agg['BigHand'] = big_hand_net
        agg['SmallHand'] = small_hand_net
        agg['TxCount'] = tx_count
        agg['BuyBrokers'] = df_detail[df_detail['Net'] > 0]['Broker'].nunique()
        agg['SellBrokers'] = df_detail[df_detail['Net'] < 0]['Broker'].nunique()
        
        return agg, df_detail
    except: return None, None

def process_uploaded_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        match_iso = re.search(r"(\d{4})[-.\s](\d{2})[-.\s](\d{2})", uploaded_file.name)
        if match_iso: date_obj = date(int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3)))
        else: date_obj = date.today()

        uploaded_file.seek(0)
        try: df_raw = pd.read_csv(uploaded_file, encoding='cp950', header=None, skiprows=2)
        except: 
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, encoding='utf-8', header=None, skiprows=2)
        return process_csv_content(df_raw, date_obj)
    except: return None, None

def process_local_file(file_path):
    try:
        with open(file_path, 'rb') as f: head = f.read(1000).decode('cp950', errors='ignore')
        date_obj = date.today() # 簡化日期判斷
        try: df_raw = pd.read_csv(file_path, encoding='cp950', header=None, skiprows=2)
        except: df_raw = pd.read_csv(file_path, encoding='utf-8', header=None, skiprows=2)
        return process_csv_content(df_raw, date_obj)
    except: return None, None

def parse_date_input(date_str, default_date):
    if not date_str: return default_date
    try:
        clean_str = re.sub(r'\D', '', str(date_str))
        if len(clean_str) == 8: return datetime.strptime(clean_str, "%Y%m%d").date()
    except: pass
    return default_date

# ============================================
# 2. 演算法與繪圖
# ============================================
def calculate_hurst(ts): return 0.5 
def kelly_criterion(win_rate, win_loss_ratio): return (win_rate * (win_loss_ratio + 1) - 1) / win_loss_ratio if win_loss_ratio > 0 else 0
def get_tier(net_vol):
    abs_net = abs(net_vol) / 1000 
    if abs_net >= 400: return "👑 超級大戶"
    elif abs_net >= 100: return "🦁 大戶"
    elif abs_net >= 50: return "🐯 中實戶"
    elif abs_net >= 10: return "🦊 小資"
    else: return "🐜 散戶"

def check_geo_insider(broker_name):
    geo_keywords = ['士林', '天母', '石牌', '北投', '蘭雅']
    for k in geo_keywords:
        if k in broker_name: return True
    return False

def check_gang_id(broker_name):
    if any(x in broker_name for x in ['虎尾', '嘉義', '富邦-建國']): return "⚡ 隔日沖"
    if any(x in broker_name for x in ['摩根', '美林', '高盛', '瑞銀']): return "🌎 外資"
    if any(x in broker_name for x in ['臺銀', '土銀', '合庫']): return "🏛️ 官股"
    return "👤 一般"

def color_pnl(val):
    if isinstance(val, str): val = float(val.replace(',','').replace('+','').replace('萬',''))
    color = '#DC3545' if val > 0 else '#28A745' if val < 0 else 'black'
    font_weight = 'bold'
    return f'color: {color}; font-weight: {font_weight}; font-size: 24px'

def plot_bar_chart(data, x_col, y_col, title, color_code, avg_col=None):
    """
    [V74 修正] 恢復均價顯示
    """
    data['Label'] = (data[x_col].abs()).round(1).astype(str) + "張"
    
    if avg_col and avg_col in data.columns:
         data['Label'] = data['Label'] + " ($" + data[avg_col].round(1).astype(str) + ")"

    fig = px.bar(data, x=x_col, y=y_col, orientation='h', text='Label', title=title)
    fig.update_traces(
        marker_color=color_code, 
        textposition='outside', 
        textfont=dict(size=26, color='black', family="Arial Black"), 
        cliponaxis=False, 
        hovertemplate="<b>%{y}</b><br>數據: %{x:.1f}<extra></extra>"
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending', 'title':None, 'tickfont':{'size':24, 'color':'black'}}, 
        xaxis={'title':"", 'showticklabels':False}, 
        margin=dict(r=200), 
        height=700, 
        font=dict(size=22, family="Microsoft JhengHei")
    )
    return fig

# ============================================
# 3. 視圖：🏠 總司令儀表板
# ============================================
def view_dashboard():
    st.header("🏠 總司令儀表板")
    
    # 1. 嘗試讀取當日快照 (最優先)
    df_detail = load_daily_snapshot()
    
    if df_detail.empty:
        st.warning("📭 目前無今日資料。請社長前往「📂 資料管理後台」上傳今日籌碼。")
        return
        
    # 從 detail 算出 aggregate
    agg = df_detail.groupby('Broker')[['Buy', 'Sell', 'BuyCost', 'SellCost']].sum().reset_index()
    agg['Net'] = agg['Buy'] - agg['Sell']
    agg['BuyAvg'] = np.where(agg['Buy']>0, agg['BuyCost']/agg['Buy'], 0)
    agg['SellAvg'] = np.where(agg['Sell']>0, agg['SellCost']/agg['Sell'], 0)
    
    total_vol = df_detail['Buy'].sum()
    buy_brokers = df_detail[df_detail['Net'] > 0]['Broker'].nunique()
    sell_brokers = df_detail[df_detail['Net'] < 0]['Broker'].nunique()
    diff_brk = sell_brokers - buy_brokers
    
    top15_buy = agg.nlargest(15, 'Net')['Net'].sum()
    top15_sell = agg.nsmallest(15, 'Net')['Net'].abs().sum()
    conc = (top15_buy + top15_sell) / total_vol * 100
    power_score = min(100, max(0, 50 + (diff_brk * 0.5) + ((conc - 30) * 1.5)))

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        color = "#28A745" if power_score > 60 else ("#DC3545" if power_score < 40 else "#FFC107")
        st.markdown(f"### 🦅 鳳凰指數")
        st.markdown(f"<h1 style='color:{color}; font-size: 80px; text-align: center; margin:0;'>{power_score:.0f}</h1>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='big-metric-box'><div class='metric-label'>籌碼集中度</div><div class='metric-value'>{conc:.1f}%</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='big-metric-box' style='border-color:#28A745'><div class='metric-label'>買家 vs 賣家</div><div class='metric-value'>{buy_brokers} vs {sell_brokers}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-metric-box' style='border-color:#28A745'><div class='metric-label'>家數差 (正=好)</div><div class='metric-value'>{diff_brk} 家</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    col_hb, col_tool = st.columns([1, 1])
    with col_hb:
        st.subheader("🥊 今日多空重拳")
        if not df_detail.empty:
            max_buy = df_detail.loc[df_detail['Buy'].idxmax()]
            max_sell = df_detail.loc[df_detail['Sell'].idxmax()]
            st.info(f"🔴 **最兇買盤**：{max_buy['Broker']} @ {max_buy['Price']}元 買 {max_buy['Buy']/1000:,.1f} 張")
            st.warning(f"🟢 **最兇賣盤**：{max_sell['Broker']} @ {max_sell['Price']}元 賣 {max_sell['Sell']/1000:,.1f} 張")
        else:
            st.warning("無當日明細資料。")
    
    with col_tool:
        st.subheader("🛠️ 戰術工具箱")
        tool_mode = st.radio("功能選擇", ["🎯 查價位", "🕵️‍♂️ 查分點"], horizontal=True)
        
        if not df_detail.empty:
            if tool_mode == "🎯 查價位":
                prices = sorted(df_detail['Price'].unique(), reverse=True)
                t_p = st.selectbox("選擇價位", prices)
                sort_m = st.radio("排序", ["🔴 買超優先", "🟢 賣超優先"], horizontal=True)
                px_d = df_detail[df_detail['Price'] == t_p].copy()
                if "買超" in sort_m: px_d = px_d.sort_values('Net', ascending=False)
                else: px_d = px_d.sort_values('Net', ascending=True)
                px_show = px_d[['Broker', 'Net']].head(5).copy()
                px_show['Net'] = px_show['Net'] / 1000
                px_show.columns = ['券商', '淨買賣(張)']
                st.dataframe(px_show.style.format("{:.1f}", subset=['淨買賣(張)']).applymap(color_pnl, subset=['淨買賣(張)']), use_container_width=True, hide_index=True)
            
            else: 
                all_bks = sorted(agg['Broker'].unique())
                t_bk = st.selectbox("選擇券商 (查看今日詳細)", all_bks)
                bk_agg = agg[agg['Broker'] == t_bk].iloc[0]
                
                st.markdown(f"""
                <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                    <div class="big-metric-box" style="flex:1; border-color: #DC3545"><div class="metric-label">淨買賣</div><div class="metric-value">{bk_agg['Net']/1000:+,.1f} 張</div></div>
                    <div class="big-metric-box" style="flex:1; border-color: #28A745"><div class="metric-label">買均 / 賣均</div><div class="metric-value" style="font-size: 28px; line-height: 1.5;">{bk_agg['BuyAvg']:.2f} / {bk_agg['SellAvg']:.2f}</div></div>
                </div>""", unsafe_allow_html=True)

                bk_detail_raw = df_detail[df_detail['Broker'] == t_bk].copy()
                if not bk_detail_raw.empty:
                    st.markdown(f"**{t_bk} 各價位明細：**")
                    bk_grp = bk_detail_raw.groupby('Price')[['Buy', 'Sell']].sum().reset_index().sort_values('Price', ascending=False)
                    bk_grp['Net'] = bk_grp['Buy'] - bk_grp['Sell']
                    bk_grp['Buy'] /= 1000
                    bk_grp['Sell'] /= 1000
                    bk_grp['Net'] /= 1000
                    bk_grp.columns = ['價位', '買進(張)', '賣出(張)', '淨買賣(張)']
                    st.dataframe(bk_grp.style.format("{:.1f}", subset=['買進(張)','賣出(張)','淨買賣(張)']).applymap(color_pnl, subset=['淨買賣(張)']), use_container_width=True, hide_index=True)
        else:
            st.info("請社長上傳今日資料以查看價位明細。")

    st.markdown("---")
    cc1, cc2 = st.columns(2)
    with cc1:
        top_buy = agg.nlargest(15, 'Net').sort_values('Net', ascending=True)
        top_buy['Abs_Zhang'] = top_buy['Net'] / 1000
        st.plotly_chart(plot_bar_chart(top_buy, 'Abs_Zhang', 'Broker', "🔴 今日買超 Top 15", '#DC3545', avg_col='BuyAvg'), use_container_width=True)
    with cc2:
        top_sell = agg.nsmallest(15, 'Net').sort_values('Net', ascending=False).sort_values('Net', ascending=True)
        top_sell['Abs_Zhang'] = top_sell['Net'].abs() / 1000
        st.plotly_chart(plot_bar_chart(top_sell, 'Abs_Zhang', 'Broker', "🟢 今日賣超 Top 15", '#28A745', avg_col='SellAvg'), use_container_width=True)

# ============================================
# 4. 視圖：🧠 AI 戰略實驗室 (移除無效功能)
# ============================================
def view_ai_strategy():
    st.header("🧠 AI 戰略實驗室")
    df_hist = load_db()
    if df_hist.empty: st.error("無歷史資料"); return

    st.info("⚠️ Hurst 與蒙地卡羅模擬暫停使用 (需收盤價)。")
    st.markdown("---")
    
    st.subheader("1. 📢 市場情緒地震儀 (Sentiment)")
    if len(df_hist) > 5:
        last_vol = df_hist.sort_values('Date').iloc[-1]['TotalVol']
        avg_vol = df_hist.groupby('Date')['TotalVol'].mean().mean()
        turnover_ratio = last_vol / avg_vol if avg_vol > 0 else 1
        st.metric("情緒貪婪指數", f"{turnover_ratio*50:.0f}") 
    st.markdown("---")

    st.subheader("2. 💰 AI 操盤手 (Kelly)")
    c_k1, c_k2, c_k3 = st.columns(3)
    win_rate = c_k1.slider("預估勝率 (%)", 10, 90, 60) / 100
    odds = c_k2.number_input("盈虧比", 0.5, 5.0, 2.0)
    kelly_pct = kelly_criterion(win_rate, odds)
    sugg_pos = max(0, kelly_pct * 0.5) 
    with c_k3: st.metric("建議投入倉位", f"{sugg_pos*100:.1f} %")

# ============================================
# 5. 視圖：📉 籌碼斷層掃描
# ============================================
def view_chip_structure():
    st.header("📉 籌碼斷層掃描")
    df_hist = load_db()
    if df_hist.empty: st.error("無歷史資料"); return
    dates = sorted(df_hist['Date'].unique())

    st.subheader("🗺️ 動態沃羅諾伊戰場")
    v_opt = st.radio("範圍", ["當日", "近 5 日", "近 10 日", "自訂"], horizontal=True)
    
    target_v = pd.DataFrame()
    if v_opt == "當日": 
        target_v = df_hist[df_hist['Date'] == dates[-1]].copy()
    else:
        if v_opt == "近 5 日": sel_dates = dates[-5:]
        elif v_opt == "近 10 日": sel_dates = dates[-10:]
        else:
            c1, c2 = st.columns(2)
            s = c1.date_input("S", dates[-5])
            e = c2.date_input("E", dates[-1])
            sel_dates = [d for d in dates if s <= d <= e]
        subset = df_hist[df_hist['Date'].isin(sel_dates)]
        target_v = subset.groupby('Broker')[['Net']].sum().reset_index()

    if not target_v.empty:
        target_v['AbsNet'] = target_v['Net'].abs() / 1000
        target_v['Net_Zhang'] = target_v['Net'] / 1000
        target_v['Tier'] = target_v['Net'].apply(get_tier)
        
        custom_scale = [[0.0, 'green'], [0.5, 'white'], [1.0, 'red']]
        max_val = max(abs(target_v['Net_Zhang'].min()), abs(target_v['Net_Zhang'].max()))
        fig_v = px.treemap(target_v, path=[px.Constant("全市場"), 'Tier', 'Broker'], values='AbsNet',
                           color='Net_Zhang', color_continuous_scale=custom_scale, range_color=[-max_val, max_val],
                           title=f"{v_opt} 主力領土")
        fig_v.update_traces(textfont=dict(size=28), hovertemplate='<b>%{label}</b><br>淨量: %{color:.1f} 張')
        st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("---")
    st.subheader("🌪️ 籌碼階級金字塔")
    if not target_v.empty:
        tiers = ["👑 超級大戶", "🦁 大戶", "🐯 中實戶", "🦊 小資", "🐜 散戶"]
        tier_stats = []
        for t in tiers:
            subset = target_v[target_v['Tier'] == t]
            buy_vol = subset[subset['Net_Zhang'] > 0]['Net_Zhang'].sum()
            sell_vol = subset[subset['Net_Zhang'] < 0]['Net_Zhang'].sum()
            tier_stats.append({'Tier': t, 'Buy': buy_vol, 'Sell': sell_vol})
        df_p = pd.DataFrame(tier_stats)
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(y=df_p['Tier'], x=df_p['Buy'], name='買方', orientation='h', marker_color='#DC3545', text=df_p['Buy'].round(1), textposition='outside'))
        fig_p.add_trace(go.Bar(y=df_p['Tier'], x=df_p['Sell'], name='賣方', orientation='h', marker_color='#28A745', text=df_p['Sell'].round(1), textposition='outside'))
        fig_p.update_layout(title="多空對峙金字塔 (張)", barmode='overlay', xaxis_title="淨買賣張數", yaxis=dict(categoryorder='array', categoryarray=tiers[::-1]), font=dict(size=20), height=500)
        st.plotly_chart(fig_p, use_container_width=True)

# ============================================
# 6. 視圖：🔍 獵殺雷達 (修復幫派辨識)
# ============================================
def view_hunter_radar():
    st.header("🔍 獵殺雷達")
    df_hist = load_db()
    if df_hist.empty: st.error("無資料"); return
    dates = sorted(df_hist['Date'].unique())

    st.subheader("📍 3030 地緣雷達")
    geo_opt = st.radio("地緣區間", ["當日", "近 5 日", "近 10 日", "自訂"], horizontal=True)
    if geo_opt == "當日": sel_dates = dates[-1:]
    elif geo_opt == "近 5 日": sel_dates = dates[-5:]
    elif geo_opt == "近 10 日": sel_dates = dates[-10:]
    else: 
        c1, c2 = st.columns(2)
        s = c1.date_input("S", dates[-5])
        e = c2.date_input("E", dates[-1])
        sel_dates = [d for d in dates if s <= d <= e]
    
    subset = df_hist[df_hist['Date'].isin(sel_dates)]
    target_geo = subset.groupby('Broker').agg({'Net':'sum', 'BuyAvg':'mean'}).reset_index()
    if not target_geo.empty:
        target_geo['IsGeo'] = target_geo['Broker'].apply(check_geo_insider)
        geo_brokers = target_geo[target_geo['IsGeo'] & (target_geo['Net'].abs() > 10000)].sort_values('Net', ascending=False)
        if not geo_brokers.empty:
            geo_show = geo_brokers[['Broker', 'Net', 'BuyAvg']].copy()
            geo_show['Net'] /= 1000
            geo_show.columns = ['地緣券商', '淨買賣(張)', '均價']
            st.dataframe(geo_show.style.format("{:.1f}", subset=['淨買賣(張)']).applymap(color_pnl, subset=['淨買賣(張)']), use_container_width=True, hide_index=True)
        else: st.success("✅ 安靜。")

    st.subheader("🩸 幫派辨識")
    # [V75] 修復：從快照讀取今日幫派數據
    df_snapshot = load_daily_snapshot()
    if not df_snapshot.empty:
        df_gang = df_snapshot.copy()
        df_gang['Gang'] = df_gang['Broker'].apply(check_gang_id)
        df_gang['Net_Zhang'] = (df_gang['Net']/1000).round(1)
        df_gang['Info'] = df_gang['Broker'] + ": " + df_gang['Net_Zhang'].astype(str) + "張"
        
        gang_stats = df_gang.groupby('Gang').agg({'Net': 'sum', 'Info': lambda x: '<br>'.join(x.tolist())}).reset_index().sort_values('Net', ascending=False)
        gang_stats['Net_Zhang'] = gang_stats['Net'] / 1000
        
        fig_g = px.bar(gang_stats, x='Net_Zhang', y='Gang', orientation='h', text_auto='.1f', 
                       title="幫派淨買賣", color='Net_Zhang', color_continuous_scale='RdYlGn', custom_data=['Info'])
        fig_g.update_traces(textfont=dict(size=24), hovertemplate="<b>%{y}</b><br>淨量: %{x} 張<br>成員:<br>%{customdata[0]}<extra></extra>")
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.warning("尚無今日資料，無法辨識幫派。")

# ============================================
# 7. 視圖：📈 趨勢戰情室 (修復均價)
# ============================================
def view_trend_analysis():
    st.header("📈 趨勢戰情室")
    df = load_db()
    if df.empty: return

    dates = sorted(df['Date'].unique())
    c1, c2 = st.columns(2)
    with c1: s_input = st.text_input("開始 (YYYYMMDD)", value=dates[0].strftime("%Y%m%d"))
    with c2: e_input = st.text_input("結束 (YYYYMMDD)", value=dates[-1].strftime("%Y%m%d"))
    s_date = parse_date_input(s_input, dates[0])
    e_date = parse_date_input(e_input, dates[-1])

    mask = (df['Date'] >= s_date) & (df['Date'] <= e_date)
    df_period = df.loc[mask].copy()
    
    brokers = sorted(df['Broker'].unique())
    target_brokers = st.multiselect("🔍 特定分點比較", brokers)
    custom_price = st.number_input("輸入假設收盤價 (算未實現)", value=100.0)

    if target_brokers:
        stats = []
        for bk in target_brokers:
            d = df_period[df_period['Broker'] == bk]
            if d.empty: continue
            net = d['Net'].sum()
            cost = d['BuyCost'].sum()/d['Buy'].sum() if d['Buy'].sum()>0 else 0
            profit = (custom_price - cost) * net
            stats.append({"券商": bk, "淨買賣(張)": net/1000, "均價": cost, "預估獲利(萬)": profit/10000})
        
        if stats:
            st.dataframe(pd.DataFrame(stats).style.format("{:,.1f}", subset=['淨買賣(張)']).format("{:,.0f}", subset=['預估獲利(萬)']).format("{:.2f}", subset=['均價']).applymap(color_pnl, subset=['預估獲利(萬)']), use_container_width=True, hide_index=True)
        
        st.markdown("### 📅 指定區間每日明細")
        detail_show = df_period[df_period['Broker'].isin(target_brokers)].sort_values(['Date', 'Broker'], ascending=[False, True]).copy()
        
        if not detail_show.empty:
            detail_show['Buy'] /= 1000
            detail_show['Sell'] /= 1000
            detail_show['Net'] /= 1000
            detail_show = detail_show[['Date', 'Broker', 'Buy', 'Sell', 'Net', 'BuyAvg']]
            detail_show.columns = ['日期', '券商', '買進(張)', '賣出(張)', '淨買賣(張)', '買均']
            st.dataframe(detail_show.style.format("{:.1f}", subset=['買進(張)','賣出(張)','淨買賣(張)']).format("{:.2f}", subset=['買均']).applymap(color_pnl, subset=['淨買賣(張)']), use_container_width=True, hide_index=True)
        else: st.warning("該區間無交易紀錄")

    else:
        group = df_period.groupby('Broker').agg({'Buy':'sum', 'Sell':'sum', 'Net':'sum', 'BuyCost':'sum', 'SellCost':'sum'}).reset_index()
        group['Net_Zhang'] = (group['Net']/1000).round(1)
        # [V74 新增] 算出區間均價
        group['BuyAvg'] = np.where(group['Buy']>0, group['BuyCost']/group['Buy'], 0)
        group['SellAvg'] = np.where(group['Sell']>0, group['SellCost']/group['Sell'], 0)

        c_t1, c_t2 = st.columns(2)
        with c_t1:
            top = group.nlargest(15, 'Net').sort_values('Net', ascending=True)
            # [V74 修正] 顯示均價
            st.plotly_chart(plot_bar_chart(top, 'Net_Zhang', 'Broker', "🏆 區間買超", '#DC3545', avg_col='BuyAvg'), use_container_width=True)
        with c_t2:
            tail = group.nsmallest(15, 'Net').sort_values('Net', ascending=False)
            tail['Abs_Zhang'] = tail['Net_Zhang'].abs()
            # [V74 修正] 顯示均價
            st.plotly_chart(plot_bar_chart(tail, 'Abs_Zhang', 'Broker', "📉 區間賣超", '#28A745', avg_col='SellAvg'), use_container_width=True)

# ============================================
# 8. 視圖：🏆 贏家與韭菜
# ============================================
def view_winners():
    st.header("🏆 贏家與韭菜名人堂")
    df_hist = load_db()
    if df_hist.empty: return
    
    range_opt = st.radio("範圍", ["近 20 日", "近 60 日", "自訂"], horizontal=True)
    dates = sorted(df_hist['Date'].unique())
    
    if range_opt == "近 20 日": d_sub = df_hist[df_hist['Date'].isin(dates[-20:])]
    elif range_opt == "近 60 日": d_sub = df_hist[df_hist['Date'].isin(dates[-60:])]
    else: 
        c1, c2 = st.columns(2)
        s = c1.date_input("S", dates[0])
        e = c2.date_input("E", dates[-1])
        d_sub = df_hist[(df_hist['Date']>=s) & (df_hist['Date']<=e)]
        
    group = d_sub.groupby('Broker').agg({'Net': 'sum', 'BuyCost': 'sum', 'Buy': 'sum'}).reset_index()
    group['AvgCost'] = group['BuyCost'] / group['Buy']
    
    winners = group.nlargest(10, 'Net')
    losers = group.nsmallest(10, 'Net')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🥇 大戶吸籌榜 (買超最多)")
        w_show = winners[['Broker', 'Net', 'AvgCost']].copy()
        w_show['Net'] /= 1000
        w_show.columns = ['券商', '淨買(張)', '成本']
        st.dataframe(w_show.style.format("{:.1f}", subset=['淨買(張)']).format("{:.2f}", subset=['成本']).applymap(color_pnl, subset=['淨買(張)']), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("🥬 大戶倒貨榜 (賣超最多)")
        l_show = losers[['Broker', 'Net', 'AvgCost']].copy()
        l_show['Net'] /= 1000
        l_show.columns = ['券商', '淨買(張)', '成本']
        st.dataframe(l_show.style.format("{:.1f}", subset=['淨買(張)']).format("{:.2f}", subset=['成本']).applymap(color_pnl, subset=['淨買(張)']), use_container_width=True, hide_index=True)

# ============================================
# 9. 視圖：🕵️‍♂️ 分點偵探
# ============================================
def view_broker_detective():
    st.header("🕵️‍♂️ 分點偵探")
    df = load_db()
    if df.empty: return
    dates = sorted(df['Date'].unique())
    brokers = sorted(df['Broker'].unique())
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: target = st.selectbox("選擇券商", brokers)
    with c2: 
        s_input = st.text_input("開始", value=dates[0].strftime("%Y%m%d"), key="bd_s")
        e_input = st.text_input("結束", value=dates[-1].strftime("%Y%m%d"), key="bd_e")
    s_date = parse_date_input(s_input, dates[0])
    e_date = parse_date_input(e_input, dates[-1])
    data = df[(df['Broker'] == target) & (df['Date'] >= s_date) & (df['Date'] <= e_date)].sort_values('Date')
    
    if not data.empty:
        with c3: calc_p = st.number_input("目前股價 (計算獲利)", value=100.0)
        total_net = data['Net'].sum() / 1000
        avg_cost = data['BuyCost'].sum() / data['Buy'].sum() if data['Buy'].sum() > 0 else 0
        est_profit = (calc_p - avg_cost) * data['Net'].sum() / 10000
        
        m1, m2 = st.columns(2)
        m1.metric("區間淨買賣", f"{total_net:+.1f} 張")
        m2.metric("平均成本", f"{avg_cost:.2f}")
        m3, m4 = st.columns(2)
        m3.metric("目前試算價", f"{calc_p}")
        m4.metric("未實現獲利", f"{est_profit:+.0f} 萬", delta_color="normal")

        data['Net_Zhang'] = data['Net'] / 1000
        fig = go.Figure()
        fig.add_trace(go.Bar(x=data['Date'], y=data['Net_Zhang'], name='淨買賣(張)', marker_color=np.where(data['Net']>0, '#DC3545', '#28A745')))
        fig.update_layout(title=f"{target} 操作軌跡", yaxis=dict(title="張數"), height=500)
        st.plotly_chart(fig, use_container_width=True)
        show = data[['Date', 'Buy', 'Sell', 'Net', 'BuyAvg']].copy()
        show.iloc[:, 1:4] /= 1000
        show.columns = ['日期', '買進(張)', '賣出(張)', '淨買賣(張)', '買均']
        st.dataframe(show.style.format("{:.1f}", subset=['買進(張)','賣出(張)','淨買賣(張)']).format("{:.2f}", subset=['買均']).applymap(color_pnl, subset=['淨買賣(張)']), use_container_width=True, hide_index=True)

# ============================================
# 10. 視圖：📂 資料管理後台 (社長專用)
# ============================================
def view_batch_import():
    st.header("📂 資料管理後台 (社長專用)")
    
    admin_pwd = st.sidebar.text_input("🔑 社長密碼 (上傳解鎖)", type="password")

    if admin_pwd == "8888":
        st.success("🔓 社長權限已解鎖！")
        
        # [V74] 社長專用上傳區 (同步更新快照)
        st.subheader("📤 上傳今日 CSV (更新首頁資訊)")
        st.info("上傳後，首頁、獵殺雷達、買賣超排行都會立即顯示最新數據。")
        uploaded_file = st.file_uploader("拖曳今日 CSV 到此處", type=['csv'], key="today_csv")
        
        if uploaded_file and st.button("🚀 更新今日戰情"):
            uploaded_file.seek(0)
            try: df_raw = pd.read_csv(uploaded_file, encoding='cp950', header=None, skiprows=2)
            except: 
                uploaded_file.seek(0)
                df_raw = pd.read_csv(uploaded_file, encoding='utf-8', header=None, skiprows=2)
            
            date_pick = date.today()
            agg, df_detail = process_csv_content(df_raw, date_pick)
            if agg is not None:
                save_to_db(agg, detail_df=df_detail)
                st.success(f"✅ 資料已更新！首頁現在顯示 {date_pick} 的數據。")
                time.sleep(1)
                st.rerun()

        st.markdown("---")
        st.caption("下方為批次歷史資料匯入 (不影響今日首頁)")
        tab1, tab2 = st.tabs(["🚀 本機掃描 (推薦)", "📤 批量拖曳上傳"])
        
        with tab1:
            folder_path = st.text_input("請輸入 CSV 資料夾路徑", value=os.getcwd())
            if st.button("🚀 開始掃描並匯入"):
                if os.path.isdir(folder_path):
                    files = glob.glob(os.path.join(folder_path, "*.csv"))
                    if files:
                        progress_bar = st.progress(0)
                        all_dfs = []
                        for i, fp in enumerate(files):
                            try:
                                agg, _ = process_local_file(fp)
                                if agg is not None: all_dfs.append(agg)
                            except: pass
                            progress_bar.progress((i+1)/len(files))
                        if all_dfs:
                            with st.spinner("存檔中..."):
                                final_df = pd.concat(all_dfs, ignore_index=True)
                                save_to_db(final_df)
                            st.success(f"🎉 成功匯入 {len(all_dfs)} 個檔案！")
        
        with tab2:
            up_files = st.file_uploader("選擇多個 CSV", type=['csv'], accept_multiple_files=True)
            if up_files and st.button("📥 解析並匯入"):
                progress_bar = st.progress(0)
                all_dfs = []
                for i, f in enumerate(up_files):
                    try:
                        agg, _ = process_uploaded_file(f)
                        if agg is not None: all_dfs.append(agg)
                    except: pass
                    progress_bar.progress((i+1)/len(up_files))
                if all_dfs:
                    with st.spinner("存檔中..."):
                        final_df = pd.concat(all_dfs, ignore_index=True)
                        save_to_db(final_df)
                    st.success("🎉 匯入完成")

    else:
        st.info("👋 這裡是後台管理區，請輸入密碼解鎖。")

# ============================================
# Main Loop (功能導航)
# ============================================
def main():
    with st.sidebar:
        st.title("🦅 Phoenix V75")
        st.caption("帝王匿名版")
        st.markdown("---")
        choice = st.radio("功能選單", [
            "🏠 總司令儀表板", 
            "🧠 AI 戰略實驗室", 
            "📈 趨勢戰情室", 
            "🔍 獵殺雷達", 
            "📉 籌碼斷層", 
            "🕵️‍♂️ 分點偵探", 
            "🏆 贏家與韭菜名人堂", 
            "📂 資料管理後台"
        ])
        st.markdown("---")
        st.info("System Ready")
    
    if choice == "🏠 總司令儀表板": view_dashboard()
    elif choice == "🧠 AI 戰略實驗室": view_ai_strategy()
    elif choice == "📈 趨勢戰情室": view_trend_analysis()
    elif choice == "🔍 獵殺雷達": view_hunter_radar()
    elif choice == "📉 籌碼斷層": view_chip_structure()
    elif choice == "🕵️‍♂️ 分點偵探": view_broker_detective()
    elif choice == "🏆 贏家與韭菜名人堂": view_winners()
    elif choice == "📂 資料管理後台": view_batch_import()

if __name__ == "__main__":
    main()