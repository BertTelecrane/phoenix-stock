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
# 0. 系統設定 & CSS (24px 帝王字體)
# ============================================
st.set_page_config(
    page_title="Phoenix V65 帝王手寫版",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    /* 1. 全局字體強制加大 */
    html, body, [class*="css"], .stMarkdown, .stDataFrame, .stTable, p, div, input, label, span, button, .stSelectbox, .stRadio {
        font-family: 'Microsoft JhengHei', 'Arial', sans-serif !important;
        font-size: 24px !important; 
        line-height: 1.6 !important;
    }
    
    /* 2. 標題特大化 */
    h1 { font-size: 48px !important; font-weight: 900 !important; color: #000; margin-bottom: 20px !important; }
    h2 { font-size: 36px !important; font-weight: bold; color: #333; margin-top: 30px !important; }
    h3 { font-size: 30px !important; font-weight: bold; color: #444; }

    /* 3. 版面間距調整 */
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    
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
    
    /* 5. 隱藏干擾元素 */
    thead tr th:first-child { display:none }
    tbody th { display:none }
    .modebar { display: none !important; }
    
    /* 6. 自訂大字體數據卡片 */
    .big-metric-box {
        background-color: #f8f9fa;
        border-left: 10px solid #DC3545;
        padding: 20px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-label { font-size: 24px; color: #555; font-weight: bold; margin-bottom: 8px; display: block; }
    .metric-value { font-size: 42px; color: #000; font-weight: 900; display: block; }
    
    /* 7. 表格樣式 */
    div[data-testid="stDataFrame"] { border: 2px solid #CCC; }
    </style>
    """, unsafe_allow_html=True)

# 檔案路徑定義
CSV_FILE = "phoenix_history.csv"
PARQUET_FILE = "phoenix_history.parquet"

# ============================================
# 1. 核心資料處理 (極速版：移除開機掃描)
# ============================================

def clean_broker_name(name):
    if pd.isna(name): return "未知"
    name = str(name)
    cleaned = re.sub(r'^[A-Za-z0-9]+\s*', '', name)
    cleaned = re.sub(r'^\d+', '', cleaned)
    return cleaned.strip()

def scrub_history_file():
    """
    手動觸發的資料庫清洗功能。
    (不再於啟動時自動執行，避免卡頓)
    """
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            if 'Broker' in df.columns:
                df['Broker'] = df['Broker'].apply(clean_broker_name)
                df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
                try: df.to_parquet(PARQUET_FILE, index=False)
                except: pass
            return True
        except:
            return False
    return False

# [關鍵] 這裡不再自動呼叫 scrub_history_file()

@st.cache_data(ttl=600)
def load_db():
    """極速讀取"""
    df = pd.DataFrame()
    # 1. 優先嘗試 Parquet
    if os.path.exists(PARQUET_FILE):
        try:
            df = pd.read_parquet(PARQUET_FILE)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.date
            # 讀取時順手清洗 (不寫入硬碟，速度快)
            if 'Broker' in df.columns:
                df['Broker'] = df['Broker'].apply(clean_broker_name)
            return df
        except: pass

    # 2. 備用嘗試 CSV
    if df.empty and os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            if 'Broker' in df.columns:
                df['Broker'] = df['Broker'].apply(clean_broker_name)
            
            cols = ['BuyCost', 'SellCost', 'TotalVol', 'BigHand', 'SmallHand', 'TxCount', 'BuyBrokers', 'SellBrokers']
            for c in cols:
                if c not in df.columns: df[c] = 0
            return df
        except: return pd.DataFrame()
        
    return pd.DataFrame()

def save_to_db(new_data_df):
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
        
        day_close = df_detail[df_detail['Price'] > 0]['Price'].iloc[-1] if not df_detail.empty else 0
        total_vol = df_detail['Buy'].sum()
        tx_count = len(df_detail)
        
        df_detail['Net'] = df_detail['Buy'] - df_detail['Sell']
        df_detail['BuyCost'] = df_detail['Price'] * df_detail['Buy']
        df_detail['SellCost'] = df_detail['Price'] * df_detail['Sell']
        
        buy_brokers = df_detail[df_detail['Net'] > 0]['Broker'].nunique()
        sell_brokers = df_detail[df_detail['Net'] < 0]['Broker'].nunique()
        
        agg = df_detail.groupby('Broker')[['Buy', 'Sell', 'BuyCost', 'SellCost']].sum().reset_index()
        agg['Net'] = agg['Buy'] - agg['Sell']
        agg['BuyAvg'] = np.where(agg['Buy']>0, agg['BuyCost']/agg['Buy'], 0)
        agg['SellAvg'] = np.where(agg['Sell']>0, agg['SellCost']/agg['Sell'], 0)
        
        agg['Date'] = date_obj
        agg['DayClose'] = day_close
        agg['TotalVol'] = total_vol
        agg['TxCount'] = tx_count
        agg['BuyBrokers'] = buy_brokers
        agg['SellBrokers'] = sell_brokers
        agg['BigHand'] = 0; agg['SmallHand'] = 0
        
        return agg, df_detail
    except: return None, None

def process_local_file(file_path):
    try:
        with open(file_path, 'rb') as f: head = f.read(1000).decode('cp950', errors='ignore')
        date_obj = smart_parse_date(os.path.basename(file_path), content_head=head, file_path=file_path)
        try: df_raw = pd.read_csv(file_path, encoding='cp950', header=None, skiprows=2)
        except: df_raw = pd.read_csv(file_path, encoding='utf-8', header=None, skiprows=2)
        return process_csv_content(df_raw, date_obj)
    except: return None, None

def process_uploaded_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        head = uploaded_file.read(1000).decode('cp950', errors='ignore')
        date_obj = smart_parse_date(uploaded_file.name, content_head=head)
        uploaded_file.seek(0)
        try: df_raw = pd.read_csv(uploaded_file, encoding='cp950', header=None, skiprows=2)
        except: 
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, encoding='utf-8', header=None, skiprows=2)
        return process_csv_content(df_raw, date_obj)
    except: return None, None

def smart_parse_date(filename, content_head=None, file_path=None):
    match_iso = re.search(r"(\d{4})[-.\s](\d{2})[-.\s](\d{2})", filename)
    if match_iso: return date(int(match_iso.group(1)), int(match_iso.group(2)), int(match_iso.group(3)))
    match_compact = re.search(r"(202\d{5})", filename)
    if match_compact: return datetime.strptime(match_compact.group(1), "%Y%m%d").date()
    if content_head:
        try:
            tw_date = re.search(r"(\d{3})/(\d{1,2})/(\d{1,2})", content_head)
            if tw_date: return date(int(tw_date.group(1)) + 1911, int(tw_date.group(2)), int(tw_date.group(3)))
        except: pass
    if file_path:
        try: return date.fromtimestamp(os.path.getmtime(file_path))
        except: pass
    return date.today()

def parse_date_input(date_str, default_date):
    """解析日期函數"""
    if not date_str: return default_date
    try:
        clean_str = re.sub(r'\D', '', str(date_str))
        if len(clean_str) == 8: return datetime.strptime(clean_str, "%Y%m%d").date()
    except: pass
    return default_date

# ============================================
# 2. 演算法與繪圖輔助
# ============================================
def calculate_hurst(ts):
    if len(ts) < 20: return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0 

def kelly_criterion(win_rate, win_loss_ratio):
    if win_loss_ratio == 0: return 0
    return (win_rate * (win_loss_ratio + 1) - 1) / win_loss_ratio

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

# [V65] 刪除 plot_bar_chart 共用函數，改為在各視圖中手動展開

# ============================================
# 3. 視圖：🏠 總司令儀表板
# ============================================
def view_dashboard():
    st.header("🏠 總司令儀表板")
    
    # [V63/V64] 首頁無上傳功能
    df = load_db()
    if df.empty:
        st.warning("📭 目前資料庫是空的。請社長前往「📂 資料管理後台」匯入資料。")
        return

    latest_date = df['Date'].max()
    st.info(f"📊 數據日期：{latest_date} (如需更新請洽社長)")
    
    df_today = df[df['Date'] == latest_date].copy()
    
    if not df_today.empty:
        # 鳳凰指數
        buy_brk = df_today['BuyBrokers'].iloc[0] if 'BuyBrokers' in df_today.columns else 0
        sell_brk = df_today['SellBrokers'].iloc[0] if 'SellBrokers' in df_today.columns else 0
        diff_brk = sell_brk - buy_brk 
        top15_buy = df_today.nlargest(15, 'Net')['Net'].sum()
        top15_sell = df_today.nsmallest(15, 'Net')['Net'].abs().sum()
        total_vol = df_today['TotalVol'].iloc[0] if df_today['TotalVol'].iloc[0] > 0 else 1
        conc = (top15_buy + top15_sell) / total_vol * 100
        power_score = min(100, max(0, 50 + (diff_brk * 0.5) + ((conc - 30) * 1.5)))
        
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            color = "#DC3545" if power_score > 60 else ("#28A745" if power_score < 40 else "#FFC107")
            st.markdown(f"### 🦅 鳳凰指數")
            st.markdown(f"<h1 style='color:{color}; font-size: 80px; text-align: center; margin:0;'>{power_score:.0f}</h1>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='big-metric-box'><div class='metric-label'>收盤價</div><div class='metric-value'>{df_today['DayClose'].iloc[0]}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-metric-box'><div class='metric-label'>籌碼集中度</div><div class='metric-value'>{conc:.1f}%</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='big-metric-box' style='border-color:#28A745'><div class='metric-label'>買家 vs 賣家</div><div class='metric-value'>{buy_brk} vs {sell_brk}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-metric-box' style='border-color:#DC3545'><div class='metric-label'>籌碼流向 (正=集中)</div><div class='metric-value'>{diff_brk} 家</div></div>", unsafe_allow_html=True)

        with st.expander("ℹ️ 鳳凰指數戰術指導"):
            st.markdown("* **>60 (紅)**：籌碼集中，偏多。\n* **<40 (綠)**：籌碼渙散，偏空。")

        st.markdown("---")

        # 買賣超排行 (手動展開圖表代碼)
        col_hb, col_tool = st.columns([1, 1])
        with col_hb:
            st.subheader("🏆 今日主力買超 (張)")
            top_buy = df_today.nlargest(15, 'Net').sort_values('Net', ascending=True)
            top_buy['Label'] = (top_buy['Net']/1000).round(1).astype(str) + "張"
            top_buy['Net_Z'] = top_buy['Net'] / 1000
            
            # 手寫買超圖
            fig_buy = px.bar(top_buy, x='Net_Z', y='Broker', orientation='h', text='Label', title="")
            fig_buy.update_traces(
                marker_color='#DC3545',
                textposition='outside',
                textfont=dict(size=26, color='black', family="Arial Black"),
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>淨買賣: %{x:.1f} 張<extra></extra>"
            )
            fig_buy.update_layout(
                yaxis={'categoryorder':'total ascending', 'title':None, 'tickfont':{'size':24, 'color':'black'}},
                xaxis={'title':"", 'showticklabels':False},
                margin=dict(r=150), height=700, font=dict(size=22, family="Microsoft JhengHei")
            )
            st.plotly_chart(fig_buy, use_container_width=True)

        with col_tool:
            st.subheader("📉 今日主力賣超 (張)")
            top_sell = df_today.nsmallest(15, 'Net').sort_values('Net', ascending=False).sort_values('Net', ascending=True)
            top_sell['Label'] = (top_sell['Net'].abs()/1000).round(1).astype(str) + "張"
            top_sell['Abs_Z'] = top_sell['Net'].abs() / 1000
            
            # 手寫賣超圖
            fig_sell = px.bar(top_sell, x='Abs_Z', y='Broker', orientation='h', text='Label', title="")
            fig_sell.update_traces(
                marker_color='#28A745',
                textposition='outside',
                textfont=dict(size=26, color='black', family="Arial Black"),
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>淨買賣: %{x:.1f} 張<extra></extra>"
            )
            fig_sell.update_layout(
                yaxis={'categoryorder':'total ascending', 'title':None, 'tickfont':{'size':24, 'color':'black'}},
                xaxis={'title':"", 'showticklabels':False},
                margin=dict(r=150), height=700, font=dict(size=22, family="Microsoft JhengHei")
            )
            st.plotly_chart(fig_sell, use_container_width=True)

        st.markdown("---")
        
        # 查分點 (當日)
        st.subheader("🕵️‍♂️ 快速查分點 (當日)")
        all_bks = sorted(df_today['Broker'].unique())
        t_bk = st.selectbox("選擇券商", all_bks)
        
        bk_agg = df_today[df_today['Broker'] == t_bk].iloc[0]
        st.markdown(f"""
        <div style="display: flex; gap: 15px; margin-bottom: 20px;">
            <div class="big-metric-box" style="flex:1; border-color: #DC3545"><div class="metric-label">淨買賣</div><div class="metric-value">{bk_agg['Net']/1000:+,.1f} 張</div></div>
            <div class="big-metric-box" style="flex:1; border-color: #28A745"><div class="metric-label">買均 / 賣均</div><div class="metric-value" style="font-size: 28px; line-height: 1.5;">{bk_agg['BuyAvg']:.2f} / {bk_agg['SellAvg']:.2f}</div></div>
        </div>""", unsafe_allow_html=True)

# ============================================
# 4. 視圖：🧠 AI 戰略實驗室
# ============================================
def view_ai_strategy():
    st.header("🧠 AI 戰略實驗室")
    df_hist = load_db()
    if df_hist.empty: st.error("無歷史資料"); return

    # Hurst
    st.subheader("1. 🌌 混沌趨勢檢測儀 (Hurst)")
    df_price = df_hist.sort_values('Date').drop_duplicates('Date').set_index('Date')['DayClose']
    if len(df_price) > 30:
        h_val = calculate_hurst(df_price.values)
        c1, c2 = st.columns([1, 2])
        with c1:
            h_color = "#DC3545" if h_val > 0.6 else ("#28A745" if h_val < 0.4 else "#FFC107")
            st.markdown(f"<h1 style='color:{h_color}; font-size: 80px;'>{h_val:.2f}</h1>", unsafe_allow_html=True)
        with c2:
            if h_val > 0.6: st.error("🔥 **強趨勢**：慣性大，適合追價。")
            elif h_val < 0.4: st.success("🌊 **震盪**：高出低進。")
            else: st.warning("☁️ **隨機**：無方向。")
    st.markdown("---")
    
    # Monte Carlo
    st.subheader("2. 🔮 蒙地卡羅模擬 (未來10天)")
    if len(df_price) > 30:
        returns = df_price.pct_change().dropna()
        mu = returns.mean(); sigma = returns.std()
        last_price = df_price.iloc[-1]
        sim_df = pd.DataFrame()
        for i in range(1000):
            p = [last_price]
            for d in range(10): p.append(p[-1] * (1 + np.random.normal(mu, sigma)))
            sim_df[i] = p
        
        fig = go.Figure()
        upper = sim_df.quantile(0.95, axis=1)
        lower = sim_df.quantile(0.05, axis=1)
        mean_path = sim_df.mean(axis=1)
        fig.add_trace(go.Scatter(y=upper, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(y=lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,0,0,0.1)', name='90% 區間'))
        fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color='red', width=3), name='平均路徑'))
        fig.update_layout(title="股價機率預測", height=500, font=dict(size=20))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    # 庫存背離
    st.subheader("3. 📉 主力庫存背離圖")
    df_trend = df_hist.groupby('Date').agg({'Net':'sum', 'DayClose':'last'}).reset_index().sort_values('Date')
    df_trend['CumNet'] = df_trend['Net'].cumsum()
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=df_trend['Date'], y=df_trend['DayClose'], name='股價', line=dict(color='black', width=3)))
    fig_d.add_trace(go.Bar(x=df_trend['Date'], y=df_trend['CumNet'], name='主力庫存', yaxis='y2', opacity=0.3, marker_color='red'))
    fig_d.update_layout(title="股價 vs 庫存", yaxis2=dict(title="庫存", overlaying='y', side='right'), height=500, font=dict(size=20))
    st.plotly_chart(fig_d, use_container_width=True)
    st.markdown("---")

    # NLP & Kelly
    st.subheader("4. 📢 情緒與資金 (Sentiment & Kelly)")
    c_s1, c_s2 = st.columns([1, 1])
    with c_s1:
        if len(df_price) > 5:
            last_vol = df_hist.sort_values('Date').iloc[-1]['TotalVol']
            avg_vol = df_hist.groupby('Date')['TotalVol'].mean().mean()
            turnover_ratio = last_vol / avg_vol if avg_vol > 0 else 1
            st.metric("情緒貪婪指數", f"{turnover_ratio*50:.0f}")
    with c_s2:
        win_rate = st.slider("勝率", 10, 90, 60) / 100
        odds = st.number_input("盈虧比", 0.5, 5.0, 2.0)
        kelly_pct = (win_rate * (odds + 1) - 1) / odds if odds > 0 else 0
        sugg_pos = max(0, kelly_pct * 0.5) 
        st.metric("建議倉位", f"{sugg_pos*100:.1f} %")

# ============================================
# 5. 視圖：📉 籌碼斷層掃描
# ============================================
def view_chip_structure():
    st.header("📉 籌碼斷層掃描")
    df_hist = load_db()
    if df_hist.empty: st.error("無歷史資料"); return
    dates = sorted(df_hist['Date'].unique())

    st.subheader("🗺️ 動態沃羅諾伊戰場 (紅買/綠賣)")
    v_opt = st.radio("範圍", ["當日", "近 5 日", "近 10 日", "自訂"], horizontal=True)
    target_v = pd.DataFrame()
    if v_opt == "當日": target_v = df_hist[df_hist['Date'] == dates[-1]].copy()
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
        target_v['Net_Z'] = target_v['Net'] / 1000
        target_v['Tier'] = target_v['Net'].apply(get_tier)
        
        custom_scale = [[0.0, 'green'], [0.5, 'white'], [1.0, 'red']]
        max_val = max(abs(target_v['Net_Z'].min()), abs(target_v['Net_Z'].max()))
        
        fig_v = px.treemap(target_v, path=[px.Constant("全市場"), 'Tier', 'Broker'], values='AbsNet',
                           color='Net_Z', color_continuous_scale=custom_scale, range_color=[-max_val, max_val],
                           title=f"{v_opt} 主力領土 (面積=張數, 紅=買/綠=賣)")
        fig_v.update_traces(textfont=dict(size=28), hovertemplate='<b>%{label}</b><br>淨量: %{color:.1f} 張<br>板塊大小: %{value:.1f} 張')
        st.plotly_chart(fig_v, use_container_width=True)

    st.markdown("---")
    st.subheader("🌪️ 籌碼階級金字塔")
    if not target_v.empty:
        tiers = ["👑 超級大戶", "🦁 大戶", "🐯 中實戶", "🦊 小資", "🐜 散戶"]
        tier_stats = []
        for t in tiers:
            subset = target_v[target_v['Tier'] == t]
            buy_vol = subset[subset['Net_Z'] > 0]['Net_Z'].sum()
            sell_vol = subset[subset['Net_Z'] < 0]['Net_Z'].sum()
            tier_stats.append({'Tier': t, 'Buy': buy_vol, 'Sell': sell_vol})
        df_p = pd.DataFrame(tier_stats)
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(y=df_p['Tier'], x=df_p['Buy'], name='買方', orientation='h', marker_color='#DC3545', text=df_p['Buy'].round(1), textposition='outside'))
        fig_p.add_trace(go.Bar(y=df_p['Tier'], x=df_p['Sell'], name='賣方', orientation='h', marker_color='#28A745', text=df_p['Sell'].round(1), textposition='outside'))
        fig_p.update_layout(title="多空對峙金字塔 (張)", barmode='overlay', xaxis_title="淨買賣張數", yaxis=dict(categoryorder='array', categoryarray=tiers[::-1]), font=dict(size=20), height=500)
        st.plotly_chart(fig_p, use_container_width=True)

# ============================================
# 6. 視圖：🔍 獵殺雷達
# ============================================
def view_hunter_radar():
    st.header("🔍 獵殺雷達")
    df_hist = load_db()
    if df_hist.empty: st.error("無資料"); return
    dates = sorted(df_hist['Date'].unique())

    st.subheader("📍 3030 地緣雷達 (士林/天母)")
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
    if 'daily_data' in st.session_state:
        df_gang = st.session_state['daily_data'].copy()
        df_gang['Gang'] = df_gang['Broker'].apply(check_gang_id)
        df_gang['Net_Z'] = (df_gang['Net']/1000).round(1)
        df_gang['Info'] = df_gang['Broker'] + ": " + df_gang['Net_Z'].astype(str) + "張"
        
        gang_stats = df_gang.groupby('Gang').agg({'Net': 'sum', 'Info': lambda x: '<br>'.join(x.tolist())}).reset_index().sort_values('Net', ascending=False)
        gang_stats['Net_Z'] = gang_stats['Net'] / 1000
        
        fig_g = px.bar(gang_stats, x='Net_Z', y='Gang', orientation='h', text_auto='.1f', 
                       title="幫派淨買賣", color='Net_Z', color_continuous_scale='RdYlGn', custom_data=['Info'])
        fig_g.update_traces(textfont=dict(size=24), hovertemplate="<b>%{y}</b><br>淨量: %{x} 張<br>成員:<br>%{customdata[0]}<extra></extra>")
        st.plotly_chart(fig_g, use_container_width=True)

# ============================================
# 7. 視圖：📈 趨勢戰情室 (手寫展開圖表)
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
    last_close = df_period.sort_values('Date').iloc[-1]['DayClose']
    
    brokers = sorted(df['Broker'].unique())
    target_brokers = st.multiselect("🔍 特定分點比較", brokers)
    custom_price = st.number_input("輸入假設收盤價 (算未實現)", value=float(last_close))

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
            detail_show = detail_show[['Date', 'Broker', 'Buy', 'Sell', 'Net', 'BuyAvg', 'DayClose']]
            detail_show.columns = ['日期', '券商', '買進(張)', '賣出(張)', '淨買賣(張)', '買均', '收盤']
            st.dataframe(detail_show.style.format("{:.1f}", subset=['買進(張)','賣出(張)','淨買賣(張)']).format("{:.2f}", subset=['買均','收盤']).background_gradient(subset=['淨買賣(張)'], cmap='RdYlGn'), use_container_width=True, hide_index=True)
    else:
        group = df_period.groupby('Broker').agg({'Buy':'sum', 'Sell':'sum', 'Net':'sum', 'BuyCost':'sum', 'SellCost':'sum'}).reset_index()
        group['Net_Z'] = (group['Net']/1000).round(1)
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            top = group.nlargest(15, 'Net').sort_values('Net', ascending=True)
            top['Label'] = top['Net_Z'].astype(str) + "張"
            
            # 手寫買超圖
            fig_buy = px.bar(top, x='Net_Z', y='Broker', orientation='h', text='Label', title="🏆 區間買超")
            fig_buy.update_traces(
                marker_color='#DC3545',
                textposition='outside',
                textfont=dict(size=26, color='black', family="Arial Black"),
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>買超: %{x} 張<extra></extra>"
            )
            fig_buy.update_layout(yaxis={'categoryorder':'total ascending', 'title':None, 'tickfont':{'size':24, 'color':'black'}}, xaxis={'title':"", 'showticklabels':False}, margin=dict(r=150), height=700, font=dict(size=22, family="Microsoft JhengHei"))
            st.plotly_chart(fig_buy, use_container_width=True)

        with c_t2:
            tail = group.nsmallest(15, 'Net').sort_values('Net', ascending=False)
            tail['Abs_Z'] = tail['Net_Z'].abs()
            tail['Label'] = tail['Abs_Z'].astype(str) + "張"
            
            # 手寫賣超圖
            fig_sell = px.bar(tail, x='Abs_Z', y='Broker', orientation='h', text='Label', title="📉 區間賣超")
            fig_sell.update_traces(
                marker_color='#28A745',
                textposition='outside',
                textfont=dict(size=26, color='black', family="Arial Black"),
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>賣超: %{text} 張<extra></extra>"
            )
            fig_sell.update_layout(yaxis={'categoryorder':'total ascending', 'title':None, 'tickfont':{'size':24, 'color':'black'}}, xaxis={'title':"", 'showticklabels':False}, margin=dict(r=150), height=700, font=dict(size=22, family="Microsoft JhengHei"))
            st.plotly_chart(fig_sell, use_container_width=True)

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
    last_price = df_hist.sort_values('Date').iloc[-1]['DayClose']
    group = d_sub.groupby('Broker').agg({'Net': 'sum', 'BuyCost': 'sum', 'Buy': 'sum'}).reset_index()
    group = group[group['Buy'] > 10000] 
    group['AvgCost'] = group['BuyCost'] / group['Buy']
    group['Profit'] = (last_price - group['AvgCost']) * group['Net'] / 10000
    winners = group.nlargest(10, 'Profit')
    losers = group.nsmallest(10, 'Profit')
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🥇 贏家榜 (跟單)")
        w_show = winners[['Broker', 'Net', 'AvgCost', 'Profit']].copy()
        w_show['Net'] /= 1000
        w_show.columns = ['券商', '淨買(張)', '成本', '獲利(萬)']
        st.dataframe(w_show.style.format("{:.1f}", subset=['淨買(張)','獲利(萬)']).format("{:.2f}", subset=['成本']).applymap(color_pnl, subset=['獲利(萬)']), use_container_width=True, hide_index=True)
    with c2:
        st.subheader("🥬 韭菜榜 (反指標)")
        l_show = losers[['Broker', 'Net', 'AvgCost', 'Profit']].copy()
        l_show['Net'] /= 1000
        l_show.columns = ['券商', '淨買(張)', '成本', '虧損(萬)']
        st.dataframe(l_show.style.format("{:.1f}", subset=['淨買(張)','虧損(萬)']).format("{:.2f}", subset=['成本']).applymap(color_pnl, subset=['虧損(萬)']), use_container_width=True, hide_index=True)

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
        last_close = data.iloc[-1]['DayClose']
        with c3: calc_p = st.number_input("目前股價 (計算獲利)", value=float(last_close))
        total_net = data['Net'].sum() / 1000
        total_buy_cost = data['BuyCost'].sum()
        total_buy_vol = data['Buy'].sum()
        avg_cost = total_buy_cost / total_buy_vol if total_buy_vol > 0 else 0
        est_profit = (calc_p - avg_cost) * data['Net'].sum() / 10000
        m1, m2 = st.columns(2)
        m1.metric("區間淨買賣", f"{total_net:+.1f} 張")
        m2.metric("平均成本", f"{avg_cost:.2f}")
        m3, m4 = st.columns(2)
        m3.metric("目前試算價", f"{calc_p}")
        m4.metric("未實現獲利", f"{est_profit:+.0f} 萬", delta_color="normal")
        data['Net_Z'] = data['Net'] / 1000
        fig = go.Figure()
        fig.add_trace(go.Bar(x=data['Date'], y=data['Net_Z'], name='淨買賣(張)', marker_color=np.where(data['Net']>0, '#DC3545', '#28A745')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['DayClose'], name='股價', yaxis='y2', line=dict(color='#FFC107', width=3)))
        fig.update_layout(title=f"{target} 操作軌跡", yaxis=dict(title="張數"), yaxis2=dict(title="股價", overlaying='y', side='right'), height=500, font=dict(size=20), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        show = data[['Date', 'Buy', 'Sell', 'Net', 'BuyAvg', 'DayClose']].copy()
        show.iloc[:, 1:4] /= 1000
        show.columns = ['日期', '買進(張)', '賣出(張)', '淨買賣(張)', '買均', '收盤']
        st.dataframe(show.style.format("{:.1f}", subset=['買進(張)','賣出(張)','淨買賣(張)']).format("{:.2f}", subset=['買均','收盤']).background_gradient(subset=['淨買賣(張)'], cmap='RdYlGn'), use_container_width=True, hide_index=True)

# ============================================
# 10. 視圖：📂 資料管理後台 (社長權限)
# ============================================
def view_batch_import():
    st.header("📂 資料管理後台")
    admin_pwd = st.sidebar.text_input("🔑 社長密碼", type="password")
    
    if admin_pwd == "8888":
        st.success("✅ 身分確認：社長好！")
        tab1, tab2, tab3 = st.tabs(["🚀 本機掃描", "📤 拖曳上傳", "🛠️ 資料庫維護"])
        
        with tab1:
            folder_path = st.text_input("CSV 資料夾路徑", value=os.getcwd())
            if st.button("🚀 掃描並匯入"):
                if os.path.isdir(folder_path):
                    files = glob.glob(os.path.join(folder_path, "*.csv"))
                    if files:
                        progress_bar = st.progress(0)
                        all_dfs = []
                        status = st.empty()
                        for i, fp in enumerate(files):
                            status.text(f"處理: {os.path.basename(fp)}")
                            agg, _ = process_local_file(fp)
                            if agg is not None: all_dfs.append(agg)
                            progress_bar.progress((i+1)/len(files))
                        if all_dfs:
                            full = pd.concat(all_dfs, ignore_index=True)
                            save_to_db(full)
                            st.success(f"成功匯入 {len(all_dfs)} 個檔案！")
                else: st.error("路徑錯誤")

        with tab2:
            uploaded = st.file_uploader("CSV", accept_multiple_files=True)
            if uploaded and st.button("上傳"):
                all_dfs = []
                for f in uploaded:
                    agg, _ = process_uploaded_file(f)
                    if agg is not None: all_dfs.append(agg)
                if all_dfs:
                    save_to_db(pd.concat(all_dfs, ignore_index=True))
                    st.success("完成！")
                    
        with tab3:
            st.warning("若發現舊資料有亂碼，可點擊此按鈕手動清洗。")
            if st.button("🛠️ 執行深度清洗"):
                if os.path.exists(CSV_FILE):
                    try:
                        with st.spinner("正在掃描與清洗..."):
                            df = pd.read_csv(CSV_FILE)
                            if 'Broker' in df.columns:
                                df['Broker'] = df['Broker'].apply(clean_broker_name)
                                df.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
                                try: df.to_parquet(PARQUET_FILE, index=False)
                                except: pass
                        st.success("清洗完成！")
                    except Exception as e: st.error(f"失敗: {e}")
    else:
        st.info("👋 這裡是社長專屬後台。若您是社員，請點選左側其他功能查看戰情。")

# ============================================
# Main Loop (功能導航)
# ============================================
def main():
    with st.sidebar:
        st.title("🦅 Phoenix V64")
        st.caption("帝王完美版")
        
        # 選單中包含「資料管理後台」，但點進去會被密碼擋住
        choice = st.radio("功能選單", [
            "🏠 總司令儀表板", "🧠 AI 戰略實驗室", "📈 趨勢戰情室", 
            "🔍 獵殺雷達", "📉 籌碼斷層", "🕵️‍♂️ 分點偵探", 
            "🏆 贏家與韭菜名人堂", "📂 資料管理後台" 
        ])
    
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