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
# 0. 系統設定 & CSS (隱形斗篷 + RWD 自適應字體)
# ============================================
st.set_page_config(
    page_title="Phoenix V104 隱形自適應版",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    /* ====================================================================
       1. 【隱形斗篷】核心代碼：徹底移除右下角身分標記
       ==================================================================== */
    
    /* 隱藏 Streamlit 右下角的 Viewer Badge (顯示帳號/公司名的那個) */
    div[class*="viewerBadge"] { display: none !important; }
    .viewerBadge_container__1QSob { display: none !important; }
    
    /* 隱藏右上角漢堡選單、頂部彩條、底部 Footer */
    #MainMenu { visibility: hidden !important; }
    header { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    [data-testid="stStatusWidget"] { visibility: hidden !important; }
    
    /* ====================================================================
       2. 【RWD 自適應字體】依據螢幕寬度自動調整大小
       ==================================================================== */
    
    /* 預設 (手機或小視窗): 字體適中，不會爆版 */
    html, body, [class*="css"] {
        font-family: 'Microsoft JhengHei', 'Arial', sans-serif !important;
        font-size: 18px !important; 
        font-weight: bold !important;
    }
    h1 { font-size: 36px !important; }
    h2 { font-size: 28px !important; }
    h3 { font-size: 24px !important; }
    .stMetricValue { font-size: 40px !important; }

    /* 平板/筆電 (寬度 > 768px): 字體稍微放大 */
    @media (min-width: 768px) {
        html, body, [class*="css"] {
            font-size: 22px !important; 
        }
        h1 { font-size: 48px !important; }
        h2 { font-size: 36px !important; }
        h3 { font-size: 30px !important; }
        .stMetricValue { font-size: 50px !important; }
        
        /* 表格字體 */
        div[data-testid="stDataFrame"] div[data-testid="stTable"], 
        div[data-testid="stDataFrame"] td, 
        div[data-testid="stDataFrame"] th {
            font-size: 22px !important;
        }
    }

    /* 大型桌機/大螢幕 (寬度 > 1200px): 字體加大 (你的老花眼模式) */
    @media (min-width: 1200px) {
        html, body, [class*="css"] {
            font-size: 28px !important; 
        }
        h1 { font-size: 60px !important; }
        h2 { font-size: 48px !important; }
        h3 { font-size: 36px !important; }
        .stMetricValue { font-size: 64px !important; }
        
        /* 表格字體特大 */
        div[data-testid="stDataFrame"] div[data-testid="stTable"], 
        div[data-testid="stDataFrame"] td, 
        div[data-testid="stDataFrame"] th {
            font-size: 28px !important;
            padding: 12px !important;
        }
        
        /* 輸入框高度撐開 */
        .stSelectbox div[data-baseweb="select"] > div,
        .stTextInput div[data-baseweb="input"] > div {
            min-height: 60px !important;
        }
    }

    /* ====================================================================
       3. 通用元件優化
       ==================================================================== */
    .modebar { display: none !important; }
    
    /* 輸入框通用設定 */
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextInput div[data-baseweb="input"] > div,
    .stNumberInput div[data-baseweb="input"] > div {
        display: flex !important;
        align-items: center !important;
    }
    
    /* 自訂大字體數據卡片 (也會隨 RWD 縮放) */
    .big-metric-box {
        background-color: #f8f9fa;
        border-left: 10px solid #DC3545;
        padding: 20px;
        margin: 15px 0;
        border-radius: 12px;
        box-shadow: 4px 4px 10px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# 檔案路徑定義
CSV_FILE = "phoenix_history.csv"
PARQUET_FILE = "phoenix_history.parquet"
DAILY_SNAPSHOT = "daily_snapshot.csv"

# ============================================
# 1. 核心資料清洗與 I/O 邏輯
# ============================================

def clean_broker_name(name):
    if pd.isna(name): return "未知"
    name = str(name)
    cleaned = re.sub(r'^[A-Za-z0-9]+\s*', '', name)
    cleaned = re.sub(r'^\d+', '', cleaned)
    return cleaned.strip()

def parse_date_input(date_str, default_date):
    if not date_str: return default_date
    try:
        clean_str = re.sub(r'\D', '', str(date_str))
        if len(clean_str) == 8: return datetime.strptime(clean_str, "%Y%m%d").date()
    except: pass
    return default_date

@st.cache_data(ttl=600)
def load_db():
    df = pd.DataFrame()
    if os.path.exists(PARQUET_FILE):
        try:
            df = pd.read_parquet(PARQUET_FILE)
            if 'Date' in df.columns: df['Date'] = pd.to_datetime(df['Date']).dt.date
            if 'Broker' in df.columns: df['Broker'] = df['Broker'].apply(clean_broker_name)
            return df
        except: pass
    if os.path.exists(CSV_FILE):
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
    if os.path.exists(DAILY_SNAPSHOT):
        try:
            df = pd.read_csv(DAILY_SNAPSHOT)
            if 'Broker' in df.columns: df['Broker'] = df['Broker'].apply(clean_broker_name)
            return df
        except: pass
    return pd.DataFrame()

def save_to_db(new_data_df, detail_df=None):
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
    else: final_db = new_data_df

    final_db = final_db.sort_values(by=['Date', 'Net'], ascending=[True, False])
    final_db.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')
    try: final_db.to_parquet(PARQUET_FILE, index=False)
    except: pass
    
    if detail_df is not None:
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
        
        day_close = 0 
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
        try: df_raw = pd.read_csv(uploaded_file, encoding='cp950', header=None, skiprows=2)
        except: 
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, encoding='utf-8', header=None, skiprows=2)
        return process_csv_content(df_raw, date.today())
    except: return None, None

def process_local_file(file_path):
    try:
        try: df_raw = pd.read_csv(file_path, encoding='cp950', header=None, skiprows=2)
        except: df_raw = pd.read_csv(file_path, encoding='utf-8', header=None, skiprows=2)
        match = re.search(r"(\d{4})[-.\s](\d{2})[-.\s](\d{2})", os.path.basename(file_path))
        d_obj = date(int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else date.today()
        return process_csv_content(df_raw, d_obj)
    except: return None, None

# ============================================
# 2. 演算法與繪圖
# ============================================
def calculate_hurst(ts):
    if len(ts) < 20: return 0.5
    try:
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

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
    return f'color: {color};'

def plot_bar_chart(data, x_col, y_col, title, color_code, avg_col=None):
    data['Label'] = (data[x_col].abs()).round(1).astype(str) + "張"
    if avg_col and avg_col in data.columns:
         data['Label'] = data['Label'] + " ($" + data[avg_col].round(1).astype(str) + ")"

    fig = px.bar(data, x=x_col, y=y_col, orientation='h', text='Label', title=title)
    
    # 圖表字體大小設定 (這裡我們給一個相對大的預設值，讓它在大螢幕清楚，手機上 Plotly 會自動縮放一些)
    fig.update_layout(
        yaxis={'categoryorder':'total ascending', 'title':None, 'tickfont':{'size':20, 'color':'black', 'family': 'Microsoft JhengHei'}}, 
        xaxis={'title':"", 'showticklabels': False}, 
        margin=dict(r=150, l=120, t=80, b=50), 
        height=850, 
        title_font=dict(size=30, family="Microsoft JhengHei", color='black'),
        hoverlabel=dict(font_size=24, font_family="Microsoft JhengHei", bgcolor="white") 
    )
    
    fig.update_traces(
        marker_color=color_code, 
        textposition='outside', 
        textfont=dict(size=24, color='black', family="Arial Black"), 
        cliponaxis=False, 
        hovertemplate="<b>%{y}</b><br>數據: %{x:.1f}<extra></extra>"
    )
    return fig

# ============================================
# 3. 視圖：🏠 總司令儀表板
# ============================================
def view_dashboard():
    st.header("🏠 總司令儀表板")
    
    df_detail = load_daily_snapshot()
    df_hist = load_db()

    buy_brk, sell_brk, top20_buy_vol, top20_sell_vol, total_vol = 0, 0, 0, 0, 1
    final_agg = pd.DataFrame()

    if not df_detail.empty:
        agg = df_detail.groupby('Broker')[['Buy', 'Sell', 'BuyCost', 'SellCost']].sum().reset_index()
        agg['Net'] = agg['Buy'] - agg['Sell']
        final_agg = agg
        
        total_vol = df_detail['Buy'].sum()
        buy_brk = df_detail[df_detail['Net'] > 0]['Broker'].nunique()
        sell_brk = df_detail[df_detail['Net'] < 0]['Broker'].nunique()
        
    elif not df_hist.empty:
        latest = df_hist['Date'].max()
        agg = df_hist[df_hist['Date'] == latest].copy()
        if not agg.empty:
             final_agg = agg
             buy_brk = agg['BuyBrokers'].iloc[0] if 'BuyBrokers' in agg.columns else 0
             sell_brk = agg['SellBrokers'].iloc[0] if 'SellBrokers' in agg.columns else 0
             total_vol = agg['TotalVol'].iloc[0] if 'TotalVol' in agg.columns else 1
    else:
        st.warning("📭 請社長上傳資料")

    if not final_agg.empty:
        top15_buy_sum = final_agg.nlargest(15, 'Net')['Net'].sum()
        top15_sell_sum = final_agg.nsmallest(15, 'Net')['Net'].abs().sum()
    else:
        top15_buy_sum = 0
        top15_sell_sum = 0

    diff_brk = sell_brk - buy_brk
    conc = (top15_buy_sum + top15_sell_sum) / total_vol * 100 if total_vol > 0 else 0
    power_score = min(100, max(0, 50 + (diff_brk * 0.5) + ((conc - 30) * 1.5)))
    
    user_price = st.number_input("請輸入今日收盤價", value=100.0)

    # 數據卡片顯示
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        color = "#28A745" if power_score > 60 else ("#DC3545" if power_score < 40 else "#FFC107")
        st.markdown(f"### 🦅 鳳凰指數")
        st.markdown(f"<h1 style='color:{color}; text-align: center; margin:0;'>{power_score:.0f}</h1>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='big-metric-box'><div class='metric-label'>收盤價</div><div class='metric-value'>{user_price}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-metric-box'><div class='metric-label'>籌碼集中度</div><div class='metric-value'>{conc:.1f}%</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='big-metric-box' style='border-color:#28A745'><div class='metric-label'>買家 vs 賣家</div><div class='metric-value'>{buy_brk} vs {sell_brk}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big-metric-box' style='border-color:#28A745'><div class='metric-label'>家數差 (正=好)</div><div class='metric-value'>{diff_brk} 家</div></div>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ 鳳凰指數說明"):
        st.markdown("""
        * **> 60 分 (綠)**：籌碼集中，主力買進。
        * **< 40 分 (紅)**：籌碼渙散，主力倒貨。
        """)

    st.markdown("---")

    col_hb, col_tool = st.columns([1, 1])
    with col_hb:
        st.subheader("🥊 今日多空重拳")
        if not df_detail.empty:
            max_buy = df_detail.loc[df_detail['Buy'].idxmax()]
            max_sell = df_detail.loc[df_detail['Sell'].idxmax()]
            st.info(f"🔴 **最兇買盤**：{max_buy['Broker']} @ {max_buy['Price']}元 買 {max_buy['Buy']/1000:,.1f} 張")
            st.warning(f"🟢 **最兇賣盤**：{max_sell['Broker']} @ {max_sell['Price']}元 賣 {max_sell['Sell']/1000:,.1f} 張")
    
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
                px_show['Net'] /= 1000
                st.dataframe(px_show.style.format("{:.1f}", subset=['Net']).applymap(color_pnl, subset=['Net']), use_container_width=True, hide_index=True)
            else: 
                all_bks = sorted(final_agg['Broker'].unique())
                t_bk = st.selectbox("選擇券商", all_bks)
                bk_detail_raw = df_detail[df_detail['Broker'] == t_bk].copy()
                if not bk_detail_raw.empty:
                    # 計算總結數據
                    t_buy = bk_detail_raw['Buy'].sum()
                    t_sell = bk_detail_raw['Sell'].sum()
                    t_net = t_buy - t_sell
                    avg_b = (bk_detail_raw['Price'] * bk_detail_raw['Buy']).sum() / t_buy if t_buy > 0 else 0
                    avg_s = (bk_detail_raw['Price'] * bk_detail_raw['Sell']).sum() / t_sell if t_sell > 0 else 0
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("總買進 (張)", f"{t_buy/1000:,.1f}", f"均價 {avg_b:.2f}")
                    m2.metric("總賣出 (張)", f"{t_sell/1000:,.1f}", f"均價 {avg_s:.2f}")
                    m3.metric("淨買賣 (張)", f"{t_net/1000:,.1f}", delta_color="normal")
                    
                    bk_grp = bk_detail_raw.groupby('Price')[['Buy', 'Sell']].sum().reset_index().sort_values('Price', ascending=False)
                    bk_grp['Net'] = bk_grp['Buy'] - bk_grp['Sell']
                    bk_grp['Buy'] /= 1000; bk_grp['Sell'] /= 1000; bk_grp['Net'] /= 1000
                    st.dataframe(bk_grp.style.format("{:.1f}", subset=['Buy','Sell','Net']).format("{:.2f}", subset=['Price']).applymap(color_pnl, subset=['Net']), use_container_width=True, hide_index=True)

    st.markdown("---")
    cc1, cc2 = st.columns(2)
    # Top 20 邏輯
    N_TOP = 20
    
    with cc1:
        if not final_agg.empty:
            final_agg['BuyAvg'] = np.where(final_agg['Buy']>0, final_agg['BuyCost']/final_agg['Buy'], 0)
            
            top_buy = final_agg.nlargest(N_TOP, 'Net').sort_values('Net', ascending=True)
            top_buy['Abs_Zhang'] = top_buy['Net'] / 1000
            
            st.plotly_chart(plot_bar_chart(top_buy, 'Abs_Zhang', 'Broker', f"🔴 今日買超 Top {N_TOP}", '#DC3545', avg_col='BuyAvg'), use_container_width=True)
            
            tb_vol = top_buy['Net'].sum() / 1000
            tb_avg = (top_buy['BuyCost'].sum() / top_buy['Buy'].sum()) if top_buy['Buy'].sum() > 0 else 0
            
            st.markdown(f"""
            <div style="background-color:#ffe6e6; padding:15px; border-radius:10px; border-left: 10px solid #DC3545;">
                <span style="color:#555; font-size:24px;">Top {N_TOP} 買方總計：</span><br>
                <span style="color:#DC3545; font-size:36px; font-weight:900;">{tb_vol:,.1f} 張</span> <span style="font-size:24px; color:#333;">(均價 {tb_avg:.2f})</span>
            </div>
            """, unsafe_allow_html=True)
            
    with cc2:
        if not final_agg.empty:
            final_agg['SellAvg'] = np.where(final_agg['Sell']>0, final_agg['SellCost']/final_agg['Sell'], 0)
            
            top_sell = final_agg.nsmallest(N_TOP, 'Net').copy()
            top_sell['Abs_Zhang'] = top_sell['Net'].abs() / 1000
            top_sell = top_sell.sort_values('Abs_Zhang', ascending=True)
            
            st.plotly_chart(plot_bar_chart(top_sell, 'Abs_Zhang', 'Broker', f"🟢 今日賣超 Top {N_TOP}", '#28A745', avg_col='SellAvg'), use_container_width=True)

            ts_vol = top_sell['Net'].abs().sum() / 1000
            ts_avg = (top_sell['SellCost'].sum() / top_sell['Sell'].sum()) if top_sell['Sell'].sum() > 0 else 0
            
            st.markdown(f"""
            <div style="background-color:#e6ffe6; padding:15px; border-radius:10px; border-left: 10px solid #28A745;">
                <span style="color:#555; font-size:24px;">Top {N_TOP} 賣方總計：</span><br>
                <span style="color:#28A745; font-size:36px; font-weight:900;">{ts_vol:,.1f} 張</span> <span style="font-size:24px; color:#333;">(均價 {ts_avg:.2f})</span>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# 4. 視圖：🧠 AI 戰略實驗室
# ============================================
def view_ai_strategy():
    st.header("🧠 AI 戰略實驗室")
    df_hist = load_db()
    
    user_price = st.number_input("請輸入目前收盤價 (以計算 Hurst)", value=100.0)

    daily_price = df_hist.groupby('Date').apply(lambda x: x['BuyCost'].sum() / x['Buy'].sum() if x['Buy'].sum() > 0 else 0) if not df_hist.empty else []
    h_val = 0.5
    if len(daily_price) > 20: h_val = calculate_hurst(daily_price.values)

    st.subheader("1. 🌌 混沌趨勢檢測儀 (Hurst)")
    c1, c2 = st.columns([1, 2])
    with c1:
        h_color = "#DC3545" if h_val > 0.6 else ("#28A745" if h_val < 0.4 else "#FFC107")
        st.markdown(f"<h1 style='color:{h_color}; margin:0;'>{h_val:.2f}</h1>", unsafe_allow_html=True)
    with c2:
        if h_val > 0.6: st.error("🔥 **強趨勢**：慣性大。")
        elif h_val < 0.4: st.success("🌊 **震盪**：高出低進。")
        else: st.warning("☁️ **隨機**：無方向。")
    
    with st.expander("ℹ️ B 大戰術指導：Hurst"):
        st.markdown("* **H > 0.7**：強趨勢 * **H < 0.3**：震盪 * **H = 0.5**：隨機")
    
    st.markdown("---")
    st.subheader("2. 📢 市場情緒地震儀")
    if not df_hist.empty:
        last_vol = df_hist.sort_values('Date').iloc[-1]['TotalVol']
        avg_vol = df_hist.groupby('Date')['TotalVol'].mean().mean()
        turnover_ratio = last_vol / avg_vol if avg_vol > 0 else 1
        st.metric("情緒貪婪指數", f"{turnover_ratio*50:.0f}") 
    else:
        st.metric("情緒貪婪指數", "--")

    with st.expander("ℹ️ B 大戰術指導：情緒背離"):
        st.markdown("情緒過熱 (指數 > 200) 請小心主力倒貨。")

    st.markdown("---")
    st.subheader("3. 💰 AI 操盤手 (Kelly)")
    c_k1, c_k2, c_k3 = st.columns(3)
    win_rate = c_k1.slider("預估勝率 (%)", 10, 90, 60) / 100
    odds = c_k2.number_input("盈虧比", 0.5, 5.0, 2.0)
    kelly_pct = kelly_criterion(win_rate, odds)
    sugg_pos = max(0, kelly_pct * 0.5) 
    with c_k3: st.metric("建議投入倉位", f"{sugg_pos*100:.1f} %")
    
    with st.expander("ℹ️ B 大戰術指導：資金控管"):
        st.markdown("凱利公式能確保長期獲利最大化。")

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
    
    if v_opt == "當日": target_v = df_hist[df_hist['Date'] == dates[-1]].copy()
    else:
        sel_dates = dates[-5:] if v_opt == "近 5 日" else dates[-10:]
        subset = df_hist[df_hist['Date'].isin(sel_dates)]
        target_v = subset.groupby('Broker')[['Net']].sum().reset_index()

    if not target_v.empty:
        target_v = target_v.groupby('Broker')[['Net']].sum().reset_index()
        target_v['AbsNet'] = target_v['Net'].abs() / 1000
        target_v['Net_Zhang'] = target_v['Net'] / 1000
        target_v['Tier'] = target_v['Net'].apply(get_tier)
        
        # 視覺權重平衡：放大中實戶
        def weight_boost(row):
            if "超級大戶" in row['Tier']: return row['AbsNet'] * 1.0  
            if "大戶" in row['Tier']: return row['AbsNet'] * 1.0      
            if "中實戶" in row['Tier']: return row['AbsNet'] * 3.0   
            return row['AbsNet'] * 0.8  
            
        target_v['W_Size'] = target_v.apply(weight_boost, axis=1)

        custom_scale = [[0.0, 'green'], [0.5, 'white'], [1.0, 'red']]
        max_val = max(abs(target_v['Net_Zhang'].min()), abs(target_v['Net_Zhang'].max()))
        fig_v = px.treemap(target_v, path=[px.Constant("全市場"), 'Tier', 'Broker'], values='W_Size',
                           color='Net_Zhang', color_continuous_scale=custom_scale, range_color=[-max_val, max_val],
                           title=f"{v_opt} 主力領土 (加權平衡顯示)")
        
        fig_v.update_traces(textfont=dict(size=24), hovertemplate='<b>%{label}</b><br>淨量: %{color:.1f} 張')
        fig_v.update_layout(hoverlabel=dict(font_size=24, font_family="Microsoft JhengHei", bgcolor="white"))
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
        fig_p.update_layout(title="多空對峙金字塔 (張)", barmode='overlay', xaxis_title="淨買賣張數", yaxis=dict(categoryorder='array', categoryarray=tiers[::-1]), font=dict(size=20), height=600, hoverlabel=dict(font_size=24))
        st.plotly_chart(fig_p, use_container_width=True)

# ============================================
# 6. 視圖：🔍 獵殺雷達
# ============================================
def view_hunter_radar():
    st.header("🔍 獵殺雷達")
    df_hist = load_db()
    if df_hist.empty: st.error("無資料"); return
    dates = sorted(df_hist['Date'].unique())

    st.subheader("📍 3030 地緣雷達")
    geo_opt = st.radio("地緣區間", ["當日", "近 5 日", "近 10 日", "自訂"], horizontal=True)
    if geo_opt == "當日": sel_dates = dates[-1:]
    else: sel_dates = dates[-5:] if geo_opt == "近 5 日" else dates[-10:]
    
    subset = df_hist[df_hist['Date'].isin(sel_dates)]
    target_geo = subset.groupby('Broker').agg({'Net':'sum', 'BuyAvg':'mean'}).reset_index()
    if not target_geo.empty:
        target_geo['IsGeo'] = target_geo['Broker'].apply(check_geo_insider)
        geo_brokers = target_geo[target_geo['IsGeo'] & (target_geo['Net'].abs() > 10000)].sort_values('Net', ascending=False)
        if not geo_brokers.empty:
            geo_show = geo_brokers[['Broker', 'Net', 'BuyAvg']].copy()
            geo_show['Net'] /= 1000
            geo_show.columns = ['地緣券商', '淨買賣(張)', '均價']
            st.dataframe(geo_show.style.format("{:.1f}", subset=['淨買賣(張)']).format("{:.2f}", subset=['均價']).applymap(color_pnl, subset=['淨買賣(張)']), use_container_width=True, hide_index=True)
        else: st.success("✅ 安靜。")

    st.subheader("🩸 幫派辨識")
    df_snapshot = load_daily_snapshot()
    if not df_snapshot.empty:
        df_gang = df_snapshot.copy()
        df_gang['Gang'] = df_gang['Broker'].apply(check_gang_id)
        df_gang['Net_Zhang'] = (df_gang['Net']/1000).round(1)
        
        df_gang['Info'] = df_gang['Broker'] + ": " + df_gang['Net_Zhang'].astype(str) + "張"
        
        gang_stats = df_gang.groupby('Gang').agg({
            'Net': 'sum', 
            'Info': lambda x: '<br>'.join(x.tolist())
        }).reset_index().sort_values('Net', ascending=False)
        
        gang_stats['Net_Zhang'] = gang_stats['Net'] / 1000
        
        fig_g = px.bar(gang_stats, x='Net_Zhang', y='Gang', orientation='h', text_auto='.1f', 
                       title="幫派淨買賣", color='Net_Zhang', color_continuous_scale='RdYlGn', 
                       custom_data=['Info']) 
        
        fig_g.update_traces(
            textfont=dict(size=24),
            hovertemplate="<b>%{y}</b><br>淨量: %{x} 張<br>成員明細:<br>%{customdata[0]}<extra></extra>"
        )
        fig_g.update_layout(hoverlabel=dict(font_size=24, font_family="Microsoft JhengHei"), height=600, font=dict(size=22))
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.warning("尚無今日資料。")

# ============================================
# 7. 視圖：📈 趨勢戰情室
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
    
    if st.button("🔄 重置 (返回 Top 15)"):
        st.session_state['target_brokers'] = []
        st.rerun()

    if 'target_brokers' not in st.session_state: st.session_state['target_brokers'] = []
    
    target_brokers = st.multiselect("🔍 特定分點比較", brokers, key='trend_multiselect')
    if target_brokers: st.session_state['target_brokers'] = target_brokers
    
    if target_brokers:
        stats = []
        for bk in target_brokers:
            d = df_period[df_period['Broker'] == bk]
            if d.empty: continue
            net = d['Net'].sum()
            cost = d['BuyCost'].sum()/d['Buy'].sum() if d['Buy'].sum()>0 else 0
            stats.append({"券商": bk, "淨買賣(張)": net/1000, "均價": cost})
        
        if stats:
            st.dataframe(pd.DataFrame(stats).style.format("{:,.1f}", subset=['淨買賣(張)']).format("{:.2f}", subset=['均價']).applymap(color_pnl, subset=['淨買賣(張)']), use_container_width=True, hide_index=True)
        
        st.markdown("### 📅 指定區間每日明細")
        detail_show = df_period[df_period['Broker'].isin(target_brokers)].sort_values(['Date', 'Broker'], ascending=[False, True]).copy()
        if not detail_show.empty:
            detail_show['Buy'] /= 1000; detail_show['Sell'] /= 1000; detail_show['Net'] /= 1000
            detail_show = detail_show[['Date', 'Broker', 'Buy', 'Sell', 'Net', 'BuyAvg']]
            detail_show.columns = ['日期', '券商', '買進(張)', '賣出(張)', '淨買賣(張)', '買均']
            st.dataframe(detail_show.style.format("{:.1f}", subset=['買進(張)','賣出(張)','淨買賣(張)']).format("{:.2f}", subset=['買均']).applymap(color_pnl, subset=['淨買賣(張)']), use_container_width=True, hide_index=True)
    else:
        group = df_period.groupby('Broker').agg({'Buy':'sum', 'Sell':'sum', 'Net':'sum', 'BuyCost':'sum', 'SellCost':'sum'}).reset_index()
        group['Net_Zhang'] = (group['Net']/1000).round(1)
        group['BuyAvg'] = np.where(group['Buy']>0, group['BuyCost']/group['Buy'], 0)
        group['SellAvg'] = np.where(group['Sell']>0, group['SellCost']/group['Sell'], 0)

        c_t1, c_t2 = st.columns(2)
        with c_t1:
            top_buy = group.nlargest(15, 'Net').sort_values('Net', ascending=True)
            top_buy['Abs_Zhang'] = top_buy['Net'] / 1000
            st.plotly_chart(plot_bar_chart(top_buy, 'Abs_Zhang', 'Broker', "🏆 區間買超", '#DC3545', avg_col='BuyAvg'), use_container_width=True)
        with c_t2:
            top_sell = group.nsmallest(15, 'Net').copy()
            top_sell['Abs_Zhang'] = top_sell['Net'].abs() / 1000
            top_sell = top_sell.sort_values('Abs_Zhang', ascending=True)
            st.plotly_chart(plot_bar_chart(top_sell, 'Abs_Zhang', 'Broker', "📉 區間賣超", '#28A745', avg_col='SellAvg'), use_container_width=True)

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
        s = c1.date_input("S", dates[0]); e = c2.date_input("E", dates[-1])
        d_sub = df_hist[(df_hist['Date']>=s) & (df_hist['Date']<=e)]
        
    group = d_sub.groupby('Broker').agg({'Net': 'sum', 'BuyCost': 'sum', 'Buy': 'sum'}).reset_index()
    group = group[group['Buy'] > 1000] 
    group['AvgCost'] = group['BuyCost'] / group['Buy']
    
    last_price = st.number_input("請輸入目前股價 (計算獲利)", value=100.0)
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
    s_date = parse_date_input(s_input, dates[0]); e_date = parse_date_input(e_input, dates[-1])
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
        fig.update_layout(title=f"{target} 操作軌跡", yaxis=dict(title="張數"), height=600, hoverlabel=dict(font_size=24), font=dict(size=22))
        st.plotly_chart(fig, use_container_width=True)
        show = data[['Date', 'Buy', 'Sell', 'Net', 'BuyAvg']].copy()
        show.iloc[:, 1:4] /= 1000
        show.columns = ['日期', '買進(張)', '賣出(張)', '淨買賣(張)', '買均']
        st.dataframe(show.style.format("{:.1f}", subset=['買進(張)','賣出(張)','淨買賣(張)']).format("{:.2f}", subset=['買均']).applymap(color_pnl, subset=['淨買賣(張)']), use_container_width=True, hide_index=True)

# ============================================
# 10. 視圖：📂 資料管理後台
# ============================================
def view_batch_import():
    st.header("📂 資料管理後台 (社長專用)")
    admin_pwd = st.sidebar.text_input("🔑 社長密碼 (上傳解鎖)", type="password")

    if admin_pwd == "8888":
        st.success("🔓 社長權限已解鎖！")
        st.subheader("📤 上傳今日 CSV (更新首頁資訊)")
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
    else: st.info("👋 這裡是後台管理區，請輸入密碼解鎖。")

# ============================================
# Main Loop (功能導航)
# ============================================
def main():
    with st.sidebar:
        st.title("🦅 Phoenix V104")
        st.caption("隱形自適應版")
        st.markdown("---")
        choice = st.radio("功能選單", ["🏠 總司令儀表板", "🧠 AI 戰略實驗室", "📈 趨勢戰情室", "🔍 獵殺雷達", "📉 籌碼斷層", "🕵️‍♂️ 分點偵探", "🏆 贏家與韭菜名人堂", "📂 資料管理後台"])
    
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