import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np

# --- UI SETUP ---
st.set_page_config(page_title="Master Confluence Backtester", layout="wide")
st.title("XAUT/USD Master Strategy: S&D + Liquidity Sweep")

st.sidebar.header("Confluence Parameters")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=1000)
lot_size = st.sidebar.number_input("Lot Size (oz)", value=1.0) 
momentum_mult = st.sidebar.slider("S&D Momentum (ATR)", 1.0, 4.0, 2.0, 0.1)
pivot_len = st.sidebar.number_input("Liquidity Swing Length", min_value=2, value=3)
rr_ratio = st.sidebar.slider("Risk Reward Ratio", 1.0, 5.0, 2.0, 0.5)
days_history = st.sidebar.slider("Days of Data (15m TF)", 5, 59, 30)

# --- DATA FETCHING ---
@st.cache_data(ttl=900)
def get_data(days):
    df = yf.download("GC=F", period=f"{days}d", interval="15m")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df

with st.spinner("Calculating Confluence Zones..."):
    df = get_data(days_history)

# --- CONFLUENCE ENGINE ---
if not df.empty:
    # Math & Indicators
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(alpha=1/14, adjust=False).mean()
    df['Body'] = abs(df['Close'] - df['Open'])

    df['Pivot_High'] = df['High'].rolling(window=pivot_len*2+1, center=True).max()
    df['Pivot_Low'] = df['Low'].rolling(window=pivot_len*2+1, center=True).min()
    df['Liquidity_High'] = df['Pivot_High'].ffill()
    df['Liquidity_Low'] = df['Pivot_Low'].ffill()

    df['Bull_Sweep'] = (df['Low'] < df['Liquidity_Low']) & (df['Close'] > df['Liquidity_Low'])
    df['Bear_Sweep'] = (df['High'] > df['Liquidity_High']) & (df['Close'] < df['Liquidity_High'])

    balance = initial_capital
    equity_curve = []
    trade_log = []
    
    in_pos = False
    pos_type = None
    entry_price, sl, tp = 0, 0, 0
    
    active_demand_zones = []
    active_supply_zones = []
    zone_shapes = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        date = df.index[i]
        prev_row = df.iloc[i-1]
        
        # 1. Manage Exits
        if in_pos:
            if pos_type == 'LONG':
                if row['Low'] <= sl: 
                    loss = (sl - entry_price) * lot_size
                    balance += loss
                    trade_log.append({'Date': date, 'Type': 'LONG', 'Entry': entry_price, 'Target': tp, 'StopLoss': sl, 'Result': 'Loss', 'PnL': loss})
                    in_pos = False
                elif row['High'] >= tp: 
                    profit = (tp - entry_price) * lot_size
                    balance += profit
                    trade_log.append({'Date': date, 'Type': 'LONG', 'Entry': entry_price, 'Target': tp, 'StopLoss': sl, 'Result': 'Win', 'PnL': profit})
                    in_pos = False
            
            elif pos_type == 'SHORT':
                if row['High'] >= sl: 
                    loss = (entry_price - sl) * lot_size
                    balance += loss
                    trade_log.append({'Date': date, 'Type': 'SHORT', 'Entry': entry_price, 'Target': tp, 'StopLoss': sl, 'Result': 'Loss', 'PnL': loss})
                    in_pos = False
                elif row['Low'] <= tp: 
                    profit = (entry_price - tp) * lot_size
                    balance += profit
                    trade_log.append({'Date': date, 'Type': 'SHORT', 'Entry': entry_price, 'Target': tp, 'StopLoss': sl, 'Result': 'Win', 'PnL': profit})
                    in_pos = False

        equity_curve.append(balance)

        # 2. Master Confluence Entries
        if not in_pos:
            # Check Long Confluence: Bull Sweep inside Active Demand
            if row['Bull_Sweep']:
                for z in active_demand_zones[:]:
                    if row['Low'] <= z['top'] and row['Low'] >= z['bottom']: # Tapped zone during sweep
                        entry_price = row['Close'] # Enter on candle close after sweep
                        sl = min(row['Low'], z['bottom']) - (row['ATR'] * 0.2)
                        risk = entry_price - sl
                        if risk > 0:
                            tp = entry_price + (risk * rr_ratio)
                            pos_type = 'LONG'
                            in_pos = True
                            active_demand_zones.remove(z)
                            break
            
            # Check Short Confluence: Bear Sweep inside Active Supply
            if not in_pos and row['Bear_Sweep']:
                for z in active_supply_zones[:]:
                    if row['High'] >= z['bottom'] and row['High'] <= z['top']:
                        entry_price = row['Close']
                        sl = max(row['High'], z['top']) + (row['ATR'] * 0.2)
                        risk = sl - entry_price
                        if risk > 0:
                            tp = entry_price - (risk * rr_ratio)
                            pos_type = 'SHORT'
                            in_pos = True
                            active_supply_zones.remove(z)
                            break

        # 3. Clean up broken zones
        active_demand_zones = [z for z in active_demand_zones if row['Close'] > z['bottom']]
        active_supply_zones = [z for z in active_supply_zones if row['Close'] < z['top']]

        # 4. Identify New S&D Zones
        is_expansion = row['Body'] > (row['ATR'] * momentum_mult)
        if is_expansion:
            if row['Close'] > row['Open']:
                zone = {'top': prev_row['High'], 'bottom': prev_row['Low'], 'start': date}
                active_demand_zones.append(zone)
                zone_shapes.append(dict(type="rect", x0=zone['start'], y0=zone['bottom'], x1=df.index[-1], y1=zone['top'], fillcolor="rgba(0, 255, 0, 0.1)", line=dict(width=0)))
            elif row['Close'] < row['Open']:
                zone = {'top': prev_row['High'], 'bottom': prev_row['Low'], 'start': date}
                active_supply_zones.append(zone)
                zone_shapes.append(dict(type="rect", x0=zone['start'], y0=zone['bottom'], x1=df.index[-1], y1=zone['top'], fillcolor="rgba(255, 0, 0, 0.1)", line=dict(width=0)))

    # Ensure equity curve aligns by prepending initial capital for the 0th index
    df['Equity'] = [initial_capital] + equity_curve[:-1]
    trade_df = pd.DataFrame(trade_log)

    # --- UI DISPLAY ---
    tab1, tab2 = st.tabs(["Dashboard & Chart", "Trade Log"])

    with tab1:
        total_trades = len(trade_df)
        win_rate = (len(trade_df[trade_df['Result'] == 'Win']) / total_trades * 100) if total_trades > 0 else 0
        net_profit = balance - initial_capital

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Balance", f"${balance:,.2f}", f"{net_profit:,.2f}")
        col2.metric("Total Trades", total_trades)
        col3.metric("Win Rate", f"{win_rate:.1f}%")
        col4.metric("Risk / Reward", f"1 : {rr_ratio}")

        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
        fig.add_trace(go.Scatter(x=df.index, y=df['Liquidity_High'], mode='lines', line=dict(color='red', width=1, dash='dot'), name='Liq High'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Liquidity_Low'], mode='lines', line=dict(color='green', width=1, dash='dot'), name='Liq Low'))

        if not trade_df.empty:
            longs = trade_df[trade_df['Type'] == 'LONG']
            shorts = trade_df[trade_df['Type'] == 'SHORT']
            
            fig.add_trace(go.Scatter(x=longs['Date'], y=longs['Entry'], mode='markers', marker=dict(color='blue', size=12, symbol='triangle-up'), name='Long Entry'))
            fig.add_trace(go.Scatter(x=longs['Date'], y=longs['Target'], mode='markers', marker=dict(color='green', size=7, symbol='circle'), name='Long Target'))
            fig.add_trace(go.Scatter(x=longs['Date'], y=longs['StopLoss'], mode='markers', marker=dict(color='red', size=7, symbol='x'), name='Long SL'))

            fig.add_trace(go.Scatter(x=shorts['Date'], y=shorts['Entry'], mode='markers', marker=dict(color='magenta', size=12, symbol='triangle-down'), name='Short Entry'))
            fig.add_trace(go.Scatter(x=shorts['Date'], y=shorts['Target'], mode='markers', marker=dict(color='green', size=7, symbol='circle'), name='Short Target'))
            fig.add_trace(go.Scatter(x=shorts['Date'], y=shorts['StopLoss'], mode='markers', marker=dict(color='red', size=7, symbol='x'), name='Short SL'))

        fig.update_layout(height=700, template='plotly_dark', title="Master Confluence: Sweeps inside S&D Zones", xaxis_rangeslider_visible=False, shapes=zone_shapes[-30:]) 
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if not trade_df.empty:
            st.dataframe(trade_df[['Date', 'Type', 'Entry', 'Target', 'StopLoss', 'Result', 'PnL']].style.map(lambda x: 'color: green' if x == 'Win' else ('color: red' if x == 'Loss' else ''), subset=['Result']))
        else:
            st.info("No Confluence Trades executed. This is a high-probability, low-frequency strategy.")

else:
    st.error("Failed to fetch data.")
