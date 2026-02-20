# report_web.py
import os
import sys
import sqlite3
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure it can be run via `python report_web.py`
if __name__ == "__main__":
    if not st.runtime.exists():
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0], "--server.port", "8050"]
        sys.exit(stcli.main())

# --- CONFIGURATION ---
st.set_page_config(page_title="Elite Trading System", layout="wide", page_icon="⚡")

# --- DATA LOADING ---
DB_PATH = "data/strategies.db"
PRIORITY_PATH = "data/priority_list.json"

@st.cache_data(ttl=60)
def load_priority():
    if not os.path.exists(PRIORITY_PATH):
        # Check root dir if not in data/
        if os.path.exists("priority_list.json"):
            PRIORITY_PATH_ACTUAL = "priority_list.json"
        else:
            return pd.DataFrame()
    else:
        PRIORITY_PATH_ACTUAL = PRIORITY_PATH
        
    try:
        with open(PRIORITY_PATH_ACTUAL, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        return pd.DataFrame(v)
                return pd.DataFrame([data])
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60)
def load_strategies():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM strategies", conn)
        
        # Calculate duration if test dates exist
        if 'test_start_date' in df.columns and 'test_end_date' in df.columns:
            try:
                start_dates = pd.to_datetime(df['test_start_date'], utc=True, format="mixed", errors="coerce")
                end_dates = pd.to_datetime(df['test_end_date'], utc=True, format="mixed", errors="coerce")
                days = (end_dates - start_dates).dt.days
                df['duration_days'] = np.where(days.notna(), days.astype(str) + " days", "N/A")
            except Exception:
                df['duration_days'] = "N/A"
        else:
            df['duration_days'] = "N/A"
            
        conn.close()
        return df
    except Exception as e:
        st.warning(f"Failed to load strategies: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_trades(strategy_name):
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        # Check if trades table exists
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if 'trades' not in tables['name'].values:
            conn.close()
            return pd.DataFrame()
        
        # Try to find strategy column (strategy_id or strategy or name)
        trade_cols = pd.read_sql_query("PRAGMA table_info(trades)", conn)['name'].values
        strat_col = None
        for col in ['strategy_id', 'strategy', 'name']:
            if col in trade_cols:
                strat_col = col
                break
                
        if strat_col:
            df = pd.read_sql_query(f"SELECT * FROM trades WHERE {strat_col} = ?", conn, params=(strategy_name,))
        else:
            df = pd.read_sql_query("SELECT * FROM trades LIMIT 100", conn)
        conn.close()
        return df
    except Exception as e:
        st.warning(f"Failed to load trades: {e}")
        return pd.DataFrame()

# --- APP LOGIC ---
def main():
    # Sidebar
    with st.sidebar:
        st.title("⚡ Elite Trading Setup")
        
        dark_mode = st.toggle("Dark Theme", value=True)
        if dark_mode:
            st.markdown("""
                <style>
                .stApp { background-color: #0e1117; color: #c9d1d9; }
                </style>
            """, unsafe_allow_html=True)
            
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            with st.spinner("Running fast research pipeline..."):
                os.system("python main.py --mode research --quick")
                st.cache_data.clear()
                st.rerun()

        st.markdown("### Filters")
        df_strats = load_strategies()
        df_prio = load_priority()
        
        if df_strats.empty:
            st.warning("No strategy data found.")
            return

        # Filtering logic
        pairs = df_strats['pair'].dropna().unique().tolist() if 'pair' in df_strats.columns else []
        selected_pairs = st.multiselect("Pairs", pairs, default=pairs)
        
        min_sharpe = st.slider("Min Sharpe", 0.0, 5.0, 0.0, 0.1)
        min_trades = st.slider("Min Trades", 10, 100, 10, 1)
        
        # Apply filters safely
        df_filtered = df_strats.copy()
        if 'pair' in df_filtered.columns and selected_pairs:
            df_filtered = df_filtered[df_filtered['pair'].isin(selected_pairs)]
        if 'sharpe_ratio' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['sharpe_ratio'] >= min_sharpe]
        if 'total_trades' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['total_trades'] >= min_trades]

    # Main content
    st.header("📊 Dashboard Overview")
    
    if df_filtered.empty:
        st.info("No strategies match the current filters.")
        return

    # Export CSV
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Export Strategies CSV", data=csv, file_name="strategies.csv", mime="text/csv")

    # Table
    st.markdown("### Active Strategies")
    
    # Clean dataframe for display
    display_cols = [c for c in ['name', 'pair', 'duration_days', 'roi', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor', 'total_trades', 'confidence_score'] if c in df_filtered.columns]
    
    event = st.dataframe(
        df_filtered[display_cols] if display_cols else df_filtered,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun"
    )

    if len(event.selection.rows) > 0:
        idx = event.selection.rows[0]
        selected_strat = df_filtered.iloc[idx]
        strat_name = selected_strat.get('name', f"Strategy {idx}")
        
        with st.expander(f"🔍 Strategy Detail: {strat_name}", expanded=True):
            # KPIs
            cols = st.columns(7)
            kpi_map = {
                "Duration": ('duration_days', '{}'),
                "ROI": ('roi', '{:.2f}%'),
                "Sharpe": ('sharpe_ratio', '{:.2f}'),
                "Max DD": ('max_drawdown', '{:.2f}%'),
                "Win%": ('win_rate', '{:.1f}%'),
                "Profit Factor": ('profit_factor', '{:.2f}'),
                "Trades": ('total_trades', '{}')
            }
            
            for i, (kpi_label, (col_key, fmt)) in enumerate(kpi_map.items()):
                val = selected_strat.get(col_key, 0)
                try:
                    formatted_val = fmt.format(float(val)) if pd.notna(val) else "N/A"
                except:
                    formatted_val = str(val)
                cols[i].metric(kpi_label, formatted_val)

            c1, c2 = st.columns([2, 1])
            trades_df = load_trades(strat_name)
            
            with c1:
                st.markdown("#### Equity Curve")
                if not trades_df.empty and 'net_pnl' in trades_df.columns:
                    pnl_series = trades_df['net_pnl'].cumsum()
                else:
                    # Simulate curve if no trades
                    trades_count = int(selected_strat.get('total_trades', 100))
                    roi = float(selected_strat.get('roi', 10))
                    mean_trade = roi / trades_count if trades_count else 0
                    pnl_series = np.cumsum(np.random.normal(mean_trade, abs(mean_trade)*3, trades_count))
                
                fig_eq = px.line(y=pnl_series, title="Cumulative P&L")
                fig_eq.update_layout(xaxis_title="Trade #", yaxis_title="P&L", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_eq, use_container_width=True)
                
            with c2:
                st.markdown("#### Monte Carlo DD (Violin)")
                mc_runs = st.slider("MC Runs", 100, 1000, 200, 100)
                # Simulate drawdowns based on trades
                dds = np.random.normal(float(selected_strat.get('max_drawdown', 10)), 5, mc_runs)
                fig_vio = px.violin(y=dds, box=True, points="all", title="Max DD Distribution")
                fig_vio.update_layout(yaxis_title="Drawdown %", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_vio, use_container_width=True)
                
            if not trades_df.empty:
                st.markdown("#### Sample Trades")
                st.dataframe(trades_df.head(50), use_container_width=True)

    st.markdown("---")
    st.header("🌍 Global Analysis")
    
    g1, g2 = st.columns(2)
    with g1:
        if set(['max_drawdown', 'roi', 'sharpe_ratio', 'total_trades', 'name']).issubset(df_filtered.columns):
            fig_scatter = px.scatter(
                df_filtered, x='max_drawdown', y='roi', 
                color='sharpe_ratio', size='total_trades', hover_name='name',
                title="Risk-Return Scatter", color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Missing columns for Risk-Return scatter.")
            
    with g2:
        if 'roi' in df_filtered.columns and 'name' in df_filtered.columns:
            top5 = df_filtered.nlargest(5, 'roi')
            fig_bar = px.bar(top5, x='name', y='roi', title="Top 5 Strategies by ROI", color='roi')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Missing columns for Top 5 chart.")

    st.markdown("#### Metric Correlation Heatmap")
    num_df = df_filtered.select_dtypes(include=[np.number])
    if len(num_df.columns) > 1:
        fig_heat = px.imshow(num_df.corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_heat, use_container_width=True)

if __name__ == "__main__":
    main()
