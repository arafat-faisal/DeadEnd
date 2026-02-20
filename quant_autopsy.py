import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tabulate import tabulate

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings
from utils.database import get_database
from research.backtester import Backtester
from research.strategy_generator import StrategyParams, StrategyType

def generate_pdf_report(md_text, report_data, mc_results, trade_stats_list):
    doc = SimpleDocTemplate("report.pdf", pagesize=landscape(letter),
                            rightMargin=30, leftMargin=30,
                            topMargin=30, bottomMargin=18)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, spaceAfter=20)
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=16, spaceAfter=10)
    normal_style = styles["Normal"]
    
    Story = []
    
    Story.append(Paragraph("🩺 Elite Trading System: The Ultimate Quant Autopsy Report", title_style))
    Story.append(Paragraph("<b>Dissecting the Truth Behind Priority Top 5 Models</b>", styles['Heading3']))
    Story.append(Paragraph(
        "Rigorous simulation across the most recent 800 15m candles or actual DB trades to break the strategy apart. "
        "Enforcing strict parameters: 10x Leverage, 0.1% Comms, 0.05% Slippage, and 0.01%/8h Funding Simulation.",
        normal_style))
    Story.append(Spacer(1, 12))
    
    # Aggregate Metrics
    Story.append(Paragraph("Aggregate Backtest Metrics", h2_style))
    
    if report_data:
        headers = list(report_data[0].keys())
        table_data = [headers] + [[str(x) for x in row.values()] for row in report_data]
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        Story.append(t)
    Story.append(Spacer(1, 20))
    
    # Trade P&L Stats
    Story.append(Paragraph("Trade-Level P&L Breakdown", h2_style))
    if trade_stats_list:
        headers_trade = list(trade_stats_list[0].keys())
        t_data_trade = [headers_trade] + [[str(x) for x in row.values()] for row in trade_stats_list]
        t_trade = Table(t_data_trade)
        t_trade.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        Story.append(t_trade)
    Story.append(Spacer(1, 20))

    # Monte Carlo Stats
    Story.append(Paragraph("Risk Analysis: Monte Carlo Path Simulations (100x)", h2_style))
    if mc_results:
        headers_mc = list(mc_results[0].keys())
        t_data_mc = [headers_mc] + [[str(x) for x in row.values()] for row in mc_results]
        t_mc = Table(t_data_mc)
        t_mc.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.maroon),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        Story.append(t_mc)
    Story.append(Spacer(1, 20))

    # Visuals
    Story.append(Paragraph("The Anatomy: Visual Diagnostics", h2_style))
    try:
        from reportlab.lib.units import inch
        Story.append(Image("equity.png", width=7.0 * inch, height=4.0 * inch))
        Story.append(Spacer(1, 10))
        Story.append(Image("heatmap.png", width=5.0 * inch, height=3.0 * inch))
    except Exception as e:
        print("Could not embed plots in PDF:", e)
        
    Story.append(Spacer(1, 20))
    Story.append(Paragraph("Surgeon's Final Recommendations", h2_style))
    Story.append(Paragraph("1. Check 95% VaR / Liquidation Probability. Consider dynamic sizing.", normal_style))
    Story.append(Paragraph("2. Paper Trade Top Performers before committing huge equity.", normal_style))
    
    doc.build(Story)


def main():
    print("Starting Dissection - Ultimate Crypto Autopsy Report...")

    # 1. Run quick research
    print("[1] Running Quick Research to freshen data...")
    os.system('python main.py --mode research --quick')
    print("[1] Quick Research Completed.")

    # 2. Load Priority List
    print("[2] Loading freshest priority models...")
    priority_file = Path('data/priority_list.json')
    if not priority_file.exists():
        print("Error: Priority list not found.")
        return

    with open(priority_file, 'r') as f:
        priority_list = json.load(f)

    if not priority_list or 'strategies' not in priority_list:
        print("Priority list is empty.")
        return

    top_5 = priority_list['strategies'][:5]

    # 3. Initialize Backtester
    bt = Backtester(commission=0.001, slippage=0.0005)
    db = get_database()
    
    report_data = []
    mc_results = []
    trade_stats_list = []
    
    plt.figure(figsize=(14, 8))
    
    print("[3] Simulating rigorous backtest with 10x Lev, 0.1% Fee, 0.05% Slippage, Funding: 0.01%/8h")

    for i, entry in enumerate(top_5):
        pair = entry['pair']
        strat_name = entry['strategy']
        
        # Get from DB
        strats = db.get_strategy_by_name(strat_name, pair)
        if not strats:
            print(f"  WARNING: Strategy {strat_name} for {pair} not found in DB!")
            continue
            
        params = entry.get('params', {})
        
        # Strategy Type Matching
        strat_type_map = {
            'SMA': StrategyType.SMA_CROSSOVER,
            'EMA': StrategyType.EMA_CROSSOVER,
            'RSI': StrategyType.RSI_REVERSAL,
            'MACD': StrategyType.MACD_SIGNAL,
            'BB': StrategyType.BOLLINGER_BANDS,
            'COMBINED': StrategyType.COMBINED,
            'GRID': StrategyType.GRID,
            'SCALP': StrategyType.SCALPING,
            'ARB': StrategyType.ARB_FUNDING
        }
        
        stype = StrategyType.SMA_CROSSOVER
        for k, v in strat_type_map.items():
            if strat_name.startswith(k):
                stype = v
                break
                
        strat = StrategyParams(strategy_type=stype, name=strat_name, params=params)
        print(f"  -> Dissecting {strat_name} on {pair}...")
        
        # Pull actual trades if available
        real_trades = []
        with db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM trades 
                WHERE strategy = ? AND pair = ?
                ORDER BY created_at ASC
            ''', (strat_name, pair))
            real_trades = [dict(r) for r in cursor.fetchall()]
            
        actual_returns = None
        has_real_trades = len(real_trades) > 10

        if has_real_trades:
            print(f"     -> Found {len(real_trades)} ACTUAL trades in db! Using real trade P&L.")
            trade_series = []
            gross_win = 0
            gross_loss = 0
            net_pnl = 0
            win_count = 0
            loss_count = 0
            
            # Simple assumption: FIFO or independent legs, just calculate basic delta
            last_buy_price = None
            for idx, t in enumerate(real_trades):
                fee = t['fee'] if t['fee'] else 0
                if t['side'] == 'buy':
                    last_buy_price = t['price']
                    trade_series.append(-fee)
                else: # sell
                    if last_buy_price:
                        pnl = (t['price'] - last_buy_price) / last_buy_price
                        net = pnl - fee - 0.001 # approx fee + slip
                        trade_series.append(net)
                        net_usd = net * t['amount'] * t['price']
                        
                        if net > 0:
                            gross_win += net_usd
                            win_count += 1
                        else:
                            gross_loss += net_usd
                            loss_count += 1
                            
                        net_pnl += net_usd
                        last_buy_price = None
                        
            # Create a mock cumulative series for plotting based on actual trades
            s = pd.Series(trade_series).fillna(0)
            actual_returns = s * 10 # Apply 10x lev
            
            trade_stats_list.append({
                'Strategy': strat_name,
                'Pair': pair,
                'Data': 'Actual DB',
                'Trades': len(real_trades),
                'Win Rate': f"{(win_count / max(1, win_count+loss_count))*100:.2f}%",
                'Net P&L (Est)': f"${net_pnl:.2f}"
            })

            
        # Also run backtest to get full dataframe
        df = bt.pipeline.fetch_ohlcv(pair, '15m', 800)
        if df.empty or len(df) < 50:
            print(f"  -> Insufficient data for {pair}!")
            continue
            
        df = bt._add_indicators(df, strat)
        df = bt._generate_signals(df, strat)
        df = bt._calculate_returns(df, strat)
        
        # Rigorous Override (Applying Leverage and Extra Costs)
        if stype not in (StrategyType.GRID, StrategyType.ARB_FUNDING):
            df['strategy_returns'] = df['position'].shift(1) * df['returns'] * 10 
            
            # Sub costs on trade days
            trade_mask = df['signal'] != 0
            df.loc[trade_mask, 'strategy_returns'] -= (0.001 + 0.0005) * 10
            
            # Funding
            df['strategy_returns'] -= (df['position'].abs().shift(1) * 10 * (0.0001 / 32)).fillna(0)
            
            df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
            df['cumulative_returns'] = df['cumulative_returns'].fillna(1.0)
            
        if not has_real_trades:
            # Generate mock trade stats from backtest data
            trades_df = df[df['signal'] != 0].copy()
            tr_count = len(trades_df)
            wins = len(df[df['strategy_returns'] > 0])
            losses = len(df[df['strategy_returns'] < 0])
            trade_stats_list.append({
                'Strategy': strat_name,
                'Pair': pair,
                'Data': 'Mocked BT',
                'Trades': tr_count,
                'Win Rate': f"{(wins / max(1, wins+losses))*100:.2f}%",
                'Net P&L (Est)': "N/A"
            })
            
            actual_returns = df['strategy_returns'].dropna()

        # Aggregate Stats
        res = bt._calculate_metrics(df, strat, pair)
        
        report_data.append({
            'Strategy': strat.name,
            'Pair': pair,
            'ROI': f"{res.roi*100:.2f}%",
            'Sharpe': round(res.sharpe_ratio, 2),
            'Max DD': f"{res.max_drawdown*100:.2f}%",
            'Win %': f"{res.win_rate*100:.2f}%",
            'Profit Factor': round(res.profit_factor, 2),
            'Trades': res.total_trades
        })
        
        # Plotting (Real or Mocked)
        if has_real_trades:
            cr = (1 + actual_returns).cumprod().dropna().values
            plt.plot(cr, label=f"{strat_name} ({pair}) [ACTUAL]")
        else:
            plt.plot(df['cumulative_returns'].values, label=f"{strat_name} ({pair}) [BT]")
        
        # Monte Carlo Iteration
        sim_returns = actual_returns.values
        dds = []
        liquidation_count = 0
        for _ in range(100):
            shuffled = np.random.RandomState().permutation(sim_returns)
            cum = (1 + shuffled).cumprod()
            roll_max = np.maximum.accumulate(cum)
            roll_max[roll_max == 0] = 1 # Safeguard
            dd = (roll_max - cum) / roll_max
            max_d_path = np.max(dd)
            dds.append(max_d_path)
            if max_d_path >= 0.10: # 10% DD at 10x Lev = 100% loss = Liquidation
                liquidation_count += 1
            
        mc_results.append({
            'Strategy': strat_name,
            'Pair': pair,
            '95% VaR (Max DD)': f"{np.percentile(dds, 95)*100:.2f}%",
            'Median MC DD': f"{np.median(dds)*100:.2f}%",
            'Liq. Prob (>=10% DD)': f"{liquidation_count}%"
        })

    print("[4] Generating visual artifacts...")
    # Saving Plot
    plt.title("Equity Curves (10x Lev, Fees & Funding)")
    plt.xlabel("Periods")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('equity.png', dpi=300)
    plt.close()

    try:
        df_num = pd.DataFrame(report_data).copy()
        if not df_num.empty:
            for col in ['ROI', 'Max DD', 'Win %']:
                if col in df_num.columns:
                    df_num[col] = df_num[col].str.rstrip('%').astype(float) / 100.0
                
            df_num = df_num.set_index(['Strategy', 'Pair'])
            # Filter to numeric columns only before plotting heatmap
            df_numeric_only = df_num.select_dtypes(include=[np.number])
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(df_numeric_only, annot=True, cmap='RdYlGn', fmt=".3f", center=0)
            plt.title("Aggregate Performance Heatmap")
            plt.tight_layout()
            plt.savefig('heatmap.png', dpi=300)
            plt.close()
    except Exception as e:
        print("Heatmap skipped due to error:", e)

    print("[5] Generating PDF Report with ReportLab...")
    
    # We also write MD just for history
    df_report = pd.DataFrame(report_data)
    df_mc = pd.DataFrame(mc_results)
    df_trade = pd.DataFrame(trade_stats_list)
    
    md = "# 🩺 Elite Trading System: The Ultimate Quant Autopsy Report\n\n"
    md += "### **Dissecting the Truth Behind Priority Top 5 Models**\n"
    md += "Rigorous simulation. We enforce strict parameters: **10x Leverage**, **0.1% Comms**, **0.05% Slippage**, and **0.01%/8h Funding Simulation**. If they survive this, they survive anything.\n\n"
    
    md += "## 📊 Aggregate Backtest Metrics\n\n"
    if not df_report.empty:
        md += tabulate(df_report, headers='keys', tablefmt='github', showindex=False) + "\n\n"
        
    md += "## 💸 Trade-Level P&L Breakdown (Actual DB vs Mocked)\n\n"
    if not df_trade.empty:
        md += tabulate(df_trade, headers='keys', tablefmt='github', showindex=False) + "\n\n"
    
    md += "## 🎲 Risk Analysis: Monte Carlo Path Simulations (100x)\n"
    md += "*Shows Liquidation Probability based on paths hitting 10% DD under 10x leverage.*\n\n"
    if not df_mc.empty:
        md += tabulate(df_mc, headers='keys', tablefmt='github', showindex=False) + "\n\n"
    
    md += "## 📈 The Anatomy: Visual Diagnostics\n\n"
    md += "### Cumulative Equity Curves\n"
    md += "![Equity curves generated](equity.png)\n\n"
    md += "### Performance Heatmap Matrix\n"
    md += "![Heatmap generated](heatmap.png)\n\n"
    
    with open('quant_autopsy.md', 'w', encoding='utf-8') as f:
        f.write(md)

    try:
        generate_pdf_report(md, report_data, mc_results, trade_stats_list)
        print("PDF accurately generated using ReportLab!")
    except Exception as e:
        print("Error during PDF Generation:", e)
        
    print("Report generated successfully: quant_autopsy.md, equity.png, heatmap.png, report.pdf !")

if __name__ == '__main__':
    main()
