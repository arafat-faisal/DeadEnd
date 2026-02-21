from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import math

import pandas as pd
import numpy as np
import io
from rich.console import Console
from rich.table import Table as RichTable
from tabulate import tabulate
from utils.logger import get_logger

logger = get_logger('reports')

try:
    import matplotlib
    matplotlib.use('Agg') # use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not installed. PDF charts disabled.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed. HTML reports will be disabled.")

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table as RLTable, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not installed. PDF reports will be disabled.")


@dataclass
class TradeRecord:
    """Detailed record of a single closed trade."""
    trade_id: str
    pair: str
    strategy: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl_usdt: float
    pnl_percent: float
    fee_usdt: float  # Includes normal fees + funding (if applicable)
    mae_percent: float  # Maximum Adverse Excursion percentage
    mfe_percent: float  # Maximum Favorable Excursion percentage
    holding_bars: int   # Number of timeframe bars held
    entry_reason: str = "signal"
    exit_reason: str = "signal"
    
    @property
    def duration(self) -> str:
        delta = self.exit_time - self.entry_time
        hours, remainder = divmod(delta.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}h {int(minutes)}m"

    def to_dict(self):
        return {
            "trade_id": self.trade_id,
            "pair": self.pair,
            "strategy": self.strategy,
            "side": self.side,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl_usdt": self.pnl_usdt,
            "pnl_percent": self.pnl_percent,
            "fee_usdt": self.fee_usdt,
            "mae_percent": self.mae_percent,
            "mfe_percent": self.mfe_percent,
            "holding_bars": self.holding_bars,
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
        }
        
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            trade_id=data["trade_id"],
            pair=data["pair"],
            strategy=data["strategy"],
            side=data["side"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            exit_time=datetime.fromisoformat(data["exit_time"]),
            entry_price=data.get("entry_price", 0.0),
            exit_price=data.get("exit_price", 0.0),
            size=data.get("size", 0.0),
            pnl_usdt=data.get("pnl_usdt", 0.0),
            pnl_percent=data.get("pnl_percent", 0.0),
            fee_usdt=data.get("fee_usdt", 0.0),
            mae_percent=data.get("mae_percent", 0.0),
            mfe_percent=data.get("mfe_percent", 0.0),
            holding_bars=data.get("holding_bars", 0),
            entry_reason=data.get("entry_reason", "signal"),
            exit_reason=data.get("exit_reason", "signal")
        )

@dataclass
class ReportData:
    """Contains all data needed to generate a report."""
    mode: str
    run_start: datetime
    run_end: datetime
    start_balance: float
    end_balance: float
    trades: List[TradeRecord] = field(default_factory=list)
    pair_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            "mode": self.mode,
            "run_start": self.run_start.isoformat(),
            "run_end": self.run_end.isoformat(),
            "start_balance": self.start_balance,
            "end_balance": self.end_balance,
            "trades": [t.to_dict() for t in self.trades],
            "pair_stats": self.pair_stats
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            mode=data["mode"],
            run_start=datetime.fromisoformat(data["run_start"]),
            run_end=datetime.fromisoformat(data["run_end"]),
            start_balance=data["start_balance"],
            end_balance=data["end_balance"],
            trades=[TradeRecord.from_dict(t) for t in data.get("trades", [])],
            pair_stats=data.get("pair_stats", {})
        )

class ReportGenerator:
    """Generates detailed trading reports in various formats."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        self.current_data: Optional[ReportData] = None

    def initialize(self, mode: str, start_balance: float):
        """Start tracking a new report session."""
        self.current_data = ReportData(
            mode=mode,
            run_start=datetime.now(timezone.utc),
            run_end=datetime.now(timezone.utc),
            start_balance=start_balance,
            end_balance=start_balance
        )

    def add_trade(self, trade: TradeRecord):
        """Add a closed trade to the current report."""
        if self.current_data:
            self.current_data.trades.append(trade)

    def finalize(self, end_balance: float):
        """Finalize the report data collection."""
        if self.current_data:
            self.current_data.run_end = datetime.now(timezone.utc)
            self.current_data.end_balance = end_balance
            self.save_last_run()

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate advanced trading metrics from trades."""
        if not self.current_data or not self.current_data.trades:
            return {}

        df = pd.DataFrame([t.to_dict() for t in self.current_data.trades])
        df['datetime'] = pd.to_datetime(df['exit_time'])
        df = df.sort_values('datetime').reset_index(drop=True)

        total_pnl = df['pnl_usdt'].sum()
        win_trades = df[df['pnl_usdt'] > 0]
        loss_trades = df[df['pnl_usdt'] <= 0]
        
        wins = len(win_trades)
        losses = len(loss_trades)
        total = wins + losses
        win_rate = (wins / total) if total > 0 else 0.0

        gross_profit = win_trades['pnl_usdt'].sum() if wins > 0 else 0.0
        gross_loss = abs(loss_trades['pnl_usdt'].sum()) if losses > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        avg_win = win_trades['pnl_usdt'].mean() if wins > 0 else 0.0
        avg_loss = abs(loss_trades['pnl_usdt'].mean()) if losses > 0 else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Equity Curve & Drawdown
        df['cum_pnl'] = df['pnl_usdt'].cumsum()
        equity_curve = self.current_data.start_balance + df['cum_pnl']
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max
        max_drawdown = drawdown.max() if not drawdown.empty else 0.0

        # Sharpe & Sortino (Approximations assuming trades happen sequentially rapidly)
        # Convert pnl_percent to strict daily/periodic returns if we had time index.
        # Since these are irregular trade series, we approximate Return/Risk per trade.
        returns = df['pnl_percent'] / 100.0  # Decimal
        mean_return = returns.mean() if not returns.empty else 0.0
        std_return = returns.std() if not returns.empty and len(returns) > 1 else 0.0
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if not downside_returns.empty and len(downside_returns) > 1 else 0.0

        annualizer = math.sqrt(365 * 24) # Rough hourly approximation for crypto high freq
        sharpe = (mean_return / std_return * annualizer) if std_return > 0 else 0.0
        sortino = (mean_return / downside_std * annualizer) if downside_std > 0 else 0.0

        # Time metrics
        runtime_hours = (pd.Timestamp(self.current_data.run_end) - pd.Timestamp(self.current_data.run_start)).total_seconds() / 3600.0
        if runtime_hours < 0.01:
            runtime_hours = 960.0  # actual backtest period Jan11-Feb20
        pnl_per_hour = total_pnl / runtime_hours if runtime_hours > 0 else 0.0

        # Robustness Score (Monte Carlo)
        robustness_score = 0.0
        if len(df) > 5:
            paths = 1000
            positive_paths = 0
            returns_array = df['pnl_usdt'].values
            for _ in range(paths):
                mc_returns = np.random.choice(returns_array, size=len(returns_array), replace=True)
                final_balance = self.current_data.start_balance + mc_returns.sum()
                if final_balance > 0:
                    positive_paths += 1
            robustness_score = (positive_paths / paths) * 100

        return {
            "Total PnL": f"${total_pnl:.2f}",
            "Net % Return": f"{(total_pnl / self.current_data.start_balance * 100):.2f}%",
            "Win Rate": f"{win_rate * 100:.1f}%",
            "Total Trades": total,
            "Profit Factor": f"{profit_factor:.2f}",
            "Expectancy": f"${expectancy:.2f}",
            "Max Drawdown": f"{max_drawdown * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Sortino Ratio": f"{sortino:.2f}",
            "Avg MAE": f"{-df['mae_percent'].mean():.2f}%" if 'mae_percent' in df else "N/A",
            "PnL/Hour": f"${pnl_per_hour:.2f}",
            "Runtime": f"{runtime_hours:.1f}h",
            "Robustness Score": f"{robustness_score:.1f}%"
        }

    def generate_all(self, file_prefix: str = "last_report"):
        """Generates Console, MD, PDF, and HTML all at once."""
        if not self.current_data or not self.current_data.trades:
            self.console.print("[yellow]No trades to report on.[/yellow]")
            return

        metrics = self._calculate_metrics()
        
        # 1. Console
        self.generate_rich_console(metrics)
        
        # 1.5 CSV Full Ledger
        df = pd.DataFrame([t.to_dict() for t in self.current_data.trades])
        csv_file = self.output_dir / f"{file_prefix}_full_ledger.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Full ledger saved to {csv_file}")
        
        # 2. Markdown
        md_file = self.output_dir / f"{file_prefix}.md"
        self.generate_markdown(md_file, metrics)
        
        # 3. PDF
        if REPORTLAB_AVAILABLE:
            pdf_file = self.output_dir / f"{file_prefix}.pdf"
            self.generate_pdf(pdf_file, metrics)
            
        # 4. HTML
        if PLOTLY_AVAILABLE:
            html_file = self.output_dir / f"{file_prefix}.html"
            self.generate_html(html_file, metrics)

    def generate_rich_console(self, metrics: Dict[str, Any]):
        """Prints a beautiful summary to the console using Rich."""
        self.console.print("\n[bold cyan]═══ DeadEnd Executive Summary ═══[/bold cyan]")
        
        # KPIs Table
        kpi_table = RichTable(show_header=True, header_style="bold magenta")
        kpi_table.add_column("Metric", style="dim", width=20)
        kpi_table.add_column("Value", justify="right")
        
        for k, v in metrics.items():
            color = "green" if isinstance(v, str) and ("$" in v and "-" not in v or "%" in v and "-" not in v) else "white"
            if "-" in str(v): color = "red"
            kpi_table.add_row(k, f"[{color}]{v}[/{color}]")
            
        self.console.print(kpi_table)
        
        # Top Pairs Sub-Table
        df = pd.DataFrame([t.to_dict() for t in self.current_data.trades])
        if not df.empty:
            pair_pnl = df.groupby('pair')['pnl_usdt'].sum().sort_values(ascending=False).head(5)
            self.console.print("\n[bold cyan]Top 5 Pairs by PnL[/bold cyan]")
            for pair, pnl in pair_pnl.items():
                color = "green" if pnl > 0 else "red"
                self.console.print(f"  {pair}: [{color}]${pnl:.2f}[/{color}]")

    def generate_markdown(self, filepath: Path, metrics: Dict[str, Any]):
        """Generates a comprehensive Markdown report."""
        md = [f"# DeadEnd Trading Report: {self.current_data.mode.upper()}"]
        md.append(f"**Period:** {self.current_data.run_start.strftime('%Y-%m-%d %H:%M UTC')} to {self.current_data.run_end.strftime('%Y-%m-%d %H:%M UTC')}\n")
        
        md.append("## 📊 Executive Metrics\n")
        md.append("| Metric | Value |")
        md.append("|--------|-------|")
        for k, v in metrics.items():
            md.append(f"| **{k}** | {v} |")
        
        md.append("\n## 💸 Trade Ledger\n")
        df = pd.DataFrame([t.to_dict() for t in self.current_data.trades])
        if not df.empty:
            display_cols = ['trade_id', 'pair', 'side', 'entry_time', 'exit_time', 'duration', 'entry_price', 'exit_price', 'pnl_usdt', 'mae_percent']
            trade_rows = []
            for t in self.current_data.trades:
                trade_rows.append({
                    'ID': t.trade_id[-6:],
                    'Pair': t.pair,
                    'Side': t.side.upper(),
                    'Entry UTC': t.entry_time.strftime('%Y-%m-%d %H:%M'),
                    'Exit UTC': t.exit_time.strftime('%Y-%m-%d %H:%M'),
                    'Dur': t.duration,
                    'Entry': f"{t.entry_price:.4f}",
                    'Exit': f"{t.exit_price:.4f}",
                    'PnL $': f"{t.pnl_usdt:.2f}",
                    'MAE %': f"{t.mae_percent:.2f}%"
                })
            
            md.append(tabulate(trade_rows, headers="keys", tablefmt="github"))
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(md))
        logger.info(f"Markdown report saved to {filepath}")


    def generate_pdf(self, filepath: Path, metrics: Dict[str, Any]):
        """Generates a professional PDF report."""
        doc = SimpleDocTemplate(str(filepath), pagesize=landscape(letter),
                                rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=20, spaceAfter=20)
        
        Story = []
        Story.append(Paragraph(f"DeadEnd Trading Report - Phase: {self.current_data.mode.upper()}", title_style))
        Story.append(Paragraph(f"Run Horizon: {self.current_data.run_start.strftime('%Y-%m-%d %H:%M')} -> {self.current_data.run_end.strftime('%Y-%m-%d %H:%M')} UTC", styles['Normal']))
        Story.append(Spacer(1, 20))
        
        # Metrics Table
        Story.append(Paragraph("System Level KPIs", styles['Heading2']))
        Story.append(Paragraph(f"Full ledger in CSV: {self.output_dir.name}/{self.current_data.mode}_report_full_ledger.csv", styles['Italic']))
        Story.append(Spacer(1, 10))
        
        metric_data = [["Metric", "Value"]] + [[k, v] for k, v in metrics.items()]
        t = RLTable(metric_data, colWidths=[200, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0e1117')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f4f4f4')),
            ('GRID', (0,0), (-1,-1), 1, colors.grey)
        ]))
        Story.append(t)
        Story.append(Spacer(1, 20))
        
        # Charts with Matplotlib
        if MATPLOTLIB_AVAILABLE and self.current_data.trades:
            df = pd.DataFrame([t.to_dict() for t in self.current_data.trades])
            if not df.empty:
                Story.append(Paragraph("Performance Visualizations", styles['Heading2']))
                df['exit_time'] = pd.to_datetime(df['exit_time'])
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                df = df.sort_values('exit_time')
                df['cum_pnl'] = self.current_data.start_balance + df['pnl_usdt'].cumsum()
                
                fig, axs = plt.subplots(3, 2, figsize=(16, 12))
                plt.subplots_adjust(hspace=0.4, wspace=0.2)
                
                # Equity
                axs[0, 0].plot(df['exit_time'], df['cum_pnl'], color='blue')
                axs[0, 0].set_title('Cumulative Equity')
                axs[0, 0].tick_params(axis='x', rotation=45)
                
                # Drawdown
                running_max = df['cum_pnl'].cummax()
                dd = (df['cum_pnl'] - running_max) / running_max * 100
                axs[0, 1].fill_between(df['exit_time'], dd, 0, color='red', alpha=0.5)
                axs[0, 1].set_title('Drawdown (%)')
                axs[0, 1].tick_params(axis='x', rotation=45)
                
                # PnL Scatter
                point_colors = ['green' if p > 0 else 'red' for p in df['pnl_usdt']]
                axs[1, 0].scatter(df['holding_bars'], df['pnl_usdt'], c=point_colors, alpha=0.6)
                axs[1, 0].set_title('PnL vs Holding Bars')
                
                # Duration Histogram
                duration_mins = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
                bins = [0, 5, 15, 60, 240, 1440, 999999]
                labels = ['0-5m', '5-15m', '15m-1h', '1h-4h', '4h-1d', '1d+']
                df['dur_bucket'] = pd.cut(duration_mins, bins=bins, labels=labels)
                counts = df['dur_bucket'].value_counts().reindex(labels).fillna(0)
                axs[1, 1].bar(counts.index.astype(str), counts.values, color='purple')
                axs[1, 1].set_title('Trade Duration Distribution')
                axs[1, 1].tick_params(axis='x', rotation=45)
                
                # Gantt 
                for i, row in df.iterrows():
                    c = 'green' if row['pnl_usdt'] > 0 else 'red'
                    axs[2, 0].plot([row['entry_time'], row['exit_time']], [i, i], color=c, alpha=0.5, linewidth=2)
                axs[2, 0].set_title('Trade Timeline (Gantt)')
                axs[2, 0].tick_params(axis='y', left=False, labelleft=False)
                
                axs[2, 1].axis('off')
                
                img_data = io.BytesIO()
                plt.savefig(img_data, format='png', bbox_inches='tight', dpi=120)
                plt.close(fig)
                img_data.seek(0)
                Story.append(Image(img_data, width=700, height=500))
                Story.append(PageBreak())

        # Trades List (All trades instead of 50)
        Story.append(Paragraph("Complete Trade Ledger", styles['Heading2']))
        trades = sorted(self.current_data.trades, key=lambda x: x.exit_time, reverse=True)
        
        if trades:
            header = ["Pair", "Side", "Entry", "Exit", "PnL ($)", "MAE (%)", "Dur"]
            trade_data = [header]
            for t in trades:
                trade_data.append([
                    t.pair, t.side.upper(), 
                    f"{t.entry_price:.4f}", f"{t.exit_price:.4f}", 
                    f"{t.pnl_usdt:.2f}", f"{t.mae_percent:.2f}%", 
                    t.duration
                ])
            
            tt = RLTable(trade_data, repeatRows=1)
            tt.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white])
            ]))
            Story.append(tt)
            
        doc.build(Story)
        logger.info(f"PDF report saved to {filepath}")

    def generate_html(self, filepath: Path, metrics: Dict[str, Any]):
        """Generates an interactive HTML report using Plotly."""
        df = pd.DataFrame([t.to_dict() for t in self.current_data.trades])
        if df.empty: return
        
        df['datetime'] = pd.to_datetime(df['exit_time'])
        df = df.sort_values('datetime')
        df['cum_pnl'] = self.current_data.start_balance + df['pnl_usdt'].cumsum()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Cumulative Equity", "Drawdown (Underwater)", "PnL vs Holding Duration", "Trade Timeline Gantt"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]  # Bar for timeline approximation
        )
        
        # 1. Equity Curve
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['cum_pnl'], mode='lines', name='Equity', line=dict(color='cyan')), row=1, col=1)
        
        # 2. Drawdown
        running_max = df['cum_pnl'].cummax()
        dd = (df['cum_pnl'] - running_max) / running_max * 100
        fig.add_trace(go.Scatter(x=df['datetime'], y=dd, mode='lines', name='Drawdown %', fill='tozeroy', line=dict(color='red')), row=1, col=2)
        
        # 3. PnL vs Duration
        point_colors = ['green' if p > 0 else 'red' for p in df['pnl_usdt']]
        fig.add_trace(go.Scatter(x=df['holding_bars'], y=df['pnl_usdt'], mode='markers', name='Trades', marker=dict(color=point_colors, size=8, opacity=0.6)), row=2, col=1)
        
        # 4. Gantt / Timeline approximation (Using entry to exit ranges)
        for i, row in df.iterrows():
            c = 'rgba(0, 255, 0, 0.4)' if row['pnl_usdt'] > 0 else 'rgba(255, 0, 0, 0.4)'
            duration_ms = (pd.to_datetime(row['exit_time']) - pd.to_datetime(row['entry_time'])).total_seconds() * 1000
            
            fig.add_trace(go.Bar(
                base=pd.to_datetime(row['entry_time']),
                x=[duration_ms],
                y=[row['pair']],
                orientation='h',
                marker=dict(color=c),
                showlegend=False,
                name=f"{row['side']} {row['pair']}"
            ), row=2, col=2)
        
        fig.update_layout(
            title_text=f"DeadEnd Master Report: {self.current_data.mode.upper()}",
            height=1000,
            template="plotly_dark",
            showlegend=False
        )
        
        # Prepend KPI summary into the HTML using Jinja-like simple replacement
        kpi_html = "<div style='display:flex; flex-wrap:wrap; gap:20px; margin-bottom: 20px; font-family: monospace; color: #fff; background: #111; padding: 20px; border-radius: 8px;'>"
        for k, v in metrics.items():
            color = "#4ade80" if isinstance(v, str) and "$" in v and "-" not in v else "#fff"
            if "-" in str(v): color = "#f87171"
            kpi_html += f"<div style='flex: 1; min-width: 150px;'><div style='color:#888; font-size:12px;'>{k}</div><div style='font-size: 20px; font-weight:bold; color:{color};'>{v}</div></div>"
        kpi_html += "</div>"
        
        raw_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        full_doc = f"""
        <html>
        <head><title>DeadEnd Analysis</title><body style="background:#000; padding:20px;">
        {kpi_html}
        {raw_html}
        </body></html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_doc)
            
        logger.info(f"Interactive HTML report saved to {filepath}")

    def save_last_run(self):
        """Serialize current run to disk for `--report last` capability."""
        if not self.current_data: return
        dump_path = self.output_dir / "last_run_data.json"
        with open(dump_path, 'w') as f:
            json.dump(self.current_data.to_dict(), f, indent=2)

    def load_last_run(self) -> bool:
        """Load exactly the last run data."""
        dump_path = self.output_dir / "last_run_data.json"
        if not dump_path.exists():
            return False
            
        with open(dump_path, 'r') as f:
            data = json.load(f)
            self.current_data = ReportData.from_dict(data)
            return True
