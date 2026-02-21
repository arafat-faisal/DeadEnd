#!/usr/bin/env python3
"""
Elite Trading System - Main Entry Point

Usage:
    python main.py --mode research        # Run strategy discovery
    python main.py --mode trade --paper   # Paper trading
    python main.py --mode trade --live    # Live trading (careful!)
    python main.py --mode both            # Research then trade
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import get_settings
from core.engine import TradingEngine, EngineMode
from config.settings import ExchangeType
from utils.logger import get_logger, setup_logging

logger = get_logger('main')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Elite Trading System - Automated crypto trading with strategy discovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run research (automatically runs Pair Discovery + Grid/Scalping)
    python main.py --mode research
    
    # Run research with specific pairs
    python main.py --mode research --pairs BTC/USDT ETH/USDT
    
    # Paper trading with top strategies
    python main.py --mode trade --paper --interval 60
    
    # Full auto mode (research then trade)
    python main.py --mode both --paper
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['research', 'trade', 'both', 'status'],
        default='research',
        help='Operating mode (default: research)'
    )
    
    parser.add_argument(
        '--pairs', '-p',
        nargs='+',
        default=None,
        help='Trading pairs (default: BTC/USDT ETH/USDT SOL/USDT)'
    )
    
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Enable paper trading mode (simulated)'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live trading mode (real money!)'
    )
    
    parser.add_argument(
        '--exchange', '-e',
        choices=['binance', 'bitget'],
        default='binance',
        help='Exchange to use (default: binance)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=60,
        help='Trading loop interval in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode with minimal strategy set'
    )
    
    parser.add_argument(
        '--max-strategies',
        type=int,
        default=None,
        help='Maximum strategies to test per pair'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Start FastAPI dashboard on port 8000 when trading'
    )
    
    parser.add_argument(
        '--telegram',
        action='store_true',
        help='Enable Telegram alerts (requires TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env)'
    )
    
    parser.add_argument(
        '--report',
        choices=['none', 'full', 'last'],
        default='none',
        help='Generate extremely detailed post-run reports: none, full, last (re-render without running)'
    )
    
    parser.add_argument(
        '--clone-best',
        action='store_true',
        help='Clone the best performing strategy to config/strategies for live trading'
    )
    
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Deploy the best strategy: runs 30-min paper, then asks for LIVE.'
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        default=None,
        help='Path to a specific strategy config JSON to run directly (e.g. config/strategies/best_ARB_FUNDING_0.0010_BTCUSDT.json)'
    )
    
    return parser.parse_args()


def print_banner():
    """Print startup banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║           ELITE TRADING SYSTEM v1.0                           ║
║           Automated Strategy Discovery & Execution            ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_status():
    """Print current system status"""
    from utils.database import get_database
    from research.priority_list import PriorityListManager
    
    settings = get_settings()
    db = get_database()
    priority_manager = PriorityListManager()
    
    print("\n=== SYSTEM STATUS ===\n")
    
    # Settings
    print(f"Trading Mode: {'PAPER' if settings.is_paper_mode else 'LIVE'}")
    print(f"Starting Balance: {settings.starting_balance} USDT")
    print(f"Futures Target: {settings.futures_target} USDT")
    print(f"Risk Per Trade: {settings.risk_per_trade:.1%}")
    print(f"Max Drawdown: {settings.max_drawdown:.1%}")
    print(f"Default Leverage: {settings.default_leverage}x")
    
    # Balance
    current = db.get_current_balance()
    if current:
        print(f"\nCurrent Balance: {current:.2f} USDT")
        drawdown = db.calculate_drawdown()
        print(f"Current Drawdown: {drawdown['current_drawdown']:.1%}")
        print(f"Peak Balance: {drawdown['peak_balance']:.2f} USDT")
    
    # Strategies
    strategies = db.get_top_strategies(limit=5)
    if strategies:
        print(f"\nTop 5 Strategies:")
        for s in strategies:
            print(f"  - {s['name']} on {s['pair']}: ROI={s['roi']*100:.1f}%, Sharpe={s['sharpe_ratio']:.2f}")
    
    # Priority list
    entries = priority_manager.load()
    if entries:
        print(f"\nPriority List ({len(entries)} entries):")
        priority_manager.print_summary(entries[:5])
    
    # Recent trades
    trades = db.get_recent_trades(limit=5)
    if trades:
        print(f"\nRecent Trades:")
        for t in trades:
            print(f"  - {t['side'].upper()} {t['amount']:.4f} {t['pair']} @ {t['price']:.2f}")


def run_research_mode(args, engine: TradingEngine):
    """Run research/backtesting mode"""
    logger.info("Starting Research Mode")
    
    results = engine.run_research(
        pairs=args.pairs,
        max_strategies=30 if getattr(args, 'quick', False) else args.max_strategies
    )
    
    if args.report == 'full':
        print("\nGenerating Research Reports...")
        engine.report_generator.generate_all("research_report")
        engine.report_generator.save_last_run()
        print("✅ Reports generated successfully in 'reports/' directory.")
    
    print(f"\n✅ Research complete: {len(results)} strategies tested")
    print("Priority list saved to data/priority_list.json")


def run_trading_mode(args, engine: TradingEngine, strategies=None):
    """Run trading mode"""
    if args.live and not args.paper:
        print("\n⚠️  WARNING: LIVE TRADING MODE ⚠️")
        print("You are about to trade with REAL MONEY!")
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            print("Aborted.")
            return
    
    logger.info(f"Starting Trading Mode ({'PAPER' if engine.paper_mode else 'LIVE'})")
    
    if getattr(args, 'dashboard', False):
        import threading
        from dashboard.main import start_dashboard
        logger.info("Starting Dashboard on http://localhost:8000")
        threading.Thread(target=start_dashboard, kwargs={"host": "0.0.0.0", "port": 8000}, daemon=True).start()
        
    engine.run_trading_loop(interval_seconds=args.interval, strategies=strategies)


def main():
    """Main entry point"""
    args = parse_args()
    
    print_banner()
    
    # Enable telegram if flag is passed
    if args.telegram:
        import os
        os.environ['TELEGRAM_ENABLED'] = 'true'
        
    # Setup logging
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    
    # Determine paper mode
    paper_mode = True  # Default to paper
    if args.live:
        paper_mode = False
    elif args.paper:
        paper_mode = True
    
    # Handle status mode separately
    if args.mode == 'status':
        print_status()
        return

    # Handle immediate report regeneration
    if getattr(args, 'report', 'none') == 'last':
        from core.reports import ReportGenerator
        logger.info("Executing standalone report regeneration for the last run...")
        rg = ReportGenerator()
        if rg.load_last_run():
            rg.generate_all("last_report_regenerated")
            print("\n✅ Standalone report generation complete. Check 'reports/' directory.")
        else:
            print("\n❌ Could not find a previous 'last_run_data.json' in 'reports/'. You must run a strategy first.")
    # Handle clone best
    if getattr(args, 'clone_best', False):
        from research.priority_list import PriorityListManager
        import json
        logger.info("Extracting best strategy from priority list...")
        pl = PriorityListManager()
        entries = pl.load()
        if entries:
            best = entries[0]
            config_dir = Path("config/strategies")
            config_dir.mkdir(parents=True, exist_ok=True)
            
            strat_name = f"best_{best.strategy}_{best.pair.replace('/', '')}.json"
            out_path = config_dir / strat_name
            
            is_arb = "ARB_FUNDING" in best.strategy
            with open(out_path, 'w') as f:
                json.dump({
                    "strategy": best.strategy,
                    "pair": best.pair,
                    "params": best.params,
                    "roi": best.expected_roi,
                    "sharpe": best.sharpe_ratio,
                    "leverage": 1,
                    "risk_pct": 0.5,
                    "mode": "paper",
                    "position_sizing": "neutral_delta" if is_arb else "standard"
                }, f, indent=4)
            print(f"\n✅ Cloned best strategy to {out_path} ready for live.")
        else:
            print("\n❌ No strategies found in priority list to clone.")
        return
        
    # Handle deploy pipeline
    if getattr(args, 'deploy', False):
        print("\n🚀 DEPLOYMENT PIPELINE INITIATED")
        import glob
        config_dir = Path("config/strategies")
        best_files = list(config_dir.glob("best_*.json"))
        if not best_files:
            print("❌ No best strategy config found in config/strategies/. Run --clone-best first.")
            return
            
        best_file = best_files[0]
        import json
        with open(best_file, 'r') as f:
            best_config = json.load(f)
            
        print(f"✅ Loaded config: {best_file.name}")
        print(f"   Strategy: {best_config.get('strategy')} | Pair: {best_config.get('pair')}")
        
        from research.priority_list import PriorityEntry
        strat_entry = PriorityEntry(
            strategy=best_config.get('strategy'),
            pair=best_config.get('pair'),
            params=best_config.get('params', {}),
            expected_roi=best_config.get('roi', 0),
            win_rate=best_config.get('risk_pct', 0),
            max_drawdown=0,
            sharpe_ratio=best_config.get('sharpe', 0)
        )
        
        print("\n⏳ Starting 30-minute Paper Trading Validation...")
        exchange_type = ExchangeType.BINANCE if args.exchange == 'binance' else ExchangeType.BITGET
        engine_paper = TradingEngine(
            mode=EngineMode.TRADING,
            exchange_type=exchange_type,
            paper_mode=True,
            report_mode='none'
        )
        
        try:
            engine_paper.run_trading_loop(interval_seconds=60, max_iterations=30, strategies=[strat_entry])
        except KeyboardInterrupt:
            print("\n⛔ Paper trading interrupted.")
            return
            
        print("\n📊 Paper Validation Summary:")
        balance = engine_paper.executor.get_balance().get('USDT', 0)
        print(f"Ending Paper Balance: {balance:.2f} USDT")
        
        print("\n⚠️  DEPLOYMENT GATE ⚠️")
        print("Paper validation complete. You are about to deploy with REAL MONEY API KEYS.")
        confirm = input("Ready for LIVE? Type YES: ")
        
        if confirm == 'YES':
            print("\n🔥 INITIATING LIVE TRADING ENGINE 🔥")
            engine_live = TradingEngine(
                mode=EngineMode.TRADING,
                exchange_type=exchange_type,
                paper_mode=False,
                report_mode='none'
            )
            engine_live.run_trading_loop(interval_seconds=args.interval, strategies=[strat_entry])
        else:
            print("Deployment aborted. Staying out of live mode.")
        return
    
    # Create engine
    exchange_type = ExchangeType.BINANCE if args.exchange == 'binance' else ExchangeType.BITGET
    
    
    engine = TradingEngine(
        mode=EngineMode(args.mode) if args.mode != 'both' else EngineMode.BOTH,
        exchange_type=exchange_type,
        paper_mode=paper_mode,
        report_mode=args.report
    )
    
    # Load specific strategy if requested
    specific_strategies = None
    if args.strategy:
        import json
        from research.priority_list import PriorityEntry
        strat_path = Path(args.strategy)
        if strat_path.exists():
            with open(strat_path, 'r') as f:
                cfg = json.load(f)
            strat_entry = PriorityEntry(
                strategy=cfg.get('strategy'),
                pair=cfg.get('pair'),
                params=cfg.get('params', {}),
                expected_roi=cfg.get('roi', 0),
                win_rate=cfg.get('risk_pct', 0),
                max_drawdown=0,
                sharpe_ratio=cfg.get('sharpe', 0)
            )
            specific_strategies = [strat_entry]
            print(f"\n✅ Targeting specific strategy: {strat_entry.strategy} on {strat_entry.pair}")
        else:
            print(f"\n❌ Strategy file not found: {args.strategy}")
            return
            
    print(f"\n📊 Mode: {args.mode.upper()}")
    print(f"💰 Trading: {'PAPER' if paper_mode else 'LIVE'}")
    print(f"🏦 Exchange: {args.exchange.upper()}")
    print(f"📈 Pairs: {args.pairs or 'default'}\n")
    
    try:
        if args.mode == 'research':
            run_research_mode(args, engine)
            
        elif args.mode == 'trade':
            run_trading_mode(args, engine, strategies=specific_strategies)
            
        elif args.mode == 'both':
            run_research_mode(args, engine)
            print("\n" + "="*50)
            print("Research complete. Starting trading...")
            print("="*50 + "\n")
            run_trading_mode(args, engine, strategies=specific_strategies)
            
    except KeyboardInterrupt:
        print("\n\n⛔ Interrupted by user")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == '__main__':
    main()
