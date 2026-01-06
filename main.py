import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib

# 强制使用 TkAgg 后端 (如果在某些无头环境报错，可以注释掉这一行，或者改为 'Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import Optional, Callable, Dict, List

# ==========================================
# 1. 数据获取与处理层 (Data Layer)
# ==========================================

def fetch_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """抓取数据并进行标准化的预处理"""
    print(f"⬇️ Fetching data for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        df.reset_index(inplace=True)

        if df.empty:
            print(f"Warning: No data found for {symbol}")
            return pd.DataFrame(columns=['Date', 'Close'])

        # 标准化日期格式
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
        return df[['Date', 'Close']]
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame(columns=['Date', 'Close'])


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """计算 RSI 指标"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_indicators_with_weekly(df: pd.DataFrame, ema_spans: List[int] = [200]) -> pd.DataFrame:
    """
    计算日线 EMA 和 周线 RSI
    注意：周线指标必须向后偏移一周(shift 1)并填充到日线，以避免未来函数。
    """
    if df.empty: return df
    df = df.copy().set_index('Date')
    
    # 1. 计算日线 EMA
    for span in ema_spans:
        df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
        
    # 2. 计算周线 RSI (Weekly RSI 7)
    # 重采样到周线 (以周五为结束)
    df_weekly = df['Close'].resample('W-FRI').last()
    weekly_rsi = calculate_rsi(df_weekly, window=7)
    
    # 关键步骤：避免未来函数 (Look-ahead Bias)
    # 本周五计算出的 RSI，只能用于下周一的决策。
    # 所以我们将周线 RSI 向后移动 1 个单位 (1周)
    weekly_rsi_shifted = weekly_rsi.shift(1)
    
    # 将周线 RSI 映射回日线 (向前填充 ffill)
    # 这样，下周一到周五看到的都是"上周五收盘确定的 RSI"
    df['Weekly_RSI_7'] = weekly_rsi_shifted.reindex(df.index, method='ffill')
    
    return df.reset_index()

# ==========================================
# 2. 策略定义层 (Strategy Layer)
# ==========================================

def strategy_lump_sum(df: pd.DataFrame, initial_amount: float = 10000.0) -> pd.DataFrame:
    """一次性投入策略 (Benchmark)"""
    df = df.copy().set_index('Date')
    start_price = df.iloc[0]['Close']
    shares = initial_amount / start_price

    result = pd.DataFrame(index=df.index)
    result['Total_Cost'] = initial_amount
    result['Market_Value'] = df['Close'] * shares
    result['ROI'] = (result['Market_Value'] - initial_amount) / initial_amount * 100
    return result

def strategy_tactical_rsi_dip_buy(
        df_merged: pd.DataFrame,
        ema_col: str,
        initial_cash: float = 10000.0,
        target_leverage: float = 2.0,  # 牛市目标
        bear_base_leverage: float = 0.75, # 熊市基础目标
        bear_dip_leverage: float = 1.0,   # 熊市抄底目标 (RSI触发后)
        rsi_dip_threshold: int = 15,      # 周线 RSI 阈值
        cooldown_days: int = 5,
        interest_rate: float = 0.05,
        hedge_asset_col: Optional[str] = None, # 修改点：通用对冲资产列名 (e.g., 'Close_SHV' or 'Close_GLD')
        daily_rebalance: bool = False,    # 每日强制再平衡
        rebalance_threshold: float = 0.0  # 偏离度阈值
) -> pd.DataFrame:
    """
    策略逻辑:
    1. 牛市 (Price > EMA): 2.0x 杠杆。
    2. 熊市 (Price < EMA):
       - 默认为 bear_base_leverage。
       - 监测到 周线RSI < 阈值，即刻加仓至 bear_dip_leverage。
    
    资金管理:
    - 如果 leverage < 1.0 且指定了 hedge_asset_col，剩余资金买入该资产 (SHV, GLD 等)。
    - 否则持有现金。
    """
    daily_interest_rate = interest_rate / 365
    df = df_merged.copy().set_index('Date')

    # 账户状态
    cash = initial_cash
    debt = 0.0
    shares_qqq = 0.0
    shares_hedge = 0.0 # 修改变量名，泛指对冲资产 (SHV or GLD)
    
    # 策略状态
    cooldown_counter = 0
    active_target_ratio = bear_base_leverage # 初始默认保守
    
    # 抄底状态标记
    has_bought_dip = False 
    
    results = []

    # --- Day 0 初始化 ---
    first_row = df.iloc[0]
    price0 = first_row['Close_QQQ']
    ema0 = first_row[ema_col]
    
    # 获取对冲资产价格 (如果有)
    price_hedge_0 = 0.0
    if hedge_asset_col and hedge_asset_col in first_row and pd.notna(first_row[hedge_asset_col]):
        price_hedge_0 = first_row[hedge_asset_col]

    if pd.notna(ema0) and price0 > ema0:
        active_target_ratio = target_leverage
        has_bought_dip = False
    else:
        active_target_ratio = bear_base_leverage
        if pd.notna(first_row['Weekly_RSI_7']) and first_row['Weekly_RSI_7'] < rsi_dip_threshold:
            active_target_ratio = bear_dip_leverage
            has_bought_dip = True
        
    # Day 0 建仓
    total_equity = cash
    target_qqq_val = total_equity * active_target_ratio
    
    if active_target_ratio > 1.0:
        target_debt = target_qqq_val - total_equity
        target_hedge_val = 0.0
    else:
        target_debt = 0.0
        target_hedge_val = total_equity - target_qqq_val
        
    shares_qqq = target_qqq_val / price0
    debt = target_debt
    
    # 购买对冲资产或保留现金
    if hedge_asset_col and price_hedge_0 > 0:
        shares_hedge = target_hedge_val / price_hedge_0
        cash = 0.0 # All in hedge asset
    else:
        shares_hedge = 0.0
        cash = target_hedge_val

    results.append({
        'Date': df.index[0],
        'Market_Value': initial_cash,
        'Leverage': active_target_ratio,
        'RSI_Signal': has_bought_dip,
        'Trade_Count': 1
    })
    
    trade_count_accum = 1

    # --- Day 1 Loop ---
    for i in range(1, len(df)):
        row = df.iloc[i]
        price_qqq = row['Close_QQQ']
        ema = row[ema_col]
        weekly_rsi = row['Weekly_RSI_7']
        
        # 获取当前对冲资产价格
        price_hedge = 0.0
        if hedge_asset_col and hedge_asset_col in row and pd.notna(row[hedge_asset_col]):
            price_hedge = row[hedge_asset_col]

        # 1. 计息 (债务利息)
        if debt > 0: debt += debt * daily_interest_rate
            
        # 2. 净值核算 (Mark to Market)
        val_qqq = shares_qqq * price_qqq
        val_hedge = shares_hedge * price_hedge
        equity = val_qqq + val_hedge + cash - debt
        
        current_leverage = (val_qqq / equity) if equity > 0 else 0

        if equity <= 0: # 爆仓处理
            equity = 0; shares_qqq = 0; shares_hedge = 0; debt = 0; cash = 0
            
        # 3. 信号逻辑
        desired_ratio = active_target_ratio 
        new_dip_status = has_bought_dip     
        is_bull_market = False              
        
        if cooldown_counter > 0:
            cooldown_counter -= 1
            if pd.notna(ema) and price_qqq > ema:
                is_bull_market = True
        elif pd.notna(ema):
            # A. 判断主趋势
            if price_qqq > ema:
                # 牛市
                desired_ratio = target_leverage
                new_dip_status = False 
                is_bull_market = True
            else:
                # 熊市
                is_bull_market = False
                # B. 判断是否触发抄底
                if pd.notna(weekly_rsi) and weekly_rsi < rsi_dip_threshold:
                    new_dip_status = True
                
                # C. 根据抄底标记决定仓位
                if new_dip_status:
                    desired_ratio = bear_dip_leverage 
                else:
                    desired_ratio = bear_base_leverage 
            
            if desired_ratio != active_target_ratio:
                cooldown_counter = cooldown_days
        
        # 4. 执行逻辑 (Execution)
        should_rebalance = False
        
        # 判定条件 1: 策略信号改变
        if desired_ratio != active_target_ratio:
            active_target_ratio = desired_ratio 
            has_bought_dip = new_dip_status
            should_rebalance = True
        
        # 判定条件 2 & 3: 常规再平衡 (仅在牛市 Price > EMA 时启用)
        elif is_bull_market:
            if daily_rebalance:
                should_rebalance = True
            elif rebalance_threshold > 0 and equity > 0:
                deviation = abs(current_leverage - active_target_ratio)
                if deviation > rebalance_threshold:
                    should_rebalance = True
        
        # 熊市 (Price < EMA) 且 信号未变 时：不进行再平衡
            
        # 执行调仓
        if should_rebalance and equity > 0:
            trade_count_accum += 1
            # 重新计算目标市值
            target_qqq_val = equity * active_target_ratio
            
            if active_target_ratio > 1.0:
                target_debt = target_qqq_val - equity
                target_hedge_val = 0.0
            else:
                target_debt = 0.0
                target_hedge_val = equity - target_qqq_val
                
            # 下单
            shares_qqq = target_qqq_val / price_qqq
            debt = target_debt
            
            # 买入对冲资产
            if hedge_asset_col and price_hedge > 0:
                shares_hedge = target_hedge_val / price_hedge
                cash = 0.0
            else:
                shares_hedge = 0.0
                cash = target_hedge_val
        else:
            has_bought_dip = new_dip_status

        final_leverage = (shares_qqq * price_qqq / equity) if equity > 0 else 0
        
        results.append({
            'Date': df.index[i],
            'Market_Value': equity,
            'Leverage': final_leverage,
            'RSI_Signal': has_bought_dip,
            'Trade_Count': trade_count_accum
        })

    result_df = pd.DataFrame(results).set_index('Date')
    result_df['ROI'] = (result_df['Market_Value'] - initial_cash) / initial_cash * 100
    result_df['Total_Cost'] = initial_cash 
    
    return result_df

# ==========================================
# 4. 分析与可视化层 (Analysis Layer)
# ==========================================

import pandas as pd
import numpy as np

def calculate_max_drawdown(df: pd.DataFrame, value_col: str = 'Market_Value') -> float:
    if df.empty: return 0.0
    df = df.copy()
    df['Peak'] = df[value_col].cummax()
    df['Drawdown'] = df['Peak'] - df[value_col]
    df['Drawdown_Pct'] = np.where(df['Peak'] > 0, df['Drawdown'] / df['Peak'], 0)
    return df['Drawdown_Pct'].max() * 100

def calculate_max_drawdown_amount(df: pd.DataFrame, value_col: str = 'Market_Value') -> float:
    """计算最大回撤的绝对金额"""
    if df.empty: return 0.0
    df = df.copy()
    df['Peak'] = df[value_col].cummax()
    df['Drawdown_Amount'] = df['Peak'] - df[value_col]
    return df['Drawdown_Amount'].max()

def calculate_max_drawdown_duration(df: pd.DataFrame, value_col: str = 'Market_Value') -> int:
    """计算最长回撤持续时间 (天数)"""
    if df.empty: return 0
    df = df.copy()
    df['Peak'] = df[value_col].cummax()
    df['Is_Drawdown'] = df[value_col] < df['Peak']
    
    df['Block'] = (df['Is_Drawdown'] != df['Is_Drawdown'].shift()).cumsum()
    drawdown_blocks = df[df['Is_Drawdown']].copy() # Create copy to avoid SettingWithCopy
    if drawdown_blocks.empty:
        return 0
    
    # --- 修复 FutureWarning ---
    # 旧代码: duration_days = drawdown_blocks.groupby('Block').apply(lambda x: (x.index.max() - x.index.min()).days).max()
    
    # 新代码: 使用聚合函数 (agg) 代替 apply，避免对分组列的操作警告，且速度更快
    # 1. 将 Index (Date) 显式转为 Column 以便聚合
    drawdown_blocks['Temp_Date'] = drawdown_blocks.index
    # 2. 对每一块计算 min date 和 max date
    block_agg = drawdown_blocks.groupby('Block')['Temp_Date'].agg(['min', 'max'])
    # 3. 计算每一块的时间差并取最大值
    duration_days = (block_agg['max'] - block_agg['min']).dt.days.max()
    
    return duration_days

def calculate_volatility(df: pd.DataFrame, value_col: str = 'Market_Value') -> float:
    """计算年化波动率"""
    if df.empty: return 0.0
    returns = df[value_col].pct_change().dropna()
    return returns.std() * np.sqrt(252) * 100

def calculate_recovery_factor(net_profit: float, max_dd_amount: float) -> float:
    """计算恢复因子: 净利润 / 最大回撤金额"""
    if max_dd_amount == 0: return 999.0 
    return net_profit / max_dd_amount

def calculate_annualized_roi(total_roi_pct: float, df_results: pd.DataFrame) -> float:
    if df_results.empty: return 0.0
    start_date = df_results.index.min()
    end_date = df_results.index.max()
    total_days = (end_date - start_date).days
    total_years = total_days / 365.25
    total_return_factor = 1.0 + (total_roi_pct / 100.0)
    if total_years > 0 and total_return_factor > 0:
        return ((total_return_factor ** (1.0 / total_years)) - 1.0) * 100.0
    return 0.0

def calculate_sharpe_ratio(df: pd.DataFrame, risk_free_rate: float = 0.03) -> float:
    if df.empty: return 0.0
    daily_returns = df['Market_Value'].pct_change().dropna()
    if daily_returns.empty or daily_returns.std() == 0: return 0.0
    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def calculate_sortino_ratio(df: pd.DataFrame, risk_free_rate: float = 0.03) -> float:
    if df.empty: return 0.0
    daily_returns = df['Market_Value'].pct_change().dropna()
    if daily_returns.empty: return 0.0
    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    negative_returns = excess_returns[excess_returns < 0]
    if negative_returns.empty or negative_returns.std() == 0: return 0.0
    downside_deviation = negative_returns.std() * np.sqrt(252)
    annualized_excess_return = excess_returns.mean() * 252
    return annualized_excess_return / downside_deviation

def print_yearly_returns(strategies_results: Dict[str, pd.DataFrame]):
    print("\n📅 Yearly Returns Comparison (%):")
    yearly_data = {}
    for name, df in strategies_results.items():
        if df.empty: continue
        yearly_equity = df['Market_Value'].resample('YE').last()
        yearly_pct = yearly_equity.pct_change() * 100
        if len(yearly_equity) > 0:
            first_year = yearly_equity.index[0].year
            initial_inv = df['Total_Cost'].iloc[0]
            first_year_ret = (yearly_equity.iloc[0] - initial_inv) / initial_inv * 100
            yearly_pct.iloc[0] = first_year_ret
        yearly_data[name] = yearly_pct

    yearly_df = pd.DataFrame(yearly_data)
    yearly_df.index = yearly_df.index.year
    print(yearly_df.to_string(float_format="%.1f"))

def plot_comparison_enhanced(strategies_results: Dict[str, pd.DataFrame], title: str = "Strategy Analysis"):
    if not strategies_results: return
    
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(12, 14),
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1, 1]}
    )
    plt.style.use('ggplot')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 1. 净值曲线 (Log Scale)
    for idx, (name, df) in enumerate(strategies_results.items()):
        if df.empty: continue
        color = colors[idx % len(colors)]
        final_val = df['Market_Value'].iloc[-1]
        ax1.semilogy(df.index, df['Market_Value'], label=f"{name} (${final_val:,.0f})", color=color, linewidth=1.5)
    
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel("Net Value ($) - Log", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    # 2. 回撤百分比
    for idx, (name, df) in enumerate(strategies_results.items()):
        if df.empty: continue
        color = colors[idx % len(colors)]
        peak = df['Market_Value'].cummax()
        dd_pct = (df['Market_Value'] - peak) / peak * 100
        ax2.plot(df.index, dd_pct, label=name, color=color, linewidth=1, alpha=0.8)

    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_title("Relative Drawdown (%)", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 3. 绝对利润曲线
    for idx, (name, df) in enumerate(strategies_results.items()):
        if df.empty: continue
        color = colors[idx % len(colors)]
        profit = df['Market_Value'] - df['Total_Cost'].iloc[0]
        ax3.plot(df.index, profit, label=f"{name}", color=color, linewidth=1.2)
        
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, label="Breakeven")
    ax3.set_ylabel("Net Profit ($)", fontsize=12)
    ax3.set_title("Absolute Net Profit ($)", fontsize=12)
    ax3.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 主程序入口 (Main)
# ==========================================

def main():
    symbol = "QQQ"
    cash_symbol = "SHV" 
    gold_symbol = "GLD" # 新增：黄金代码
    
    start_date = "2005-01-01"
    end_date = None
    ema_std = 200
    ema_mid = 120

    # 1. 准备主资产数据 (QQQ)
    price_df = fetch_data(symbol, start_date=start_date, end_date=end_date)
    if price_df.empty: return
    
    data = calculate_indicators_with_weekly(price_df, ema_spans=[ema_mid, ema_std])
    
    # 2. 准备对冲资产数据 (SHV, GLD)
    shv_df = fetch_data(cash_symbol, start_date=start_date, end_date=end_date)
    gld_df = fetch_data(gold_symbol, start_date=start_date, end_date=end_date) # 抓取 GLD
    
    # 3. 合并数据
    data_for_merge = data.rename(columns={'Close': 'Close_QQQ'})
    
    # 合并 SHV
    if not shv_df.empty:
        shv_df_renamed = shv_df.rename(columns={'Close': 'Close_SHV'})
        df_merged = pd.merge(data_for_merge, shv_df_renamed[['Date', 'Close_SHV']], on='Date', how='left')
    else:
        df_merged = data_for_merge.copy()
        df_merged['Close_SHV'] = np.nan
        
    # 合并 GLD (新增)
    if not gld_df.empty:
        gld_df_renamed = gld_df.rename(columns={'Close': 'Close_GLD'})
        df_merged = pd.merge(df_merged, gld_df_renamed[['Date', 'Close_GLD']], on='Date', how='left')
    else:
        df_merged['Close_GLD'] = np.nan
        
    print(f"✅ Merged Data prepared. Range: {df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}")

    # 4. 运行策略
    strategies = {}
    
    # Benchmark
    strategies['Buy & Hold (QQQ)'] = strategy_lump_sum(data, initial_amount=100000)

    # 纯EMA，牛市2.0x，熊市1.0x
    strategies['EMA200'] = strategy_tactical_rsi_dip_buy(
        df_merged,
        ema_col=f'EMA_{ema_mid}',
        initial_cash=100000,
        target_leverage=2.0, 
        bear_base_leverage=1, 
        bear_dip_leverage=1.0,
        rsi_dip_threshold=20,    
        hedge_asset_col='Close_SHV', # 传入 SHV 列名
        daily_rebalance=False, 
        rebalance_threshold=0.0
    )

    # 牛市2.0x，熊市保留0.3x QQQ + 0.7x SHV
    strategies['Bear 0.7x Cash'] = strategy_tactical_rsi_dip_buy(
        df_merged,
        ema_col=f'EMA_{ema_mid}',
        initial_cash=100000,
        target_leverage=2.0, 
        bear_base_leverage=0.3, 
        bear_dip_leverage=0.3,
        rsi_dip_threshold=20,    
        hedge_asset_col='Close_SHV', # 传入 SHV 列名
        daily_rebalance=False, 
        rebalance_threshold=0.0
    )

    # 牛市2.0x，熊市保留0.3x QQQ + 0.7x 现金，周线RSI7出现小于20抄底，满仓0杠杆，1.0x QQQ
    strategies['Bear 0.7x SHV & Buy Dip'] = strategy_tactical_rsi_dip_buy(
        df_merged,
        ema_col=f'EMA_{ema_mid}',
        initial_cash=100000,
        target_leverage=2.0, 
        bear_base_leverage=0.3, 
        bear_dip_leverage=1,
        rsi_dip_threshold=20,    
        hedge_asset_col='Close_SHV', # 传入 SHV 列名
        daily_rebalance=False, 
        rebalance_threshold=0.0
    )

    # 牛市2.0x，熊市保留0.3x QQQ + 0.7x 现金 + RSI 周线抄底，同时牛市的时候杠杆率偏差2.0x超过10%的时候，重新再平衡回到2.0x
    strategies['SHV & Buy Dip & Rebalance'] = strategy_tactical_rsi_dip_buy(
        df_merged,
        ema_col=f'EMA_{ema_std}',
        initial_cash=100000,
        target_leverage=2.0, 
        bear_base_leverage=0.3, 
        bear_dip_leverage=1.0,
        rsi_dip_threshold=20,    
        hedge_asset_col='Close_SHV', # 传入 SHV 列名
        daily_rebalance=False, 
        rebalance_threshold=0.1
    )

    # 牛市2.0x + 再平衡，熊市保留0.3x QQQ + 0.7x 黄金，RSI周线底部卖出黄金，换成QQQ 1.0x
    strategies['Rebalance + GLD Buffer'] = strategy_tactical_rsi_dip_buy(
        df_merged,
        ema_col=f'EMA_{ema_std}',
        initial_cash=100000,
        target_leverage=2.0, 
        bear_base_leverage=0.3, 
        bear_dip_leverage=1.0,
        rsi_dip_threshold=20,    
        hedge_asset_col='Close_GLD', # 传入 GLD 列名
        daily_rebalance=False, 
        rebalance_threshold=0.1 
    )

    # 同上，不进行再平衡，但采用黄金缓冲
    strategies['No Reb + GLD Buffer'] = strategy_tactical_rsi_dip_buy(
        df_merged,
        ema_col=f'EMA_{ema_std}',
        initial_cash=100000,
        target_leverage=2.0, 
        bear_base_leverage=0.3, 
        bear_dip_leverage=1.0,
        rsi_dip_threshold=20,    
        hedge_asset_col='Close_GLD', # 传入 GLD 列名
        daily_rebalance=False, 
        rebalance_threshold=0.0
    )

    # 5. 输出摘要
    print("\n📊 Detailed Performance Metrics:")
    print("-" * 120)
    summary_data = []

    for name, result in strategies.items():
        if result.empty: continue
        last_row = result.iloc[-1]
        
        net_value = last_row['Market_Value']
        initial_cost = result['Total_Cost'].iloc[0]
        net_profit = net_value - initial_cost
        
        max_dd_pct = calculate_max_drawdown(result, 'Market_Value')
        max_dd_amount = calculate_max_drawdown_amount(result, 'Market_Value')
        max_dd_days = calculate_max_drawdown_duration(result, 'Market_Value')
        volatility = calculate_volatility(result, 'Market_Value')
        
        recovery_factor = calculate_recovery_factor(net_profit, max_dd_amount)
        annualized_roi = calculate_annualized_roi(last_row['ROI'], result)
        sharpe = calculate_sharpe_ratio(result, risk_free_rate=0.03)
        sortino = calculate_sortino_ratio(result, risk_free_rate=0.03)
        
        trade_count = result.iloc[-1]['Trade_Count'] if 'Trade_Count' in result.columns else 0

        summary_data.append({
            "Strategy": name,
            "CAGR %": annualized_roi,
            "Vol %": volatility,
            "Sharpe": sharpe,
            "Max DD %": max_dd_pct,
            "Max DD $": max_dd_amount,
            "DD Days": max_dd_days,
            "Rec Factor": recovery_factor,
            "Trades": trade_count,
            "Net Value": net_value,
            "Sortino": sortino,
        })

    summary_df = pd.DataFrame(summary_data).sort_values("Sortino", ascending=False)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    formatters = {
        "CAGR %": "{:.2f}%".format,
        "Vol %": "{:.2f}%".format,
        "Sharpe": "{:.4f}".format,
        "Max DD %": "{:.2f}%".format,
        "Max DD $": "${:,.0f}".format,
        "DD Days": "{:.0f}".format,
        "Rec Factor": "{:.2f}".format,
        "Trades": "{:.0f}".format,
        "Net Value": "${:,.0f}".format,
        "Sortino": "{:.4f}".format,
    }
    
    print(summary_df.to_string(index=False, formatters=formatters))
    print("-" * 120)

    # 6. 打印分年度收益
    print_yearly_returns(strategies)

    # 7. 绘图
    plot_comparison_enhanced(strategies, title=f"Strategy: SHV vs GLD Buffer Comparison")

if __name__ == "__main__":
    main()