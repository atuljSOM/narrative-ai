"""
Enhanced charts for Beauty/Supplements stores
Focuses on actionable insights rather than generic comparisons
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
try:
    import seaborn as sns
    sns.set_palette("husl")
except Exception:
    sns = None  # Seaborn is optional; fall back to Matplotlib defaults

# Set beautiful defaults for Beauty vertical
plt.style.use('seaborn-v0_8-whitegrid')

def repurchase_curve_chart(g: pd.DataFrame, aligned: dict, out_path: str) -> str:
    """
    Shows customer repurchase timeline - critical for Beauty
    Highlights the 21-45 day winback window and 60-120 dormant period
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Calculate days since last purchase distribution
    if 'days_since_last' in g.columns:
        days = g['days_since_last'].dropna()
        
        # Create bins for visualization
        bins = [0, 7, 14, 21, 30, 45, 60, 90, 120, 180, 365]
        labels = ['0-7', '8-14', '15-21', '22-30', '31-45', '46-60', '61-90', '91-120', '121-180', '180+']
        
        # Categorize and count
        binned = pd.cut(days, bins=bins, labels=labels, include_lowest=True)
        counts = binned.value_counts().sort_index()
        
        # Create bar chart with action zones highlighted
        colors = []
        for label in counts.index:
            if label in ['22-30', '31-45']:  # Winback zone
                colors.append('#3b82f6')  # Blue - primary action
            elif label in ['61-90', '91-120']:  # Dormant zone
                colors.append('#f59e0b')  # Amber - secondary action
            else:
                colors.append('#e5e7eb')  # Gray - neutral
        
        bars = ax.bar(range(len(counts)), counts.values, color=colors)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        
        # Add action zone annotations
        ax.axvspan(2.5, 4.5, alpha=0.1, color='blue', label='Winback Zone')
        ax.axvspan(6.5, 7.5, alpha=0.1, color='orange', label='Dormant Zone')
        
        # Styling
        ax.set_xlabel('Days Since Last Purchase', fontsize=11)
        ax.set_ylabel('Number of Customers', fontsize=11)
        ax.set_title('Customer Repurchase Timeline & Action Zones', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, fancybox=True)
        
        # Add insight text
        total = float(counts.sum())
        winback_base = float(counts.get('22-30', 0) + counts.get('31-45', 0))
        winback_pct = (winback_base / total * 100.0) if total > 0 else 0.0
        ax.text(0.02, 0.98, f'{winback_pct:.0f}% in winback zone', 
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    else:
        # Fallback if no days_since_last data
        ax.text(
            0.5,
            0.5,
            "Repurchase timeline needs 'days_since_last' (derived from Created at + Customer Email/customer_id)",
            ha='center', va='center', transform=ax.transAxes, wrap=True
        )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


def product_velocity_chart(g: pd.DataFrame, aligned: dict, out_path: str, df: Optional[pd.DataFrame] = None) -> str:
    """
    Enhanced version with replenishment cycle detection.
    Top-left: product velocity; Top-right: product repeat rates;
    Bottom-left: replenishment cycles; Bottom-right: subscription readiness.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    top_index = []
    if 'Created at' in g.columns:
        # Calculate product velocity (units per month)
        g['Created at'] = pd.to_datetime(g['Created at'])
        recent_30 = g[g['Created at'] >= g['Created at'].max() - pd.Timedelta(days=30)]
        # Top products by volume (prefer raw line items if available)
        if df is not None and all(c in df.columns for c in ['Lineitem name','Lineitem quantity','Created at']):
            d = df.copy()
            d['Created at'] = pd.to_datetime(d['Created at'], errors='coerce')
            recent_li = d[d['Created at'] >= d['Created at'].max() - pd.Timedelta(days=30)]
            # Choose identity column dynamically for customer counts
            id_col = 'Customer Email' if 'Customer Email' in recent_li.columns else ('customer_id' if 'customer_id' in recent_li.columns else None)
            if id_col is not None:
                product_counts = recent_li.groupby('Lineitem name').agg({
                    'Lineitem quantity': 'sum',
                    id_col: 'nunique'
                }).rename(columns={id_col: 'customers'}).sort_values('Lineitem quantity', ascending=False).head(8)
            else:
                product_counts = recent_li.groupby('Lineitem name').agg({
                    'Lineitem quantity': 'sum'
                }).sort_values('Lineitem quantity', ascending=False).head(8)
            top_index = product_counts.index
        elif 'lineitem_any' in g.columns:
            product_counts = recent_30.groupby('lineitem_any').agg({
                'units_per_order': 'sum',
                'customer_id': 'nunique'
            }).sort_values('units_per_order', ascending=False).head(8)
            top_index = product_counts.index
        else:
            product_counts = pd.DataFrame()
            top_index = []
        
        # Chart 1: Product velocity
        if df is not None and 'Lineitem quantity' in product_counts.columns:
            units = product_counts['Lineitem quantity'].values
        else:
            units = product_counts['units_per_order'].values if 'units_per_order' in product_counts.columns else []
        products = [str(p)[:15] + '...' if len(str(p)) > 15 else str(p) for p in top_index]
        
        bars1 = ax1.barh(range(len(products)), units, color='#8b5cf6')
        ax1.set_yticks(range(len(products)))
        ax1.set_yticklabels(products)
        ax1.set_xlabel('Units Sold (30 days)', fontsize=11)
        ax1.set_title('Top Products by Velocity', fontsize=12, fontweight='bold')
        
        # Add values on bars
        for i, (bar, val) in enumerate(zip(bars1, units)):
            ax1.text(val, bar.get_y() + bar.get_height()/2, f'{int(val)}', 
                    ha='left', va='center', fontsize=9, color='black')
        
        # Chart 2: Repeat purchase rate by product
        # Compute repeat rates for top 5 products
        repeat_rates = []
        # Use email or customer_id for repeat rate computation
        id_col = None
        if df is not None:
            if 'Customer Email' in df.columns:
                id_col = 'Customer Email'
            elif 'customer_id' in df.columns:
                id_col = 'customer_id'
        if df is not None and id_col is not None and all(c in df.columns for c in ['Lineitem name','Name','Created at']):
            d_all = df.copy()
            d_all['Created at'] = pd.to_datetime(d_all['Created at'], errors='coerce')
            # Use a 180-day horizon to measure repeats; adjust if needed
            horizon_start = d_all['Created at'].max() - pd.Timedelta(days=180)
            d_win = d_all[d_all['Created at'] >= horizon_start]
            for product in list(top_index)[:5]:
                sub = d_win[d_win['Lineitem name'] == product]
                # Count unique orders per customer for this product
                per_cust_orders = sub.groupby(id_col)['Name'].nunique()
                if per_cust_orders.shape[0] == 0:
                    rr = 0.0
                else:
                    rr = float((per_cust_orders > 1).mean() * 100.0)
                repeat_rates.append(rr)
        elif 'lineitem_any' in g.columns:
            # Fallback to order-level proxy (less accurate)
            for product in list(top_index)[:5]:
                product_customers = g[g['lineitem_any'] == product]['customer_id'].value_counts()
                repeat_rate = (product_customers > 1).mean() * 100 if product_customers.shape[0] > 0 else 0.0
                repeat_rates.append(float(repeat_rate))
        else:
            repeat_rates = [0.0] * min(5, len(top_index))
        
        colors2 = ['#10b981' if r > 30 else '#f59e0b' if r > 20 else '#ef4444' 
                  for r in repeat_rates]
        
        bars2 = ax2.bar(range(len(repeat_rates)), repeat_rates, color=colors2)
        ax2.set_xticks(range(len(repeat_rates)))
        ax2.set_xticklabels([str(p)[:10] + '...' if len(str(p)) > 10 else str(p) 
                             for p in list(top_index)[:5]], rotation=45, ha='right')
        ax2.set_ylabel('Repeat Purchase Rate (%)', fontsize=11)
        ax2.set_title('Subscription Potential', fontsize=12, fontweight='bold')
        ax2.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='Good for subscription')
        ax2.legend(loc='upper right', fontsize=9)
        
        # Add values on bars
        for bar, val in zip(bars2, repeat_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.0f}%', 
                    ha='center', va='bottom', fontsize=9)
    
    else:
        ax1.text(0.5, 0.5, 'No product data available', 
                ha='center', va='center', transform=ax1.transAxes)
        ax2.axis('off')

    # NEW: ax3 - Replenishment Cycles by Product
    try:
        if df is not None and 'Lineitem name' in df.columns and 'Created at' in df.columns:
            dfx = df.copy()
            dfx['Created at'] = pd.to_datetime(dfx['Created at'], errors='coerce')
            id_col = 'Customer Email' if 'Customer Email' in dfx.columns else ('customer_id' if 'customer_id' in dfx.columns else None)
            group_keys = [id_col, 'Lineitem name'] if id_col is not None else ['Lineitem name']
            grp = dfx.dropna(subset=['Created at']).groupby(group_keys)['Created at'].apply(list)

            replenishment_data = []
            for key, dates in grp.items():
                # key may be a tuple (customer, product) or just product if id_col missing
                product = key[1] if isinstance(key, tuple) else key
                if len(dates) > 1:
                    dates_sorted = sorted(pd.to_datetime(dates))
                    intervals = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
                    if intervals:
                        replenishment_data.append({
                            'product': product,
                            'median_days': float(np.median(intervals)),
                            'std_days': float(np.std(intervals))
                        })

            if replenishment_data:
                repl_df = pd.DataFrame(replenishment_data)
                top_products_repl = repl_df.groupby('product').agg({
                    'median_days': 'median',
                    'std_days': 'mean'
                }).sort_values('median_days').head(10)

                y_pos = range(len(top_products_repl))
                ax3.barh(y_pos, top_products_repl['median_days'],
                         xerr=top_products_repl['std_days'],
                         color='#6366f1', alpha=0.7)
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels([str(p)[:20] for p in top_products_repl.index])
                ax3.set_xlabel('Days Between Purchases')
                ax3.set_title('Product Replenishment Cycles', fontweight='bold')
                ax3.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Monthly')
                ax3.axvline(x=60, color='orange', linestyle='--', alpha=0.5, label='Bi-monthly')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'Insufficient replenishment data', ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'Raw line-item data required', ha='center', va='center', transform=ax3.transAxes)
    except Exception as e:
        ax3.text(0.5, 0.5, f'Error computing cycles: {e}', ha='center', va='center', transform=ax3.transAxes)

    # NEW: ax4 - Subscription Readiness Score
    try:
        subscription_scores: List[float] = []
        prod_labels: List[str] = []
        if df is not None and 'Lineitem name' in df.columns and 'Created at' in df.columns:
            dfx2 = df.copy()
            dfx2['Created at'] = pd.to_datetime(dfx2['Created at'], errors='coerce')
            # Determine identity and order columns
            id_col_sr = 'Customer Email' if 'Customer Email' in dfx2.columns else ('customer_id' if 'customer_id' in dfx2.columns else None)
            order_col_sr = 'Name' if 'Name' in dfx2.columns else ('order_id' if 'order_id' in dfx2.columns else None)
            if id_col_sr is not None and order_col_sr is not None:
                for product in list(top_index)[:5]:
                    product_data = dfx2[dfx2['Lineitem name'] == product]
                    if product_data.empty:
                        continue
                    # Repeat rate based on unique orders per customer for this product
                    repeat_rate = float((product_data.groupby(id_col_sr)[order_col_sr].nunique() > 1).mean())
                    intervals: List[float] = []
                    customers_iter = product_data[id_col_sr].dropna().unique()
                    for customer in customers_iter:
                        cust_dates = pd.to_datetime(product_data[product_data[id_col_sr] == customer]['Created at']).dropna()
                        if cust_dates.shape[0] > 1:
                            cust_intervals = cust_dates.sort_values().diff().dt.days.dropna()
                            intervals.extend(cust_intervals.tolist())
                    if intervals and np.mean(intervals) > 0:
                        consistency = 1 - (np.std(intervals) / np.mean(intervals))
                        consistency = float(max(0.0, min(1.0, consistency)))
                    else:
                        consistency = 0.0
                    score = (repeat_rate * 0.6 + consistency * 0.4) * 100.0
                    subscription_scores.append(float(score))
                    prod_labels.append(str(product))
        colors4 = ['#10b981' if s > 60 else '#f59e0b' if s > 40 else '#ef4444' for s in subscription_scores]
        ax4.bar(range(len(subscription_scores)), subscription_scores, color=colors4)
        ax4.set_xticks(range(len(subscription_scores)))
        ax4.set_xticklabels([str(p)[:12] for p in prod_labels], rotation=45, ha='right')
        ax4.set_ylabel('Subscription Readiness Score')
        ax4.set_title('Products Ready for Subscription', fontweight='bold')
        ax4.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Ready')
        ax4.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Maybe')
        ax4.legend()
    except Exception as e:
        ax4.text(0.5, 0.5, f'Error computing readiness: {e}', ha='center', va='center', transform=ax4.transAxes)
    
    plt.suptitle('Product Performance & Subscription Opportunities', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


def product_performance_compact_chart(g: pd.DataFrame, aligned: dict, out_path: str,
                                      df: Optional[pd.DataFrame] = None,
                                      top_n: int = 5) -> str:
    """
    Compact Product Performance chart: focuses on the two highest-signal visuals.
    Left: Top products by 30D velocity (units). Right: Subscription readiness score.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Compute top products by 30D units
    top_index = []
    units = []
    try:
        if df is not None and all(c in df.columns for c in ['Lineitem name','Lineitem quantity','Created at']):
            d = df.copy(); d['Created at'] = pd.to_datetime(d['Created at'], errors='coerce')
            recent_li = d[d['Created at'] >= d['Created at'].max() - pd.Timedelta(days=30)]
            product_counts = recent_li.groupby('Lineitem name').agg({'Lineitem quantity': 'sum'})
            product_counts = product_counts.sort_values('Lineitem quantity', ascending=False).head(top_n)
            top_index = product_counts.index.tolist()
            units = product_counts['Lineitem quantity'].tolist()
        elif 'Created at' in g.columns and 'lineitem_any' in g.columns and 'units_per_order' in g.columns:
            gg = g.copy(); gg['Created at'] = pd.to_datetime(gg['Created at'], errors='coerce')
            recent_30 = gg[gg['Created at'] >= gg['Created at'].max() - pd.Timedelta(days=30)]
            product_counts = recent_30.groupby('lineitem_any')['units_per_order'].sum().sort_values(ascending=False).head(top_n)
            top_index = product_counts.index.tolist(); units = product_counts.values.tolist()
    except Exception:
        pass

    # Left panel: velocity
    products = [str(p)[:18] + '…' if len(str(p)) > 18 else str(p) for p in top_index]
    ax1.barh(range(len(products)), units, color='#6366f1')
    ax1.set_yticks(range(len(products)))
    ax1.set_yticklabels(products)
    ax1.invert_yaxis()
    ax1.set_xlabel('Units (30 days)')
    ax1.set_title('Top Products by Velocity', fontsize=12, fontweight='bold')

    # Right panel: subscription readiness (repeat + consistency proxy)
    subscription_scores: List[float] = []
    prod_labels: List[str] = []
    if df is not None and 'Lineitem name' in df.columns and 'Created at' in df.columns:
        dfx2 = df.copy(); dfx2['Created at'] = pd.to_datetime(dfx2['Created at'], errors='coerce')
        id_col_comp = 'Customer Email' if 'Customer Email' in dfx2.columns else ('customer_id' if 'customer_id' in dfx2.columns else None)
        order_col_comp = 'Name' if 'Name' in dfx2.columns else ('order_id' if 'order_id' in dfx2.columns else None)
        if id_col_comp is not None and order_col_comp is not None:
            for product in list(top_index)[:top_n]:
                product_data = dfx2[dfx2['Lineitem name'] == product]
                if product_data.empty:
                    continue
                repeat_rate = float((product_data.groupby(id_col_comp)[order_col_comp].nunique() > 1).mean())
                intervals: List[float] = []
                customers_iter = product_data[id_col_comp].dropna().unique()
                for customer in customers_iter:
                    cust_dates = pd.to_datetime(product_data[product_data[id_col_comp] == customer]['Created at']).dropna()
                    if cust_dates.shape[0] > 1:
                        cust_intervals = cust_dates.sort_values().diff().dt.days.dropna()
                        intervals.extend(cust_intervals.tolist())
                if intervals and np.mean(intervals) > 0:
                    consistency = 1 - (np.std(intervals) / np.mean(intervals))
                    consistency = float(max(0.0, min(1.0, consistency)))
                else:
                    consistency = 0.0
                score = (repeat_rate * 0.6 + consistency * 0.4) * 100.0
                subscription_scores.append(float(score))
                prod_labels.append(str(product))
    colors = ['#10b981' if s > 60 else '#f59e0b' if s > 40 else '#ef4444' for s in subscription_scores]
    ax2.bar(range(len(subscription_scores)), subscription_scores, color=colors)
    ax2.set_xticks(range(len(subscription_scores)))
    ax2.set_xticklabels([str(p)[:12] + ('…' if len(str(p))>12 else '') for p in prod_labels], rotation=45, ha='right')
    ax2.set_ylabel('Readiness Score')
    ax2.set_title('Subscription Readiness', fontsize=12, fontweight='bold')
    ax2.axhline(y=60, color='green', linestyle='--', alpha=0.4)
    ax2.axhline(y=40, color='orange', linestyle='--', alpha=0.3)

    plt.suptitle('Product Performance (Compact)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


def customer_value_segments_chart(g: pd.DataFrame, aligned: dict, out_path: str) -> str:
    """
    RFM-style segmentation showing where value sits
    Helps justify winback and VIP actions
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if all(col in g.columns for col in ['AOV', 'customer_id', 'Created at']):
        # Calculate customer metrics
        customer_summary = g.groupby('customer_id').agg({
            'AOV': 'mean',
            'Name': 'count',
            'Created at': 'max'
        }).rename(columns={'Name': 'order_count', 'Created at': 'last_order'})
        
        # Calculate recency
        max_date = customer_summary['last_order'].max()
        customer_summary['recency_days'] = (max_date - customer_summary['last_order']).dt.days
        
        # Create segments
        def segment_customers(row):
            if row['order_count'] >= 3 and row['recency_days'] <= 30:
                return 'Champions'
            elif row['order_count'] >= 3 and row['recency_days'] <= 90:
                return 'Loyal'
            elif row['order_count'] == 1 and row['recency_days'] <= 30:
                return 'New'
            elif row['order_count'] >= 2 and row['recency_days'] > 60:
                return 'At Risk'
            elif row['recency_days'] > 90:
                return 'Lost'
            else:
                return 'Developing'
        
        customer_summary['segment'] = customer_summary.apply(segment_customers, axis=1)
        
        # Count and value by segment
        segment_stats = customer_summary.groupby('segment').agg({
            'AOV': 'mean',
            'segment': 'count'
        }).rename(columns={'segment': 'count'})
        
        segment_stats['total_value'] = segment_stats['AOV'] * segment_stats['count']
        segment_stats = segment_stats.sort_values('total_value', ascending=True)
        
        # Create horizontal bar chart
        colors_map = {
            'Champions': '#10b981',
            'Loyal': '#3b82f6',
            'At Risk': '#f59e0b',
            'Lost': '#ef4444',
            'New': '#8b5cf6',
            'Developing': '#6b7280'
        }
        
        colors = [colors_map.get(seg, '#6b7280') for seg in segment_stats.index]
        
        bars = ax.barh(range(len(segment_stats)), segment_stats['total_value'], color=colors)
        
        # Annotations
        for i, (idx, row) in enumerate(segment_stats.iterrows()):
            # Value on bar
            ax.text(row['total_value'], i, f" ${row['total_value']:.0f}", 
                   va='center', fontsize=10)
            # Count in parentheses
            ax.text(0, i, f"{idx} ({row['count']:.0f})", 
                   ha='right', va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks([])
        ax.set_xlabel('Total Segment Value ($)', fontsize=11)
        ax.set_title('Customer Value Segments - Where to Focus', fontsize=13, fontweight='bold')
        
        # Add action callouts
        if 'At Risk' in segment_stats.index:
            at_risk_count = segment_stats.loc['At Risk', 'count']
            ax.text(0.98, 0.02, f'🎯 {at_risk_count:.0f} customers need winback', 
                   transform=ax.transAxes, ha='right', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    else:
        ax.text(0.5, 0.5, 'Insufficient data for segmentation', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


def action_impact_forecast_chart(actions: List[Dict], aligned: dict, out_path: str, chosen_window: Optional[str] = None) -> str:
    """
    Shows expected impact of recommended actions
    Makes the "why" clear visually
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if actions and len(actions) > 0:
        # Determine monthly baseline and leave action impacts in monthly units
        cw = (chosen_window or '').upper()
        if cw == 'L7':
            # Approximate month as 4x L7
            baseline_monthly = float(aligned.get('L7', {}).get('net_sales') or 0.0) * 4.0
            baseline_label = '4× L7 (monthly)'
        elif cw == 'L56':
            # 56d ~ 8 weeks ~ 2 months; use L56 directly but scale to monthly
            base_window = aligned.get('L56', {})
            baseline_monthly = float((base_window.get('net_sales') or 0.0)) / 2.0
            baseline_label = 'L56 ÷ 2 (monthly)'
        else:
            # Default: L28 is already ~monthly
            base_window = aligned.get('L28', {})
            baseline_monthly = float(base_window.get('net_sales') or 0.0)
            baseline_label = 'L28 (monthly)'
        
        # Build cumulative impact (adjusted for diminishing returns and channel overlap)
        action_names = []
        impacts = [baseline_monthly]
        cumulative = baseline_monthly
        used_channels = set()
        # Position-based diminishing schedule (top 3)
        pos_schedule = [1.00, 0.90, 0.80]
        # Global portfolio cap to avoid overstating combined lift
        portfolio_cap = 0.50 * baseline_monthly if baseline_monthly > 0 else float('inf')
        
        for idx, action in enumerate(actions[:3]):  # Top 3 actions
            # expected_$ has been scaled to monthly in the engine
            expected = float(action.get('expected_$', 0) or 0.0)

            # Channel-overlap interference: penalize by prior-used channel overlap
            chans = set()
            try:
                meta_ch = action.get('channels') or []
                if isinstance(meta_ch, dict):
                    # if structured, take keys truthy
                    meta_ch = [k for k, v in meta_ch.items() if v]
                chans = set(str(c).lower() for c in meta_ch)
            except Exception:
                chans = set()
            overlap = len(chans & used_channels)
            channel_factor = (0.90 ** overlap) if overlap > 0 else 1.0

            # Position-based diminishing factor
            pos_factor = pos_schedule[idx] if idx < len(pos_schedule) else 0.75

            adjusted = expected * channel_factor * pos_factor

            # Apply portfolio cap across combined lifts
            current_lift = cumulative - baseline_monthly
            remaining_cap = portfolio_cap - current_lift
            if remaining_cap < float('inf'):
                adjusted = max(0.0, min(adjusted, remaining_cap))

            cumulative += adjusted
            impacts.append(cumulative)
            
            # Shorten action names
            name = action.get('title', 'Unknown')
            if len(name) > 20:
                name = name[:17] + '...'
            action_names.append(name)

            # Accumulate used channels for interference on subsequent actions
            used_channels |= chans
        
        # Create waterfall chart
        x = range(len(impacts))
        
        # Baseline bar
        ax.bar(0, baseline_monthly, color='#6b7280', label=f'Baseline ({baseline_label})')
        
        # Action bars (stacked)
        bottom = baseline_monthly
        colors = ['#3b82f6', '#10b981', '#8b5cf6']
        for i, (name, impact) in enumerate(zip(action_names, impacts[1:])):
            height = impact - bottom
            ax.bar(i+1, height, bottom=bottom, color=colors[i % 3], label=name)
            
            # Add impact label
            ax.text(i+1, bottom + height/2, f'+${height:.0f}', 
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            
            bottom = impact
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(['Current'] + [f'+ Action {i+1}' for i in range(len(action_names))])
        ax.set_ylabel('Expected Monthly Revenue ($)', fontsize=11)
        ax.set_title('Revenue Impact Forecast - This Month\'s Actions', fontsize=13, fontweight='bold')
        # Clarify units and interaction assumptions
        ax.text(0.5, -0.12, f'Units: monthly. Baseline source: {baseline_label}. Combined impact adjusted for channel overlap and diminishing returns; capped at 50% of baseline.',
                transform=ax.transAxes, ha='center', va='top', fontsize=9, color='#6b7280')
        
        # Add total impact annotation
        total_lift = impacts[-1] - baseline_monthly
        lift_pct = (total_lift / baseline_monthly * 100) if baseline_monthly > 0 else 0
        ax.text(0.98, 0.98, f'Total Expected Lift: +${total_lift:.0f} ({lift_pct:.1f}%)', 
               transform=ax.transAxes, ha='right', va='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        ax.legend(loc='upper left', frameon=True, fancybox=True)
        ax.grid(axis='y', alpha=0.3)
    
    else:
        ax.text(0.5, 0.5, 'No actions to forecast', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


def stock_vs_demand_chart(inventory_metrics: pd.DataFrame, out_path: str, top_n: int = 10) -> str:
    """Visualize stock (available_net) vs projected monthly demand for top SKUs by velocity.
    - Colors indicate coverage: green (>=28d), amber (14-28d), red (<14d).
    - Adds shortfall labels where demand exceeds stock.
    Expects columns: sku, product, available_net, daily_velocity, cover_days.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        inv = inventory_metrics.copy()
        if inv is None or inv.empty:
            ax.text(0.5, 0.5, 'No inventory data', ha='center', va='center', transform=ax.transAxes)
        else:
            # Compute projected monthly demand (28 days)
            inv['proj_demand'] = pd.to_numeric(inv.get('daily_velocity', 0), errors='coerce').fillna(0) * 28.0
            inv['available_net'] = pd.to_numeric(inv.get('available_net', 0), errors='coerce').fillna(0)
            inv['cover_days'] = pd.to_numeric(inv.get('cover_days', np.inf), errors='coerce')
            # Top by highest projected demand
            top = inv.sort_values('proj_demand', ascending=False).head(top_n).copy()
            labels = [str(x)[:18] + ('…' if len(str(x))>18 else '') for x in (top.get('product').fillna(top.get('sku')))]
            x = np.arange(len(top))
            width = 0.4
            bars1 = ax.bar(x - width/2, top['available_net'], width=width, label='Available (net)', color='#10b981')
            bars2 = ax.bar(x + width/2, top['proj_demand'], width=width, label='Projected Demand (28d)', color='#3b82f6')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel('Units')
            ax.set_title('Stock vs Projected Monthly Demand (Top SKUs)', fontsize=13, fontweight='bold')
            ax.legend()
            # Coverage color markers atop bars
            for i, (cv, an, dm) in enumerate(zip(top['cover_days'], top['available_net'], top['proj_demand'])):
                color = '#10b981' if cv >= 28 else ('#f59e0b' if cv >= 14 else '#ef4444')
                ax.plot([x[i], x[i]], [max(an, dm) * 1.02, max(an, dm) * 1.06], color=color, linewidth=4)
                # Shortfall annotation
                if dm > an:
                    short = float(dm - an)
                    ax.text(x[i] + width/2, dm + max(1.0, dm*0.02), f'-{int(short)}', ha='center', va='bottom', fontsize=9, color='#ef4444')
            ax.margins(y=0.2)
    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


def cohort_retention_chart(df: pd.DataFrame, out_path: str) -> str:
    """
    Shows cohort retention — essential for Beauty/Supplements LTV.
    Left: monthly cohort retention heatmap (last 6 cohorts, first 6 months).
    Right: average retention curve (first 12 months).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    try:
        required_any_id = ['Customer Email', 'customer_id']
        required_common = ['Created at']
        if df is not None and all(c in df.columns for c in required_common) and any(c in df.columns for c in required_any_id):
            d = df.copy()
            d['Created at'] = pd.to_datetime(d['Created at'], errors='coerce')
            d = d.dropna(subset=['Created at'])
            if d.empty:
                raise ValueError('no dates')

            # Use email if available; else customer_id
            id_col = 'Customer Email' if 'Customer Email' in d.columns else 'customer_id'
            # First purchase (cohort) per customer
            first = d.groupby(id_col)['Created at'].min().reset_index()
            first.columns = [id_col, 'cohort_date']
            first['cohort'] = first['cohort_date'].dt.to_period('M')
            dc = d.merge(first, on=id_col, how='left')
            dc['order_month'] = dc['Created at'].dt.to_period('M')
            # Months since first purchase as integer (robust to pandas period arithmetic)
            try:
                dc['months_since'] = (dc['order_month'].astype(int) - dc['cohort'].astype(int)).astype(int)
            except Exception:
                # Fallback: compute via start-of-month timestamps
                om_start = dc['order_month'].dt.start_time
                co_start = dc['cohort'].dt.start_time
                dc['months_since'] = ((om_start.dt.year - co_start.dt.year) * 12 + (om_start.dt.month - co_start.dt.month)).astype(int)

            # Retention matrix
            ret = dc.groupby(['cohort', 'months_since'])[id_col].nunique().reset_index()
            cohort_sizes = dc.groupby('cohort')[id_col].nunique()
            ret = ret.merge(cohort_sizes.rename('cohort_size'), left_on='cohort', right_index=True)
            # Use the same ID column for counts to avoid KeyError on 'Customer Email'
            ret['retention_rate'] = ret[id_col] / ret['cohort_size']

            pivot = ret.pivot(index='cohort', columns='months_since', values='retention_rate').fillna(0.0)

            # Heatmap: last 6 cohorts, first 6 months
            rows = list(pivot.index)[-6:]
            # Ensure integer-like month offsets
            def _to_int_safe(x):
                try:
                    return int(x)
                except Exception:
                    return None
            cols = [c for c in pivot.columns if (_to_int_safe(c) is not None and 0 <= _to_int_safe(c) <= 5)]
            data = pivot.loc[rows, cols] if rows else pivot.iloc[[], :]

            if sns is not None and not data.empty:
                sns.heatmap(data, annot=True, fmt='.0%', cmap='YlOrRd', ax=ax1, vmin=0, vmax=1)
                ax1.set_title('Monthly Cohort Retention', fontweight='bold')
                ax1.set_xlabel('Months Since First Purchase')
                ax1.set_ylabel('Cohort')
            else:
                # Fallback simple image
                if data.empty:
                    ax1.text(0.5, 0.5, 'Insufficient cohort data', ha='center', va='center', transform=ax1.transAxes)
                else:
                    im = ax1.imshow(data.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
                    ax1.set_xticks(range(len(data.columns))); ax1.set_xticklabels(data.columns)
                    ax1.set_yticks(range(len(data.index))); ax1.set_yticklabels([str(i) for i in data.index])
                    ax1.set_title('Monthly Cohort Retention', fontweight='bold')
                    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

            # Average retention curve (first 12 months)
            avg_ret = pivot.mean(axis=0)
            xs = []
            ys = []
            for c, v in avg_ret.items():
                ci = _to_int_safe(c)
                if ci is not None and 0 <= ci <= 11:
                    xs.append(ci)
                    ys.append(float(v) * 100.0)
            xs, ys = (list(zip(*sorted(zip(xs, ys)))) if xs else ([], []))
            ax2.plot(xs, ys, marker='o', linewidth=2, markersize=6, color='#6366f1')
            ax2.fill_between(xs, [0]*len(xs), ys, alpha=0.25, color='#6366f1')
            ax2.set_xlabel('Months Since First Purchase')
            ax2.set_ylabel('Average Retention Rate (%)')
            ax2.set_title('Average Retention Curve', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Industry Avg')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Good')
            ax2.legend()

            for month in [1, 3, 6]:
                if month in xs:
                    val = ys[xs.index(month)]
                    ax2.annotate(f'{val:.0f}%', xy=(month, val), xytext=(month, val + 5), ha='center', fontweight='bold')
        else:
            missing = []
            if df is not None:
                if 'Created at' not in df.columns:
                    missing.append('Created at')
                if ('Customer Email' not in df.columns) and ('customer_id' not in df.columns):
                    missing.append('Customer Email or customer_id')
            msg = 'Missing columns: ' + ', '.join(missing) if missing else 'Missing required columns'
            ax1.text(0.5, 0.5, f"{msg}. Use exact column names.", ha='center', va='center', transform=ax1.transAxes, wrap=True)
            ax2.axis('off')
    except Exception as e:
        ax1.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax1.transAxes)
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


def first_to_second_purchase_chart(df: pd.DataFrame, out_path: str) -> str:
    """
    Critical for Beauty: Shows time to second purchase and conversion rate.
    Left: distribution of days to second purchase with median.
    Right: simple funnel of 2+ and 3+ orders.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    try:
        id_col = 'Customer Email' if (df is not None and 'Customer Email' in df.columns) else ('customer_id' if (df is not None and 'customer_id' in df.columns) else None)
        order_col = 'order_id' if (df is not None and 'order_id' in df.columns) else ('Name' if (df is not None and 'Name' in df.columns) else None)
        if df is not None and id_col is not None and 'Created at' in df.columns:
            d = df.copy()
            d['Created at'] = pd.to_datetime(d['Created at'], errors='coerce')
            d = d.dropna(subset=['Created at'])

            # Optional recent windowing via env (days). Example: F2S_WINDOW_DAYS=365
            try:
                win_days = int(os.getenv('F2S_WINDOW_DAYS', '0') or '0')
            except Exception:
                win_days = 0
            if win_days and win_days > 0:
                cutoff = d['Created at'].max() - pd.Timedelta(days=win_days)
                d = d[d['Created at'] >= cutoff]

            # De-duplicate to one record per order per customer to avoid line-item duplicates
            if order_col is not None and order_col in d.columns:
                # Keep the first timestamp per (customer, order)
                d = (
                    d[[id_col, order_col, 'Created at']]
                    .sort_values(['Created at'])
                    .groupby([id_col, order_col], as_index=False)['Created at']
                    .min()
                )
            else:
                # Fallback: drop duplicates on (customer, timestamp)
                d = d[[id_col, 'Created at']].drop_duplicates([id_col, 'Created at'])

            # Purchase sequences per customer using de-duplicated orders
            seq = (
                d.sort_values('Created at')
                 .groupby(id_col)['Created at']
                 .apply(list)
                 .reset_index()
            )

            days_to_second: List[float] = []
            for _, row in seq.iterrows():
                dates = row['Created at']
                if isinstance(dates, list) and len(dates) >= 2:
                    days = (dates[1] - dates[0]).days
                    days_to_second.append(days)

            if days_to_second:
                ax1.hist(days_to_second, bins=30, color='#6366f1', alpha=0.7, edgecolor='black')
                med = float(np.median(days_to_second))
                ax1.axvline(x=med, color='red', linestyle='--', label=f'Median: {med:.0f} days')
                ax1.set_xlabel('Days to Second Purchase')
                ax1.set_ylabel('Number of Customers')
                ax1.set_title('Time to Second Purchase Distribution', fontweight='bold')
                ax1.legend()
            else:
                ax1.text(0.5, 0.5, 'No second purchases observed', ha='center', va='center', transform=ax1.transAxes)

            # Funnel: 2+ and 3+
            total_customers = int(seq.shape[0])
            two_plus = int((seq['Created at'].apply(lambda x: len(x) if isinstance(x, list) else 0) >= 2).sum())
            three_plus = int((seq['Created at'].apply(lambda x: len(x) if isinstance(x, list) else 0) >= 3).sum())

            stages = ['All\nCustomers', '2+\nOrders', '3+\nOrders']
            values = [total_customers, two_plus, three_plus]
            colors = ['#e5e7eb', '#6366f1', '#10b981']
            bars = ax2.bar(stages, values, color=colors)
            ax2.set_ylabel('Number of Customers')
            ax2.set_title('Customer Order Frequency Funnel', fontweight='bold')

            for i, (stage, val) in enumerate(zip(stages, values)):
                ax2.text(i, val, f'{val}', ha='center', va='bottom', fontweight='bold')
                if i > 0 and values[i-1] > 0:
                    conv_rate = (val / values[i-1] * 100)
                    ax2.text(i - 0.5, values[i-1] / 2, f'{conv_rate:.0f}%',
                             ha='center', va='center', fontweight='bold', color='red', fontsize=12)
        else:
            need = []
            if df is not None:
                if 'Created at' not in df.columns:
                    need.append('Created at')
                if ('Customer Email' not in df.columns) and ('customer_id' not in df.columns):
                    need.append('Customer Email or customer_id')
            msg = 'Missing columns: ' + ', '.join(need) if need else 'Missing required columns'
            ax1.text(0.5, 0.5, f"{msg}. Use exact column names.", ha='center', va='center', transform=ax1.transAxes, wrap=True)
            ax2.axis('off')
    except Exception as e:
        ax1.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax1.transAxes)
        ax2.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


def generate_charts(g: pd.DataFrame, aligned: dict, actions: List[Dict], out_dir: str,
                   df: Optional[pd.DataFrame] = None,
                   chosen_window: Optional[str] = None,
                   charts_mode: Optional[str] = None,
                   inventory_metrics: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """Enhanced chart generation for Beauty/Supplements"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    charts = {}
    
    # Generate each chart
    try:
        charts['repurchase_timeline'] = repurchase_curve_chart(
            g, aligned, str(Path(out_dir) / "repurchase_timeline.png")
        )
    except Exception as e:
        print(f"Warning: Could not generate repurchase timeline chart: {e}")
    
    try:
        mode = (charts_mode or os.getenv('CHARTS_MODE', 'detailed') or 'detailed')
        mode = str(mode).strip().lower()
        if mode in {'compact','minimal'}:
            charts['product_velocity'] = product_performance_compact_chart(
                g, aligned, str(Path(out_dir) / "product_velocity.png"), df=df, top_n=5
            )
        else:
            charts['product_velocity'] = product_velocity_chart(
                g, aligned, str(Path(out_dir) / "product_velocity.png"), df=df
            )
    except Exception as e:
        print(f"Warning: Could not generate product velocity chart: {e}")
    
    try:
        charts['customer_segments'] = customer_value_segments_chart(
            g, aligned, str(Path(out_dir) / "customer_segments.png")
        )
    except Exception as e:
        print(f"Warning: Could not generate customer segments chart: {e}")
    
    try:
        charts['impact_forecast'] = action_impact_forecast_chart(
            actions.get('actions', []), aligned, str(Path(out_dir) / "impact_forecast.png"), chosen_window=chosen_window
        )
    except Exception as e:
        print(f"Warning: Could not generate impact forecast chart: {e}")

    # Inventory: Stock vs Demand chart
    try:
        if inventory_metrics is not None and not getattr(inventory_metrics, 'empty', False):
            charts['stock_vs_demand'] = stock_vs_demand_chart(
                inventory_metrics, str(Path(out_dir) / "stock_vs_demand.png"), top_n=10
            )
    except Exception as e:
        print(f"Warning: Could not generate stock vs demand chart: {e}")

    if df is not None:
        try:
            charts['cohort_retention'] = cohort_retention_chart(
                df, str(Path(out_dir) / "cohort_retention.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate cohort retention chart: {e}")
        try:
            charts['first_to_second'] = first_to_second_purchase_chart(
                df, str(Path(out_dir) / "first_to_second.png")
            )
        except Exception as e:
            print(f"Warning: Could not generate first-to-second purchase chart: {e}")
    
    return charts
