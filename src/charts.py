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
        ax.text(0.5, 0.5, 'Insufficient repurchase data', 
                ha='center', va='center', transform=ax.transAxes)
    
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
            product_counts = recent_li.groupby('Lineitem name').agg({
                'Lineitem quantity': 'sum',
                'Customer Email': 'nunique'
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
        if df is not None and all(c in df.columns for c in ['Lineitem name','Name','Customer Email','Created at']):
            d_all = df.copy()
            d_all['Created at'] = pd.to_datetime(d_all['Created at'], errors='coerce')
            # Use a 180-day horizon to measure repeats; adjust if needed
            horizon_start = d_all['Created at'].max() - pd.Timedelta(days=180)
            d_win = d_all[d_all['Created at'] >= horizon_start]
            for product in list(top_index)[:5]:
                sub = d_win[d_win['Lineitem name'] == product]
                # Count unique orders per customer for this product
                per_cust_orders = sub.groupby('Customer Email')['Name'].nunique()
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
            grp = dfx.dropna(subset=['Created at']).groupby(['Customer Email', 'Lineitem name'])['Created at'].apply(list)

            replenishment_data = []
            for (customer, product), dates in grp.items():
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
        if df is not None and 'Lineitem name' in df.columns and 'Name' in df.columns and 'Created at' in df.columns:
            dfx2 = df.copy()
            dfx2['Created at'] = pd.to_datetime(dfx2['Created at'], errors='coerce')
            for product in list(top_index)[:5]:
                product_data = dfx2[dfx2['Lineitem name'] == product]
                if product_data.empty:
                    continue
                repeat_rate = float((product_data.groupby('Customer Email')['Name'].nunique() > 1).mean())
                intervals: List[float] = []
                for customer in product_data['Customer Email'].dropna().unique():
                    cust_dates = pd.to_datetime(product_data[product_data['Customer Email'] == customer]['Created at']).dropna()
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
    if df is not None and 'Lineitem name' in df.columns and 'Name' in df.columns and 'Created at' in df.columns:
        dfx2 = df.copy(); dfx2['Created at'] = pd.to_datetime(dfx2['Created at'], errors='coerce')
        for product in list(top_index)[:top_n]:
            product_data = dfx2[dfx2['Lineitem name'] == product]
            if product_data.empty:
                continue
            repeat_rate = float((product_data.groupby('Customer Email')['Name'].nunique() > 1).mean())
            intervals: List[float] = []
            for customer in product_data['Customer Email'].dropna().unique():
                cust_dates = pd.to_datetime(product_data[product_data['Customer Email'] == customer]['Created at']).dropna()
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
        elif cw == 'L56':
            # 56d ~ 8 weeks ~ 2 months; use L56 directly but scale to monthly
            base_window = aligned.get('L56', {})
            baseline_monthly = float((base_window.get('net_sales') or 0.0)) / 2.0
        else:
            # Default: L28 is already ~monthly
            base_window = aligned.get('L28', {})
            baseline_monthly = float(base_window.get('net_sales') or 0.0)
        
        # Build cumulative impact
        action_names = []
        impacts = [baseline_monthly]
        cumulative = baseline_monthly
        
        for action in actions[:3]:  # Top 3 actions
            # expected_$ has been scaled to monthly in the engine
            expected = float(action.get('expected_$', 0) or 0.0)
            cumulative += expected
            impacts.append(cumulative)
            
            # Shorten action names
            name = action.get('title', 'Unknown')
            if len(name) > 20:
                name = name[:17] + '...'
            action_names.append(name)
        
        # Create waterfall chart
        x = range(len(impacts))
        
        # Baseline bar
        ax.bar(0, baseline_monthly, color='#6b7280', label='Current Baseline (monthly)')
        
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
        # Clarify units
        ax.text(0.5, -0.12, 'Baseline and lifts normalized to monthly values.',
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


def cohort_retention_chart(df: pd.DataFrame, out_path: str) -> str:
    """
    Shows cohort retention — essential for Beauty/Supplements LTV.
    Left: monthly cohort retention heatmap (last 6 cohorts, first 6 months).
    Right: average retention curve (first 12 months).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    try:
        required = ['Customer Email', 'Created at', 'Name']
        if df is not None and all(c in df.columns for c in required):
            d = df.copy()
            d['Created at'] = pd.to_datetime(d['Created at'], errors='coerce')
            d = d.dropna(subset=['Created at'])
            if d.empty:
                raise ValueError('no dates')

            # First purchase (cohort) per customer
            first = d.groupby('Customer Email')['Created at'].min().reset_index()
            first.columns = ['Customer Email', 'cohort_date']
            first['cohort'] = first['cohort_date'].dt.to_period('M')

            dc = d.merge(first, on='Customer Email', how='left')
            dc['order_month'] = dc['Created at'].dt.to_period('M')
            # Months since first purchase as integer
            dc['months_since'] = (dc['order_month'].astype('period[M]') - dc['cohort'].astype('period[M]')).astype(int)

            # Retention matrix
            ret = dc.groupby(['cohort', 'months_since'])['Customer Email'].nunique().reset_index()
            cohort_sizes = dc.groupby('cohort')['Customer Email'].nunique()
            ret = ret.merge(cohort_sizes.rename('cohort_size'), left_on='cohort', right_index=True)
            ret['retention_rate'] = ret['Customer Email'] / ret['cohort_size']

            pivot = ret.pivot(index='cohort', columns='months_since', values='retention_rate').fillna(0.0)

            # Heatmap: last 6 cohorts, first 6 months
            rows = list(pivot.index)[-6:]
            cols = [c for c in pivot.columns if isinstance(c, (int, np.integer)) and 0 <= int(c) <= 5]
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
            # Ensure integer-like x for plotting
            xs = sorted([int(c) for c in avg_ret.index if isinstance(c, (int, np.integer)) and 0 <= int(c) <= 11])
            ys = [float(avg_ret.get(x, 0.0)) * 100.0 for x in xs]
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
            ax1.text(0.5, 0.5, 'Raw df with Customer Email, Name, Created at required', ha='center', va='center', transform=ax1.transAxes)
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
        if df is not None and 'Customer Email' in df.columns and 'Created at' in df.columns:
            d = df.copy()
            d['Created at'] = pd.to_datetime(d['Created at'], errors='coerce')
            d = d.dropna(subset=['Created at'])

            # Purchase sequences per customer
            seq = (
                d.sort_values('Created at')
                 .groupby('Customer Email')['Created at']
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
            ax1.text(0.5, 0.5, 'Raw df with Customer Email and Created at required', ha='center', va='center', transform=ax1.transAxes)
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
                   charts_mode: Optional[str] = None) -> Dict[str, str]:
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
