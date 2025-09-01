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
    Shows top products by velocity and their reorder rates
    Critical for identifying which products to push for subscription
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
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
    
    plt.suptitle('Product Performance & Subscription Opportunities', 
                fontsize=14, fontweight='bold', y=1.02)
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
        # Determine weekly baseline and scale factor for lifts
        cw = (chosen_window or '').upper()
        if cw == 'L7':
            baseline_weekly = float(aligned.get('L7', {}).get('net_sales') or 0.0)
            scale = 1.0
        else:
            # Default to L28 if not L7; scale to weekly (~/4). If L56, scale by 8.
            if cw == 'L56':
                weeks = 8.0
            else:
                weeks = 4.0
            base_window = aligned.get('L28', {}) if cw != 'L56' else aligned.get('L56', {})
            baseline_weekly = float((base_window.get('net_sales') or 0.0)) / weeks if weeks else 0.0
            scale = 1.0 / weeks if weeks else 1.0
        
        # Build cumulative impact
        action_names = []
        impacts = [baseline_weekly]
        cumulative = baseline_weekly
        
        for action in actions[:3]:  # Top 3 actions
            expected = float(action.get('expected_$', 0) or 0.0) * scale
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
        ax.bar(0, baseline_weekly, color='#6b7280', label='Current Baseline (weekly)')
        
        # Action bars (stacked)
        bottom = baseline_weekly
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
        ax.set_ylabel('Expected Weekly Revenue ($)', fontsize=11)
        ax.set_title('Revenue Impact Forecast - This Week\'s Actions', fontsize=13, fontweight='bold')
        # Clarify units
        ax.text(0.5, -0.12, 'Baseline and lifts normalized to weekly values (scaled from chosen window).',
                transform=ax.transAxes, ha='center', va='top', fontsize=9, color='#6b7280')
        
        # Add total impact annotation
        total_lift = impacts[-1] - baseline_weekly
        lift_pct = (total_lift / baseline_weekly * 100) if baseline_weekly > 0 else 0
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


def generate_charts(g: pd.DataFrame, aligned: dict, actions: List[Dict], out_dir: str, df: Optional[pd.DataFrame] = None, chosen_window: Optional[str] = None) -> Dict[str, str]:
    """
    Generate all charts for Beauty/Supplements vertical
    Returns dict of chart_name: file_path
    """
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
    
    return charts
