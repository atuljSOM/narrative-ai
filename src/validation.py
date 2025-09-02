"""
Enhanced Data Validation Engine for Aura
Performs critical data consistency checks with anomaly detection and sanity checks
Prepared for multi-source validation including inventory
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats

# Base class must be defined before specific checks
class ValidationCheck:
    """Base class for validation checks"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns: {
            'status': 'green'|'amber'|'red',
            'message': str,
            'details': dict,
            'severity': 'info'|'warning'|'critical'
        }
        """
        raise NotImplementedError

class AOVConsistencyCheck(ValidationCheck):
    """Validate AOV in aligned snapshot matches order-level reality."""

    def __init__(self):
        super().__init__(
            "AOV Consistency",
            "Checks AOV against order-level Subtotal/Discount or Total-Shipping-Taxes"
        )

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('df')
        aligned = data.get('aligned', {})
        if df is None or df.empty:
            return {
                'status': 'red', 'message': 'No data to compute AOV', 'details': {}, 'severity': 'critical'
            }

        # Compute AOV in the same window as aligned['L28'] (recent 28d ending at anchor)
        d = df.copy()
        if 'Created at' in d.columns:
            d['Created at'] = pd.to_datetime(d['Created at'], errors='coerce')
        # Match KPI exclusions: drop cancelled/refunded before windowing
        if 'Cancelled at' in d.columns:
            canc = pd.to_datetime(d['Cancelled at'], errors='coerce')
            d = d[canc.isna()]
        if 'Financial Status' in d.columns:
            d = d[~d['Financial Status'].astype(str).str.contains('refunded|chargeback', case=False, na=False)]
        anchor = aligned.get('anchor')
        try:
            anchor = pd.to_datetime(anchor) if anchor is not None else pd.to_datetime(d['Created at'].dropna().max())
        except Exception:
            anchor = pd.to_datetime(d['Created at'].dropna().max()) if 'Created at' in d.columns else None

        window_days = int(((aligned or {}).get('L28') or {}).get('window_days') or 28)
        if anchor is not None and 'Created at' in d.columns:
            recent_end = anchor.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
            recent_start = recent_end.normalize() - pd.Timedelta(days=window_days - 1)
            d = d[(d['Created at'] >= recent_start) & (d['Created at'] <= recent_end)]

        # Deduplicate by order for AOV
        d_orders = d.drop_duplicates(subset=['Name']) if 'Name' in d.columns else d
        def _money(s):
            return pd.to_numeric(s, errors='coerce') if s is not None else pd.Series(dtype=float)
        # Compute all variants when possible for diagnostics
        aov_sub, aov_tot, aov_li = None, None, None
        net_sum_sub, net_sum_tot, net_sum_li = None, None, None
        if all(c in d_orders.columns for c in ['Subtotal','Total Discount']):
            nets_sub = _money(d_orders['Subtotal']) - _money(d_orders['Total Discount'])
            aov_sub = float(nets_sub.dropna().mean()) if not nets_sub.dropna().empty else None
            net_sum_sub = float(nets_sub.dropna().sum()) if not nets_sub.dropna().empty else None
        if all(c in d_orders.columns for c in ['Total','Shipping','Taxes']):
            nets_tot = _money(d_orders['Total']) - _money(d_orders['Shipping']) - _money(d_orders['Taxes'])
            aov_tot = float(nets_tot.dropna().mean()) if not nets_tot.dropna().empty else None
            net_sum_tot = float(nets_tot.dropna().sum()) if not nets_tot.dropna().empty else None
        # Line-items per-order fallback (mirrors KPI revenue fallback path)
        if all(c in d.columns for c in ['Lineitem price','Lineitem quantity','Name']):
            li = d[['Name','Lineitem price','Lineitem quantity','Lineitem discount']].copy() if 'Lineitem discount' in d.columns else d[['Name','Lineitem price','Lineitem quantity']].copy()
            li['Lineitem price'] = pd.to_numeric(li['Lineitem price'], errors='coerce')
            li['Lineitem quantity'] = pd.to_numeric(li['Lineitem quantity'], errors='coerce')
            if 'Lineitem discount' in li.columns:
                li['Lineitem discount'] = pd.to_numeric(li['Lineitem discount'], errors='coerce')
            else:
                li['Lineitem discount'] = 0.0
            li['line_net'] = (li['Lineitem price'] * li['Lineitem quantity']) - li['Lineitem discount']
            per_order = li.groupby('Name', dropna=False)['line_net'].sum().reset_index(name='order_net')
            aov_li = float(per_order['order_net'].dropna().mean()) if not per_order['order_net'].dropna().empty else None
            net_sum_li = float(per_order['order_net'].dropna().sum()) if not per_order['order_net'].dropna().empty else None

        # Choose the method that mirrors KPI preference (subtotal-discount, then total-ship-tax, then line-items)
        if aov_sub is not None:
            aov_calc = aov_sub
            method_used = 'subtotal_minus_discount'
        elif aov_tot is not None:
            aov_calc = aov_tot
            method_used = 'total_minus_shipping_taxes'
        elif aov_li is not None:
            aov_calc = aov_li
            method_used = 'line_items_per_order'
        else:
            aov_calc = None
            method_used = 'unknown'

        aov_aligned = None
        try:
            aov_aligned = float((aligned or {}).get('L28', {}).get('aov'))
        except Exception:
            aov_aligned = None

        if aov_calc is None or aov_calc == 0 or aov_aligned is None or aov_aligned == 0:
            return {
                'status': 'amber',
                'message': 'Insufficient data to validate AOV precisely',
                'details': {'aov_calc': aov_calc, 'aov_aligned': aov_aligned},
                'severity': 'warning'
            }

        diff_pct = abs(aov_calc - aov_aligned) / max(aov_aligned, 1e-9)
        # Also compute optional alt delta for visibility
        alt_delta = None
        if method_used == 'subtotal_minus_discount' and aov_tot is not None:
            alt_delta = abs(aov_tot - aov_aligned) / max(aov_aligned, 1e-9)
        elif method_used == 'total_minus_shipping_taxes' and aov_sub is not None:
            alt_delta = abs(aov_sub - aov_aligned) / max(aov_aligned, 1e-9)
        elif method_used == 'line_items_per_order':
            # Provide alternative if either order-level method exists
            if aov_sub is not None:
                alt_delta = abs(aov_sub - aov_aligned) / max(aov_aligned, 1e-9)
            elif aov_tot is not None:
                alt_delta = abs(aov_tot - aov_aligned) / max(aov_aligned, 1e-9)
        # Surface aligned KPI components if present
        aligned_orders = None
        aligned_net_sales = None
        try:
            aligned_orders = int((aligned or {}).get('L28', {}).get('orders'))
        except Exception:
            pass
        try:
            aligned_net_sales = float((aligned or {}).get('L28', {}).get('net_sales'))
        except Exception:
            pass

        orders_in_window = None
        try:
            orders_in_window = int(d_orders['Name'].nunique()) if 'Name' in d_orders.columns else int(len(d_orders))
        except Exception:
            orders_in_window = None

        if diff_pct <= 0.1:
            return {
                'status': 'green',
                'message': f'✓ AOV consistent (Δ {diff_pct:.1%})',
                'details': {
                    'aov_calc': aov_calc,
                    'aov_aligned': aov_aligned,
                    'delta_pct': diff_pct,
                    'aov_subtotal_minus_discount': aov_sub,
                    'aov_total_minus_shipping_taxes': aov_tot,
                    'aov_line_items_per_order': aov_li,
                    'alt_delta_pct': alt_delta,
                    'method_used': method_used,
                    'orders_in_window': orders_in_window,
                    'aligned_orders': aligned_orders,
                    'net_sales_subtotal_minus_discount': net_sum_sub,
                    'net_sales_total_minus_shipping_taxes': net_sum_tot,
                    'net_sales_line_items_per_order': net_sum_li,
                    'aligned_net_sales': aligned_net_sales,
                },
                'severity': 'info'
            }
        elif diff_pct <= 0.25:
            return {
                'status': 'amber',
                'message': f'⚠️ AOV differs by {diff_pct:.0%} — check parsing and returns',
                'details': {
                    'aov_calc': aov_calc,
                    'aov_aligned': aov_aligned,
                    'delta_pct': diff_pct,
                    'aov_subtotal_minus_discount': aov_sub,
                    'aov_total_minus_shipping_taxes': aov_tot,
                    'aov_line_items_per_order': aov_li,
                    'alt_delta_pct': alt_delta,
                    'method_used': method_used,
                    'orders_in_window': orders_in_window,
                    'aligned_orders': aligned_orders,
                    'net_sales_subtotal_minus_discount': net_sum_sub,
                    'net_sales_total_minus_shipping_taxes': net_sum_tot,
                    'net_sales_line_items_per_order': net_sum_li,
                    'aligned_net_sales': aligned_net_sales,
                },
                'severity': 'warning'
            }
        else:
            return {
                'status': 'red',
                'message': f'❌ AOV mismatch {diff_pct:.0%} — verify monetary fields',
                'details': {
                    'aov_calc': aov_calc,
                    'aov_aligned': aov_aligned,
                    'delta_pct': diff_pct,
                    'aov_subtotal_minus_discount': aov_sub,
                    'aov_total_minus_shipping_taxes': aov_tot,
                    'aov_line_items_per_order': aov_li,
                    'alt_delta_pct': alt_delta,
                    'method_used': method_used,
                    'orders_in_window': orders_in_window,
                    'aligned_orders': aligned_orders,
                    'net_sales_subtotal_minus_discount': net_sum_sub,
                    'net_sales_total_minus_shipping_taxes': net_sum_tot,
                    'net_sales_line_items_per_order': net_sum_li,
                    'aligned_net_sales': aligned_net_sales,
                },
                'severity': 'critical'
            }


class AttributionMatchCheck(ValidationCheck):
    """Lightweight attribution/consistency check between line items and orders."""

    def __init__(self):
        super().__init__(
            "Attribution Consistency",
            "Verifies line-item revenue broadly matches order revenue"
        )

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('df')
        if df is None or df.empty:
            return {'status':'amber','message':'No data to verify attribution','details':{},'severity':'warning'}
        def _money(s): return pd.to_numeric(s, errors='coerce')
        if all(c in df.columns for c in ['Lineitem price','Lineitem quantity']):
            li_rev = (_money(df['Lineitem price']) * pd.to_numeric(df['Lineitem quantity'], errors='coerce')).sum(skipna=True)
        else:
            li_rev = np.nan
        if 'Subtotal' in df.columns:
            ord_rev = _money(df.drop_duplicates(subset=['Name'])['Subtotal'] if 'Name' in df.columns else df['Subtotal']).sum(skipna=True)
        else:
            ord_rev = np.nan
        if np.isnan(li_rev) or np.isnan(ord_rev) or ord_rev == 0:
            return {'status':'amber','message':'Insufficient fields for attribution check','details':{},'severity':'warning'}
        ratio = float(li_rev / ord_rev)
        if 0.8 <= ratio <= 1.2:
            return {'status':'green','message':'✓ Line items align with orders','details':{'ratio':ratio},'severity':'info'}
        elif 0.6 <= ratio <= 1.5:
            return {'status':'amber','message':f'⚠️ Line items to orders ratio {ratio:.2f}','details':{'ratio':ratio},'severity':'warning'}
        else:
            return {'status':'red','message':f'❌ Severe mismatch ratio {ratio:.2f}','details':{'ratio':ratio},'severity':'critical'}


class InventoryGuardrailCheck(ValidationCheck):
    """Guardrail on quantities and product fields to catch bad exports."""

    def __init__(self):
        super().__init__(
            "Inventory Guardrails",
            "Checks presence of line items and non-negative quantities"
        )

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('df')
        if df is None or df.empty:
            return {'status':'amber','message':'No line items present','details':{},'severity':'warning'}
        has_cols = all(c in df.columns for c in ['Lineitem name','Lineitem quantity'])
        if not has_cols:
            return {'status':'amber','message':'Missing line-item columns','details':{},'severity':'warning'}
        qty = pd.to_numeric(df['Lineitem quantity'], errors='coerce')
        neg_or_zero = (qty <= 0).mean()
        if neg_or_zero > 0.2:
            return {'status':'red','message':f'❌ {neg_or_zero:.0%} non-positive quantities','details':{},'severity':'critical'}
        return {'status':'green','message':'✓ Quantities look reasonable','details':{},'severity':'info'}

 

class DataAnomalyCheck(ValidationCheck):
    """
    Detects statistical anomalies that could indicate data quality issues.
    Uses z-scores and IQR methods to flag suspicious patterns.
    """
    
    def __init__(self):
        super().__init__(
            "Data Anomalies",
            "Detects outliers and suspicious patterns in order data"
        )
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('df')
        
        if df is None or df.empty:
            return {
                'status': 'red',
                'message': 'No data for anomaly detection',
                'details': {},
                'severity': 'critical'
            }
        
        anomalies = []
        
        # 1. AOV Spike Detection
        if 'Subtotal' in df.columns and 'Total Discount' in df.columns:
            # Calculate AOV per order
            df_orders = df.drop_duplicates(subset=['Name']) if 'Name' in df.columns else df
            aov = (pd.to_numeric(df_orders['Subtotal'], errors='coerce') - 
                   pd.to_numeric(df_orders['Total Discount'], errors='coerce'))
            
            # Remove nulls and zeros
            aov_clean = aov[aov > 0].dropna()
            
            if len(aov_clean) > 10:
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(aov_clean))
                extreme_outliers = (z_scores > 4).sum()
                
                # IQR method for more robust detection
                Q1 = aov_clean.quantile(0.25)
                Q3 = aov_clean.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers_iqr = ((aov_clean < lower_bound) | (aov_clean > upper_bound)).sum()
                
                if extreme_outliers > 0 or outliers_iqr > len(aov_clean) * 0.05:
                    max_aov = aov_clean.max()
                    median_aov = aov_clean.median()
                    if max_aov > median_aov * 10:
                        anomalies.append(f"AOV outlier: Max ${max_aov:.0f} vs median ${median_aov:.0f} (10x difference!)")
        
        # 2. Temporal Anomalies
        if 'Created at' in df.columns:
            df['Created at'] = pd.to_datetime(df['Created at'], errors='coerce')
            
            # Check for impossible dates
            future_orders = df[df['Created at'] > datetime.now()]
            if len(future_orders) > 0:
                anomalies.append(f"❌ {len(future_orders)} orders with future dates")
            
            # Check for suspiciously old orders in recent export
            very_old = df[df['Created at'] < datetime.now() - timedelta(days=1095)]  # 3+ years
            if len(very_old) > len(df) * 0.3:
                anomalies.append(f"30%+ orders are 3+ years old - verify date parsing")
            
            # Detect date clustering (all orders on same day)
            dates = df['Created at'].dt.date.value_counts()
            if len(dates) > 0:
                top_day_pct = dates.iloc[0] / len(df) * 100
                if top_day_pct > 50:
                    anomalies.append(f"{top_day_pct:.0f}% orders on single day - possible test data")
        
        # 3. Discount Rate Anomalies
        if 'Total Discount' in df.columns and 'Subtotal' in df.columns:
            df_clean = df.dropna(subset=['Total Discount', 'Subtotal'])
            discount_rates = pd.to_numeric(df_clean['Total Discount'], errors='coerce') / pd.to_numeric(df_clean['Subtotal'], errors='coerce')
            discount_rates = discount_rates[(discount_rates >= 0) & (discount_rates <= 1)]
            
            if len(discount_rates) > 0:
                # Flag if >20% of orders have >50% discount
                high_discount = (discount_rates > 0.5).mean()
                if high_discount > 0.2:
                    anomalies.append(f"⚠️ {high_discount:.0%} orders with >50% discount")
                
                # Flag if ALL orders have exact same discount
                if discount_rates.nunique() == 1 and len(discount_rates) > 10:
                    anomalies.append(f"All orders have identical discount rate: {discount_rates.iloc[0]:.0%}")
        
        # 4. Customer Pattern Anomalies
        if 'Customer Email' in df.columns:
            customer_orders = df.groupby('Customer Email')['Name'].nunique() if 'Name' in df.columns else df.groupby('Customer Email').size()
            
            # Check for test customers
            test_emails = df[df['Customer Email'].astype(str).str.contains('test|example|sample', case=False, na=False)]
            if len(test_emails) > 0:
                anomalies.append(f"Found {len(test_emails)} orders with test email patterns")
            
            # Check for single customer domination
            if len(customer_orders) > 0:
                top_customer_orders = customer_orders.max()
                if top_customer_orders > len(df) * 0.15:
                    anomalies.append(f"Single customer has {top_customer_orders} orders (>15% of total)")
        
        # 5. Product Name Anomalies
        if 'Lineitem name' in df.columns:
            products = df['Lineitem name'].astype(str)
            
            # Check for placeholder products
            placeholder_patterns = products.str.contains('test|sample|placeholder|xxx|delete', case=False, na=False)
            if placeholder_patterns.sum() > 0:
                anomalies.append(f"Found {placeholder_patterns.sum()} line items with test/placeholder names")
            
            # Check for missing product names
            missing_names = products.isna() | (products == '') | (products == 'nan')
            if missing_names.sum() > len(products) * 0.1:
                anomalies.append(f"{missing_names.mean():.0%} line items missing product names")
        
        # Determine overall status
        if len(anomalies) == 0:
            return {
                'status': 'green',
                'message': '✓ No data anomalies detected',
                'details': {'checks_passed': 'All patterns within normal ranges'},
                'severity': 'info'
            }
        elif len(anomalies) <= 2:
            return {
                'status': 'amber',
                'message': f'⚠️ {len(anomalies)} anomalies detected - review recommended',
                'details': {'anomalies': anomalies},
                'severity': 'warning'
            }
        else:
            return {
                'status': 'red',
                'message': f'❌ {len(anomalies)} critical anomalies - data quality issues likely',
                'details': {'anomalies': anomalies[:5]},  # Show top 5
                'severity': 'critical'
            }

class MetricConsistencyCheck(ValidationCheck):
    """
    Validates that key metrics make business sense and are internally consistent.
    """
    
    def __init__(self):
        super().__init__(
            "Metric Consistency",
            "Validates that KPIs are realistic and internally consistent"
        )
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        aligned = data.get('aligned', {})
        df = data.get('df')
        
        warnings = []
        
        # Check L28 metrics if available
        l28 = aligned.get('L28', {})
        if l28:
            # AOV sanity check
            aov = l28.get('aov')
            if aov:
                if aov < 10:
                    warnings.append(f"AOV ${aov:.2f} unusually low - check currency/decimals")
                elif aov > 5000:
                    warnings.append(f"AOV ${aov:.2f} unusually high - verify not B2B")
            
            # Repeat rate sanity
            repeat_rate = l28.get('repeat_share')
            if repeat_rate is not None:
                if repeat_rate > 0.8:
                    warnings.append(f"Repeat rate {repeat_rate:.0%} unusually high - verify calculation")
                elif repeat_rate < 0.05 and l28.get('orders', 0) > 100:
                    warnings.append(f"Repeat rate {repeat_rate:.0%} suspiciously low for {l28.get('orders')} orders")
            
            # Week-over-week changes
            delta = l28.get('delta', {})
            
            # Flag extreme changes
            for metric, change in delta.items():
                if change is not None and abs(change) > 2.0:  # >200% change
                    warnings.append(f"{metric} changed {change:.0%} - verify data completeness")
            
            # Revenue vs orders consistency
            if l28.get('net_sales') and l28.get('orders') and l28.get('aov'):
                expected_revenue = l28['orders'] * l28['aov']
                actual_revenue = l28['net_sales']
                discrepancy = abs(expected_revenue - actual_revenue) / actual_revenue if actual_revenue else 0
                if discrepancy > 0.1:  # >10% discrepancy
                    warnings.append(f"Revenue inconsistency: orders × AOV ≠ net_sales (off by {discrepancy:.0%})")
        
        # Check for impossible metrics
        if df is not None and not df.empty:
            if 'Total Discount' in df.columns and 'Subtotal' in df.columns:
                # Check for negative revenues
                net = pd.to_numeric(df['Subtotal'], errors='coerce') - pd.to_numeric(df['Total Discount'], errors='coerce')
                if (net < 0).any():
                    neg_count = (net < 0).sum()
                    warnings.append(f"{neg_count} orders with negative net revenue")
        
        if len(warnings) == 0:
            return {
                'status': 'green',
                'message': '✓ All metrics internally consistent',
                'details': {},
                'severity': 'info'
            }
        elif len(warnings) <= 2:
            return {
                'status': 'amber',
                'message': f'⚠️ Metric warnings: {warnings[0]}',
                'details': {'warnings': warnings},
                'severity': 'warning'
            }
        else:
            return {
                'status': 'red',
                'message': f'❌ Multiple metric inconsistencies detected',
                'details': {'warnings': warnings[:5]},
                'severity': 'critical'
            }

class TransactionVolumeCheck(ValidationCheck):
    """
    Enhanced version with actual sanity checks for order volumes.
    """
    
    def __init__(self):
        super().__init__(
            "Transaction Volume",
            "Validates order count consistency and completeness"
        )
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('df')
        aligned = data.get('aligned', {})
        
        if df is None or df.empty:
            return {
                'status': 'red',
                'message': 'No data available for validation',
                'details': {},
                'severity': 'critical'
            }
        
        issues = []
        
        # Check for reasonable order volume
        total_orders = len(df['Name'].unique()) if 'Name' in df.columns else len(df)
        
        # Business logic checks
        if total_orders < 10:
            issues.append(f"Only {total_orders} orders - insufficient for analysis")
            severity = 'critical'
        elif total_orders < 100:
            issues.append(f"Low order volume ({total_orders}) - recommendations may be unreliable")
            severity = 'warning'
        
        # Check for duplicate orders
        if 'Name' in df.columns:
            order_counts = df['Name'].value_counts()
            true_duplicates = order_counts[order_counts > 10]  # Same order appearing >10 times is suspicious
            if len(true_duplicates) > 0:
                issues.append(f"{len(true_duplicates)} orders appear 10+ times (likely duplicates)")
        
        # Financial completeness check
        if 'Financial Status' in df.columns:
            financial_complete = df['Financial Status'].notna().mean()
            if financial_complete < 0.95:
                missing = int((1 - financial_complete) * total_orders)
                issues.append(f"{missing} orders missing financial status")
        
        # Check for refund rate
        if 'Financial Status' in df.columns:
            refunded = df['Financial Status'].astype(str).str.contains('refunded', case=False, na=False).mean()
            if refunded > 0.15:
                issues.append(f"High refund rate: {refunded:.0%}")
        
        # Cancelled orders check
        if 'Cancelled at' in df.columns:
            cancelled = pd.to_datetime(df['Cancelled at'], errors='coerce').notna().mean()
            if cancelled > 0.1:
                issues.append(f"High cancellation rate: {cancelled:.0%}")
        
        # Currency consistency
        if 'Currency' in df.columns:
            currencies = df['Currency'].nunique()
            if currencies > 1:
                currency_mix = df['Currency'].value_counts().to_dict()
                issues.append(f"Mixed currencies detected: {currency_mix}")
        
        if len(issues) == 0:
            status = 'green'
            severity = 'info'
            message = f"✓ Transaction volume verified: {total_orders} clean orders"
        elif len(issues) <= 2:
            status = 'amber'
            severity = 'warning'
            message = f"⚠️ Minor issues: {' | '.join(issues[:2])}"
        else:
            status = 'red'
            severity = 'critical'
            message = f"❌ Critical issues: {' | '.join(issues[:3])}"
        
        return {
            'status': status,
            'message': message,
            'details': {
                'total_orders': total_orders,
                'issues': issues,
                'financial_completeness': f"{financial_complete:.0%}" if 'Financial Status' in df.columns else 'N/A'
            },
            'severity': severity
        }


class DataValidationEngine:
    """Orchestrates all validation checks and renders results."""

    def __init__(self):
        self.checks: List[ValidationCheck] = [
            TransactionVolumeCheck(),
            AOVConsistencyCheck(),
            DataAnomalyCheck(),
            MetricConsistencyCheck(),
        ]

    def run_all_checks(self, df: pd.DataFrame, aligned: Dict[str, Any], actions: List[Dict[str,Any]] | None = None, qa: Dict[str, Any] | None = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            'checks': {}, 'overall_status': 'green', 'validation_score': 100,
            'critical_issues': [], 'warnings': [], 'summary': '', 'data_quality_grade': 'A'
        }
        status_map = {'green': 100, 'amber': 70, 'red': 40}
        scores: List[int] = []
        worst = 'green'
        def worse(a,b):
            order = {'green':0,'amber':1,'red':2}
            return a if order[a] >= order[b] else b
        for chk in self.checks:
            out = chk.run({'df': df, 'aligned': aligned, 'actions': actions, 'qa': qa})
            results['checks'][chk.name] = out
            st = out.get('status','amber')
            scores.append(status_map.get(st,70))
            worst = worse(worst, st)
            if st == 'red':
                results['critical_issues'].append(out.get('message',''))
            elif st == 'amber':
                results['warnings'].append(out.get('message',''))
        avg = int(round(sum(scores)/len(scores))) if scores else 70
        results['validation_score'] = max(0, min(100, avg))
        results['overall_status'] = worst
        results['summary'] = (
            'All checks passed' if worst=='green' else
            'Some issues detected' if worst=='amber' else 'Critical data issues detected'
        )
        # Letter grade for UI
        grade = 'A' if avg>=90 else ('B+' if avg>=80 else ('B' if avg>=70 else ('C' if avg>=60 else 'D')))
        results['data_quality_grade'] = grade
        return results

    def to_html_panel(self, results: Dict[str, Any]) -> str:
        status = results.get('overall_status','amber')
        score = results.get('validation_score',0)
        return f"""
        <div class='validation-panel validation-{status}'>
          <h3>Data Validation Report</h3>
          <p>Overall status: <strong>{status.upper()}</strong> — Score: <strong>{score}/100</strong></p>
        </div>
        """
