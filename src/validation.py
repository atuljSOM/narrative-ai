"""
Data Validation Engine for Aura
Performs critical data consistency checks across potential multi-source inputs
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

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

class TransactionVolumeCheck(ValidationCheck):
    """
    Compares order volume from Shopify vs payment gateway (simulated for MVP).
    In production, would compare with Stripe API data.
    """
    
    def __init__(self):
        super().__init__(
            "Transaction Volume",
            "Validates order count consistency across systems"
        )
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('df')
        qa = data.get('qa', {})
        
        if df is None or df.empty:
            return {
                'status': 'red',
                'message': 'No data available for validation',
                'details': {},
                'severity': 'critical'
            }
        
        # For MVP: Check for duplicate orders and missing data
        total_orders = len(df['Name'].unique()) if 'Name' in df.columns else 0
        
        # Check for potential duplicates
        duplicates = df[df.duplicated(subset=['Name'], keep=False)] if 'Name' in df.columns else pd.DataFrame()
        duplicate_rate = len(duplicates) / len(df) if len(df) > 0 else 0
        
        # Check financial status completeness
        financial_complete = df['Financial Status'].notna().mean() if 'Financial Status' in df.columns else 0
        
        # Simulate payment gateway comparison (in production, would call Stripe API)
        expected_payments = total_orders
        missing_payments = int(total_orders * (1 - financial_complete))
        discrepancy_pct = (missing_payments / expected_payments * 100) if expected_payments > 0 else 0
        
        if discrepancy_pct > 5:
            status = 'red'
            severity = 'critical'
            message = f"⚠️ {missing_payments} orders ({discrepancy_pct:.1f}%) missing payment status"
        elif discrepancy_pct > 2:
            status = 'amber'
            severity = 'warning'
            message = f"Minor discrepancy: {missing_payments} orders without payment confirmation"
        else:
            status = 'green'
            severity = 'info'
            message = f"✓ Transaction volume verified: {total_orders} orders processed"
        
        return {
            'status': status,
            'message': message,
            'details': {
                'total_orders': total_orders,
                'missing_payment_status': missing_payments,
                'duplicate_rate': f"{duplicate_rate:.1%}",
                'discrepancy_pct': f"{discrepancy_pct:.1f}%"
            },
            'severity': severity
        }

class AOVConsistencyCheck(ValidationCheck):
    """
    Verifies AOV stability across two independent Shopify calculations:
      1) Order-level: sum(Subtotal − Total Discount) / #unique_orders
      2) Line-item:  (Σ(Lineitem price * Lineitem quantity) − Σ(Lineitem discount)) / #unique_orders

    Reports percent difference and flags per thresholds:
      green <5%, amber 5–12%, red >12%.
    """

    def __init__(self):
        super().__init__(
            "AOV Consistency",
            "Validates AOV via order-level vs line-item paths"
        )

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('df')

        if df is None or df.empty:
            return {
                'status': 'red',
                'message': 'No data for AOV validation',
                'details': {},
                'severity': 'critical'
            }

        d = df.copy()

        # Exclude cancelled/refunded if present (defensive)
        if 'Cancelled at' in d.columns:
            canc = pd.to_datetime(d['Cancelled at'], errors='coerce')
            d = d[canc.isna()]
        if 'Financial Status' in d.columns:
            d = d[~d['Financial Status'].astype(str).str.contains('refunded|chargeback', case=False, na=False)]

        # Determine unique order count (Shopify 'Name') AFTER filtering
        order_count = int(d['Name'].nunique()) if 'Name' in d.columns else int(len(d))

        # ---- Order-level AOV (de-duplicate orders) ----
        aov_orders = None
        if {'Subtotal', 'Total Discount'}.issubset(d.columns) and order_count > 0:
            # Drop duplicates to avoid summing order-level fields multiple times
            if 'Name' in d.columns:
                d_orders = d.sort_values('Created at' if 'Created at' in d.columns else 'Name')
                d_orders = d_orders.drop_duplicates(subset=['Name'])
            else:
                d_orders = d
            subtotal_sum = pd.to_numeric(d_orders['Subtotal'], errors='coerce').sum()
            discount_sum = pd.to_numeric(d_orders['Total Discount'], errors='coerce').sum()
            net_sales_orders = float(subtotal_sum - discount_sum)
            aov_orders = float(net_sales_orders / order_count) if order_count else None

        # ---- Line-item AOV (independent path) ----
        aov_line = None
        if {'Lineitem price', 'Lineitem quantity'}.issubset(d.columns) and order_count > 0:
            li_price = pd.to_numeric(d['Lineitem price'], errors='coerce')
            li_qty   = pd.to_numeric(d['Lineitem quantity'], errors='coerce')
            li_rev   = float((li_price * li_qty).sum())
            if 'Lineitem discount' in d.columns:
                li_disc = pd.to_numeric(d['Lineitem discount'], errors='coerce').sum()
            else:
                li_disc = 0.0
            net_sales_line = float(li_rev - (li_disc if not np.isnan(li_disc) else 0.0))
            aov_line = float(net_sales_line / order_count) if order_count else None

        # ---- Validate presence of both paths ----
        if not (aov_orders and aov_orders > 0) or not (aov_line and aov_line > 0):
            return {
                'status': 'amber',
                'message': 'Insufficient data to cross-check AOV (need order-level and line-item fields)',
                'details': {
                    'aov_orders': None if not (aov_orders and aov_orders > 0) else round(aov_orders, 2),
                    'aov_line': None if not (aov_line and aov_line > 0) else round(aov_line, 2),
                    'order_count': order_count
                },
                'severity': 'warning'
            }

        # ---- Compare percent difference ----
        diff_pct = float(abs(aov_orders - aov_line) / aov_orders * 100.0) if aov_orders else 0.0

        if diff_pct > 12.0:
            status, severity = 'red', 'critical'
            message = f"⚠️ AOV methods differ by {diff_pct:.1f}% (orders ${aov_orders:.2f} vs line ${aov_line:.2f})"
        elif diff_pct >= 5.0:
            status, severity = 'amber', 'warning'
            message = f"Minor AOV variance {diff_pct:.1f}% (orders ${aov_orders:.2f} vs line ${aov_line:.2f})"
        else:
            status, severity = 'green', 'info'
            message = f"✓ AOV stable: orders ${aov_orders:.2f} vs line ${aov_line:.2f} (Δ{diff_pct:.1f}%)"

        return {
            'status': status,
            'message': message,
            'details': {
                'aov_orders': f"${aov_orders:.2f}",
                'aov_line': f"${aov_line:.2f}",
                'difference_pct': f"{diff_pct:.1f}%",
                'order_count': order_count
            },
            'severity': severity
        }

class AttributionMatchCheck(ValidationCheck):
    """
    For MVP: Checks for attribution data quality.
    In production: Would compare Shopify attribution vs Meta/Google APIs.
    """
    
    def __init__(self):
        super().__init__(
            "Attribution Match",
            "Validates marketing attribution consistency"
        )
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('df')
        
        # For MVP: Check for UTM parameters or source data if available
        # In production, this would compare with actual ad platform APIs
        
        if df is None or df.empty:
            return {
                'status': 'amber',
                'message': 'No attribution data available',
                'details': {},
                'severity': 'warning'
            }
        
        # Check if we have any attribution columns (simplified for MVP)
        attribution_cols = [c for c in df.columns if any(
            x in c.lower() for x in ['source', 'utm', 'campaign', 'channel', 'referr']
        )]
        
        if not attribution_cols:
            # No attribution data - common for basic Shopify exports
            return {
                'status': 'amber',
                'message': 'No attribution data in export. Consider enabling UTM tracking.',
                'details': {
                    'recommendation': 'Enable UTM parameters in marketing campaigns'
                },
                'severity': 'info'
            }
        
        # If we have attribution, check completeness
        for col in attribution_cols[:1]:  # Check first attribution column
            fill_rate = df[col].notna().mean()
            
            if fill_rate < 0.5:
                status = 'amber'
                severity = 'warning'
                message = f"Attribution data sparse ({fill_rate:.0%} complete)"
            else:
                status = 'green'
                severity = 'info'
                message = f"✓ Attribution data available ({fill_rate:.0%} coverage)"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'attribution_columns': attribution_cols,
                    'fill_rate': f"{fill_rate:.0%}"
                },
                'severity': severity
            }
        
        return {
            'status': 'green',
            'message': '✓ Attribution tracking configured',
            'details': {},
            'severity': 'info'
        }

class InventoryGuardrailCheck(ValidationCheck):
    """
    Checks for potential inventory issues that could affect recommended actions.
    For MVP: Analyzes sales velocity patterns.
    """
    
    def __init__(self):
        super().__init__(
            "Inventory Guardrail",
            "Validates inventory levels for recommended actions"
        )
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('df')
        actions = data.get('actions', [])
        
        if df is None or df.empty:
            return {
                'status': 'amber',
                'message': 'No data for inventory validation',
                'details': {},
                'severity': 'warning'
            }
        
        # Analyze top SKUs velocity
        if 'Lineitem name' in df.columns and 'Created at' in df.columns:
            df['Created at'] = pd.to_datetime(df['Created at'], errors='coerce')
            
            # Last 30 days sales by SKU
            recent = df[df['Created at'] >= df['Created at'].max() - timedelta(days=30)]
            
            if not recent.empty and 'Lineitem quantity' in recent.columns:
                sku_velocity = recent.groupby('Lineitem name')['Lineitem quantity'].sum().sort_values(ascending=False)
                top_sku = sku_velocity.index[0] if len(sku_velocity) > 0 else None
                top_sku_velocity = sku_velocity.iloc[0] if len(sku_velocity) > 0 else 0
                
                # Check if any action targets the top SKU
                amplify_action = any('bestseller' in str(a.get('play_id', '')).lower() for a in actions)
                
                if amplify_action and top_sku_velocity > 100:
                    status = 'amber'
                    severity = 'warning'
                    message = f"⚠️ High velocity on {top_sku} ({int(top_sku_velocity)} units/month). Verify inventory before amplifying."
                elif top_sku_velocity > 200:
                    status = 'amber'
                    severity = 'warning'
                    message = f"Very high sales velocity detected. Consider inventory levels."
                else:
                    status = 'green'
                    severity = 'info'
                    message = "✓ Sales velocity within normal range"
                
                return {
                    'status': status,
                    'message': message,
                    'details': {
                        'top_sku': top_sku,
                        'monthly_velocity': int(top_sku_velocity),
                        'recommendation': 'Upload inventory CSV for precise stock validation' if status == 'amber' else None
                    },
                    'severity': severity
                }
        
        return {
            'status': 'green',
            'message': '✓ No inventory concerns detected',
            'details': {},
            'severity': 'info'
        }

class DataValidationEngine:
    """
    Main validation engine that runs all checks and aggregates results.
    """
    
    def __init__(self):
        self.checks = [
            TransactionVolumeCheck(),
            AOVConsistencyCheck(),
            AttributionMatchCheck(),
            InventoryGuardrailCheck()
        ]
    
    def run_all_checks(self, df: pd.DataFrame, aligned: Dict[str, Any] = None, 
                       actions: List[Dict] = None, qa: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run all validation checks and return aggregated results.
        """
        data = {
            'df': df,
            'aligned': aligned or {},
            'actions': actions or [],
            'qa': qa or {}
        }
        
        results = {}
        overall_status = 'green'
        critical_issues = []
        warnings = []
        
        for check in self.checks:
            result = check.run(data)
            results[check.name] = result
            
            # Update overall status
            if result['status'] == 'red':
                overall_status = 'red'
                if result['severity'] == 'critical':
                    critical_issues.append(result['message'])
            elif result['status'] == 'amber' and overall_status != 'red':
                overall_status = 'amber'
                if result['severity'] == 'warning':
                    warnings.append(result['message'])
        
        # Calculate validation score (0-100)
        status_scores = {'green': 100, 'amber': 70, 'red': 30}
        scores = [status_scores.get(r['status'], 50) for r in results.values()]
        validation_score = np.mean(scores)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'validation_score': round(validation_score),
            'critical_issues': critical_issues,
            'warnings': warnings,
            'checks': results,
            'summary': self._generate_summary(overall_status, validation_score, len(critical_issues), len(warnings))
        }
    
    def _generate_summary(self, status: str, score: float, critical_count: int, warning_count: int) -> str:
        """Generate a human-readable summary."""
        if status == 'green':
            return f"✓ All validation checks passed (Score: {score:.0f}/100)"
        elif status == 'amber':
            return f"⚠️ {warning_count} warnings detected. Review recommended. (Score: {score:.0f}/100)"
        else:
            return f"⚠️ {critical_count} critical issues require attention. (Score: {score:.0f}/100)"
    
    def to_html_panel(self, results: Dict[str, Any]) -> str:
        """
        Generate HTML for the validation panel in the briefing.
        This will be included in the Jinja template.
        """
        status_icons = {
            'green': '✓',
            'amber': '⚠️',
            'red': '❌'
        }
        
        status_colors = {
            'green': '#10b981',
            'amber': '#f59e0b', 
            'red': '#ef4444'
        }
        
        panel_html = f"""
        <div class="validation-panel" style="background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 20px 0;">
            <h3 style="margin: 0 0 12px 0; color: #111827; font-size: 18px;">Data Validation Report</h3>
            <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 16px;">
                <span style="font-size: 24px; font-weight: bold; color: {status_colors[results['overall_status']]};">
                    {results['validation_score']}/100
                </span>
                <span style="color: #6b7280; font-size: 14px;">{results['summary']}</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;">
        """
        
        for check_name, check_result in results['checks'].items():
            icon = status_icons[check_result['status']]
            color = status_colors[check_result['status']]
            
            panel_html += f"""
                <div style="background: white; border: 1px solid #e5e7eb; border-radius: 6px; padding: 12px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="color: {color}; font-size: 18px;">{icon}</span>
                        <span style="font-weight: 600; color: #374151; font-size: 14px;">{check_name}</span>
                    </div>
                    <p style="margin: 0; color: #6b7280; font-size: 12px; line-height: 1.4;">
                        {check_result['message']}
                    </p>
                </div>
            """
        
        panel_html += """
            </div>
        </div>
        """
        
        return panel_html
