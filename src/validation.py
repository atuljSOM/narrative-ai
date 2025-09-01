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
    Verifies AOV consistency across different calculation methods.
    Flags hidden fees or calculation errors.
    """
    
    def __init__(self):
        super().__init__(
            "AOV Consistency",
            "Validates average order value calculations"
        )
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('df')
        aligned = data.get('aligned', {})
        
        if df is None or df.empty:
            return {
                'status': 'red',
                'message': 'No data for AOV validation',
                'details': {},
                'severity': 'critical'
            }
        
        # Method 1: From aligned summary (if available)
        aov_aligned = aligned.get('L28', {}).get('aov') or aligned.get('L7', {}).get('aov')
        
        # Method 2: Direct calculation from orders
        if 'Subtotal' in df.columns and 'Total Discount' in df.columns:
            net_sales = (pd.to_numeric(df['Subtotal'], errors='coerce') - 
                        pd.to_numeric(df['Total Discount'], errors='coerce')).sum()
            order_count = df['Name'].nunique() if 'Name' in df.columns else len(df)
            aov_direct = net_sales / order_count if order_count > 0 else 0
        else:
            aov_direct = None
        
        # Method 3: From Total (if available)
        if 'Total' in df.columns:
            total_sum = pd.to_numeric(df['Total'], errors='coerce').sum()
            order_count = df['Name'].nunique() if 'Name' in df.columns else len(df)
            aov_total = total_sum / order_count if order_count > 0 else 0
        else:
            aov_total = None
        
        # Compare methods
        aovs = [a for a in [aov_aligned, aov_direct, aov_total] if a is not None and a > 0]
        
        if len(aovs) < 2:
            return {
                'status': 'amber',
                'message': 'Insufficient data for AOV cross-validation',
                'details': {'aov_calculated': aovs[0] if aovs else None},
                'severity': 'warning'
            }
        
        aov_mean = np.mean(aovs)
        aov_std = np.std(aovs)
        cv = (aov_std / aov_mean * 100) if aov_mean > 0 else 0
        
        if cv > 10:
            status = 'red'
            severity = 'critical'
            message = f"⚠️ AOV calculations differ by {cv:.1f}% - possible data issue"
        elif cv > 5:
            status = 'amber'
            severity = 'warning'
            message = f"Minor AOV variance ({cv:.1f}%) detected across methods"
        else:
            status = 'green'
            severity = 'info'
            message = f"✓ AOV consistent: ${aov_mean:.2f} (±{cv:.1f}%)"
        
        return {
            'status': status,
            'message': message,
            'details': {
                'aov_mean': f"${aov_mean:.2f}",
                'aov_methods': {
                    'from_summary': f"${aov_aligned:.2f}" if aov_aligned else "N/A",
                    'from_orders': f"${aov_direct:.2f}" if aov_direct else "N/A",
                    'from_totals': f"${aov_total:.2f}" if aov_total else "N/A"
                },
                'coefficient_variation': f"{cv:.1f}%"
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