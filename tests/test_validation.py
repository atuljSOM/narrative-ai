"""
Test the data validation module
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.validation import (
    DataValidationEngine, 
    TransactionVolumeCheck,
    AOVConsistencyCheck,
    AttributionMatchCheck,
    InventoryGuardrailCheck
)

def test_transaction_volume_check():
    """Test transaction volume validation"""
    check = TransactionVolumeCheck()
    
    # Test with good data
    df = pd.DataFrame({
        'Name': [f'#{i}' for i in range(100)],
        'Financial Status': ['paid'] * 100
    })
    result = check.run({'df': df})
    assert result['status'] == 'green'
    
    # Test with missing payment status
    df_bad = pd.DataFrame({
        'Name': [f'#{i}' for i in range(100)],
        'Financial Status': ['paid'] * 50 + [np.nan] * 50
    })
    result = check.run({'df': df_bad})
    assert result['status'] == 'red'
    assert 'critical' in result['severity']

def test_aov_consistency_check():
    """Test AOV consistency validation"""
    check = AOVConsistencyCheck()
    
    # Create consistent data
    df = pd.DataFrame({
        'Name': [f'#{i}' for i in range(50)],
        'Subtotal': [100.0] * 50,
        'Total Discount': [10.0] * 50,
        'Total': [95.0] * 50,  # Including shipping/tax
        'Shipping': [5.0] * 50,
        'Taxes': [0.0] * 50
    })
    
    aligned = {'L28': {'aov': 90.0}}  # Matches our calculation
    
    result = check.run({'df': df, 'aligned': aligned})
    assert result['status'] in ['green', 'amber']  # Should be consistent
    
def test_full_validation_engine():
    """Test the complete validation engine"""
    validator = DataValidationEngine()
    
    # Create sample data
    df = pd.DataFrame({
        'Name': [f'#{i}' for i in range(200)],
        'Created at': pd.date_range('2025-07-01', periods=200),
        'Financial Status': ['paid'] * 190 + [np.nan] * 10,
        'Subtotal': np.random.normal(100, 20, 200),
        'Total Discount': np.random.uniform(0, 20, 200),
        'Total': np.random.normal(105, 20, 200),
        'Shipping': [5.0] * 200,
        'Taxes': [0.0] * 200,
        'Lineitem name': np.random.choice(['SKU_A', 'SKU_B', 'SKU_C'], 200),
        'Lineitem quantity': np.random.randint(1, 5, 200),
        'Customer Email': [f'customer{i%50}@example.com' for i in range(200)]
    })
    
    aligned = {
        'L28': {
            'aov': 100.0,
            'orders': 100,
            'net_sales': 10000.0
        }
    }
    
    actions = [
        {'play_id': 'bestseller_amplify', 'title': 'Amplify Bestseller'}
    ]
    
    results = validator.run_all_checks(df=df, aligned=aligned, actions=actions)
    
    # Verify structure
    assert 'overall_status' in results
    assert 'validation_score' in results
    assert 'checks' in results
    assert len(results['checks']) == 4  # We have 4 checks
    
    # Verify score is between 0 and 100
    assert 0 <= results['validation_score'] <= 100
    
    # Verify HTML generation
    html = validator.to_html_panel(results)
    assert 'validation-panel' in html
    assert 'Data Validation Report' in html
    
    print(f"Validation Score: {results['validation_score']}/100")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Summary: {results['summary']}")


def test_debug_window_alignment():
    """Debug helper to print absolute values used in windowed validation comparisons."""
    import pandas as pd
    import numpy as np

    # Build a sample df with monetary fields and dates
    n = 100
    dates = pd.date_range('2025-06-01', periods=n)
    df = pd.DataFrame({
        'Name': [f'#{i}' for i in range(n)],
        'Created at': dates,
        'Cancelled at': [np.nan]*n,
        'Financial Status': ['paid'] * n,
        'Subtotal': np.random.normal(100, 15, n),
        'Total Discount': np.random.uniform(0, 20, n),
        'Total': np.random.normal(105, 15, n),
        'Shipping': [5.0] * n,
        'Taxes': [0.0] * n,
        'Lineitem name': np.random.choice(['Alpha Serum 30ml', 'Beta Cream 50ml', 'Gamma Cleanser'], n),
        'Lineitem quantity': np.random.randint(1, 3, n),
        'Customer Email': [f'user{i%25}@example.com' for i in range(n)]
    })

    # Aligned object stub with anchor and window_days for L28
    anchor = pd.to_datetime(df['Created at']).max()
    aligned = {
        'anchor': anchor,
        'L28': {
            'window_days': 28,
            # For comparability, compute a KPI-like AOV over the same window here
        }
    }

    # Compute window bounds as in validation
    recent_end = anchor.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    recent_start = recent_end.normalize() - pd.Timedelta(days=aligned['L28']['window_days'] - 1)

    # Apply KPI-like exclusions (cancelled/refunded)
    d = df.copy()
    d['Created at'] = pd.to_datetime(d['Created at'], errors='coerce')
    d['Cancelled at'] = pd.to_datetime(d['Cancelled at'], errors='coerce')
    d = d[d['Cancelled at'].isna()]
    d = d[~d['Financial Status'].astype(str).str.contains('refunded|chargeback', case=False, na=False)]
    in_win = d[(d['Created at'] >= recent_start) & (d['Created at'] <= recent_end)]
    d_orders = in_win.drop_duplicates(subset=['Name']) if 'Name' in in_win.columns else in_win

    # KPI-like nets: Subtotal - Total Discount; else Total - Shipping - Taxes
    def _money(s):
        return pd.to_numeric(s, errors='coerce')
    # Compute both variants for visibility
    nets_sub = None
    nets_tot = None
    if all(c in d_orders.columns for c in ['Subtotal', 'Total Discount']):
        nets_sub = _money(d_orders['Subtotal']) - _money(d_orders['Total Discount'])
    if all(c in d_orders.columns for c in ['Total', 'Shipping', 'Taxes']):
        nets_tot = _money(d_orders['Total']) - _money(d_orders['Shipping']) - _money(d_orders['Taxes'])
    aov_sub = float(nets_sub.dropna().mean()) if nets_sub is not None and not nets_sub.dropna().empty else np.nan
    aov_tot = float(nets_tot.dropna().mean()) if nets_tot is not None and not nets_tot.dropna().empty else np.nan
    # Choose KPI-like method (subtotal-discount first)
    aov_calc = aov_sub if not np.isnan(aov_sub) else aov_tot
    aligned['L28']['aov'] = aov_calc  # target for check so diff≈0

    # Run the AOVConsistencyCheck (should be green with Δ≈0)
    check = AOVConsistencyCheck()
    out = check.run({'df': df, 'aligned': aligned})

    # Print absolute numbers for visibility
    print("\n[DEBUG] Window alignment and AOV comparison:")
    print(f"Anchor: {anchor}")
    print(f"Window: {recent_start.date()} to {recent_end.date()} ({aligned['L28']['window_days']}d)")
    print(f"Orders in window (pre-dedupe): {len(in_win)}; unique orders: {len(d_orders)}")
    print(f"AOV_subtotal_minus_discount = {aov_sub:.2f}")
    print(f"AOV_total_minus_shipping_taxes = {aov_tot:.2f}")
    print(f"AOV_calc (chosen) = {aov_calc:.2f}")
    print(f"AOV_aligned (L28) = {aligned['L28']['aov']:.2f}")
    print(f"Validation check => status={out['status']}, message={out['message']}")

    # Sanity: With aligned derived from the same window, expect green
    assert out['status'] in ['green', 'amber']

if __name__ == '__main__':
    test_transaction_volume_check()
    print("✓ Transaction volume check passed")
    
    test_aov_consistency_check()
    print("✓ AOV consistency check passed")
    
    test_full_validation_engine()
    print("✓ Full validation engine passed")
    
    print("\nAll validation tests passed!")
