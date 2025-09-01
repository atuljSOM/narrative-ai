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

if __name__ == '__main__':
    test_transaction_volume_check()
    print("✓ Transaction volume check passed")
    
    test_aov_consistency_check()
    print("✓ AOV consistency check passed")
    
    test_full_validation_engine()
    print("✓ Full validation engine passed")
    
    print("\nAll validation tests passed!")