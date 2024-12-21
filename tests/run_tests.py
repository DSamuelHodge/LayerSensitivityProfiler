import unittest
import sys
import os

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from tests.test_mps import TestMPS
from tests.test_tdvp import TestTDVP
from tests.test_layer_sensitivity_profiler import TestLayerSensitivityProfiler

def run_tests():
    """Run all unit tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMPS))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTDVP))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLayerSensitivityProfiler))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return 0 if all tests passed, 1 otherwise
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests())
