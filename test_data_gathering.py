import unittest
import os
import pandas as pd
import sys

# Add directory to sys path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from fetchers.transfermarkt import parse_value_to_millions
from processing.prepare_dataset import normalize_team_name, prepare_dataset

class TestDataGathering(unittest.TestCase):
    
    def test_value_parser(self):
        """Tests that Transfermarkt market value strings are parsed correctly into float Millions."""
        self.assertEqual(parse_value_to_millions("€1.55bn"), 1550.0)
        self.assertEqual(parse_value_to_millions("€982.00m"), 982.0)
        self.assertEqual(parse_value_to_millions("€25.00m"), 25.0)
        self.assertEqual(parse_value_to_millions("€500k"), 0.5)
        self.assertEqual(parse_value_to_millions("-"), 0.0)
        self.assertEqual(parse_value_to_millions(""), 0.0)
        self.assertEqual(parse_value_to_millions(None), 0.0)

    def test_name_normalization(self):
        """Tests that team names from different sources are normalized to their standard format."""
        self.assertEqual(normalize_team_name("Korea Republic"), "South Korea")
        self.assertEqual(normalize_team_name("Cote d'Ivoire"), "Côte d'Ivoire")
        self.assertEqual(normalize_team_name("US"), "United States")
        self.assertEqual(normalize_team_name("USA"), "United States")
        self.assertEqual(normalize_team_name("Turkey"), "Türkiye")
        self.assertEqual(normalize_team_name("Brazil"), "Brazil")

    def test_dataset_preparation(self):
        """Tests the complete dataset preparation pipeline output."""
        df = prepare_dataset()
        
        # Verify shape
        self.assertEqual(len(df), 48, "Dataset must contain exactly 48 qualified teams.")
        
        # Verify columns exist
        expected_columns = [
            "team", "group", "fifa_rank", "fifa_points", 
            "squad_value_m", "recent_results", "recent_form_index"
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Column '{col}' must be present in the prepared dataset.")
            
        # Verify no NaN values in critical features
        self.assertFalse(df['fifa_rank'].isna().any(), "FIFA Rank cannot contain NaN.")
        self.assertFalse(df['squad_value_m'].isna().any(), "Squad Value cannot contain NaN.")
        self.assertFalse(df['recent_form_index'].isna().any(), "Recent Form Index cannot contain NaN.")

if __name__ == "__main__":
    unittest.main()
