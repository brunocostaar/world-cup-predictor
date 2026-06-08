import unittest
import os
import sys
import json

# Add directory to sys path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from model.predict import get_match_lambda, predict_match, simulate_group_stage, simulate_knockout_stage

class TestPredictions(unittest.TestCase):
    
    def setUp(self):
        # Load groups config
        with open(config.GROUPS_JSON_PATH, 'r') as f:
            self.groups = json.load(f)

    def test_lambdas(self):
        """Tests that expected goals (lambdas) are calculated and positive."""
        l1, l2 = get_match_lambda("Brazil", "Argentina")
        self.assertGreater(l1, 0)
        self.assertGreater(l2, 0)
        
        # Test strength difference effects: France vs Haiti
        l_france, l_haiti = get_match_lambda("France", "Haiti")
        self.assertGreater(l_france, l_haiti, "France should have a higher goal expectation than Haiti.")

    def test_predict_match_outcomes(self):
        """Tests that match prediction yields valid goals and winners."""
        # Test non-knockout
        goals_A, goals_B, winner = predict_match("Brazil", "Argentina", is_knockout=False)
        self.assertGreaterEqual(goals_A, 0)
        self.assertGreaterEqual(goals_B, 0)
        if goals_A > goals_B:
            self.assertEqual(winner, "Brazil")
        elif goals_B > goals_A:
            self.assertEqual(winner, "Argentina")
        else:
            self.assertIsNone(winner)
            
        # Test knockout match shootout (never returns None for winner)
        _, _, winner_ko = predict_match("Brazil", "Argentina", is_knockout=True)
        self.assertIsNotNone(winner_ko)
        self.assertTrue(winner_ko in ["Brazil", "Argentina"])

    def test_tournament_stages(self):
        """Tests that the group and knockout stage simulations return valid advance lists."""
        # Simulate Group stage
        winners, runners, thirds = simulate_group_stage(self.groups)
        
        # Checking counts: 12 winners, 12 runners, 8 third-place teams
        self.assertEqual(len(winners), 12)
        self.assertEqual(len(runners), 12)
        self.assertEqual(len(thirds), 8)
        
        # Simulate Knockout stage
        stages = simulate_knockout_stage(winners, runners, thirds)
        
        # Verify stages map has champions and runner-ups
        teams_reached_stage = list(stages.values())
        self.assertIn("Champion", teams_reached_stage)
        self.assertIn("Runner-Up", teams_reached_stage)
        
        # Verify counts in knockout stages
        self.assertEqual(teams_reached_stage.count("Champion"), 1)
        self.assertEqual(teams_reached_stage.count("Runner-Up"), 1)
        self.assertEqual(teams_reached_stage.count("Semi-Finals"), 2)
        self.assertEqual(teams_reached_stage.count("Quarter-Finals"), 4)
        self.assertEqual(teams_reached_stage.count("Round of 16"), 8)
        self.assertEqual(teams_reached_stage.count("Round of 32"), 16)

if __name__ == "__main__":
    unittest.main()
