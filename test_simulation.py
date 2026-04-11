import unittest
import numpy as np

from state import GridEnvironmentState
from rules import SimpleDiscreteSpreadRule, ThermodynamicSpreadRule
from engine import WildfireSimulationEngine

class TestWildfireCellularAutomata(unittest.TestCase):
    
    def test_state_initialization(self):
        """Ensure grid variables initialize safely within their logical bounds."""
        state = GridEnvironmentState(10, 15)
        self.assertEqual(state.total_rows, 10)
        self.assertEqual(state.total_columns, 15)
        self.assertTrue(np.all((state.fuel_levels >= 0.0) & (state.fuel_levels <= 1.0)))
        self.assertFalse(np.any(state.is_actively_burning))

    def test_deep_copy_state(self):
        """Ensure states don't mutate each other unintentionally in synchronous updates."""
        state_one = GridEnvironmentState(5, 5)
        state_one.is_actively_burning[2, 2] = True
        
        state_two = state_one.duplicate_state()
        state_two.is_actively_burning[2, 2] = False
        
        self.assertTrue(state_one.is_actively_burning[2, 2], "State duplicate did not perform deep copy.")

    def test_simple_discrete_rule(self):
        """Test basic ignition spread logic mapping."""
        state = GridEnvironmentState(5, 5)
        state.fuel_levels = np.ones((5, 5))
        state.is_actively_burning[2, 2] = True
        
        rule = SimpleDiscreteSpreadRule()
        next_state = rule.calculate_next_state(state)
        
        # Immediate neighbors of [2, 2] should catch fire
        self.assertTrue(next_state.is_actively_burning[2, 3])
        self.assertTrue(next_state.is_actively_burning[1, 2])
        # Far-away cells should remain untouched
        self.assertFalse(next_state.is_actively_burning[0, 0])

    def test_thermodynamic_rule_moisture_barrier(self):
        """Ensure cells with very high moisture do not ignite immediately."""
        state = GridEnvironmentState(3, 3)
        state.fuel_levels = np.ones((3, 3))
        state.moisture_levels = np.ones((3, 3)) # Impenetrable moisture
        state.heat_levels = np.zeros((3, 3))
        
        # Ignite center cell
        state.is_actively_burning[1, 1] = True
        state.heat_levels[1, 1] = 1.0
        
        rule = ThermodynamicSpreadRule()
        next_state = rule.calculate_next_state(state)
        
        # High moisture should prevent immediate spread to neighbor
        self.assertFalse(next_state.is_actively_burning[0, 1])

    def test_engine_turn_advancement(self):
        """Tests that the simulation engine reliably increments timesteps."""
        rule = SimpleDiscreteSpreadRule()
        engine = WildfireSimulationEngine(5, 5, rule)
        self.assertEqual(engine.simulation_turn_count, 0)
        engine.advance_one_turn()
        self.assertEqual(engine.simulation_turn_count, 1)

if __name__ == "__main__":
    unittest.main()