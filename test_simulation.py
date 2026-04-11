import unittest
import numpy as np

from layers import LayeredGridEnvironmentState
from rules import SimpleDiscreteSpreadRule, ThermodynamicSpreadRule
from engine import WildfireSimulationEngine

class TestWildfireCellularAutomata3D(unittest.TestCase):
    
    def test_layered_state_initialization(self):
        """Ensures grid initializes variables safely inside multi-layer bounds."""
        state = LayeredGridEnvironmentState(10, 15, 3)
        self.assertEqual(len(state.layers), 3)
        
        layer = state.layers['layer_0']
        self.assertEqual(layer.total_rows, 10)
        self.assertEqual(layer.total_columns, 15)
        self.assertTrue(np.all((layer.fuel_levels >= 0.0) & (layer.fuel_levels <= 1.0)))
        self.assertFalse(np.any(layer.is_actively_burning))

    def test_deep_copy_state(self):
        """Ensures complete recursive deep copying works without cross-mutation."""
        state_one = LayeredGridEnvironmentState(5, 5, 2)
        state_one.layers['layer_1'].is_actively_burning[2, 2] = True
        
        state_two = state_one.duplicate_state()
        state_two.layers['layer_1'].is_actively_burning[2, 2] = False
        
        self.assertTrue(state_one.layers['layer_1'].is_actively_burning[2, 2])

    def test_3d_simple_discrete_rule_upward_bias(self):
        """Test vertical spread logic maps upward more intensely than downward."""
        state = LayeredGridEnvironmentState(5, 5, 2)
        state.layers['layer_0'].fuel_levels = np.ones((5, 5))
        state.layers['layer_1'].fuel_levels = np.ones((5, 5))
        
        # Ignite center of the ground layer
        state.layers['layer_0'].is_actively_burning[2, 2] = True
        
        rule = SimpleDiscreteSpreadRule()
        next_state = rule.calculate_next_state(state)
        
        # Due to strong upward bias, the layer directly above should ignite instantly
        self.assertTrue(next_state.layers['layer_1'].is_actively_burning[2, 2])

    def test_thermodynamic_rule_3d_heat_transfer(self):
        """Test 3D thermodynamic gradient convection logic."""
        state = LayeredGridEnvironmentState(3, 3, 2)
        
        # Add high moisture barrier to prevent ignition, purely testing heat
        state.layers['layer_1'].moisture_levels = np.ones((3, 3))
        state.layers['layer_1'].fuel_levels = np.ones((3, 3))
        
        state.layers['layer_0'].fuel_levels = np.ones((3, 3))
        state.layers['layer_0'].is_actively_burning[1, 1] = True
        state.layers['layer_0'].heat_levels[1, 1] = 1.0
        
        rule = ThermodynamicSpreadRule()
        next_state = rule.calculate_next_state(state)
        
        heat_layer_1 = next_state.layers['layer_1'].heat_levels[1, 1]
        self.assertTrue(heat_layer_1 > 0.0, "Thermodynamic heat did not properly transfer upwards.")

    def test_engine_turn_advancement(self):
        rule = SimpleDiscreteSpreadRule()
        engine = WildfireSimulationEngine(5, 5, 3, rule)
        self.assertEqual(engine.simulation_turn_count, 0)
        engine.advance_one_turn()
        self.assertEqual(engine.simulation_turn_count, 1)

if __name__ == "__main__":
    unittest.main()