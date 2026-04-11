import unittest
import numpy as np

from layers import LayeredGridEnvironmentState
from rules import SimpleDiscreteSpreadRule, ThermodynamicSpreadRule
from engine import WildfireSimulationEngine
from environments import PrairieBuilder, SwampBuilder, ForestBuilder

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
        builder = ForestBuilder()
        engine = WildfireSimulationEngine(5, 5, 3, rule, builder)
        self.assertEqual(engine.simulation_turn_count, 0)
        engine.advance_one_turn()
        self.assertEqual(engine.simulation_turn_count, 1)

class TestWildfireCellularAutomataEnvironments(unittest.TestCase):
    def test_prairie_generation_empty_air(self):
        """Validates that prairie upper layers are mostly empty air."""
        builder = PrairieBuilder()
        state = builder.build(10, 10, 3)
        
        air_cells = np.sum(state.layers['layer_1'].fuel_levels == 0.0)
        self.assertTrue(air_cells > 80, "Prairie upper layer was not mostly empty air.")

    def test_swamp_generation_high_moisture(self):
        """Validates swamp ground has uniquely high moisture preventing quick spread."""
        builder = SwampBuilder()
        state = builder.build(10, 10, 3)
        
        avg_moisture = np.mean(state.layers['layer_0'].moisture_levels)
        self.assertTrue(avg_moisture > 0.7, "Swamp bottom layer was not moist enough.")

    def test_falling_ember_physics(self):
        """Simulates an ember falling from the canopy through empty air to the ground."""
        state = LayeredGridEnvironmentState(3, 3, 3)
        
        # Ground has fuel
        state.layers['layer_0'].fuel_levels = np.ones((3, 3))
        # Layer 1 is empty air (fuel 0)
        state.layers['layer_1'].fuel_levels = np.zeros((3, 3))
        # Layer 2 (Canopy) has fuel and is burning at the center
        state.layers['layer_2'].fuel_levels = np.ones((3, 3))
        state.layers['layer_2'].is_actively_burning[1, 1] = True
        state.layers['layer_2'].heat_levels[1, 1] = 1.0
        
        rule = ThermodynamicSpreadRule()
        next_state = rule.calculate_next_state(state)
        
        # Ember drops through Layer 1 (empty) directly onto Layer 0, delivering high heat
        heat_ground = next_state.layers['layer_0'].heat_levels[1, 1]
        heat_air = next_state.layers['layer_1'].heat_levels[1, 1]
        
        self.assertTrue(heat_ground > 0.5, "Ember failed to drop through empty air to ignite ground.")
        self.assertTrue(heat_air < heat_ground, "Air caught an ember when it shouldn't have.")

    def test_engine_integration(self):
        engine = WildfireSimulationEngine(5, 5, 3, SimpleDiscreteSpreadRule(), ForestBuilder())
        self.assertEqual(engine.simulation_turn_count, 0)
        engine.advance_one_turn()
        self.assertEqual(engine.simulation_turn_count, 1)

if __name__ == "__main__":
    unittest.main()