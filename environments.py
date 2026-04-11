import numpy as np
from abc import ABC, abstractmethod

from layers import LayeredGridEnvironmentState

class EnvironmentBuilder(ABC):
    """Abstract Factory for generating specialized grid environments."""
    
    @abstractmethod
    def build(self, rows: int, cols: int, num_layers: int) -> LayeredGridEnvironmentState:
        """Constructs a layered grid representing a specific biome."""
        pass


class PrairieBuilder(EnvironmentBuilder):
    """Dry scrublands with sparse, isolated trees."""
    
    def build(self, rows: int, cols: int, num_layers: int) -> LayeredGridEnvironmentState:
        state = LayeredGridEnvironmentState(rows, cols, num_layers)
        
        # Layer 0: Flammable dry scrub
        layer_0 = state.layers['layer_0']
        layer_0.fuel_levels = np.random.uniform(0.4, 0.8, (rows, cols))
        layer_0.moisture_levels = np.random.uniform(0.1, 0.3, (rows, cols))
        
        # Top layers: Empty air, except for rare trees (2% coverage)
        tree_mask = np.random.random((rows, cols)) < 0.02
        
        for i in range(1, num_layers):
            layer = state.layers[f'layer_{i}']
            layer.fuel_levels = np.zeros((rows, cols))
            layer.moisture_levels = np.zeros((rows, cols))
            
            if i == num_layers - 1:
                # Canopy
                layer.fuel_levels[tree_mask] = np.random.uniform(0.5, 0.9, np.sum(tree_mask))
                layer.moisture_levels[tree_mask] = 0.3
            else:
                # Trunks
                layer.fuel_levels[tree_mask] = 1.0
                layer.moisture_levels[tree_mask] = 0.5
                
        return state


class SwampBuilder(EnvironmentBuilder):
    """Moist, hard-to-ignite ground with a sparse canopy above."""
    
    def build(self, rows: int, cols: int, num_layers: int) -> LayeredGridEnvironmentState:
        state = LayeredGridEnvironmentState(rows, cols, num_layers)
        
        # Layer 0: Extremely wet swamp ground
        layer_0 = state.layers['layer_0']
        layer_0.fuel_levels = np.random.uniform(0.3, 0.7, (rows, cols))
        layer_0.moisture_levels = np.random.uniform(0.8, 1.0, (rows, cols)) 
        
        # Top layers: Flammable canopy, but only covers ~40% of the map
        canopy_mask = np.random.random((rows, cols)) < 0.4
        
        for i in range(1, num_layers):
            layer = state.layers[f'layer_{i}']
            layer.fuel_levels = np.zeros((rows, cols))
            layer.moisture_levels = np.zeros((rows, cols))
            
            if i == num_layers - 1:
                layer.fuel_levels[canopy_mask] = np.random.uniform(0.6, 1.0, np.sum(canopy_mask))
                layer.moisture_levels[canopy_mask] = 0.4
            else:
                layer.fuel_levels[canopy_mask] = 0.9
                layer.moisture_levels[canopy_mask] = 0.8 
                
        return state


class ForestBuilder(EnvironmentBuilder):
    """Dense canopy above, varying undergrowth, connected by hard-to-ignite trunks."""
    
    def build(self, rows: int, cols: int, num_layers: int) -> LayeredGridEnvironmentState:
        state = LayeredGridEnvironmentState(rows, cols, num_layers)
        
        # Layer 0: Patchy undergrowth
        layer_0 = state.layers['layer_0']
        undergrowth_mask = np.random.random((rows, cols)) < 0.7
        layer_0.fuel_levels = np.where(undergrowth_mask, np.random.uniform(0.3, 0.7, (rows, cols)), 0.0)
        layer_0.moisture_levels = np.random.uniform(0.3, 0.5, (rows, cols))
        
        # Top Layers: Thick canopy (80% coverage) and dense trunks
        canopy_mask = np.random.random((rows, cols)) < 0.8
        
        for i in range(1, num_layers):
            layer = state.layers[f'layer_{i}']
            layer.fuel_levels = np.zeros((rows, cols))
            layer.moisture_levels = np.zeros((rows, cols))
            
            if i == num_layers - 1:
                layer.fuel_levels[canopy_mask] = np.random.uniform(0.7, 1.0, np.sum(canopy_mask))
                layer.moisture_levels[canopy_mask] = 0.3
            else:
                # Tree trunks: High fuel density, but very high moisture makes upward spread difficult
                layer.fuel_levels[canopy_mask] = 1.0
                layer.moisture_levels[canopy_mask] = 0.7 
                
        return state