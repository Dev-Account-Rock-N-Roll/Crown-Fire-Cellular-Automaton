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
        # FIX: Lowered moisture to realistic drought levels (5% to 12%)
        layer_0.moisture_levels = np.random.uniform(0.05, 0.12, (rows, cols))
        
        # FIX: Populate Alexandridis CA variables
        layer_0.p_veg = np.full((rows, cols), 0.4)    # 0.4 = Grasslands/Shrubs
        layer_0.p_den = np.full((rows, cols), -0.4)   # -0.4 = Sparse
        layer_0.veg_height = np.full((rows, cols), 0.5) 
        
        # Top layers: Empty air, except for rare trees (2% coverage)
        tree_mask = np.random.random((rows, cols)) < 0.02
        
        for i in range(1, num_layers):
            layer = state.layers[f'layer_{i}']
            layer.fuel_levels = np.zeros((rows, cols))
            layer.moisture_levels = np.zeros((rows, cols))
            
            # Default empty air attributes
            layer.p_veg = np.full((rows, cols), -1.0)
            layer.p_den = np.full((rows, cols), -0.4)
            layer.veg_height = np.zeros((rows, cols))
            
            if i == num_layers - 1:
                # Canopy
                layer.fuel_levels[tree_mask] = np.random.uniform(0.5, 0.9, size=int(np.sum(tree_mask)))
                layer.moisture_levels[tree_mask] = 0.10
                layer.p_veg[tree_mask] = 0.2          # 0.2 = Oak/Broadleaf
                layer.p_den[tree_mask] = 0.0          # Normal density
                layer.veg_height[tree_mask] = 5.0
            else:
                # Trunks
                layer.fuel_levels[tree_mask] = 1.0
                layer.moisture_levels[tree_mask] = 0.15
                layer.p_veg[tree_mask] = 0.2
                layer.p_den[tree_mask] = 0.0
                layer.veg_height[tree_mask] = 2.0
                
        return state


class SwampBuilder(EnvironmentBuilder):
    """Moist, hard-to-ignite ground with a sparse canopy above."""
    
    def build(self, rows: int, cols: int, num_layers: int) -> LayeredGridEnvironmentState:
        state = LayeredGridEnvironmentState(rows, cols, num_layers)
        
        # Layer 0: Extremely wet swamp ground
        layer_0 = state.layers['layer_0']
        layer_0.fuel_levels = np.random.uniform(0.3, 0.7, (rows, cols))
        # FIX: Swamps are wet, but 80% to 100% acts as an absolute mathematical firewall.
        # 25% to 40% will cause fires to sputter out naturally without breaking the math.
        layer_0.moisture_levels = np.random.uniform(0.25, 0.40, (rows, cols)) 
        
        layer_0.p_veg = np.full((rows, cols), 0.0)    # 0.0 = Riverine vegetation
        layer_0.p_den = np.full((rows, cols), 0.0)    # Normal density
        layer_0.veg_height = np.full((rows, cols), 1.0)
        
        # Top layers: Flammable canopy, but only covers ~40% of the map
        canopy_mask = np.random.random((rows, cols)) < 0.4
        
        for i in range(1, num_layers):
            layer = state.layers[f'layer_{i}']
            layer.fuel_levels = np.zeros((rows, cols))
            layer.moisture_levels = np.zeros((rows, cols))
            
            layer.p_veg = np.full((rows, cols), -1.0)
            layer.p_den = np.full((rows, cols), -0.4)
            layer.veg_height = np.zeros((rows, cols))
            
            if i == num_layers - 1:
                layer.fuel_levels[canopy_mask] = np.random.uniform(0.6, 1.0, size=int(np.sum(canopy_mask)))
                layer.moisture_levels[canopy_mask] = 0.15
                layer.p_veg[canopy_mask] = 0.3        # 0.3 = Juniper
                layer.p_den[canopy_mask] = -0.4       # Sparse
                layer.veg_height[canopy_mask] = 6.0
            else:
                layer.fuel_levels[canopy_mask] = 0.9
                layer.moisture_levels[canopy_mask] = 0.20
                layer.p_veg[canopy_mask] = 0.3
                layer.p_den[canopy_mask] = -0.4
                layer.veg_height[canopy_mask] = 3.0
                
        return state


class ForestBuilder(EnvironmentBuilder):
    """Dense canopy above, varying undergrowth, connected by hard-to-ignite trunks."""
    
    def build(self, rows: int, cols: int, num_layers: int) -> LayeredGridEnvironmentState:
        state = LayeredGridEnvironmentState(rows, cols, num_layers)
        
        # Layer 0: Patchy undergrowth
        layer_0 = state.layers['layer_0']
        undergrowth_mask = np.random.random((rows, cols)) < 0.7
        layer_0.fuel_levels = np.where(undergrowth_mask, np.random.uniform(0.3, 0.7, (rows, cols)), 0.0)
        
        # FIX: Highly flammable dry brush (5% to 15% moisture)
        layer_0.moisture_levels = np.random.uniform(0.05, 0.15, (rows, cols))
        
        layer_0.p_veg = np.where(undergrowth_mask, 0.4, -1.0) # 0.4 = Pine needles / Fir
        layer_0.p_den = np.where(undergrowth_mask, 0.0, -0.4) # Normal Density
        layer_0.veg_height = np.where(undergrowth_mask, 1.0, 0.0)
        
        # Top Layers: Thick canopy (80% coverage) and dense trunks
        canopy_mask = np.random.random((rows, cols)) < 0.8
        
        for i in range(1, num_layers):
            layer = state.layers[f'layer_{i}']
            layer.fuel_levels = np.zeros((rows, cols))
            layer.moisture_levels = np.zeros((rows, cols))
            
            layer.p_veg = np.full((rows, cols), -1.0)
            layer.p_den = np.full((rows, cols), -0.4)
            layer.veg_height = np.zeros((rows, cols))
            
            if i == num_layers - 1:
                layer.fuel_levels[canopy_mask] = np.random.uniform(0.7, 1.0, size=int(np.sum(canopy_mask)))
                layer.moisture_levels[canopy_mask] = 0.08     # Dry canopy catches instantly
                layer.p_veg[canopy_mask] = 0.4                # Aleppo Pine
                layer.p_den[canopy_mask] = 0.3                # Dense Canopy
                layer.veg_height[canopy_mask] = 10.0
            else:
                # Tree trunks: High fuel density, hold slightly more water
                layer.fuel_levels[canopy_mask] = 1.0
                layer.moisture_levels[canopy_mask] = 0.20 
                layer.p_veg[canopy_mask] = 0.2
                layer.p_den[canopy_mask] = 0.0
                layer.veg_height[canopy_mask] = 5.0
                
        return state