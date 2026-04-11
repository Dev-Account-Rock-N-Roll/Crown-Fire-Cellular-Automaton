from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import convolve

from layers import LayeredGridEnvironmentState

class WildfireTransitionRule(ABC):
    """Abstract Base Class for 3D CA transition rules (Strategy Pattern)."""
    
    @abstractmethod
    def calculate_next_state(self, current_state: LayeredGridEnvironmentState) -> LayeredGridEnvironmentState:
        """Computes the next evolutionary state of the layered 3D grid."""
        pass


class ThermodynamicSpreadRule(WildfireTransitionRule):
    """3D thermodynamic rule modeling intra-layer diffusion and inter-layer heat convection."""
    
    def __init__(self):
        self.heat_diffusion_kernel = np.array([
            [0.10, 0.20, 0.10],
            [0.20, 0.40, 0.20],
            [0.10, 0.20, 0.10]
        ])
        self.ignition_heat_threshold = 0.4    
        self.heat_generated_by_fire = 0.8     
        self.cooling_retention_rate = 0.95    
        self.evaporation_rate = 0.2           
        self.fuel_consumption_rate = 0.05
        
        # 3D Specific constants (fire heats upwards intensely, downwards weakly)
        self.upward_heat_transfer = 0.45 
        self.downward_heat_transfer = 0.05 

    def calculate_next_state(self, current_state: LayeredGridEnvironmentState) -> LayeredGridEnvironmentState:
        next_state = current_state.duplicate_state()
        num_layers = len(current_state.layers)

        layer_heat_updates = []

        # Step 1: Pre-calculate heat diffusion across all 3D layers safely
        for i in range(num_layers):
            layer = current_state.layers[f'layer_{i}']
            
            diffused_heat = convolve(layer.heat_levels, self.heat_diffusion_kernel, mode='constant')
            diffused_heat *= self.cooling_retention_rate
            diffused_heat[layer.is_actively_burning] += self.heat_generated_by_fire
            
            # Convection interactions with 3D neighbors
            if i > 0: 
                layer_below = current_state.layers[f'layer_{i-1}']
                diffused_heat += layer_below.heat_levels * self.upward_heat_transfer
            if i < num_layers - 1: 
                layer_above = current_state.layers[f'layer_{i+1}']
                diffused_heat += layer_above.heat_levels * self.downward_heat_transfer
                
            layer_heat_updates.append(diffused_heat)

        # Step 2: Apply thermodynamic phase transitions simultaneously
        for i in range(num_layers):
            curr_layer = current_state.layers[f'layer_{i}']
            next_layer = next_state.layers[f'layer_{i}']
            
            next_layer.heat_levels = layer_heat_updates[i]

            next_layer.moisture_levels -= next_layer.heat_levels * self.evaporation_rate
            next_layer.moisture_levels = np.clip(next_layer.moisture_levels, 0.0, 1.0)

            ignition_vulnerable = (next_layer.heat_levels > self.ignition_heat_threshold) & \
                                  (next_layer.moisture_levels <= 0.1) & \
                                  (next_layer.fuel_levels > 0.0) & \
                                  (~curr_layer.is_actively_burning)
            next_layer.is_actively_burning[ignition_vulnerable] = True

            next_layer.fuel_levels[next_layer.is_actively_burning] -= self.fuel_consumption_rate
            fuel_depleted = next_layer.fuel_levels <= 0.0
            next_layer.is_actively_burning[fuel_depleted] = False
            next_layer.fuel_levels = np.clip(next_layer.fuel_levels, 0.0, 1.0)

        return next_state


class SimpleDiscreteSpreadRule(WildfireTransitionRule):
    """3D discrete rule checking adjacent cellular neighbors across x, y, and z axes."""
    
    def __init__(self):
        self.neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    def calculate_next_state(self, current_state: LayeredGridEnvironmentState) -> LayeredGridEnvironmentState:
        next_state = current_state.duplicate_state()
        num_layers = len(current_state.layers)
        
        for i in range(num_layers):
            curr_layer = current_state.layers[f'layer_{i}']
            next_layer = next_state.layers[f'layer_{i}']
            
            burning_neighbors = convolve(
                curr_layer.is_actively_burning.astype(int), 
                self.neighbor_kernel, 
                mode='constant'
            )
            
            # Fire climbs up easily (weight 2), falls downwards poorly (weight 1)
            if i > 0: 
                burning_below = current_state.layers[f'layer_{i-1}'].is_actively_burning
                burning_neighbors += burning_below.astype(int) * 2 
            if i < num_layers - 1: 
                burning_above = current_state.layers[f'layer_{i+1}'].is_actively_burning
                burning_neighbors += burning_above.astype(int) * 1 
                
            vulnerable_to_fire = (burning_neighbors >= 1) & \
                                 (next_layer.fuel_levels > 0.0) & \
                                 (~curr_layer.is_actively_burning)
            
            next_layer.is_actively_burning[vulnerable_to_fire] = True
            next_layer.fuel_levels[curr_layer.is_actively_burning] -= 0.15
            
            depleted_mask = next_layer.fuel_levels <= 0.0
            next_layer.is_actively_burning[depleted_mask] = False
            next_layer.fuel_levels = np.clip(next_layer.fuel_levels, 0.0, 1.0)

        return next_state