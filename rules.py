from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import convolve

from layers import LayeredGridEnvironmentState

class WildfireTransitionRule(ABC):
    @abstractmethod
    def calculate_next_state(self, current_state: LayeredGridEnvironmentState) -> LayeredGridEnvironmentState:
        pass


class ThermodynamicSpreadRule(WildfireTransitionRule):
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
        
        self.upward_heat_transfer = 0.45 
        self.downward_heat_transfer = 0.05 
        self.falling_ember_heat = 1.5 

    def calculate_next_state(self, current_state: LayeredGridEnvironmentState) -> LayeredGridEnvironmentState:
        next_state = current_state.duplicate_state()
        num_layers = len(current_state.layers)
        layer_heat_updates = []

        # Step 1: Pre-calculate local heat diffusion and standard rising convection
        for i in range(num_layers):
            layer = current_state.layers[f'layer_{i}']
            diffused_heat = convolve(layer.heat_levels, self.heat_diffusion_kernel, mode='constant')
            diffused_heat *= self.cooling_retention_rate
            diffused_heat[layer.is_actively_burning] += self.heat_generated_by_fire
            
            if i > 0: 
                layer_below = current_state.layers[f'layer_{i-1}']
                diffused_heat += layer_below.heat_levels * self.upward_heat_transfer
            if i < num_layers - 1: 
                layer_above = current_state.layers[f'layer_{i+1}']
                diffused_heat += layer_above.heat_levels * self.downward_heat_transfer
                
            layer_heat_updates.append(diffused_heat)

        # Step 2: Simulate Gravity & Sparks. Embers drop and bypass empty space.
        falling_embers = np.zeros((current_state.total_rows, current_state.total_columns))
        for i in range(num_layers - 1, -1, -1):
            curr_layer = current_state.layers[f'layer_{i}']
            
            # Embers hit the layer, applying intense heat downward
            layer_heat_updates[i] += falling_embers * self.falling_ember_heat
            
            # Embers are 'caught' if there is enough fuel to physically hit
            caught_mask = curr_layer.fuel_levels > 0.1
            falling_embers[caught_mask] = 0.0
            
            # New embers spark off actively burning cells
            falling_embers += curr_layer.is_actively_burning.astype(float)

        # Step 3: Apply thermodynamic phase transitions simultaneously
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
    def __init__(self):
        self.neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    def calculate_next_state(self, current_state: LayeredGridEnvironmentState) -> LayeredGridEnvironmentState:
        next_state = current_state.duplicate_state()
        num_layers = len(current_state.layers)
        
        # Track falling sparks cascading down the layers
        falling_sparks = np.zeros((current_state.total_rows, current_state.total_columns), dtype=int)
        
        for i in range(num_layers - 1, -1, -1):
            curr_layer = current_state.layers[f'layer_{i}']
            next_layer = next_state.layers[f'layer_{i}']
            
            burning_neighbors = convolve(curr_layer.is_actively_burning.astype(int), self.neighbor_kernel, mode='constant')
            
            if i > 0: 
                burning_below = current_state.layers[f'layer_{i-1}'].is_actively_burning
                burning_neighbors += burning_below.astype(int) * 2 
            
            # Massive infectious multiplier for falling sparks
            burning_neighbors += falling_sparks * 4 
                
            vulnerable_to_fire = (burning_neighbors >= 1) & \
                                 (next_layer.fuel_levels > 0.0) & \
                                 (~curr_layer.is_actively_burning)
            
            next_layer.is_actively_burning[vulnerable_to_fire] = True
            next_layer.fuel_levels[curr_layer.is_actively_burning] -= 0.15
            
            depleted_mask = next_layer.fuel_levels <= 0.0
            next_layer.is_actively_burning[depleted_mask] = False
            next_layer.fuel_levels = np.clip(next_layer.fuel_levels, 0.0, 1.0)

            # Resolve sparks passing through air vs hitting objects
            caught_mask = curr_layer.fuel_levels > 0.0
            falling_sparks[caught_mask] = 0
            falling_sparks += curr_layer.is_actively_burning.astype(int)

        return next_state