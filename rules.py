from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import convolve

from state import GridEnvironmentState

class WildfireTransitionRule(ABC):
    """Abstract Base Class for CA transition rules (Strategy Pattern)."""
    
    @abstractmethod
    def calculate_next_state(self, current_state: GridEnvironmentState) -> GridEnvironmentState:
        """Computes the next evolutionary state of the grid."""
        pass


class ThermodynamicSpreadRule(WildfireTransitionRule):
    """Complex rule modeling heat diffusion, moisture evaporation, and fuel consumption."""
    
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

    def calculate_next_state(self, current_state: GridEnvironmentState) -> GridEnvironmentState:
        next_state = current_state.duplicate_state()

        # 1. Heat diffuses to neighbors and dissipates slightly into the atmosphere
        diffused_heat = convolve(current_state.heat_levels, self.heat_diffusion_kernel, mode='constant')
        diffused_heat *= self.cooling_retention_rate
        diffused_heat[current_state.is_actively_burning] += self.heat_generated_by_fire
        next_state.heat_levels = diffused_heat.clip(0.0, 1.0)

        # 2. Moisture evaporates based on local heat
        next_state.moisture_levels -= next_state.heat_levels * self.evaporation_rate
        next_state.moisture_levels = np.clip(next_state.moisture_levels, 0.0, 1.0)

        # 3. Ignite cells meeting threshold criteria
        ignition_vulnerable = (next_state.heat_levels > self.ignition_heat_threshold) & \
                              (next_state.moisture_levels <= 0.1) & \
                              (next_state.fuel_levels > 0.0) & \
                              (~current_state.is_actively_burning)
        next_state.is_actively_burning[ignition_vulnerable] = True

        # 4. Burn fuel and extinguish if empty
        next_state.fuel_levels[next_state.is_actively_burning] -= self.fuel_consumption_rate
        fuel_depleted = next_state.fuel_levels <= 0.0
        next_state.is_actively_burning[fuel_depleted] = False
        next_state.fuel_levels = np.clip(next_state.fuel_levels, 0.0, 1.0)

        return next_state


class SimpleDiscreteSpreadRule(WildfireTransitionRule):
    """Simple neighbor-counting rule demonstrating architectural modularity."""
    
    def __init__(self):
        self.neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    def calculate_next_state(self, current_state: GridEnvironmentState) -> GridEnvironmentState:
        next_state = current_state.duplicate_state()
        
        burning_neighbors = convolve(
            current_state.is_actively_burning.astype(int), 
            self.neighbor_kernel, 
            mode='constant'
        )
        
        vulnerable_to_fire = (burning_neighbors >= 1) & \
                             (next_state.fuel_levels > 0.0) & \
                             (~current_state.is_actively_burning)
        
        next_state.is_actively_burning[vulnerable_to_fire] = True
        next_state.fuel_levels[current_state.is_actively_burning] -= 0.15
        
        depleted_mask = next_state.fuel_levels <= 0.0
        next_state.is_actively_burning[depleted_mask] = False
        next_state.fuel_levels = np.clip(next_state.fuel_levels, 0.0, 1.0)

        return next_state