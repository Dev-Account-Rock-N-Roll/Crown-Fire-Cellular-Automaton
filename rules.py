from abc import ABC, abstractmethod
import numpy as np
import math
from scipy.ndimage import convolve

from layers import LayeredGridEnvironmentState

class WildfireTransitionRule(ABC):
    @abstractmethod
    def calculate_next_state(self, current_state: LayeredGridEnvironmentState) -> LayeredGridEnvironmentState:
        pass
# ---------------------------------------------------------
# Alexandridis et al. 2011 CA Model Implementation
# ---------------------------------------------------------
class AlexandridisWildfireRule(WildfireTransitionRule):
    """
    Wildfire spread modeling based on Alexandridis et al. (2011).
    Uses a stochastic cellular automaton to predict fire front evolution.
    
    Includes thermodynamic factors:
    - Wind field and velocity
    - Ground slope and elevation
    - Vegetation type, height, and density
    - Fuel moisture
    - Spotting phenomena (flying pop-corn-like embers)
    """

    def __init__(self, cell_resolution: float = 5.0):
        # Physical size of each grid cell side (in meters)
        self.cell_resolution = cell_resolution
        
        # Operational parameters configured per Table 4 from the paper
        self.base_spread_prob = 0.60       # P_0
        self.slope_coefficient = 0.063     # a_s
        self.moisture_coeff_a = 3.258      # a
        self.moisture_coeff_b = 0.111      # b
        self.height_power_law = 0.932      # d
        self.wind_coeff_1 = 0.045          # c_1
        self.wind_coeff_2 = 0.191          # c_2
        
        # Spotting effect parameters (Rule 5)
        self.spotting_lambda = 0.2         # Average pinecones emitted per burning cell
        self.spotting_base_prob = 0.1      # Base probability that a landed spot fire ignites
        self.spotting_mean_thrust = 15.0   # Base metric thrust scalar for the log-normal throw
        
        # Surrounding Moore neighborhood directions: (row_delta, col_delta)
        self.neighborhood_directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

    @staticmethod
    def _shift_grid(matrix: np.ndarray, row_shift: int, col_shift: int, fill_value: float = 0.0) -> np.ndarray:
        """
        Shifts a 2D matrix by a given row and column delta.
        Fills the undefined boundary areas cleanly with `fill_value` without wrapping around.
        """
        rows, cols = matrix.shape
        shifted_matrix = np.full((rows, cols), fill_value, dtype=matrix.dtype)
        
        src_row_start = max(0, row_shift)
        src_row_end = rows if row_shift >= 0 else rows + row_shift
        src_col_start = max(0, col_shift)
        src_col_end = cols if col_shift >= 0 else cols + col_shift
        
        dst_row_start = max(0, -row_shift)
        dst_row_end = rows if row_shift <= 0 else rows - row_shift
        dst_col_start = max(0, -col_shift)
        dst_col_end = cols if col_shift <= 0 else cols - col_shift
        
        shifted_matrix[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = \
            matrix[src_row_start:src_row_end, src_col_start:src_col_end]
            
        return shifted_matrix

    def calculate_next_state(self, current_state: LayeredGridEnvironmentState) -> LayeredGridEnvironmentState:
        next_state = current_state.duplicate_state()
        wind_velocity = current_state.wind_speed
        wind_angle_rad = math.radians(current_state.wind_direction)
        
        for layer_key, current_layer in current_state.layers.items():
            next_layer = next_state.layers[layer_key]
            rows, cols = current_state.total_rows, current_state.total_columns
            
            # Retrieve specialized topological properties (fallback to defaults if standard layer)
            elevation = getattr(current_layer, 'elevation', np.zeros((rows, cols)))
            veg_type_prob = getattr(current_layer, 'p_veg', np.zeros((rows, cols)))
            veg_density_prob = getattr(current_layer, 'p_den', np.zeros((rows, cols)))
            veg_height = getattr(current_layer, 'veg_height', np.ones((rows, cols)))
            
            # Equation 3: Moisture Effect (Cm is percentage 0 to 100)
            moisture_percentage = current_layer.moisture_levels * 100.0
            prob_moisture = self.moisture_coeff_a * np.exp(-self.moisture_coeff_b * moisture_percentage)
            
            # Equation 4: Vegetation Height Effect
            prob_height = veg_height ** self.height_power_law
            
            # Tracking the probability that a vulnerable cell avoids ignition from ALL its neighbors
            prob_avoids_ignition = np.ones((rows, cols))
            
            # Assess spread from every direction in the Moore neighborhood
            for row_delta, col_delta in self.neighborhood_directions:
                # Euclidean distance between cell centers
                travel_distance = self.cell_resolution * (math.sqrt(2) if row_delta != 0 and col_delta != 0 else 1.0)
                
                # Cartesian vector mapped from grid shifts (row increases downwards, thus -row_delta)
                dx, dy = col_delta, -row_delta
                propagation_angle = math.atan2(dy, dx)
                
                # Equation 5: Wind Effect
                theta_wind = wind_angle_rad - propagation_angle
                wind_directional_term = math.cos(theta_wind) - 1
                prob_wind = math.exp(self.wind_coeff_1 * wind_velocity) * \
                            math.exp(wind_velocity * self.wind_coeff_2 * wind_directional_term)
                
                # Identify which neighbors in this specific direction are actively burning
                burning_neighbor_mask = self._shift_grid(
                    current_layer.is_actively_burning, 
                    row_delta, col_delta, 
                    fill_value=False
                )
                
                # Equation 6 & 8: Topography Slope Effect
                neighbor_elevation = self._shift_grid(elevation, row_delta, col_delta, fill_value=0.0)
                slope_angle_rad = np.arctan((elevation - neighbor_elevation) / travel_distance)
                prob_slope = np.exp(self.slope_coefficient * np.degrees(slope_angle_rad))
                
                # Equation 1: Comprehensive Probability of local fire transmission
                prob_burn = self.base_spread_prob * (1 + veg_type_prob) * (1 + veg_density_prob) * \
                            prob_wind * prob_slope * prob_moisture * prob_height
                
                prob_burn = np.clip(prob_burn, 0.0, 1.0)
                
                # Accumulate the survival sequence: Cell must survive exposure from this specific neighbor
                prob_avoids_ignition *= (1.0 - (prob_burn * burning_neighbor_mask.astype(float)))
            
            # Calculate aggregate transmission and resolve probabilistic ignitions
            prob_ignition = 1.0 - prob_avoids_ignition
            random_draws = np.random.random((rows, cols))
            
            # Rule 1 & 2: Empty cells cannot ignite. Vulnerable cells catch fire based on aggregated probability
            vulnerable_cells = (~current_layer.is_actively_burning) & (current_layer.fuel_levels > 0.0)
            ignitions = vulnerable_cells & (random_draws < prob_ignition)
            
            # Rule 3 & 4: Burning cells consume all fuel in one discrete timestep and die out
            next_layer.fuel_levels[current_layer.is_actively_burning] = 0.0
            next_layer.is_actively_burning[current_layer.is_actively_burning] = False
            
            next_layer.is_actively_burning[ignitions] = True

            # Rule 5: Spotting Mechanism Implementation (Pinecone Embers)
            self._apply_spotting_mechanics(current_layer, next_layer, wind_velocity, wind_angle_rad)
            
        return next_state

    def _apply_spotting_mechanics(self, current_layer, next_layer, wind_velocity: float, wind_angle_rad: float):
        """
        Calculates and applies the long-distance spotting jumps simulating flying pinecones.
        """
        burning_indices = np.argwhere(current_layer.is_actively_burning)
        if len(burning_indices) == 0 or self.spotting_lambda <= 0:
            return
            
        pinecone_counts = np.random.poisson(self.spotting_lambda, len(burning_indices))
        
        for (row_idx, col_idx), emission_count in zip(burning_indices, pinecone_counts):
            for _ in range(emission_count):
                # Pinecone ejects in a uniform random angle relative to the primary wind vector
                relative_eject_angle = np.random.uniform(0, 2 * math.pi)
                absolute_flight_angle = wind_angle_rad + relative_eject_angle
                
                # Equation 9: Distance traveled by the pinecone based on wind carrying it
                base_thrust = abs(np.random.normal(loc=self.spotting_mean_thrust, scale=5.0))
                wind_assistance = math.exp(wind_velocity * self.wind_coeff_2 * (math.cos(relative_eject_angle) - 1))
                travel_distance = base_thrust * wind_assistance
                
                # Calculate landing coordinate indices (scaled to cells)
                dx = (travel_distance * math.cos(absolute_flight_angle)) / self.cell_resolution
                dy = (travel_distance * math.sin(absolute_flight_angle)) / self.cell_resolution
                
                target_row = int(row_idx - dy) 
                target_col = int(col_idx + dx)
                
                if 0 <= target_row < current_layer.is_actively_burning.shape[0] and \
                   0 <= target_col < current_layer.is_actively_burning.shape[1]:
                    
                    # Ensure destination has fuel and isn't already burning
                    if next_layer.fuel_levels[target_row, target_col] > 0 and not current_layer.is_actively_burning[target_row, target_col]:
                        # Check stochastic success of the brand establishing a firebed
                        if np.random.random() < self.spotting_base_prob:
                            next_layer.is_actively_burning[target_row, target_col] = True



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

        self.neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) 

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
            
            # Add wind effect to heat diffusion
            if current_state.wind_speed > 0:
                wind_rad = math.radians(current_state.wind_direction)
                wind_x = current_state.wind_speed * math.cos(wind_rad)
                wind_y = current_state.wind_speed * math.sin(wind_rad)
                shift_x = int(round(wind_x))
                shift_y = int(round(wind_y))
                if shift_x != 0 or shift_y != 0:
                    shifted_heat = np.roll(np.roll(layer.heat_levels, shift_x, axis=1), shift_y, axis=0)
                    diffused_heat += shifted_heat * 0.1 * current_state.wind_speed
            
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
            
            # Embers are 'caught' if there is enough fuel to physically hit
            caught_mask = curr_layer.fuel_levels > 0.1
            
            # Embers hit the layer, applying intense heat downward only if caught
            layer_heat_updates[i] += falling_embers * self.falling_ember_heat * caught_mask.astype(float)
            
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
            
            # Add probabilistic ignition based on burning neighbors
            burning_neighbors = convolve(curr_layer.is_actively_burning.astype(int), self.neighbor_kernel, mode='constant')
            # Include vertical neighbors
            if i > 0:
                burning_neighbors += current_state.layers[f'layer_{i-1}'].is_actively_burning.astype(int)
            if i < num_layers - 1:
                burning_neighbors += current_state.layers[f'layer_{i+1}'].is_actively_burning.astype(int)
            prob = np.clip(burning_neighbors * 0.08, 0, 0.8)
            random_draw = np.random.random((current_state.total_rows, current_state.total_columns))
            ignition_vulnerable &= (random_draw < prob)
            
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