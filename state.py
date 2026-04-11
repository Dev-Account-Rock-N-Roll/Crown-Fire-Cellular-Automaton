import numpy as np

class GridEnvironmentState:
    """Represents the continuous and discrete states of the wildfire grid."""
    
    def __init__(self, total_rows: int, total_columns: int):
        self.total_rows = total_rows
        self.total_columns = total_columns
        
        # Fuel starts randomly distributed between 0.5 and 1.0
        self.fuel_levels = np.random.uniform(0.5, 1.0, (total_rows, total_columns))
        
        # Introduce bare spots (e.g., rocks or water bodies)
        bare_ground_mask = np.random.random((total_rows, total_columns)) < 0.05
        self.fuel_levels[bare_ground_mask] = 0.0
        
        # Heat and moisture distributions
        self.heat_levels = np.zeros((total_rows, total_columns))
        self.moisture_levels = np.random.uniform(0.2, 0.6, (total_rows, total_columns))
        self.is_actively_burning = np.zeros((total_rows, total_columns), dtype=bool)

    def duplicate_state(self) -> 'GridEnvironmentState':
        """Deep copy for synchronous Cellular Automata generation updates."""
        copied_state = GridEnvironmentState(self.total_rows, self.total_columns)
        copied_state.fuel_levels = np.copy(self.fuel_levels)
        copied_state.heat_levels = np.copy(self.heat_levels)
        copied_state.moisture_levels = np.copy(self.moisture_levels)
        copied_state.is_actively_burning = np.copy(self.is_actively_burning)
        return copied_state