from layers import LayeredGridEnvironmentState
from rules import WildfireTransitionRule

class WildfireSimulationEngine:
    """Manages 3D state transitions and physics rule assignments."""
    
    def __init__(self, rows: int, columns: int, num_layers: int, initial_rule: WildfireTransitionRule):
        self.total_rows = rows
        self.total_columns = columns
        self.num_layers = num_layers
        self.current_state = LayeredGridEnvironmentState(rows, columns, num_layers)
        self.active_transition_rule = initial_rule
        self.simulation_turn_count = 0

    def advance_one_turn(self):
        """Applies the current 3D rule to advance the multi-layered state."""
        self.current_state = self.active_transition_rule.calculate_next_state(self.current_state)
        self.simulation_turn_count += 1

    def reset_environment(self):
        """Re-initializes a fresh randomized 3D grid stack."""
        self.current_state = LayeredGridEnvironmentState(self.total_rows, self.total_columns, self.num_layers)
        self.simulation_turn_count = 0

    def swap_transition_rule(self, new_rule: WildfireTransitionRule):
        """Dynamically hot-swaps the active physics ruleset."""
        self.active_transition_rule = new_rule