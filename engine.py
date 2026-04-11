from state import GridEnvironmentState
from rules import WildfireTransitionRule

class WildfireSimulationEngine:
    """Manages state transitions and rule assignments."""
    
    def __init__(self, rows: int, columns: int, initial_rule: WildfireTransitionRule):
        self.total_rows = rows
        self.total_columns = columns
        self.current_state = GridEnvironmentState(rows, columns)
        self.active_transition_rule = initial_rule
        self.simulation_turn_count = 0

    def advance_one_turn(self):
        """Applies the current rule to advance the state."""
        self.current_state = self.active_transition_rule.calculate_next_state(self.current_state)
        self.simulation_turn_count += 1

    def reset_environment(self):
        """Re-initializes a fresh randomized grid."""
        self.current_state = GridEnvironmentState(self.total_rows, self.total_columns)
        self.simulation_turn_count = 0

    def swap_transition_rule(self, new_rule: WildfireTransitionRule):
        """Dynamically hot-swaps the active physics ruleset."""
        self.active_transition_rule = new_rule