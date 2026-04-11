from layers import LayeredGridEnvironmentState
from rules import WildfireTransitionRule
from environments import EnvironmentBuilder

class WildfireSimulationEngine:
    def __init__(self, rows: int, columns: int, num_layers: int, initial_rule: WildfireTransitionRule, initial_builder: EnvironmentBuilder):
        self.total_rows = rows
        self.total_columns = columns
        self.num_layers = num_layers
        self.env_builder = initial_builder
        self.active_transition_rule = initial_rule
        self.simulation_turn_count = 0
        
        self.reset_environment()

    def advance_one_turn(self):
        self.current_state = self.active_transition_rule.calculate_next_state(self.current_state)
        self.simulation_turn_count += 1

    def reset_environment(self):
        """Generates a fresh state using the current environmental builder."""
        self.current_state = self.env_builder.build(self.total_rows, self.total_columns, self.num_layers)
        self.simulation_turn_count = 0

    def swap_transition_rule(self, new_rule: WildfireTransitionRule):
        self.active_transition_rule = new_rule
        
    def swap_environment_builder(self, new_builder: EnvironmentBuilder):
        self.env_builder = new_builder
        self.reset_environment()