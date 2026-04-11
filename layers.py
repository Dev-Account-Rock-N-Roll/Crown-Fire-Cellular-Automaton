from state import GridEnvironmentState


class LayeredGridEnvironmentState(GridEnvironmentState):
    def __init__(self, total_rows: int, total_columns: int, number_of_layers: int):
        super().__init__(total_rows, total_columns)
        self.layers = {}
        for i in range(number_of_layers):
            self.layers[f'layer_{i}'] = GridEnvironmentState(total_rows, total_columns)
    
    def duplicate_state(self) -> 'LayeredGridEnvironmentState':
        """Deep copy for synchronous Cellular Automata generation updates."""
        copied_state = LayeredGridEnvironmentState(self.total_rows, self.total_columns, len(self.layers))
        for layer_name, layer_state in self.layers.items():
            copied_state.layers[layer_name] = layer_state.duplicate_state()
        return copied_state

