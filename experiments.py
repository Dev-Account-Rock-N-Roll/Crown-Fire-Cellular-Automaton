from rules import ThermodynamicSpreadRule
from environments import ForestBuilder
from engine import WildfireSimulationEngine

class Experiment:
    def __init__(self, name: str, description: str, setup_function, report_function):
        self.name = name
        self.description = description
        self.setup_function = setup_function
        self.report_function = report_function
    

    def run_experiment(self, simulation_engine: WildfireSimulationEngine, steps: int):
        print(f"Running Experiment: {self.name}")
        print(self.description)
        self.setup_function(simulation_engine)

        for step in range(steps):
            simulation_engine.advance_one_turn()
        return self.report_function(simulation_engine.state_history)

def create_experiments():
    experiments = []
    
    def setup_forest(sim_engine):
        sim_engine.swap_environment_builder(ForestBuilder())
        sim_engine.swap_transition_rule(ThermodynamicSpreadRule())
    
    def report_forest(simulation_state_history):
        return simulation_state_history 

    experiments.append(Experiment(
        name="Dense Forest Fire",
        description="Simulates a fire in a dense forest with a strong thermodynamic spread rule.",
        setup_function=setup_forest,
        report_function=report_forest
    ))
    return experiments

