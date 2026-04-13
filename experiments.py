import matplotlib.pyplot as plt

from rules import ThermodynamicSpreadRule
from environments import ForestBuilder
from engine import WildfireSimulationEngine
from state import GridEnvironmentState

class Experiment:
    def __init__(self, name: str, description: str, setup_function, report_function):
        self.name = name
        self.description = description
        self.setup_function = setup_function
        self.report_function = report_function
    
    #todo: ADD TYPE NOTATIONS       
    def run_experiment(self, simulation_engine: WildfireSimulationEngine, steps: int) -> list[GridEnvironmentState]: 
        print(f"Running Experiment: {self.name}")
        print(self.description)
        self.setup_function(simulation_engine)
        print(simulation_engine.current_state.layers['layer_0'].wind_speed)
        for step in range(steps):
            simulation_engine.advance_one_turn()
        return simulation_engine.state_history

def create_experiments():
    experiments = []
    Wind_speed_pairings = [{"canopy_wind": 0, "understory_wind": 0}, {"canopy_wind": 30, "understory_wind": 0}, {"canopy_wind": 30, "understory_wind": 30}]
    for i in range(3):
        def setup_forest(sim_engine :WildfireSimulationEngine, wind_speed_index=i):
            sim_engine.swap_environment_builder(ForestBuilder())
            sim_engine.swap_transition_rule(ThermodynamicSpreadRule())
            sim_engine.current_state.layers['layer_0'].wind_speed = Wind_speed_pairings[wind_speed_index]["understory_wind"] * 2
            sim_engine.current_state.layers['layer_1'].wind_speed = Wind_speed_pairings[wind_speed_index]["canopy_wind"]
            sim_engine.ignite_random_fire(3)  # Ignite 3 random cells to start the fire
        def report_forest(simulation_state_history):
            return simulation_state_history 
        experiments.append(Experiment(
            name="Dense Forest Fire",
            description="Simulates a fire in a dense forest with a strong thermodynamic spread rule.",
            setup_function=setup_forest,
            report_function=report_forest
        ))
    return experiments

def chart_fuel_load_over_time(experiment_result_collection : list):
    wind_speed_colors = ['green', 'orange', 'red']
    for i, simulation in enumerate(experiment_result_collection):
        #understory_wind_speed = simulation[-1]..layers['layer_0'].wind_speed
        #canopy_wind_speed = simulation[-1].current_state.layers['layer_1'].wind_speed
        understory_wind_speed = simulation[-1].layers['layer_0'].wind_speed
        canopy_wind_speed = simulation[-1].layers['layer_1'].wind_speed
        plt.plot([state.layers['layer_0'].fuel_levels.sum() for state in simulation], color=wind_speed_colors[i], linestyle='dashed', label=f"Layer 0 - Wind {understory_wind_speed} m/s")
        plt.plot([state.layers['layer_1'].fuel_levels.sum() for state in simulation], color=wind_speed_colors[i], linestyle='solid', label=f"Layer 1 - Wind {canopy_wind_speed} m/s")
    captions = ["Layer 0 (Ground)", "Layer 1 (Canopy)"]
    #plt.legend(captions)
    plt.legend()
    plt.title("Total Fuel Load Over Time")    
    plt.xlabel("Time Steps")
    plt.ylabel("Total Fuel Load")
    plt.grid()
    plt.show()

experiments = create_experiments()
results = []
sim_engine = WildfireSimulationEngine(30, 40, 2, ThermodynamicSpreadRule(), ForestBuilder())
initial_state = sim_engine.current_state.duplicate_state()  # Capture initial state for baseline comparison
for exp in experiments:
    sim_engine.current_state = initial_state # Reset before each experiment
    results.append(exp.run_experiment(sim_engine, steps=50))
chart_fuel_load_over_time(results)