import sys
import argparse
import unittest
import tkinter as tk

from rules import ThermodynamicSpreadRule
from environments import ForestBuilder
from engine import WildfireSimulationEngine
from gui import WildfireSimulatorGUI
import test_simulation 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Layered Wildfire Cellular Automata")
    parser.add_argument('--test', action='store_true', help="Run the test suite instead of the GUI")
    args = parser.parse_args()

    if args.test:
        sys.argv = [sys.argv[0]]
        unittest.main(module=test_simulation)
    else:
        main_window = tk.Tk()
        
        physics_rule = ThermodynamicSpreadRule()
        initial_environment = ForestBuilder()
        
        # 30 Rows, 40 Columns mapped across 4 distinct Altitude Layers
        engine = WildfireSimulationEngine(30, 40, 2, physics_rule, initial_environment)
        gui = WildfireSimulatorGUI(main_window, engine)
        
        main_window.focus_force() 
        main_window.mainloop()