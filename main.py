import sys
import argparse
import unittest
import tkinter as tk

from rules import ThermodynamicSpreadRule
from engine import WildfireSimulationEngine
from gui import WildfireSimulatorGUI
import test_simulation 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wildfire Cellular Automata Simulator")
    parser.add_argument('--test', action='store_true', help="Run the test suite instead of the GUI")
    args = parser.parse_args()

    if args.test:
        # Strip argparse args so unittest doesn't get confused
        sys.argv = [sys.argv[0]]
        unittest.main(module=test_simulation)
    else:
        main_window = tk.Tk()
        physics_rule = ThermodynamicSpreadRule()
        
        # 60 rows, 80 columns generates a 960x720 internal map width
        engine = WildfireSimulationEngine(60, 80, physics_rule)
        gui = WildfireSimulatorGUI(main_window, engine)
        
        # Focus the window to immediately allow keyboard input
        main_window.focus_force() 
        main_window.mainloop()