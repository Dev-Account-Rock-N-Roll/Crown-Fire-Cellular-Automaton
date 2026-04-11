import tkinter as tk
from tkinter import ttk
import numpy as np

from engine import WildfireSimulationEngine
from rules import ThermodynamicSpreadRule, SimpleDiscreteSpreadRule

class WildfireSimulatorGUI:
    """Handles visual rendering, cross-platform UI components, and keyboard inputs."""
    
    def __init__(self, root_window: tk.Tk, simulation_engine: WildfireSimulationEngine):
        self.root_window = root_window
        self.root_window.title("Wildfire Cellular Automata Simulator")
        self.simulation = simulation_engine
        
        self.cell_pixel_size = 12
        self.is_auto_playing = False
        self.auto_play_delay_ms = 100
        
        self._setup_user_interface()
        self._setup_keyboard_bindings()
        self._initialize_canvas_grid()
        self.refresh_canvas_colors()

    def _setup_user_interface(self):
        """Builds a responsive, cross-platform layout using standard grid/pack strategies."""
        canvas_width = self.simulation.total_columns * self.cell_pixel_size
        canvas_height = self.simulation.total_rows * self.cell_pixel_size
        
        self.control_frame = tk.Frame(self.root_window, height=500, width=canvas_width, bg="green")
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=20, pady=15)

        self.play_pause_button = ttk.Button(self.control_frame, text="Play (Space)", command=self.action_toggle_play_pause, width=15)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        self.step_button = ttk.Button(self.control_frame, text="Step (Right Arrow)", command=self.action_step_forward, width=20)
        self.step_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(self.control_frame, text="Reset Map", command=self.action_reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.rule_toggle_button = ttk.Button(self.control_frame, text="Toggle Physics Rule", command=self.action_toggle_rule)
        self.rule_toggle_button.pack(side=tk.LEFT, padx=20)

        self.info_container = ttk.Frame(self.control_frame)
        self.info_container.pack(side=tk.RIGHT, fill=tk.X)

        self.turn_string_var = tk.StringVar(value="Turn: 0")
        self.turn_label = tk.Label(self.info_container, textvariable=self.turn_string_var, font=("Helvetica", 14, "bold"))
        self.turn_label.pack(side=tk.TOP, anchor="e")

        self.rule_string_var = tk.StringVar(value="Rule: ThermodynamicSpreadRule")
        self.rule_label = tk.Label(self.info_container, textvariable=self.rule_string_var, font=("Helvetica", 10))
        self.rule_label.pack(side=tk.TOP, anchor="e")

        self.grid_canvas = tk.Canvas(self.root_window, width=canvas_width, height=canvas_height, bg="black")
        self.grid_canvas.pack(side=tk.TOP, padx=10, pady=10)
        self.grid_canvas.bind("<Button-1>", self.handle_canvas_click)

    def _setup_keyboard_bindings(self):
        """Maps physical keys to simulation controls."""
        self.root_window.bind("<space>", lambda event: self.action_toggle_play_pause())
        self.root_window.bind("<Left>", lambda event: self.action_step_backward())
        self.root_window.bind("<Right>", lambda event: self.action_step_forward())

    def _initialize_canvas_grid(self):
        """Allocates rectangles for high-performance recoloring rather than redrawing."""
        self.canvas_rectangles = []
        self.current_rendered_colors = []
        
        for row_index in range(self.simulation.total_rows):
            row_of_rectangles = []
            row_colors = []
            for col_index in range(self.simulation.total_columns):
                x_start = col_index * self.cell_pixel_size
                y_start = row_index * self.cell_pixel_size
                rect_id = self.grid_canvas.create_rectangle(
                    x_start, y_start, 
                    x_start + self.cell_pixel_size, y_start + self.cell_pixel_size, 
                    outline="", fill="black"
                )
                row_of_rectangles.append(rect_id)
                row_colors.append("black")
            self.canvas_rectangles.append(row_of_rectangles)
            self.current_rendered_colors.append(row_colors)

    def refresh_canvas_colors(self):
        """Translates numerical grid states into visually distinguishable colors."""
        state = self.simulation.current_state
        red_channel = np.zeros_like(state.fuel_levels)
        green_channel = np.zeros_like(state.fuel_levels)
        blue_channel = np.zeros_like(state.fuel_levels)

        burning_mask = state.is_actively_burning
        alive_mask = (state.fuel_levels > 0.0) & ~burning_mask
        ash_mask = (state.fuel_levels <= 0.0) & ~burning_mask

        # Burning Cells: Vivid orange/red
        red_channel[burning_mask] = 1.0
        green_channel[burning_mask] = state.fuel_levels[burning_mask] * 0.7
        
        # Healthy Vegetation: Green with heat blending
        red_channel[alive_mask] = np.clip(state.heat_levels[alive_mask], 0.0, 1.0)
        green_channel[alive_mask] = 0.3 + (state.fuel_levels[alive_mask] * 0.7)
        
        # Ash/Rock: Dark gray
        red_channel[ash_mask], green_channel[ash_mask], blue_channel[ash_mask] = 0.2, 0.2, 0.2

        red_channel = np.clip(red_channel, 0.0, 1.0)
        green_channel = np.clip(green_channel, 0.0, 1.0)
        blue_channel = np.clip(blue_channel, 0.0, 1.0)

        # Performance loop: Only modify cells that have visibly changed color
        for r in range(state.total_rows):
            for c in range(state.total_columns):
                hex_color = f"#{int(red_channel[r, c] * 255):02x}{int(green_channel[r, c] * 255):02x}{int(blue_channel[r, c] * 255):02x}"
                if hex_color != self.current_rendered_colors[r][c]:
                    self.grid_canvas.itemconfig(self.canvas_rectangles[r][c], fill=hex_color)
                    self.current_rendered_colors[r][c] = hex_color

        self.turn_string_var.set(f"Turn: {self.simulation.simulation_turn_count}")
        self.rule_string_var.set(f"Rule: {self.simulation.active_transition_rule.__class__.__name__}")

    def handle_canvas_click(self, event):
        """Allows users to manually spark a fire directly on the grid."""
        col_index = event.x // self.cell_pixel_size
        row_index = event.y // self.cell_pixel_size
        state = self.simulation.current_state
        
        if 0 <= row_index < state.total_rows and 0 <= col_index < state.total_columns:
            state.heat_levels[row_index, col_index] = 2.0
            state.moisture_levels[row_index, col_index] = 0.0
            state.is_actively_burning[row_index, col_index] = True
            self.refresh_canvas_colors()

    def action_step_backward(self):
        """Placeholder for potential future implementation of reverse stepping."""
        pass 

    def action_step_forward(self):
        """Advances the simulation by one timestep, pausing playback if active."""
        if self.is_auto_playing:
            self.action_toggle_play_pause() 
        
        self.simulation.advance_one_turn()
        self.refresh_canvas_colors()

    def action_toggle_play_pause(self):
        """Handles the spacebar and button toggle for automated progression."""
        if self.is_auto_playing:
            self.is_auto_playing = False
            self.play_pause_button.config(text="Play (Space)")
        else:
            self.is_auto_playing = True
            self.play_pause_button.config(text="Pause (Space)")
            self._execute_automatic_loop()

    def action_reset_simulation(self):
        """Halts simulation and completely refreshes the cellular grid."""
        if self.is_auto_playing:
            self.action_toggle_play_pause()
            
        self.simulation.reset_environment()
        self.refresh_canvas_colors()

    def action_toggle_rule(self):
        """Hot-swaps the underlying mathematics of the cellular automaton."""
        if isinstance(self.simulation.active_transition_rule, ThermodynamicSpreadRule):
            self.simulation.swap_transition_rule(SimpleDiscreteSpreadRule())
        else:
            self.simulation.swap_transition_rule(ThermodynamicSpreadRule())
        self.refresh_canvas_colors()

    def _execute_automatic_loop(self):
        """Recursive timed callback for the Tkinter mainloop."""
        if self.is_auto_playing:
            self.simulation.advance_one_turn()
            self.refresh_canvas_colors()
            self.root_window.after(self.auto_play_delay_ms, self._execute_automatic_loop)