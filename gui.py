import math
import tkinter as tk
from tkinter import ttk
import numpy as np

from engine import WildfireSimulationEngine
from rules import ThermodynamicSpreadRule, SimpleDiscreteSpreadRule
from environments import PrairieBuilder, SwampBuilder, ForestBuilder

class WildfireSimulatorGUI:
    # ... (__init__ and other untouched internal math logic skipped for brevity if identical, but included in full here)
    def __init__(self, root_window: tk.Tk, simulation_engine: WildfireSimulationEngine):
        self.root_window = root_window
        self.root_window.title("3D Real-World Wildfire Simulator")
        self.simulation = simulation_engine
        
        self.cell_pixel_size = 8
        self.is_auto_playing = False
        self.auto_play_delay_ms = 100
        
        self.num_layers = self.simulation.num_layers
        self.layer_spacing = 20
        self.grid_columns = math.ceil(math.sqrt(self.num_layers))
        self.grid_rows = math.ceil(self.num_layers / self.grid_columns)
        
        self._setup_user_interface()
        self._setup_keyboard_bindings()
        self._initialize_canvas_grid()
        self.refresh_canvas_colors()

    def _get_layer_offset(self, layer_index: int):
        grid_x = layer_index % self.grid_columns
        grid_y = layer_index // self.grid_columns
        layer_width = self.simulation.total_columns * self.cell_pixel_size
        layer_height = self.simulation.total_rows * self.cell_pixel_size
        
        offset_x = self.layer_spacing + grid_x * (layer_width + self.layer_spacing)
        offset_y = self.layer_spacing + grid_y * (layer_height + self.layer_spacing + 30)
        return offset_x, offset_y

    def _setup_user_interface(self):
        layer_width = self.simulation.total_columns * self.cell_pixel_size
        layer_height = self.simulation.total_rows * self.cell_pixel_size
        
        canvas_width = self.layer_spacing + self.grid_columns * (layer_width + self.layer_spacing)
        canvas_height = self.layer_spacing + self.grid_rows * (layer_height + self.layer_spacing + 30)
        
        self.control_frame = tk.Frame(self.root_window, height=80, width=canvas_width, bg="darkgreen")
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=20, pady=15)

        self.play_pause_button = ttk.Button(self.control_frame, text="Play (Space)", command=self.action_toggle_play_pause, width=15)
        self.play_pause_button.pack(side=tk.LEFT, padx=5)

        self.step_button = ttk.Button(self.control_frame, text="Step (Right Arrow)", command=self.action_step_forward, width=20)
        self.step_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(self.control_frame, text="Reset Map", command=self.action_reset_simulation)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.rule_toggle_button = ttk.Button(self.control_frame, text="Toggle Physics", command=self.action_toggle_rule)
        self.rule_toggle_button.pack(side=tk.LEFT, padx=5)
        
        # New Dropdown for Environment Toggling
        self.env_string_var = tk.StringVar(value="Forest")
        self.env_combobox = ttk.Combobox(self.control_frame, textvariable=self.env_string_var, values=["Prairie", "Swamp", "Forest"], state="readonly", width=12)
        self.env_combobox.pack(side=tk.LEFT, padx=15)
        self.env_combobox.bind("<<ComboboxSelected>>", self.action_change_environment)

        self.info_container = ttk.Frame(self.control_frame)
        self.info_container.pack(side=tk.RIGHT, fill=tk.X)

        self.turn_string_var = tk.StringVar(value="Turn: 0")
        self.turn_label = tk.Label(self.info_container, textvariable=self.turn_string_var, font=("Helvetica", 14, "bold"))
        self.turn_label.pack(side=tk.TOP, anchor="e")

        self.rule_string_var = tk.StringVar(value="Rule: ThermodynamicSpreadRule")
        self.rule_label = tk.Label(self.info_container, textvariable=self.rule_string_var, font=("Helvetica", 10))
        self.rule_label.pack(side=tk.TOP, anchor="e")

        self.grid_canvas = tk.Canvas(self.root_window, width=canvas_width, height=canvas_height, bg="#111111")
        self.grid_canvas.pack(side=tk.TOP, padx=10, pady=10)
        self.grid_canvas.bind("<Button-1>", self.handle_canvas_click)

    def _setup_keyboard_bindings(self):
        self.root_window.bind("<space>", lambda event: self.action_toggle_play_pause())
        self.root_window.bind("<Left>", lambda event: self.action_step_backward())
        self.root_window.bind("<Right>", lambda event: self.action_step_forward())

    def _initialize_canvas_grid(self):
        self.canvas_rectangles = {}
        self.current_rendered_colors = {}
        
        for i in range(self.num_layers):
            layer_name = f'layer_{i}'
            self.canvas_rectangles[layer_name] = []
            self.current_rendered_colors[layer_name] = []
            
            offset_x, offset_y = self._get_layer_offset(i)
            self.grid_canvas.create_text(offset_x, offset_y - 15, text=f"Altitude Level: {i}", fill="white", anchor="w", font=("Helvetica", 10, "bold"))
            
            for row_index in range(self.simulation.total_rows):
                row_rects = []
                row_colors = []
                for col_index in range(self.simulation.total_columns):
                    x_start = offset_x + col_index * self.cell_pixel_size
                    y_start = offset_y + row_index * self.cell_pixel_size
                    rect_id = self.grid_canvas.create_rectangle(
                        x_start, y_start, 
                        x_start + self.cell_pixel_size, y_start + self.cell_pixel_size, 
                        outline="", fill="black"
                    )
                    row_rects.append(rect_id)
                    row_colors.append("black")
                self.canvas_rectangles[layer_name].append(row_rects)
                self.current_rendered_colors[layer_name].append(row_colors)

    def refresh_canvas_colors(self):
        state = self.simulation.current_state
        for i in range(self.num_layers):
            layer_name = f'layer_{i}'
            layer_state = state.layers[layer_name]
            
            red_channel = np.zeros_like(layer_state.fuel_levels)
            green_channel = np.zeros_like(layer_state.fuel_levels)
            blue_channel = np.zeros_like(layer_state.fuel_levels)

            burning_mask = layer_state.is_actively_burning
            alive_mask = (layer_state.fuel_levels > 0.0) & ~burning_mask
            ash_mask = (layer_state.fuel_levels <= 0.0) & ~burning_mask
            air_mask = (layer_state.fuel_levels == 0.0) & ~burning_mask & ~ash_mask

            red_channel[burning_mask] = 1.0
            green_channel[burning_mask] = layer_state.fuel_levels[burning_mask] * 0.7
            
            red_channel[alive_mask] = np.clip(layer_state.heat_levels[alive_mask], 0.0, 1.0)
            green_channel[alive_mask] = 0.3 + (layer_state.fuel_levels[alive_mask] * 0.7)
            
            red_channel[ash_mask], green_channel[ash_mask], blue_channel[ash_mask] = 0.2, 0.2, 0.2
            
            # Pure air (0 fuel generated) should be pure black/background
            red_channel[air_mask], green_channel[air_mask], blue_channel[air_mask] = 0.0, 0.0, 0.0

            red_channel = np.clip(red_channel, 0.0, 1.0)
            green_channel = np.clip(green_channel, 0.0, 1.0)
            blue_channel = np.clip(blue_channel, 0.0, 1.0)

            for r in range(layer_state.total_rows):
                for c in range(layer_state.total_columns):
                    hex_color = f"#{int(red_channel[r, c] * 255):02x}{int(green_channel[r, c] * 255):02x}{int(blue_channel[r, c] * 255):02x}"
                    if hex_color != self.current_rendered_colors[layer_name][r][c]:
                        self.grid_canvas.itemconfig(self.canvas_rectangles[layer_name][r][c], fill=hex_color)
                        self.current_rendered_colors[layer_name][r][c] = hex_color

        self.turn_string_var.set(f"Turn: {self.simulation.simulation_turn_count}")
        self.rule_string_var.set(f"Rule: {self.simulation.active_transition_rule.__class__.__name__}")

    def handle_canvas_click(self, event):
        for i in range(self.num_layers):
            offset_x, offset_y = self._get_layer_offset(i)
            layer_width = self.simulation.total_columns * self.cell_pixel_size
            layer_height = self.simulation.total_rows * self.cell_pixel_size
            
            if offset_x <= event.x <= offset_x + layer_width and \
               offset_y <= event.y <= offset_y + layer_height:
                
                col_index = (event.x - offset_x) // self.cell_pixel_size
                row_index = (event.y - offset_y) // self.cell_pixel_size
                
                layer_name = f'layer_{i}'
                layer_state = self.simulation.current_state.layers[layer_name]
                
                # Cannot spark thin air
                if layer_state.fuel_levels[row_index, col_index] > 0.0:
                    layer_state.heat_levels[row_index, col_index] = 2.0
                    layer_state.moisture_levels[row_index, col_index] = 0.0
                    layer_state.is_actively_burning[row_index, col_index] = True
                    self.refresh_canvas_colors()
                break

    def action_step_backward(self):
        pass 

    def action_step_forward(self):
        if self.is_auto_playing:
            self.action_toggle_play_pause() 
        self.simulation.advance_one_turn()
        self.refresh_canvas_colors()

    def action_toggle_play_pause(self):
        if self.is_auto_playing:
            self.is_auto_playing = False
            self.play_pause_button.config(text="Play (Space)")
        else:
            self.is_auto_playing = True
            self.play_pause_button.config(text="Pause (Space)")
            self._execute_automatic_loop()

    def action_reset_simulation(self):
        if self.is_auto_playing:
            self.action_toggle_play_pause()
        self.simulation.reset_environment()
        self.refresh_canvas_colors()

    def action_toggle_rule(self):
        if isinstance(self.simulation.active_transition_rule, ThermodynamicSpreadRule):
            self.simulation.swap_transition_rule(SimpleDiscreteSpreadRule())
        else:
            self.simulation.swap_transition_rule(ThermodynamicSpreadRule())
        self.refresh_canvas_colors()
        
    def action_change_environment(self, event=None):
        selection = self.env_string_var.get()
        if selection == "Prairie":
            new_builder = PrairieBuilder()
        elif selection == "Swamp":
            new_builder = SwampBuilder()
        else:
            new_builder = ForestBuilder()
            
        if self.is_auto_playing:
            self.action_toggle_play_pause()
            
        self.simulation.swap_environment_builder(new_builder)
        self.refresh_canvas_colors()

    def _execute_automatic_loop(self):
        if self.is_auto_playing:
            self.simulation.advance_one_turn()
            self.refresh_canvas_colors()
            self.root_window.after(self.auto_play_delay_ms, self._execute_automatic_loop)