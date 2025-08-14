"""

CHEM 273_Project 2

Motion Simulation of E.Coli 
in Concentration Gradient
-Python Code

-Group 6-
David Houshangi
Paul Rubrio
Yejin Yang
Christian Fernandez
Zander Rothering


"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
from typing import Tuple, List, Callable
from scipy import interpolate

# CONCENTRATION PROFILE CLASSES - GAUSSIAN FUNCTIONS# 

class ConcentrationProfile:
    """
    Class defining various radial concentration profiles using Gaussian functions.
    
    Mathematical Background:
    - All profiles use 2D radial Gaussian functions: C(r) = A * exp(-r²/σ²)
    - Where r = √((x-x₀)² + (y-y₀)²) is the Euclidean distance from center
    - A = amplitude (maximum concentration)
    - σ² = variance (controls width/steepness of gradient)
    """
    
    @staticmethod
    def radial_gradient(x: np.ndarray, y: np.ndarray, center: Tuple[float, float] = (0, 0), 
                       max_conc: float = 1.0, sigma: float = 50) -> np.ndarray:
        """
        Standard 2D Radial Gaussian concentration gradient.
        
        Mathematical Form: C(r) = A * exp(-r²/σ²)
        
        Numerical Methods Used:
        1. Euclidean distance calculation: r = √((x-x₀)² + (y-y₀)²)
        2. Exponential decay function: exp(-r²/σ²)
        3. Element-wise operations on numpy arrays for vectorization
        
        Parameters:
        -----------
        x, y : np.ndarray
            Coordinate arrays (meshgrid format)
        center : tuple
            Center coordinates (x₀, y₀) of concentration source
        max_conc : float
            Maximum concentration amplitude (A)
        sigma : float
            Standard deviation parameter (σ² = 50, moderate width)
            
        Returns:
        --------
        np.ndarray : Concentration values at each (x,y) point
        """
        # NUMERICAL METHOD 1: Euclidean distance calculation
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        # NUMERICAL METHOD 2: Gaussian function evaluation
        return max_conc * np.exp(-r**2 / sigma)
    
    @staticmethod
    def steep_radial_gradient(x: np.ndarray, y: np.ndarray, center: Tuple[float, float] = (0, 0), 
                             max_conc: float = 1.0) -> np.ndarray:
        """
        Steep (narrow) 2D Radial Gaussian concentration gradient.
        
        Mathematical Form: C(r) = A * exp(-r²/25)
        
        Numerical Characteristics:
        - Small σ² = 25 creates steep concentration drop-off
        - Higher gradient magnitude |∇C| = (2r/σ²) * C(r)
        - Bacteria experience stronger directional signals
        
        Test Function Properties:
        - Peak at center: C(0,0) = max_conc
        - Half-maximum at r = √(25 * ln(2)) ≈ 4.16
        - Effective range: ~3σ = 15 units
        """
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return max_conc * np.exp(-r**2 / 25)
    
    @staticmethod
    def shallow_radial_gradient(x: np.ndarray, y: np.ndarray, center: Tuple[float, float] = (0, 0), 
                               max_conc: float = 1.0) -> np.ndarray:
        """
        Shallow (wide) 2D Radial Gaussian concentration gradient.
        
        Mathematical Form: C(r) = A * exp(-r²/100)
        
        Numerical Characteristics:
        - Large σ² = 100 creates gradual concentration drop-off
        - Lower gradient magnitude |∇C| = (2r/σ²) * C(r)
        - Bacteria experience weaker directional signals
        
        Test Function Properties:
        - Peak at center: C(0,0) = max_conc
        - Half-maximum at r = √(100 * ln(2)) ≈ 8.33
        - Effective range: ~3σ = 30 units
        """
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return max_conc * np.exp(-r**2 / 100)
        
# INDIVIDUAL BACTERIUM CLASS - STOCHASTIC DIFFERENTIAL EQUATION SOLVER
class EcoliBacterium:
    """
    Individual E. coli bacterium implementing chemotaxis behavior.
    
    Numerical Methods Implemented:
    1. Euler method for position integration: x(t+dt) = x(t) + v*dt
    2. Finite difference gradient estimation: ∇C ≈ (C(t) - C(t-4dt))/4dt
    3. Stochastic process simulation (random walk + biased walk)
    4. State machine.implementation (tumble/run states)
    
    Biological Model:
    - Run-and-tumble motion with chemotactic bias
    - Temporal gradient sensing (memory-based)
    - State-dependent velocity modulation
    """
    def __init__(self, x: float = 0, y: float = 0, dt: float = 0.1):  #Initialize bacterium with position and numerical parameters.
        
        # POSITION STATE VARIABLES
        self.x = x
        self.y = y
        self.dt = dt  # Numerical integration time step
        self.history_x = [x]
        self.history_y = [y]
        self.concentration_history = [] # For gradient calculation
        self.state = 'tumble'  # 'tumble' or 'run'
        self.run_direction = 0  # angle in radians
        self.tumble_steps = 0
        self.run_speed = 2.0
        self.tumble_speed = 0.5
        
    def get_concentration(self, concentration_func: Callable) -> float:
        """Get concentration at current position"""
        return concentration_func(np.array([self.x]), np.array([self.y]))[0]
    
    def calculate_gradient(self) -> float:
        """
        Estimate concentration gradient using finite difference method.
        
        Numerical Method: Backward finite difference
        ∇C ≈ (C(t) - C(t-4Δt)) / (4Δt)
        
        This approximates temporal derivative dC/dt experienced by bacterium
        moving through concentration field.
        
        Returns:
        --------
        float : Estimated concentration gradient (positive = increasing)
        """
        if len(self.concentration_history) < 4:
            return 0
        
        # Compare current concentration with 4 steps ago
        current_conc = self.concentration_history[-1]
        past_conc = self.concentration_history[-4]
        return current_conc - past_conc
    
    def tumble_step(self):
        """
        Perform random walk step (tumble behavior).
        
        Numerical Methods:
        1. Uniform random number generation: θ ~ U(0, 2π)
        2. Euler integration: x(t+dt) = x(t) + v*cos(θ)*dt
        3. Trigonometric function evaluation
        
        Stochastic Process: Isotropic random walk
        """
        # RANDOM DIRECTION GENERATION
        angle = random.uniform(0, 2 * np.pi)
        
        # EULER METHOD INTEGRATION
        dx = self.tumble_speed * np.cos(angle) * self.dt
        dy = self.tumble_speed * np.sin(angle) * self.dt

        # POSITION UPDATE
        self.x += dx
        self.y += dy
        self.tumble_steps += 1
    
    def run_step(self):
        """
        Perform directed movement step (run behavior).
        
        Numerical Methods:
        1. Euler integration with fixed direction
        2. Trigonometric function evaluation
        
        Deterministic Process: Ballistic motion
        """
        # EULER METHOD INTEGRATION (directed motion)
        dx = self.run_speed * np.cos(self.run_direction) * self.dt
        dy = self.run_speed * np.sin(self.run_direction) * self.dt

        # POSITION UPDATE
        self.x += dx
        self.y += dy
    
    def update_state(self, gradient: float):
        """Update state based on concentration gradient"""
        if self.state == 'tumble':
            if self.tumble_steps >= 4:  # After 4 tumble steps
                if gradient > 0:  # Concentration increasing
                    self.state = 'run'
                    # Set run direction based on recent movement
                    if len(self.history_x) >= 2:
                        dx = self.history_x[-1] - self.history_x[-2]
                        dy = self.history_y[-1] - self.history_y[-2]
                        self.run_direction = np.arctan2(dy, dx)
                self.tumble_steps = 0
        else:  # running state (STOCHASTIC SWITCHING: gradient-dependent + random component)
            if gradient <= 0 or random.random() < 0.1:  # Switch to tumble if gradient decreases
                self.state = 'tumble'
                self.tumble_steps = 0
    
    def step(self, concentration_func: Callable):
        """Perform one simulation step"""
        # CONCENTRATION SAMPLING
        current_conc = self.get_concentration(concentration_func)
        self.concentration_history.append(current_conc)
        
        # GRADIENT ESTIMATION
        gradient = self.calculate_gradient()
        
        # STATE UPDATE
        self.update_state(gradient)
        
        # MOTION INTEGRATION
        if self.state == 'tumble':
            self.tumble_step() # Stochastic motion
        else:
            self.run_step() # Deterministic motion
        
        # TRAJECTORY RECORDING
        self.history_x.append(self.x)
        self.history_y.append(self.y)
      
class ChemotaxisSimulation:
    """Main simulation class for E. coli chemotaxis"""
    
    def __init__(self, n_bacteria: int = 100, grid_size: float = 50, dt: float = 0.1):
        
        # MAIN ADJUSTABLE PARAMETER
        self.n_bacteria = n_bacteria
        self.grid_size = grid_size
        self.dt = dt
        
        # Create spatial grid
        self.x_grid = np.linspace(-grid_size, grid_size, 200)
        self.y_grid = np.linspace(-grid_size, grid_size, 200)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Initialize bacteria randomly around the periphery
        self.bacteria = []
        for _ in range(n_bacteria):
            # Start bacteria at random positions, biased toward periphery
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(grid_size * 0.3, grid_size * 0.8)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            self.bacteria.append(EcoliBacterium(x, y, dt))
        
        self.time_steps = 0
        self.concentration_profile = None
        
        # Store snapshots for time evolution analysis
        self.position_snapshots = []
        self.snapshot_times = []
    
    def set_concentration_profile(self, profile_func: Callable):
        """Set the concentration profile function(Gaussian test function)"""
        self.concentration_profile = profile_func
    
    def step(self):
        """Perform one simulation time step for all bacteria"""
        for bacterium in self.bacteria:
            bacterium.step(lambda x, y: self.concentration_profile(x, y))
        self.time_steps += 1
    
    def run_simulation(self, n_steps: int, snapshot_interval: int = 50):
        """Run simulation for n steps and take snapshots"""
        for step in range(n_steps):
            self.step()
            
            # Take snapshots at regular intervals
            if step % snapshot_interval == 0:
                x_pos, y_pos = self.get_positions()
                self.position_snapshots.append((x_pos.copy(), y_pos.copy()))
                self.snapshot_times.append(self.time_steps)
    
    def get_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current positions of all bacteria"""
        x_pos = np.array([b.x for b in self.bacteria])
        y_pos = np.array([b.y for b in self.bacteria])
        return x_pos, y_pos
    
    def plot_concentration_profile(self, ax=None):
        """
        Visualize the Gaussian concentration profile.
        
        Numerical Methods:
        1. Function evaluation on 2D grid
        2. Contour plot generation
        3. Colormap interpolation
        """
        if ax is None: 
            fig, ax = plt.subplots(figsize=(8, 6))
        
        Z = self.concentration_profile(self.X, self.Y) #Evaluate Gaussian Function on Grid

        # CONTOUR VISUALIZATION
        contour = ax.contourf(self.X, self.Y, Z, levels=20, cmap='inferno', alpha=0.7)        
        ax.contour(self.X, self.Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        plt.colorbar(contour, ax=ax, label='Concentration')

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Radial Concentration Profile')
        ax.set_xlim(-self.grid_size, self.grid_size)
        ax.set_ylim(-self.grid_size, self.grid_size)
        return ax
    
    def plot_bacteria_positions(self, ax=None, show_trails=False, alpha=0.7):
        """Plot current bacteria positions"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos, y_pos = self.get_positions()
        
        # TRAJECTORY VISUALIZATION 
        if show_trails:
            trail_sample = min(20, len(self.bacteria))  # Limit trails to avoid clutter
            for bacterium in self.bacteria[::len(self.bacteria)//trail_sample]:
                ax.plot(bacterium.history_x, bacterium.history_y, 'k-', alpha=0.7, linewidth=0.5)
        
        # Plot current positions with size based on population
        marker_size = max(1, 50 / np.sqrt(self.n_bacteria))
        ax.scatter(x_pos, y_pos, c='red', s=marker_size, alpha=alpha, 
                  label=f'E. coli (N={self.n_bacteria})')
        
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(f'E.coli Positions (N={self.n_bacteria}) at t={self.time_steps}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-self.grid_size, self.grid_size)
        ax.set_ylim(-self.grid_size, self.grid_size)
        return ax
    
    def plot_stacked_frequency_evolution(self, ax=None, bins=100, stack_offset=None):
        """
        Plot stacked radial frequency distribution with vertical offset and smooth lines.
        
        Numerical Methods:
        1. Radial distance calculation: r = √(x² + y²)
        2. Histogram binning: np.histogram()
        3. Cubic interpolation for smooth curves
        4. Statistical frequency analysis
        5. Vertical stacking with offset calculation
        
        This is a key analysis method showing chemotactic accumulation over time.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define colors for different time points
        colors = plt.cm.plasma(np.linspace(0, 1, len(self.position_snapshots)))
        
        # Calculate bin edges
        bin_edges = np.linspace(0, self.grid_size, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create smooth x values for interpolation
        x_smooth = np.linspace(0, self.grid_size, 300)  # 300 points for smooth curve
        
        # Calculate automatic offset if not provided
        if stack_offset is None:
            max_count = 0
            for i, ((x_pos, y_pos), time) in enumerate(zip(self.position_snapshots, self.snapshot_times)):
                distances = np.sqrt(x_pos**2 + y_pos**2)  #Radial distance calculation
                counts, _ = np.histogram(distances, bins=bin_edges) #Frequency histogram calculation
                max_count = max(max_count, np.max(counts))
            stack_offset = max_count * 1.2  # 20% padding
        
        # Plot each time point with vertical offset
        for i, ((x_pos, y_pos), time) in enumerate(zip(self.position_snapshots, self.snapshot_times)):
            distances = np.sqrt(x_pos**2 + y_pos**2)
            
            # Calculate frequency (counts) in each bin
            counts, _ = np.histogram(distances, bins=bin_edges)
            
            # Create interpolation function for smooth curves
            f = interpolate.interp1d(bin_centers, counts, kind='cubic', 
                                    bounds_error=False, fill_value=0)
            
            # Get smooth counts
            counts_smooth = f(x_smooth)
            
            # Apply vertical offset
            offset_counts_smooth = counts_smooth + (i * stack_offset)
            
            # Plot smooth filled area
            ax.fill_between(x_smooth, i * stack_offset, offset_counts_smooth, 
                           color=colors[i], alpha=0.7, label=f't={time}')
            
            # Add smooth line on top
            ax.plot(x_smooth, offset_counts_smooth, color=colors[i], linewidth=2)
            
            # Add time label on the right side
            ax.text(self.grid_size * 1.02, i * stack_offset + np.max(counts)/2, 
                   f't={time}', fontsize=10, va='center', 
                   color=colors[i], fontweight='bold')
        
        ax.set_xlabel('Distance from center')
        ax.set_ylabel('Frequency (Number of bacteria) + Offset')
        ax.set_title(f'Stacked Radial Frequency Evolution (N={self.n_bacteria})')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.grid_size * 1.15)
        
        # Add legend with smaller font
        ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left', fontsize=8)
        
        return ax
    
    def plot_radial_frequency_evolution(self, ax=None, bins=20):
        """Plot radial frequency distribution for different time points (original method)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define colors for different time points
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.position_snapshots)))
        
        # Calculate bin edges
        bin_edges = np.linspace(0, self.grid_size, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        for i, ((x_pos, y_pos), time) in enumerate(zip(self.position_snapshots, self.snapshot_times)):
            distances = np.sqrt(x_pos**2 + y_pos**2)
            
            # Calculate frequency (counts) in each bin
            counts, _ = np.histogram(distances, bins=bin_edges)
            
            # Plot as line plot
            ax.plot(bin_centers, counts, color=colors[i], linewidth=2, 
                   marker='o', markersize=4, label=f't={time}')
        
        ax.set_xlabel('Distance from center')
        ax.set_ylabel('Frequency (Number of bacteria)')
        ax.set_title(f'Radial Frequency Distribution Over Time (N={self.n_bacteria})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.grid_size)
        return ax
    
    def plot_time_evolution(self, ax=None):
        """Plot time evolution of bacterial positions"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot Gaussian concentration profile background
        Z = self.concentration_profile(self.X, self.Y)
        contour = ax.contourf(self.X, self.Y, Z, levels=15, cmap='inferno', alpha=0.3)
        
        # Plot snapshots with different colors
        colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(self.position_snapshots)))
        
        for i, ((x_pos, y_pos), time) in enumerate(zip(self.position_snapshots, self.snapshot_times)):
            marker_size = max(1, 30 / np.sqrt(self.n_bacteria))
            ax.scatter(x_pos, y_pos, c=[colors[i]], s=marker_size, 
                      alpha=0.6, label=f't={time}')
        
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(f'Time Evolution of E. coli (N={self.n_bacteria})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xlim(-self.grid_size, self.grid_size)
        ax.set_ylim(-self.grid_size, self.grid_size)
        return ax
    
    def plot_example_trajectory(self, bacterium_idx=0, ax=None):
        """
        Visualize single bacterium trajectory as test of individual behavior.
        
        Test Function: Validates individual chemotactic response to Gaussian field
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        bacterium = self.bacteria[bacterium_idx]
        x_hist = np.array(bacterium.history_x)
        y_hist = np.array(bacterium.history_y)
        
        # Plot trajectory
        ax.plot(x_hist, y_hist, 'k-', alpha=0.7, linewidth=1.5, label='Trajectory')
        
        # Mark start and end
        ax.scatter(x_hist[0], y_hist[0], c='green', s=100, marker='o', 
                  label='Start', zorder=5, edgecolor='black')
        ax.scatter(x_hist[-1], y_hist[-1], c='red', s=100, marker='s', 
                  label='End', zorder=5, edgecolor='black')
        
        # Add concentration profile background
        Z = self.concentration_profile(self.X, self.Y)
        contour = ax.contourf(self.X, self.Y, Z, levels=15, cmap='inferno', alpha=0.3)
        
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title(f'Example E. coli Trajectory (N={self.n_bacteria})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-self.grid_size, self.grid_size)
        ax.set_ylim(-self.grid_size, self.grid_size)
        return ax

def run_population_analysis(population_sizes=None):
    """
    Comprehensive test suite for different population sizes and Gaussian profiles.
    
    Test Functions:
    1. Standard Gaussian: σ² = 50 (moderate gradient)
    2. Steep Gaussian: σ² = 25 (strong gradient) 
    3. Shallow Gaussian: σ² = 100 (weak gradient)
    
    Parameters:
    -----------
    population_sizes : list
        ADJUSTABLE POPULATION PARAMETER LIST
        Default: [10, 100, 1000] bacteria
    
    Returns:
    --------
    dict : Nested dictionary with simulation results for analysis
    """
    
    if population_sizes is None:
        population_sizes = [10, 100, 1000]  # ← MODIFY HERE FOR DIFFERENT POPULATIONS

    #Test Function Suite- Different Gaussian Profiles    
    gradient_types = {
        'Standard': ConcentrationProfile.radial_gradient,  # σ² = 50
        'Steep': ConcentrationProfile.steep_radial_gradient,  # σ² = 25
        'Shallow': ConcentrationProfile.shallow_radial_gradient  # σ² = 100
    }
    
    results = {}

    # Population size loop
    for pop_size in population_sizes:
        results[pop_size] = {}
        
        #Gaussian profile loop
        for grad_name, grad_func in gradient_types.items():
            # Run simulation
            sim = ChemotaxisSimulation(n_bacteria=pop_size, grid_size=50) # ← POPULATION PARAMETER
            sim.set_concentration_profile(grad_func)  # Set Gaussian test function
            
            # Record initial positions
            initial_pos = sim.get_positions()
            
            # Run simulation with snapshots
            sim.run_simulation(400, snapshot_interval=80)
            
            # Record final positions
            final_pos = sim.get_positions()
            
            results[pop_size][grad_name] = {
                'simulation': sim,
                'initial_positions': initial_pos,
                'final_positions': final_pos
            }
    
    return results

def create_stacked_frequency_plots(results):
    """Create stacked frequency plots for each concentration profile"""
    
    population_sizes = list(results.keys())
    gradient_types = ['Standard', 'Steep', 'Shallow']
    
    # Create stacked frequency plots for each gradient type
    for grad_type in gradient_types:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for i, pop_size in enumerate(population_sizes):
            sim = results[pop_size][grad_type]['simulation']
            sim.plot_stacked_frequency_evolution(axes[i])
            axes[i].set_title(f'{grad_type} Gradient - N={pop_size}')
        
        plt.suptitle(f'Stacked Frequency Evolution - {grad_type} Concentration Profile', 
                     fontsize=16)
        plt.tight_layout()
        plt.show()

def create_population_comparison_plots(results):
    """Create comprehensive visualization plots comparing populations"""
    
    population_sizes = list(results.keys())
    gradient_types = ['Standard', 'Steep', 'Shallow']
    
    # 1. Concentration Profiles
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, grad_type in enumerate(gradient_types):
        sim = results[100][grad_type]['simulation']  # Use N=100 for profile display
        sim.plot_concentration_profile(axes[i])
        axes[i].set_title(f'{grad_type} Radial Gradient')
    plt.suptitle('Concentration Profiles', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 2. Stacked Frequency Evolution Plots
    create_stacked_frequency_plots(results)
    
    # 3. Original Overlapping Frequency Evolution
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for i, pop_size in enumerate(population_sizes):
        for j, grad_type in enumerate(gradient_types):
            ax = axes[i, j]
            sim = results[pop_size][grad_type]['simulation']
            sim.plot_radial_frequency_evolution(ax)
            ax.set_title(f'N={pop_size}, {grad_type} Gradient')
    
    plt.suptitle('Original Radial Frequency Evolution Over Time', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 4. Final Distributions for Different Population Sizes
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for i, pop_size in enumerate(population_sizes):
        for j, grad_type in enumerate(gradient_types):
            ax = axes[i, j]
            sim = results[pop_size][grad_type]['simulation']
            
            # Plot concentration profile background
            sim.plot_concentration_profile(ax)
            # Plot bacteria positions
            sim.plot_bacteria_positions(ax, alpha=0.8)
            ax.set_title(f'N={pop_size}, {grad_type} Gradient')
    
    plt.suptitle('Final E. coli Distributions', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 5. Example Trajectories
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, pop_size in enumerate(population_sizes):
        sim = results[pop_size]['Standard']['simulation']
        sim.plot_example_trajectory(0, axes[i])
        axes[i].set_title(f'Example Trajectory (N={pop_size})')
    
    plt.suptitle('Individual Bacterial Trajectories', fontsize=16)
    plt.tight_layout()
    plt.show()

# Main execution and demonstration
if __name__ == "__main__":
    # Quick demonstration with stacked frequency plot
    sim = ChemotaxisSimulation(n_bacteria=200)
    sim.set_concentration_profile(ConcentrationProfile.radial_gradient)
    
    # Show initial state
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sim.plot_concentration_profile(axes[0, 0])
    sim.plot_bacteria_positions(axes[0, 1])
    axes[0, 0].set_title('Concentration Profile')
    axes[0, 1].set_title('Initial Distribution')
    
    # Run simulation
    sim.run_simulation(400, snapshot_interval=80)
    
    # Show final state and stacked frequency
    sim.plot_bacteria_positions(axes[1, 0], show_trails=True)
    sim.plot_stacked_frequency_evolution(axes[1, 1])  # NEW STACKED PLOT
    axes[1, 0].set_title('Final Distribution with Trails')
    axes[1, 1].set_title('Stacked Frequency Evolution')
    
    plt.suptitle('Quick Demonstration - Stacked Radial Chemotaxis', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Show example trajectory
    fig, ax = plt.subplots(figsize=(10, 8))
    sim.plot_example_trajectory(0, ax)
    plt.show()
    
    # Run comprehensive analysis
    results = run_population_analysis()
    
    # Create visualization plots
    create_population_comparison_plots(results)