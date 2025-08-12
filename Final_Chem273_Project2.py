import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
from typing import Tuple, List, Callable

class ConcentrationProfile:
    #Class to define radial concentration profiles"""
    
    @staticmethod
    def radial_gradient(x: np.ndarray, y: np.ndarray, center: Tuple[float, float] = (0, 0), 
                       max_conc: float = 1.0, sigma: float = 50) -> np.ndarray:
        """Standard Radial concentration gradient - higher at center"""
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return max_conc * np.exp(-r**2 / sigma)
    
    @staticmethod
    def steep_radial_gradient(x: np.ndarray, y: np.ndarray, center: Tuple[float, float] = (0, 0), 
                             max_conc: float = 1.0) -> np.ndarray:
        """Steeper radial concentration gradient"""
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return max_conc * np.exp(-r**2 / 25)
    
    @staticmethod
    def shallow_radial_gradient(x: np.ndarray, y: np.ndarray, center: Tuple[float, float] = (0, 0), 
                               max_conc: float = 1.0) -> np.ndarray:
        """Shallow radial concentration gradient"""
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return max_conc * np.exp(-r**2 / 100)

class EcoliBacterium:
    """Individual E. coli bacterium with chemotaxis behavior"""
    def __init__(self, x: float = 0, y: float = 0, dt: float = 0.1):
        self.x = x
        self.y = y
        self.dt = dt
        self.history_x = [x]
        self.history_y = [y]
        self.concentration_history = []
        self.state = 'tumble'  # 'tumble' or 'run'
        self.run_direction = 0  # angle in radians
        self.tumble_steps = 0
        self.run_speed = 2.0
        self.tumble_speed = 0.5
        
    def get_concentration(self, concentration_func: Callable) -> float:
        """Get concentration at current position"""
        return concentration_func(np.array([self.x]), np.array([self.y]))[0]
    
    def calculate_gradient(self) -> float:
        """Calculate concentration gradient from last 4 time steps"""
        if len(self.concentration_history) < 4:
            return 0
        
        # Compare current concentration with 4 steps ago
        current_conc = self.concentration_history[-1]
        past_conc = self.concentration_history[-4]
        return current_conc - past_conc
    
    def tumble_step(self):
        """Perform random walk (tumble) step"""
        angle = random.uniform(0, 2 * np.pi)
        dx = self.tumble_speed * np.cos(angle) * self.dt
        dy = self.tumble_speed * np.sin(angle) * self.dt
        
        self.x += dx
        self.y += dy
        self.tumble_steps += 1
    
    def run_step(self):
        """Perform directed movement (run) step"""
        dx = self.run_speed * np.cos(self.run_direction) * self.dt
        dy = self.run_speed * np.sin(self.run_direction) * self.dt
        
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
        else:  # implement running
            if gradient <= 0 or random.random() < 0.1:  # Switch to tumble if gradient decreases
                self.state = 'tumble'
                self.tumble_steps = 0
    
    def step(self, concentration_func: Callable):
        """Perform one simulation step"""
        # Record current concentration
        current_conc = self.get_concentration(concentration_func)
        self.concentration_history.append(current_conc)
        
        # Calculate gradient
        gradient = self.calculate_gradient()
        
        # Update state
        self.update_state(gradient)
        
        # Perform movement
        if self.state == 'tumble':
            self.tumble_step()
        else:
            self.run_step()
        
        # Record position
        self.history_x.append(self.x)
        self.history_y.append(self.y)

class ChemotaxisSimulation:
    """Main simulation class for E. coli chemotaxis"""
    
    def __init__(self, n_bacteria: int = 100, grid_size: float = 50, dt: float = 0.1):
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
        """Set the concentration profile function"""
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
        """Plot the concentration profile"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        Z = self.concentration_profile(self.X, self.Y)
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
        
        # Plot trails if requested
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
    
    def plot_radial_distribution(self, ax=None, bins=30):
        """Plot radial distribution histogram"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos, y_pos = self.get_positions()
        distances = np.sqrt(x_pos**2 + y_pos**2)
        
        ax.hist(distances, bins=bins, alpha=0.7, density=True, 
               label=f'N={self.n_bacteria}, t={self.time_steps}', edgecolor='black')
        ax.set_xlabel('Distance from center')
        ax.set_ylabel('Probability density')
        ax.set_title(f'Radial Distribution (N={self.n_bacteria})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_time_evolution(self, ax=None):
        """Plot time evolution of bacterial positions"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot concentration profile background
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
        """Plot trajectory of a single bacterium with state information"""
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

def run_population_analysis():
    """Run analysis with different population sizes"""
    
    populations = [10, 100, 1000]
    gradient_types = {
        'Standard': ConcentrationProfile.radial_gradient,
        'Steep': ConcentrationProfile.steep_radial_gradient,
        'Shallow': ConcentrationProfile.shallow_radial_gradient
    }
    
    results = {}
    
    for pop_size in populations:
        results[pop_size] = {}
        
        for grad_name, grad_func in gradient_types.items():
            # Run simulation
            sim = ChemotaxisSimulation(n_bacteria=pop_size, grid_size=50)
            sim.set_concentration_profile(grad_func)
            
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

def create_population_comparison_plots(results):
    """Create comprehensive visualization plots comparing populations"""
    
    populations = [10, 100, 1000]
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
    
    # 2. Final Distributions for Different Population Sizes
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for i, pop_size in enumerate(populations):
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
    
    # 3. Radial Distribution Histograms
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    for i, pop_size in enumerate(populations):
        for j, grad_type in enumerate(gradient_types):
            ax = axes[i, j]
            sim = results[pop_size][grad_type]['simulation']
            sim.plot_radial_distribution(ax)
            ax.set_title(f'N={pop_size}, {grad_type} Gradient')
    
    plt.suptitle('Radial Distribution Profiles', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 4. Time Evolution for Each Population Size (Standard Gradient)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, pop_size in enumerate(populations):
        sim = results[pop_size]['Standard']['simulation']
        sim.plot_time_evolution(axes[i])
        axes[i].set_title(f'Time Evolution (N={pop_size})')
    
    plt.suptitle('Time Evolution - Standard Gradient', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # 5. Example Trajectories
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, pop_size in enumerate(populations):
        sim = results[pop_size]['Standard']['simulation']
        sim.plot_example_trajectory(0, axes[i])
        axes[i].set_title(f'Example Trajectory (N={pop_size})')
    
    plt.suptitle('Individual Bacterial Trajectories', fontsize=16)
    plt.tight_layout()
    plt.show()

# Main execution and demonstration for package
if __name__ == "__main__":
    # Quick demonstration with single population
    sim = ChemotaxisSimulation(n_bacteria=100)
    sim.set_concentration_profile(ConcentrationProfile.radial_gradient)
    
    # Show initial state
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    sim.plot_concentration_profile(axes[0, 0])
    sim.plot_bacteria_positions(axes[0, 1])
    axes[0, 0].set_title('Concentration Profile')
    axes[0, 1].set_title('Initial Distribution')
    
    # Run simulation
    sim.run_simulation(300, snapshot_interval=60)
    
    # Show final states
    sim.plot_bacteria_positions(axes[1, 0], show_trails=True)
    sim.plot_radial_distribution(axes[1, 1])
    axes[1, 0].set_title('Final Distribution with Trails')
    axes[1, 1].set_title('Radial Distribution')
    
    plt.suptitle('Quick Demonstration - Radial Chemotaxis', fontsize=14)
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
