import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random
import time

class RoboticArm:
    def __init__(self):
        # Link lengths (in arbitrary units)
        self.l1 = 2.0  # Base to first joint
        self.l2 = 1.5  # First to second joint
        self.l3 = 1.5  # Second to third joint
        self.l4 = 1.0  # Third to fourth joint
        self.l5 = 0.5  # Fourth to end effector
        
        # Total reach
        self.max_reach = self.l1 + self.l2 + self.l3 + self.l4 + self.l5
        
        # Current joint angles
        self.theta = np.zeros(5)
        
        # Base position
        self.base_position = np.array([0., 0., 0.])

    def forward_kinematics(self, theta):
        """Calculate end effector position given joint angles"""
        t1, t2, t3, t4, t5 = theta
        
        x = (self.l2 * np.cos(t1) * np.cos(t2) + 
             self.l3 * np.cos(t1) * np.cos(t2 + t3) +
             self.l4 * np.cos(t1) * np.cos(t2 + t3 + t4) +
             self.l5 * np.cos(t1) * np.cos(t2 + t3 + t4 + t5))
        
        y = (self.l2 * np.sin(t1) * np.cos(t2) +
             self.l3 * np.sin(t1) * np.cos(t2 + t3) +
             self.l4 * np.sin(t1) * np.cos(t2 + t3 + t4) +
             self.l5 * np.sin(t1) * np.cos(t2 + t3 + t4 + t5))
        
        z = (self.l1 + self.l2 * np.sin(t2) +
             self.l3 * np.sin(t2 + t3) +
             self.l4 * np.sin(t2 + t3 + t4) +
             self.l5 * np.sin(t2 + t3 + t4 + t5))
        
        return self.base_position + np.array([x, y, z])

    def calculate_manipulability(self, theta):
        """Calculate manipulability measure at given configuration"""
        # Numerical Jacobian
        current_pos = self.forward_kinematics(theta)
        jacobian = np.zeros((3, 5))
        epsilon = 0.0001
        
        for i in range(5):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            pos_plus = self.forward_kinematics(theta_plus)
            jacobian[:, i] = (pos_plus - current_pos) / epsilon
        
        # Manipulability measure
        return np.sqrt(np.linalg.det(np.dot(jacobian, jacobian.T)))

    def inverse_kinematics(self, target_pos, learning_rate=0.01, max_iterations=1000, tolerance=0.001):
        current_theta = self.theta.copy()
        for _ in range(max_iterations):
            current_pos = self.forward_kinematics(current_theta)
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < tolerance:
                break
            
            jacobian = np.zeros((3, 5))
            epsilon = 0.0001
            
            for i in range(5):
                theta_plus = current_theta.copy()
                theta_plus[i] += epsilon
                pos_plus = self.forward_kinematics(theta_plus)
                jacobian[:, i] = (pos_plus - current_pos) / epsilon
            
            delta_theta = learning_rate * np.dot(jacobian.T, error)
            current_theta += delta_theta
            current_theta = np.clip(current_theta, -np.pi, np.pi)
        
        return current_theta

def generate_bowl_surface(center=[0, 0, 4], radius=2):
    """Generate points on an inverted bowl surface"""
    x = np.linspace(center[0]-radius, center[0]+radius, 20)
    y = np.linspace(center[1]-radius, center[1]+radius, 20)
    X, Y = np.meshgrid(x, y)
    
    # Inverted bowl equation
    Z = -0.5 * ((X-center[0])**2 + (Y-center[1])**2) / radius + center[2]
    
    return X, Y, Z, center

def generate_fire_points(bowl_center, radius=1.5, num_points=30):
    """Generate points representing fire locations throughout the bowl surface"""
    fire_points = []
    
    # Generate points in concentric circles to cover the entire surface
    num_circles = 5  # Number of concentric circles
    
    for circle_idx in range(num_circles):
        # Calculate radius for this circle (from center to edge)
        r = radius * (circle_idx + 1) / num_circles
        
        # Calculate number of points for this circle
        # More points for outer circles, fewer for inner circles
        if circle_idx == 0:
            # Center point
            points_in_circle = 1
        else:
            points_in_circle = int(num_points * circle_idx / num_circles)
        
        # Generate points on this circle
        if circle_idx == 0:
            # Just the center point
            x = [bowl_center[0]]
            y = [bowl_center[1]]
        else:
            t = np.linspace(0, 2*np.pi, points_in_circle, endpoint=False)
            x = bowl_center[0] + r * np.cos(t)
            y = bowl_center[1] + r * np.sin(t)
        
        # Calculate z coordinates based on bowl shape
        for i in range(len(x)):
            xi, yi = x[i], y[i]
            zi = -0.5 * ((xi-bowl_center[0])**2 + (yi-bowl_center[1])**2) / 2 + bowl_center[2]
            fire_points.append([xi, yi, zi])
    
    return np.array(fire_points)

def optimize_base_position_genetic(arm, fire_points):
    """Find optimal base position for the robot arm using a genetic algorithm"""
    start_time = time.time()
    
    # Fitness function (what we want to maximize)
    def fitness_function(base_pos):
        # Set new base position
        arm.base_position = np.array([base_pos[0], base_pos[1], 0])  # Keep z=0
        
        # Calculate average manipulability over all target points
        total_manipulability = 0
        reachable_points = 0
        
        # Use a subset of fire points for faster evaluation
        sample_points = fire_points
        if len(fire_points) > 10:
            indices = np.random.choice(len(fire_points), 10, replace=False)
            sample_points = fire_points[indices]
        
        for point in sample_points:
            # Check if point is within reach
            distance = np.linalg.norm(point - arm.base_position)
            if distance > arm.max_reach:
                continue
                
            # Try to find IK solution with a simplified check
            reachable_points += 1
            try:
                theta = arm.inverse_kinematics(point)
                manip = arm.calculate_manipulability(theta)
                total_manipulability += manip
            except:
                pass
        
        # Return 0 if no points are reachable
        if reachable_points == 0:
            return 0
            
        # Add distance penalty to keep robot at reasonable distance
        distance_to_center = np.linalg.norm(base_pos[:2] - fire_points.mean(axis=0)[:2])
        distance_penalty = max(0, distance_to_center - arm.max_reach)
        
        # Calculate final fitness (higher is better)
        fitness = (total_manipulability/max(1, reachable_points)) * (reachable_points/len(sample_points))
        fitness = fitness / (1 + 0.1 * distance_penalty)
        
        return fitness
    
    # GA parameters
    population_size = 20
    generations = 10
    mutation_rate = 0.2
    elite_size = 2
    
    # Initial bounds for search space
    bowl_center = fire_points.mean(axis=0)
    search_radius = arm.max_reach * 1.5
    bounds = [
        (bowl_center[0] - search_radius, bowl_center[0] + search_radius),
        (bowl_center[1] - search_radius, bowl_center[1] + search_radius)
    ]
    
    # Initialize population
    population = []
    for _ in range(population_size):
        individual = [
            random.uniform(bounds[0][0], bounds[0][1]),
            random.uniform(bounds[1][0], bounds[1][1])
        ]
        population.append(individual)
    
    # Calculate initial position based on bowl center (strategic starting point)
    initial_pos = [bowl_center[0] - 3, bowl_center[1]]
    population[0] = initial_pos  # Ensure the initial guess is in the population
    
    # Cache for fitness values to avoid redundant calculations
    fitness_cache = {}
    
    def get_fitness(individual):
        # Convert to tuple for hashing
        indiv_tuple = tuple(individual)
        if indiv_tuple not in fitness_cache:
            fitness_cache[indiv_tuple] = fitness_function(individual)
        return fitness_cache[indiv_tuple]
    
    # Evolution
    best_individual = None
    best_fitness = -float('inf')
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [get_fitness(indiv) for indiv in population]
        
        # Find best individual
        gen_best_idx = np.argmax(fitness_scores)
        gen_best_individual = population[gen_best_idx]
        gen_best_fitness = fitness_scores[gen_best_idx]
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = gen_best_individual
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individuals
        sorted_indices = np.argsort(fitness_scores)[::-1]
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]])
        
        # Selection
        def select_individual(population, fitness_scores):
            # Tournament selection
            tournament_size = 3
            selected_indices = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in selected_indices]
            winner_idx = selected_indices[np.argmax(tournament_fitness)]
            return population[winner_idx]
        
        # Generate new individuals
        while len(new_population) < population_size:
            # Select parents
            parent1 = select_individual(population, fitness_scores)
            parent2 = select_individual(population, fitness_scores)
            
            # Crossover
            crossover_point = random.random()
            child = [
                parent1[0] * crossover_point + parent2[0] * (1 - crossover_point),
                parent1[1] * crossover_point + parent2[1] * (1 - crossover_point)
            ]
            
            # Mutation
            if random.random() < mutation_rate:
                mutation_strength = 0.5  # How much to mutate
                child[0] += random.uniform(-1, 1) * mutation_strength
                child[1] += random.uniform(-1, 1) * mutation_strength
                
                # Ensure within bounds
                child[0] = min(max(child[0], bounds[0][0]), bounds[0][1])
                child[1] = min(max(child[1], bounds[1][0]), bounds[1][1])
            
            new_population.append(child)
        
        # Replace old population
        population = new_population
    
    # Final evaluation with all points
    print(f"Genetic algorithm completed in {time.time() - start_time:.2f} seconds.")
    return np.array([best_individual[0], best_individual[1], 0])

class Simulation:
    def __init__(self):
        self.arm = RoboticArm()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Generate bowl surface
        self.bowl_center = [0, 0, 0]
        self.X, self.Y, self.Z, _ = generate_bowl_surface(self.bowl_center)
        
        # Generate fire points throughout the bowl surface
        self.fire_points = generate_fire_points(self.bowl_center)
        self.current_point_idx = 0
        
        # Track extinguished areas to visualize coverage
        self.extinguished_areas = []
        
        # Find optimal base position using genetic algorithm
        print("Optimizing base position...")
        start_time = time.time()
        optimal_base = optimize_base_position_genetic(self.arm, self.fire_points)
        self.arm.base_position = optimal_base
        print(f"Optimal base position: {optimal_base}")
        print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        
        # Initialize animation
        self.anim = None
        
    def update(self, frame):
        self.ax.cla()
        
        # Plot bowl surface
        self.ax.plot_surface(self.X, self.Y, self.Z, alpha=0.3, color='gray')
        
        # Plot fire points
        for i, point in enumerate(self.fire_points):
            if i < self.current_point_idx:
                color = 'blue'  # Extinguished
            else:
                color = 'red'   # Still burning
            self.ax.scatter(point[0], point[1], point[2], color=color)
        
        # Visualize extinguished areas as small circles
        for area in self.extinguished_areas:
            # Draw a small circle around each extinguished point
            center, radius = area
            theta = np.linspace(0, 2*np.pi, 20)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            
            # Calculate z values based on bowl shape
            z = np.zeros_like(x)
            for i in range(len(x)):
                z[i] = -0.5 * ((x[i]-self.bowl_center[0])**2 + (y[i]-self.bowl_center[1])**2) / 2 + self.bowl_center[2]
            
            self.ax.plot(x, y, z, 'b-', alpha=0.3)
        
        # Calculate target position
        if frame < len(self.fire_points):
            target_pos = self.fire_points[frame]
            self.current_point_idx = frame
            
            # Calculate joint angles using inverse kinematics
            theta = self.arm.inverse_kinematics(target_pos)
            
            # Update arm position
            pos = self.arm.forward_kinematics(theta)
            
            # Record extinguished area (simulate water spray coverage)
            self.extinguished_areas.append((target_pos, 0.3))  # 0.3 is the spray radius
            
            # Plot robotic arm
            self.plot_arm(theta)
            
            # Show spray effect
            self.plot_spray(pos, target_pos)
        
        # Plot base position
        self.ax.scatter(*self.arm.base_position, color='black', s=100, marker='s')
        
        # Set plot limits and labels
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([0, 6])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('5-DOF Robot Arm Fire Extinguishing Simulation')
        
    def plot_arm(self, theta):
        """Plot the robotic arm segments"""
        points = np.zeros((6, 3))
        
        # Base position
        points[0] = self.arm.base_position
        
        # Calculate joint positions
        current_pos = points[0]
        t1, t2, t3, t4, t5 = theta
        
        # First joint
        points[1] = current_pos + [0, 0, self.arm.l1]
        
        # Remaining joints
        transforms = [
            (self.arm.l2, t1, t2),
            (self.arm.l3, t1, t2 + t3),
            (self.arm.l4, t1, t2 + t3 + t4),
            (self.arm.l5, t1, t2 + t3 + t4 + t5)
        ]
        
        for i, (length, angle1, angle2) in enumerate(transforms):
            points[i+2] = points[i+1] + [
                length * np.cos(angle1) * np.cos(angle2),
                length * np.sin(angle1) * np.cos(angle2),
                length * np.sin(angle2)
            ]
        
        # Plot arm segments
        for i in range(5):
            self.ax.plot([points[i][0], points[i+1][0]],
                        [points[i][1], points[i+1][1]],
                        [points[i][2], points[i+1][2]], 'k-', linewidth=2)
        
        # Plot joints
        self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       color='black', s=50)
    
    def plot_spray(self, nozzle_pos, target_pos):
        """Visualize water spray from nozzle to target"""
        # Direction vector from nozzle to target
        direction = target_pos - nozzle_pos
        
        # Spray particles
        num_particles = 20
        spray_length = np.linalg.norm(direction)
        
        # Create spray pattern
        for _ in range(num_particles):
            # Randomize spray trajectory slightly
            rand_factor = 0.1
            offset = np.random.normal(0, rand_factor, 3)
            
            # Scale offset to be perpendicular to spray direction
            offset = offset - np.dot(offset, direction) * direction / np.sum(direction**2)
            
            # Create spray particle trajectory
            t = np.linspace(0, 1, 10)
            x = np.array([nozzle_pos[0] + tt * direction[0] + tt * offset[0] * spray_length for tt in t])
            y = np.array([nozzle_pos[1] + tt * direction[1] + tt * offset[1] * spray_length for tt in t])
            z = np.array([nozzle_pos[2] + tt * direction[2] + tt * offset[2] * spray_length for tt in t])
            
            self.ax.plot(x, y, z, 'c-', alpha=0.3, linewidth=1)
    
    def run(self):
        self.anim = FuncAnimation(
            self.fig, self.update,
            frames=len(self.fire_points),
            interval=100,
            repeat=False
        )
        plt.show()

# Run the simulation
if __name__ == "__main__":
    sim = Simulation()
    sim.run()
