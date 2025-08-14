import numpy as np
import matplotlib.pyplot as plt

def Concentration_Grad(x, y, xnut, ynut, sigma): #Calculates concentration gradient to move particles
    r = -((x - xnut) ** 2 + (y - ynut) ** 2) #Calcualtes r(radius) a point is from the nutrient "seed=center" to form gradient. Neg values means farther = lower concentration
    return np.exp(r / (2 * sigma ** 2)) #Creates Concentration in Gaussian fashion so that it is slowly decreasing. Conc will always be between 0 and 1

def Gradient_Ascent(x, y, xnut, ynut, sigma, epsilon): #Gets the gradient change in x and y directions to use for run phase
    dC_dx = (Concentration_Grad(x + epsilon, y, xnut, ynut, sigma) - Concentration_Grad(x - epsilon, y, xnut, ynut, sigma)) / (2 * epsilon)
    dC_dy = (Concentration_Grad(x, y + epsilon, xnut, ynut, sigma) - Concentration_Grad(x, y - epsilon, xnut, ynut, sigma)) / (2 * epsilon)
    return dC_dx, dC_dy #Calcs conc gradient at points immediately next to current x and y to determine change in gradient.
    #Numerator calcualtes the change in concentration between points and 2*epsilon calculates the change in distance
    #Returns change in gradient in x and y direction for run phase use

def Tumble_Phase(x, y, theta, tumble_speed): #Performs tumble phase by having bacteria turn in random direction and move
    x += tumble_speed * np.cos(theta) #Moves E. Coli in random x direction from current x
    y += tumble_speed * np.sin(theta) #Moves E. Coli in random y direction from current y
    return x, y #Returns the updated x and y value

def Run_Phase(x, y, run_speed, xnut, ynut, sigma, epsilon): #Calculates direction gradient is higher and moves E. coli towards it
    dC_dx, dC_dy = Gradient_Ascent(x, y, xnut, ynut, sigma, epsilon) #Calculates gradient
    theta = np.arctan2(dC_dy, dC_dx) #Turns E. coli in direction of higher gradient. We use arctan2 as it is the 4-quadrant inverse tangent
    x += run_speed * np.cos(theta) #Moves E. coli in x direction based on gradient
    y += run_speed * np.sin(theta) #Moves E. coli in y direction based on gradient
    return x, y #Returns the updated x and y value

def Bact_Move(x0, y0, xnut, ynut, sigma, iterations, run_speed, tumble_speed, bact_num, epsilon): #Moves E.coli based on run and tumble phases
    
    Bact_positions = np.zeros((bact_num, iterations + 1, 2)) #Creates a 3D array to add all E. coli postions.
    #1st dimension Bacteria, 2nd dimension step, and 3rd spatial cordinates x and y
    
    for b in range(bact_num): #Iterates through each E. coli to calculate positions through each iteration
        x, y = x0, y0 #Sets initial E. coli positions
        Bact_positions[b, 0, :] = [x, y] #Adds initial position to position array
        step_count = 0 #Sets step count to 0. Will perform 4 tumble steps then a run step before resetting to 0

        for i in range(iterations): #Iterates through number of movement steps E. coli will have
            if step_count < 4: #Tumble phase for first 4 steps
                theta = np.random.uniform(0, 2 * np.pi) #Calculates a random angle to turn between 0 and 2 pi (full circle)
                x, y = Tumble_Phase(x, y, theta, tumble_speed) #Moves E. coli x and y direction based on random angle turned
            else: #Run phase for 5th step
                x, y = Run_Phase(x, y, run_speed, xnut, ynut, sigma, epsilon) #Moves E. coli toward ascending nutrient gradient

            Bact_positions[b, i + 1, :] = [x, y] #Adds position at each iteration to array containing the bacteria positions. Positions will be different for each E. coli.
            
            step_count += 1 #After each step adds 1 to step counter to determine when run phase is reached
            if step_count > 4: #Determines if run phase has been completed.
                step_count = 0 #If run phase complete step count returns to 0

    return Bact_positions #Returns array of E. coli positions

def Conc_Grad_Plotting(grad_size, x_vals, y_vals, xnut, ynut, sigma): #Plots gradient to have Movement plot overlayyed.
    X, Y = np.meshgrid(x_vals, y_vals) #Creates matrix of coordinates for plotting concentration gradient
    Conc_Plot = Concentration_Grad(X, Y, xnut, ynut, sigma) #Calculates concentration based on values in meshgrid
    plt.imshow(Conc_Plot, extent=[-grad_size, grad_size, -grad_size, grad_size], origin='lower', cmap='binary', alpha=1.0)
    plt.colorbar(label='Concentration') #Uses plt.imshow as it is able to render 2D scalar data as a pseudocolor image.

def Movement_Plotting(Bact_positions): #Plotting for path of each bacteria in 2D space.
    for path in Bact_positions: #Iterates through each E. colis path indexing into the bacteria
        plt.plot(path[:, 0], path[:, 1], color='white', marker='o', markersize=1, linewidth=0.5) #Plots each time step by indexing into all time steps for x and y values

class Bact_Move_in_Conc_Grad:
    def __init__(self):
        self.tumble_speed = 1 #"Speed" at which bacteria moves during each tumble step. Can be arbitrary number but must be smaller than run speed.
        self.run_speed = 5 #"Speed" at which E. coli moves during run phase. Can be arbitrary number but must be larger than tumble speed or E. coli won't ascend gradient
        self.bact_num = 10 #Number of E. coli placed into nutrient gradient
        self.epsilon = 0.001 #Represents an arbitrarily small value to change x and y by for gradient ascent calculation. 
        self.iterations = 1000 #Number of movements the E. coli make
        self.sigma = 200 #Gradient standard deviation. Higher value means greater spread.
        self.conc_size = 100 #Arbitrary size over which concentration gradient is spread.
        self.xnut = np.random.uniform(-self.conc_size, self.conc_size) #Random value at which nutrient gradient is highest for x
        self.ynut = np.random.uniform(-self.conc_size, self.conc_size) #Random value at which nutrient gradient is highest for y
        self.x0 = np.random.uniform(-self.conc_size, self.conc_size) #Random value at which E. coli positions starts for x
        self.y0 = np.random.uniform(-self.conc_size, self.conc_size) #Random value at which E. coli positions starts for y
        self.x_vals = np.linspace(-self.conc_size, self.conc_size, 1000) #X values for plotting concentration gradient
        self.y_vals = np.linspace(-self.conc_size, self.conc_size, 1000) #Y values for plotting concentration gradient

    def Run_Ecoli_Movement(self):
        Bact_positions = Bact_Move(self.x0, self.y0, self.xnut, self.ynut, self.sigma, self.iterations, self.run_speed, self.tumble_speed, self.bact_num, self.epsilon)
        Conc_Grad_Plotting(self.conc_size, self.x_vals, self.y_vals, self.xnut, self.ynut, self.sigma)
        Movement_Plotting(Bact_positions)

        plt.title("Movement of E. Coli in Nutrient Gradient")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.grid(True, color='gray', alpha=0.3)
        plt.show()

Bact_Move_in_Conc_Grad().Run_Ecoli_Movement()
