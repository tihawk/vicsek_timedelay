def initialise():
    """Simulation Variables"""
    # Set these before running!!!
    # number of particles
    global N
    N = 512
    
    # size of system
    global box_size
    box_size = 11.5
    
    # noise intensity
    global eta
    eta = 0.45
    
    # neighbour radius
    global r
    r = 1.
    
    # time step
    global t
    t = 0
    global delta_t
    delta_t = 1
    
    # maximum time steps
    global T
    T = 10000*delta_t
    
    # velocity of particles
    global vel
    vel = 0.05
    
    # length of time delay
    global timeDelay
    timeDelay = 0
    
    # are we running for static correlation (true) or spattemp corr (false)
    global isStatic
    isStatic = True
    
    # wavenumber for calculating the spattemp correlation
    global corrCalcK
    corrCalcK = 1.0588235294117647
    
    # the length of the dataset for the spattemp correlation (in units of time)
    global timeLength
    timeLength = 150
    
    # the time at which to start the spattemp corr calculations (in ratio of T)
    global corrCalcStart
    corrCalcStart = 0.1*T
    """END Sim Vars"""