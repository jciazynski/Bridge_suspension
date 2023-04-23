import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
####values to input####
x1 = 100 #[m] where the second part of the bridge is attached 
L0 = 40 #[m] free length of the ropes
k = 100000000 #N / m (value chosen to look decent on the plot) stiffness coefficient of ropes
m = 80470000 #kg #golden gate bridge mass
g = 9.81 #N/kg 
x0 = x1/2 #m if ropes are of the same lentgh and k coefficient, x-axis equilibrium postition will be between their starting points


init = np.array([-5],dtype = 'int64') #this is an initial guess of the starting y coordinate of the mass
def nonlinear(init): #used in fsolve
    y0 = init[0]
    f0 = 2*k*np.absolute(y0)+(2*k*L0*y0)/(x0**2+y0**2)**0.5-m*g #physics calculations, force conservation law
    return f0
y0 = float(fsolve(nonlinear,init)) 
r0 = np.array([x0,y0])  #starting positions
V0 = np.array([0,0]) #starting velocity, we are in equilibrium, so V0 = 0
Y0 = np.hstack((r0,V0)) #x0,y0,Vx0,Vy0 matrix used in solve_ivp
t0 = 0 #[s]
te = 40 #[s] how long simulation should be

def accelerations(t,Y): #Y[0] is x (position), Y[1] us y (position), Y[2] is Vx, and Y[3] is Vy
    R1 = np.sqrt(Y[0]**2+Y[1]**2)
    R2 = np.sqrt((x1-Y[0])**2+Y[1]**2)
    sina = (np.absolute(Y[1]))/R1
    sinb = (np.absolute(Y[1]))/R2
    cosa = Y[0]/R1
    cosb = (x1-Y[0])/R2
    ay = (k*(R1-L0)*sina/m) + (0.8*k*(R2-L0)*sinb/m) - g
    ax =( (-k*(R1-L0)*cosa) + (0.8*k*(R2-L0) * cosb) ) / m
    return np.array([ax,ay])

def movements(t,Y): #solve_ivp syntax
    return np.hstack([Y[2:],accelerations(t,Y)]) #x,y,Vx,Vy into Vx,Vy,ax,ay
sol = solve_ivp(movements,(t0,te),Y0,max_step=0.01) #to obtain even smaller error on the third plot, change maxstep to 0.001 (but please be ready either to make simulation shorter or wait a bit longer)
#thanks to solve_ivp, we can simulate movements of the "bridge" with higher or smaller accuracy
t = sol.t
Y = sol.y

#calculating energies, checking whether law of conservation of energy was not broken

#physics again, formulas for energies 
def potentialenergy(Y): #this value will be negative
    return m * g * (Y[1,:]) 
def kineticenergy(Y):
    return 0.5*m*(Y[2,:]**2+Y[3,:]**2) 
def elasticenergy1(Y):
    return 0.5*k*(((Y[0]**2+Y[1]**2)**0.5)-L0)**2
def elasticenergy2(Y):
    return 0.5*0.8*k*(((x1-Y[0])**2+Y[1]**2)**0.5-L0)**2
Kenergy = kineticenergy(Y)
Penergy = potentialenergy(Y)
Eenergy1 = elasticenergy1(Y)
Eenergy2 = elasticenergy2(Y)

def movement_plot(t,Y): #a nice upgrade would be to make an animation
    #plotting movement of the mass
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    #plotting points where ropes are attached
    ax1.plot(0,0,'xr',label = 'r1start',lw = 100)
    ax1.plot(x1,0,'xr',label = 'r2start',lw = 100)
    #starting position of a mass
    ax1.plot(x0,y0,'xb',label = 'starting position of a mass',lw = 100)

    ax1.plot(Y[0,:],Y[1,:],'-r',label='trajectory of the mass (oscillatory)',lw=0.2)
    ax1.plot(Y[0,-1],Y[1,-1],'xg',label='ending position of the mass',lw = 3)
    xvalues1 = [0,x0]
    xvalues2 = [x1,x0]
    yvalues = [0,y0]
    ax1.plot(xvalues1,yvalues,color = 'black',label = "rope of the bridge")
    ax1.plot(xvalues2,yvalues,color = 'black',label = "rope of the bridge")
    ax1.legend()
    plt.title(f'm={m}[kg], k={k}[N/m],x-coordinate of the second rope={x1}[m], free length of the rope L={L0}[m], time={te}[s] and startpoint {x0,y0}[m]')
    plt.xlabel('axis x [m]')
    plt.ylabel('axis y [m]')
    plt.axis('equal')
    plt.show()

def energy_plot(t,Y):
    #plotting all energies, and their sum (should be constant)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(t,Penergy,label = 'potential energy',lw = 2)
    ax2.plot(t,Kenergy,label = 'kinetic energy',lw = 2)
    ax2.plot(t,Eenergy1,label = 'elasticity energyx1',lw = 2)
    ax2.plot(t,Eenergy2,label = 'elasticity energyy1',lw = 2)
    ener = Kenergy+Penergy+Eenergy1+Eenergy2
    ax2.plot(t,ener,label = 'sum of energies',lw = 2)
    plt.title('Conservation of energy of the bridge during the simulation')
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    ax2.legend()
    plt.show()
def energy_delta(t,Y):
    #checking the difference between the starting energy and relative energy (should be 0, or something really small)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    energysum = Kenergy+Penergy+Eenergy1+Eenergy2
    ax3.plot(t,energysum[0]-energysum,label = "delta energy")
    plt.title(f'Difference between starting energy = {round(energysum[0],0)} J, and relative energy') 
    plt.xlabel('Time [s]')
    plt.ylabel('Starting energy minus relative energy [J]')
    ax3.legend()
    plt.show()

movement_plot(t,Y)
energy_plot(t,Y)
energy_delta(t,Y)

