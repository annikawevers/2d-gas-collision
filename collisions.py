import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import combinations
from scipy.optimize import curve_fit

'''
The following program creates an animation of the collisions of particles
representing a 2-dimensional gas. It then plots the Maxwell-Boltzmann
distribution of speed, and the Boltzmann distribution of energy as
histograms fit with the analytical solutions. Lastly, a text file is created
that identifies the unknown temperature value.
'''

# Define initial conditions
npoint = 400  # number of particles
nframe = 1000  # number of time steps
xmin, xmax, ymin, ymax = 0, 1, 0, 1
Dt = 0.00002  # time step
kb = 1.38*(10**(-23))  # boltzmann constant
m = 2.672*(10**(-26))  # mass
r = 0.0015

fig, ax = plt.subplots()
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)


def update_point(num):  # Given template
    global x, y, vx, vy
    # print(num)
    indx = np.where((x < xmin) | (x > xmax))
    indy = np.where((y < ymin) | (y > ymax))
    vx[indx] = -vx[indx]
    vy[indy] = -vy[indy]
    xx = np.asarray(list(combinations(x, 2)))
    yy = np.asarray(list(combinations(y, 2)))
    dd = (xx[:, 0]-xx[:, 1])**2+(yy[:, 0]-yy[:, 1])**2  # distance

    col = np.where(dd < ((2*r)**2))  # find where the collisions are
    arr = np.arange(0, 400)
    loc = np.asarray(list(combinations(arr, 2)))

    for i in col:  # Updating final velocities after collisions
        p1 = loc[i, 0]
        p2 = loc[i, 1]
        x1 = x[p1]
        x2 = x[p2]
        y1 = y[p1]
        y2 = y[p2]
        v1x = vx[p1]
        v2x = vx[p2]
        v1y = vy[p1]
        v2y = vy[p2]
        dot = (v1x-v2x)*(x1-x2) + (v1y-v2y)*(y1-y2)
        b = ((x1-x2)**2 + (y1-y2)**2)
        vx[p1] = v1x - (dot/b)*(x1-x2)
        vy[p1] = v1y - (dot/b)*(y1-y2)
        vx[p2] = v2x - (dot/b)*(x2-x1)
        vy[p2] = v2y - (dot/b)*(y2-y1)

    dx = Dt*vx
    dy = Dt*vy
    x = x + dx
    y = y + dy
    data = np.stack((x, y), axis=-1)
    im.set_offsets(data)


x = np.random.random(npoint)
y = np.random.random(npoint)
vx = -500.*np.ones(npoint)
vy = np.zeros(npoint)
vx[np.where(x <= 0.5)] = -vx[np.where(x <= 0.5)]
colors = np.array(['r']*npoint)
colors[np.where(x <= 0.5)] = 'b'
im = ax.scatter(x, y, c=colors)
im.set_sizes([10])

animation = animation.FuncAnimation(fig, update_point, nframe,
                                    interval=10, repeat=False)
animation.save('collisions.mp4')
plt.clf()


# Define probability functions
def f(v, T):  # maxwell distribution of speed
    return (m*v/(kb*T))*np.exp(-(m*v**2)/(2*kb*T))


def g(E, T):  # boltzmann distribution
    return (1/(kb*T))*np.exp(-E/(kb*T))


# Plot histograms
vf = np.sqrt((vx**2) + (vy**2))
E = 0.5*m*(vf**2)
mean = np.mean(vf)

# Speed Distribution
plt.subplot(211)
n1, bins1, patches1 = plt.hist(vf, bins=25, label="speed distribution",
                               density=True)
bins1 = bins1[:-1] + (bins1[1] - bins1[0])/2  # midpoint of bins
#  Curve fit with guess for T
T, _ = curve_fit(f, bins1, n1, p0=[245., ])
plt.plot(bins1, f(bins1, T), 'r', label="Fit")
plt.legend()
plt.ylabel('Probability')
plt.xlabel('Velocity (m/s)')
plt.title('Probability Distribution of Speed')

# Energy Distribution
plt.subplot(212)
n2, bins2, patches2 = plt.hist(E, bins=25, label="energy distribution",
                               density=True)
plt.plot(bins2, g(bins2, T), 'r', label="Fit")  # using calculated T
plt.ylabel('Probability')
plt.xlabel('Energy (J)')
plt.title('Probability Distribution of Energy')
plt.legend()

plt.tight_layout()
plt.savefig('distributions.pdf')

# Write sentence giving value of temperature
f = open('collisions.txt', 'w')
f.write('The value found when fitting the speed distribution'
        ' is {:.5} Kelvin.'.format(T[0]))
f.close
