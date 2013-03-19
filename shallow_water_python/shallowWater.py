import numpy as np
import pylab as pl

######################################################
## - usage from command line:                        #
##     python shallowWater.py --n=1000 --TMAX=0.05   #
##     (where:                                       #
##     n = number of grid cells                      #
##     TMAX = end time simulation)                   #
## - usage from within ipython:                      #
##     from shallowWater import *                    #
##     shallowWater(100,1.,0.01)                     #
######################################################

# Courant Friedrichs Levy condition (CFL) is a necessary condition for convergence
CFL = .25# set so it fulfilles CFL condition, experiment with it
# gravitational constant
g = 9.80665

## shallow water solver 1 dimension
def shallowWater(n,XMAX,TMAX):

    # dx = cell size
    dx = 1.*XMAX/n
    # x = cell center
    x = np.arange(dx/2, XMAX, dx)# set to be cell centers
    # initialize height h and momentum hu
    h, hu = initialize(x,XMAX)

    # add ghost cells
    h = addGhostCells(h)
    hu = addGhostCells(hu)

    # plot initial conditions with interactive mode on
    pl.ion()
    plotVars(x,h,hu,0)

    # time starts at zero
    tsum=0.
    # loop over time
    while tsum<TMAX:

        # apply boundary conditions
        h = neumannBoundaryConditions(h)
        hu = neumannBoundaryConditions(hu)
        #h = periodicBoundaryConditions(h)
        #hu = periodicBoundaryConditions(hu)

        # calculate fluxes at cell interfaces and largest eigenvalue
        fhp, fhup, eigp = fluxes(h[1:],hu[1:])
        fhm, fhum, eigm = fluxes(h[:-1],hu[:-1])
        maxeig = np.maximum(eigp, eigm)# maximum of eigp and eigm

        # calculate time step according to CFL-condition
        dt = calculateDt(dx,maxeig,tsum,TMAX)
        # advance time
        tsum = tsum+dt
        lambd = 1.*dt/dx

        # R = Lax-Friedrichs Flux
        Rh = LxFflux(h,fhp, fhm, lambd)
        Rhu = LxFflux(hu, fhup, fhum, lambd)
        print Rhu[np.floor(n / 3):n - np.floor(n / 3)]
        h[1:-1] -= lambd * .5 *(Rh[:-1] - Rh[1:])# update cell average (tip: depends on Rh and lambda)
        hu[1:-1] -= lambd * .5 *(Rhu[:-1] - Rhu[1:])# update cell average (tip: depends on Rhu and lambda)
        plotVars(x,h,hu,tsum)

    #end while (time loop)


def plotVars(x,h,hu,time):
    time = '{0:.5f}'.format(time)
    pl.figure(1)
    pl.clf()
    pl.subplot(211)
    pl.title('water height h(t='+time+')')
    pl.plot(x,h[1:-1],label='u')
    pl.xlabel('x')
    pl.legend()
    pl.subplot(212)
    pl.title('momentum hu(t='+time+')')
    pl.plot(x,hu[1:-1],label='hu')
    pl.xlabel('x')
    pl.legend()
    pl.draw()

def initialize(x,XMAX):
    # initialize water height with two peaks
    c = 0.01
    h = .5 + .5*np.exp(-(x-.6*XMAX)**2/(2*c**2))
    h += np.exp(-(x-.4*XMAX)**2/(2*c**2))
    # momentum is zero
    hu = 0.*x
    return h, hu

def addGhostCells(var):
    return np.hstack([0.,var,0.])

def neumannBoundaryConditions(var):
    var[0] = var[1]
    var[-1] = var[-2]
    return var # var with neumann boundary conditions

def periodicBoundaryConditions(var):
    var[0] = var[-2]
    var[-1] = var[1] 
    return var # var with periodic boundary conditions

def calculateDt(dx,maxeig,tsum,TMAX):
	dt = dx/maxeig
	if tsum + dt > TMAX:
		dt = TMAX - tsum

	return dt #stepsize dtdt

def fluxes(h,hu):
    fh,fhu,lambd1,lambd2 = fluxAndLambda(h,hu)
    maxeig = np.maximum(np.amax(lambd1), np.amax(lambd2))# calculate largest eigenvalue
    return fh, fhu, maxeig

def LxFflux(q, fqp, fqm, lambd):
	LxF = .5 * (q[:-1] + q[1:]) - .5 * lambd * (fqm - fqp)
	return LxF # Lax-Friedrichs Flux

def fluxAndLambda(h,hu):
	u = hu/h
	fh = hu
	fhu = h*u*u + .5*g*h*h
	lambda1 = u + np.sqrt(g*h)
	lambda2 = u - np.sqrt(g*h)
	#fluxes fh and fhu and eigenvalues lambda1, lambda2
	return fh, fhu, lambda1, lambda2

if __name__ == "__main__":

    from optparse import OptionParser
    usage = "usage: %prog [var=value]"
    p = OptionParser(usage)
    p.add_option("--n", type="int", help="number of points")
    p.add_option("--XMAX", type="float", help="length of domain")
    p.add_option("--TMAX", type="float", help="time of simulation")
    (opts, args) = p.parse_args()
    if opts.n == None:
        n = 100
    else:
        n = opts.n
    if opts.XMAX == None:
        XMAX = 1.0
    else:
        XMAX = opts.XMAX
    if opts.TMAX == None:
        TMAX = .02
    else:
        TMAX = opts.TMAX

    shallowWater(n,XMAX,TMAX)
