#!/usr/bin/env python
import scipy.constants as cst
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as npla
from scipy.integrate import odeint as solv

c2 = cst.c**2
kappa = 8*np.pi*cst.G/c2**2
k = 1.475*10**(-3)*(cst.fermi**3/(cst.eV*10**6))**(2/3)*c2**(5/3)

#Equation of state
def PEQS(rho):
    return k*rho**(5/3)

#Inverted equation of state
def RhoEQS(P):
    return (P/k)**(3/5)

#Lagrangian
def Lagrangian(P, option):
    rho = RhoEQS(P)
    if option == 0:
        return -c2*rho+3*P
    elif option == 1:
        return -c2*rho
    elif option == 2:
        return P
    else:
        print('not a valid option')

#Equation for b
def b(r, m):
    return (1-(c2*m*kappa/(4*np.pi*r)))**(-1)

#Equation for da/dr
def adota(r, P, m, Psi, Phi):
    A = (b(r, m)/r)
    B = (1-(1/b(r, m))+P*kappa*r**2*Phi**(-1/2)-2*r*Psi/(b(r,m)*Phi))
    C = (1+r*Psi/(2*Phi))**(-1)
    return A*B*C

#Equation for D00
def D00(r, P, m, Psi, Phi, option):
    ADOTA = adota(r, P, m, Psi, Phi)
    rho = RhoEQS(P)
    Lm = Lagrangian(P, option)
    T = -c2*rho + 3*P
    A = Psi*ADOTA/(2*Phi*b(r,m))
    B = kappa*(Lm-T)/(3*Phi**(1/2))
    return A+B

#Equation for db/dr
def bdotb(r, P, m, Psi, Phi, option):
    rho = RhoEQS(P)
    A = -b(r,m)/r
    B = 1/r
    C = b(r,m)*r*(-D00(r, P, m, Psi, Phi, option)+kappa*c2*rho*Phi**(-1/2))
    return A+B+C

#Equation for dP/dr
def f1(r, P, m, Psi, Phi, option):
    ADOTA = adota(r, P, m, Psi, Phi)
    Lm = Lagrangian(P, option)
    rho = RhoEQS(P)
    return -(ADOTA/2)*(P+rho*c2)+(Psi/(2*Phi))*(Lm-P)

#Equation for dm/dr
def f2(r, P, m, Psi, Phi, option):
    rho = RhoEQS(P)
    A = 4*np.pi*rho*(Phi**(-1/2))*r**2
    B = 4*np.pi*(-D00(r, P, m, Psi, Phi, option)/(kappa*c2))*r**2
    return A+B

#Equation for dPsi/dr
def f4(r, P, m, Psi, Phi, option):
    ADOTA = adota(r, P, m, Psi, Phi)
    BDOTB = bdotb(r, P, m, Psi, Phi, option)
    rho = RhoEQS(P)
    Lm = Lagrangian(P, option)
    T = -c2*rho + 3*P
    A = (-Psi/2)*(ADOTA-BDOTB+4/r)
    B = b(r,m)*kappa*Phi**(1/2)*(T-Lm)/3
    return A+B

#Equation for dPhi/dr
def f3(r, P, m, Psi, Phi, option):
    return Psi

#Define for dy/dr
def dy_dr(y, r, option):
    P, M, Phi, Psi = y
    dy_dt = [f1(r, P, M, Psi, Phi, option), f2(r, P, M, Psi, Phi, option),f3(r, P, M, Psi, Phi, option),f4(r, P, M, Psi, Phi, option) ]
    return dy_dt

#Define for dy/dr out of the star
def dy_dr_out(y, r, P, option):
    M, Phi, Psi = y
    dy_dt = [f2(r, P, M, Psi, Phi, option),f3(r, P, M, Psi, Phi, option),f4(r, P, M, Psi, Phi, option) ]
    return dy_dt

class TOV():

    def __init__(self, initDensity, initPsi, initPhi, radiusMax, Npoint, option):
#Init value
        self.initDensity = initDensity
        self.initPressure = PEQS(initDensity)
        self.initPsi = initPsi
        self.initPhi = initPhi
        self.initMass = 0.000000000000001
        self.option = option
#Computation variable
        self.radiusMax = radiusMax
        self.Npoint = Npoint
#Star data
        self.Nstar = 0
        self.massStar = 0
        self.pressureStar = 0
        self.radiusStar = 0
#Output vector
        self.pressure = 0
        self.mass = 0
        self.Phi = 0
        self.Psi = 0
        self.radius = 0
        self.metric11 = 0
        self.metric00 = 0

    def Compute(self):
        y0 = [self.initPressure,self.initMass,self.initPhi,self.initPsi]
        r = np.linspace(0.01,self.radiusMax,self.Npoint)

        # Inside the star----------------------------------------------------------------------------
        sol = solv(dy_dr, y0, r, args=(self.option,))
        i = 0
        while(sol[i,0]-sol[i+1,0]>0 and sol[i,0]>10**10):
            i = i+1
        self.pressure = sol[0:i:1,0]
        self.Nstar = np.size(self.pressure)
        self.mass = sol[0:i:1,1]
        self.Phi = sol[0:i:1,2]
        self.Psi = sol[0:i:1,3]
        self.radius = r[0:i:1]
        # Value at the radius of star----------------------------------------------------------------
        self.massStar = self.mass[-1]
        self.radiusStar = self.radius[-1]
        self.pressureStar = self.pressure[-1]
        # Outside the star---------------------------------------------------------------------------
        y0 = [self.massStar, self.Phi[-1],self.Psi[-1]]
        r = np.linspace(self.radiusStar,self.radiusMax,self.Npoint)
        sol = solv(dy_dr_out, y0, r, args=(0,self.option))
        self.pressure = np.concatenate([self.pressure, np.zeros(self.Npoint)])
        self.mass = np.concatenate([self.mass, sol[:,0]])
        self.Phi = np.concatenate([self.Phi, sol[:,1]])
        self.Psi = np.concatenate([self.Psi,  sol[:,2]])
        self.radius = np.concatenate([self.radius, r])
        self.metric11 = b(self.radius, self.mass)
        #self.metric00 =

    def PlotEvolution(self):
        plt.subplot(221)
        plt.plot([x/10**3 for x in self.radius], [x for x in self.pressure])
        plt.xlabel('Radius r (km)')
        plt.title('Pressure P (Pa)', fontsize=12)
        plt.axvline(x=self.radiusStar, '-r')

        plt.subplot(222)
        plt.plot([x/10**3 for x in self.radius], [x/(1.989*10**30) for x in self.mass])
        plt.xlabel('Radius r (km)')
        plt.title('Mass $M/M_{\odot}$', fontsize=12)
        plt.axvline(x=self.radiusStar, '-r')

        plt.subplot(223)
        plt.plot([x/10**3 for x in self.radius], self.Phi)
        plt.xlabel('Radius r (km)')
        plt.title('Dilaton field Φ', fontsize=12)
        plt.axvline(x=self.radiusStar, '-r')

        plt.subplot(224)
        plt.plot([x/10**3 for x in self.radius], self.Psi)
        plt.xlabel('Radius r (km)')
        plt.title('Ψ (derivative of Φ)', fontsize=12)
        plt.axvline(x=self.radiusStar, '-r')

        plt.show()



    def Phi_infini(self):
        return self.Phi[-1]

    def Phi_r(self):
        M = np.zeros(len(self.Phi))
        for i in range(len(self.Phi)):
            M[i] = self.Phi[i]-self.initPhi
        plt.plot([x/10**3 for x in self.radius[:]], M)
