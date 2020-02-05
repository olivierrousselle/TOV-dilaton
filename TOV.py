#!/usr/bin/env python
import scipy.constants as cst
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as npla

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

#Equation for dPhi/dr
def f3(r, P, m, Psi, Phi, option):
    ADOTA = adota(r, P, m, Psi, Phi)
    BDOTB = bdotb(r, P, m, Psi, Phi, option)
    rho = RhoEQS(P)
    Lm = Lagrangian(P, option)
    T = -c2*rho + 3*P
    A = (-Psi/2)*(ADOTA-BDOTB+4/r)
    B = b(r,m)*kappa*Phi**(1/2)*(T-Lm)/3
    return A+B

#Equation for dPsi/dr
def f4(r, P, m, Psi, Phi, option):
    return Psi

class TOV():

    def __init__(self, initRadius, initDensity, initPsi, initPhi, radiusStep, option, dilaton):
#Init value
        self.initDensity = initDensity
        self.initPressure = PEQS(initDensity)
        self.initPsi = initPsi
        self.initPhi = initPhi
        self.initMass = 0
        self.initRadius = initRadius
#Computation variable
        self.limitCompute = 5000
        self.radiusStep = radiusStep
#Star data
        self.massStar = 0
        self.pressureStar = 0
        self.radiusStar = 0
        self.stepStar = 0
        self.LambdaStar = 0
#Output vector
        self.pressure = 0
        self.mass = 0
        self.Phi = 0
        self.Psi = 0
        self.metric00 = 0
        self.metric11 = 0
        self.radius = 0

        self.PhiInf = 0
        self.option = option
        self.dilaton = dilaton

    def Compute(self):
# Initialisation ===================================================================================
        dr = self.radiusStep
        n = 0
        P =        [self.initPressure]
        m =        [self.initMass]
        Psi =      [self.initPsi]
        Phi =      [self.initPhi]
        r =        [self.initRadius]
        metric11 = [b(r[0],m[0])]
        metric00 = [1]      # Coordinate time as proper time at center
        # Inside the star----------------------------------------------------------------------------
        while(P[n]>10**26):
            if(n == self.limitCompute):
                break
                print('end')
            else:
                P.append(     P[n]   + dr*f1(r[n], P[n], m[n], Psi[n], Phi[n], self.option ))
                m.append(     m[n]   + dr*f2(r[n], P[n], m[n], Psi[n], Phi[n], self.option ))
                if self.dilaton:
                    Phi.append(   Phi[n] + dr*f4(r[n], P[n], m[n], Psi[n], Phi[n], self.option ))
                    Psi.append(   Psi[n] + dr*f3(r[n], P[n], m[n], Psi[n], Phi[n], self.option ))
                else:
                    Phi.append(   Phi[n] )
                    Psi.append(   Psi[n] )
                metric11.append(b(r[n], m[n]))
                metric00.append(metric00[n] + dr*metric00[n]*adota(r[n], P[n], m[n], Psi[n], Phi[n]))
                n = n+1
                r.append(self.initRadius+n*dr)
        P.pop()
        m.pop()
        Psi.pop()
        Phi.pop()
        r.pop()
        metric11.pop()
        metric00.pop()
        n = n-1
        self.pressure = P #star pressure
        self.mass = m #star mass
        self.stepStar = n #index where code stop at 0 pressure

        # Outside the star--------------------------------------------------------------------------
        while(n<self.limitCompute):
            if self.dilaton:
                Phi.append( Phi[n] + dr*f4(r[n], 0, m[n], Psi[n], Phi[n], self.option ))
                Psi.append( Psi[n] + dr*f3(r[n], 0, m[n], Psi[n], Phi[n], self.option ))
            else:
                Phi.append( Phi[n] )
                Psi.append( Psi[n] )
            P.append( P[n] + dr*f1(r[n], P[n], m[n], Psi[n], Phi[n], self.option ))
            m.append( m[n] + dr*f2(r[n], 0, m[n], Psi[n], Phi[n], self.option ))
            metric11.append(b(r[n], m[n]))
            metric00.append(metric00[n] + dr*metric00[n]*adota(r[n],0, m[n], Psi[n], Phi[n]))
            n = n+1
            r.append(self.initRadius+n*dr)

        # Star property
        self.massStar = self.mass[self.stepStar]
        self.radiusStar = r[self.stepStar]
        self.pressureStar = self.pressure[self.stepStar]
        self.Psi = Psi
        self.Phi = Phi
        self.radius = r
        self.metric11 = metric11
        self.metric00 = metric00

    def ComputeGR(self):
        dr = self.radiusStep
        n = 0 #Integration parameter
        P =        [self.initPressure]
        m =        [self.initMass]
        r =        [self.initRadius]
        metric11 = [b(r[0],m[0])]
        metric00 = [1]      # Coordinate time as proper time at center
        while(P[n]>10**26):
            if(n == self.limitCompute):
                break
                print('end')
            else:
                P.append( P[n] + dr*f1(r[n], P[n], m[n], 0, 1, self.option ))
                m.append( m[n] + dr*f2(r[n], P[n], m[n], 0, 1, self.option ))
                metric11.append(b(r[n], m[n]))
                metric00.append(metric00[n] + dr*metric00[n]*adota(r[n],0, m[n], 0, 1))
                n = n+1
                r.append(self.initRadius+n*dr)
        P.pop()
        m.pop()
        r.pop()
        metric00.pop()
        metric11.pop()
        n = n-1
        self.massStar = m[n]
        self.radiusStar = r[n]
        self.pressureStar = P[n]
        self.pressure = P
        self.mass = m
        self.metric00 = metric00
        self.metric11 = metric11


    def PlotEvolution(self):
        plt.figure()
        plt.plot([x/10**3 for x in self.radius[0:self.stepStar*2]], [x for x in self.pressure[0:self.stepStar*2]])
        plt.xlabel('Radius r (km)')
        plt.title('Pressure P (Pa)', fontsize=12)

        plt.figure()
        plt.plot([x/10**3 for x in self.radius[0:self.stepStar*2]], [x/(1.989*10**30) for x in self.mass[0:self.stepStar*2]])
        plt.xlabel('Radius r (km)')
        plt.title('Mass $M/M_{\odot}$', fontsize=12)

        plt.figure()
        plt.plot([x/10**3 for x in self.radius[0:self.stepStar*20]], self.Phi[0:self.stepStar*20])
        plt.xlabel('Radius r (km)')
        plt.title('Dilaton field Φ', fontsize=12)

        plt.figure()
        plt.plot([x/10**3 for x in self.radius[0:self.stepStar*20]], self.Psi[0:self.stepStar*20])
        plt.xlabel('Radius r (km)')
        plt.title('Ψ (derivative of Φ)', fontsize=12)

        plt.figure()
        ax3 = plt.subplot(1,2,1)
        plt.plot([x/10**3 for x in self.radius], self.metric00)
        plt.xlabel('Radius r (km)')
        plt.ylabel('a')

        ax4 = plt.subplot(1,2,2)
        plt.plot([x/10**3 for x in self.radius], self.metric11)
        plt.xlabel('Radius r (km)')
        plt.ylabel('b')

        plt.show()

    def Phi_infini(self):
        return self.Phi[self.stepStar*10]

    def Phi_r(self):
        M = np.zeros(len(self.Phi))
        for i in range(len(self.Phi)):
            M[i] = self.Phi[i]-self.initPhi
        plt.plot([x/10**3 for x in self.radius[:]], M)
