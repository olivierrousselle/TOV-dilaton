#!/usr/bin/env python
import scipy.constants as cst
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as npla
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz as integcum
from scipy.integrate import trapz as integ
"""
TOV equation solver in the context of Entangled Relativity (ER).
Based on "Compact Objects in Entangled Relativity" (2011.14629.pdf).
"""
c2 = cst.c**2
kappa = 8*np.pi*cst.G/c2**2
k = 1.475*10**(-3)*(cst.fermi**3/(cst.eV*10**6))**(2/3)*c2**(5/3)
massSun = 1.989*10**30

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
    return (1-(c2*m*kappa/(4*np.pi*r)))**(-1) # Eq (29) in 2011.14629.pdf

#Equation for (da/dr)/a
def adota(r, P, m, Psi, Phi):
    A = (b(r, m)/r)
    B = (1-(1/b(r, m))+P*kappa*r**2*Phi**(-1/2)-2*r*Psi/(b(r,m)*Phi))
    C = (1+r*Psi/(2*Phi))**(-1)
    return A*B*C # Eq (17) in 2011.14629.pdf

#Equation for D00
def D00(r, P, m, Psi, Phi, option):
    ADOTA = adota(r, P, m, Psi, Phi)
    rho = RhoEQS(P)
    Lm = Lagrangian(P, option)
    T = -c2*rho + 3*P
    A = Psi*ADOTA/(2*Phi*b(r,m)) #
    B = kappa*(Lm-T)/(3*Phi**(1/2))
    return A+B # Eq (11) in 2011.14629.pdf

#Equation for db/dr
def bdotb(r, P, m, Psi, Phi, option):
    rho = RhoEQS(P)
    A = -b(r,m)/r
    B = 1/r
    C = b(r,m)*r*(-D00(r, P, m, Psi, Phi, option)+kappa*c2*rho*Phi**(-1/2))
    return A+B+C # Eq (10) in 2011.14629.pdf

#Equation for dP/dr
def f1(r, P, m, Psi, Phi, option, P0):
    ADOTA = adota(r, P*P0, m*massSun, Psi, Phi)
    Lm = Lagrangian(P*P0, option)
    rho = RhoEQS(P*P0)
    return -(ADOTA/2)*(P+rho*c2/P0)+(Psi/(2*Phi))*(Lm/P0-P) # Eq (20) in 2011.14629.pdf

#Equation for dm/dr
def f2(r, P, m, Psi, Phi, option,P0):
    rho = RhoEQS(P*P0)
    A = 4*np.pi*rho*(Phi**(-1/2))*r**2
    B = 4*np.pi*(-D00(r, P*P0, m*massSun, Psi, Phi, option)/(kappa*c2))*r**2
    return (A+B)/massSun # Eq (23) in 2011.14629.pdf

#Equation for dPsi/dr
def f4(r, P, m, Psi, Phi, option, dilaton_active,P0):
    ADOTA = adota(r, P*P0, m*massSun, Psi, Phi)
    BDOTB = bdotb(r, P*P0, m*massSun, Psi, Phi, option)
    rho = RhoEQS(P*P0)
    Lm = Lagrangian(P*P0, option)
    T = -c2*rho + 3*P*P0
    A = (-Psi/2)*(ADOTA-BDOTB+4/r)
    B = b(r,m*massSun)*kappa*Phi**(1/2)*(T-Lm)/3
    if dilaton_active:
        return A+B # Eq (21) in 2011.14629.pdf
    else:
        return 0

#Equation for dPhi/dr
def f3(r, P, m, Psi, Phi, option, dilaton_active):
    if dilaton_active:
        return Psi # Eq (24) in 2011.14629.pdf
    else:
        return 0

#Define for dy/dr
def dy_dr(r, y, option, dilaton_active,P0):
    P, M, Phi, Psi = y
    dy_dt = [f1(r, P, M, Psi, Phi, option,P0), f2(r, P, M, Psi, Phi, option,P0),f3(r, P, M, Psi, Phi, option, dilaton_active),f4(r, P, M, Psi, Phi, option, dilaton_active,P0) ]
    return dy_dt

#Define for dy/dr out of the star
def dy_dr_out(r, y, P, option, dilaton_active,P0):
    M, Phi, Psi = y
    dy_dt = [f2(r, P, M, Psi, Phi, option, P0),f3(r, P, M, Psi, Phi, option, dilaton_active),f4(r, P, M, Psi, Phi, option, dilaton_active, P0) ]
    return dy_dt

class TOV():
    """
    * Initialization
        - initDensity: initial (meaning at star center) value of density [MeV/fm3]
        - initPsi: initial value for psi (usually to 1).
        - initPhi: initial value for the derivative of psi (usually taken to 0).
        - radiusMax_in: For star interior, the solver integrates until it reach radiusMax_in.
        - radiusMax_out: For star exterior, the solver integrates until it reach radiusMax_out.
        - Npoint: Number at which the solution is evaluated (t_span in solve_ivp).
        - lag_type: Select lagrangian.
            0 -> Lm=T
            1 -> Lm=-c²rho
            2 -> Lm=P
        - dilaton_active:
            True -> Solves for equation of ER.
            False -> Solves for equation of GR.
        - log_active: Consol outputs.
            True -> activates consol output
    * ComputeTOV
    * Compute
    * Plot
    * PlotMetric
    """
    def __init__(self, initDensity, initPsi, initPhi, radiusMax_in, radiusMax_out, Npoint, lag_type, dilaton_active, log_active):
#Init value
        self.initDensity = initDensity*cst.eV*10**6/(cst.c**2*cst.fermi**3)
        self.initPressure = PEQS(initDensity)
        self.initPsi = initPsi
        self.initPhi = initPhi
        self.initMass = 0
        self.option = lag_type
        self.dilaton_active = dilaton_active
        self.log_active = log_active

#Computation variable
        self.radiusMax_in = radiusMax_in
        self.radiusMax_out = radiusMax_out
        self.Npoint = Npoint
#Star data
        self.Nstar = 0
        self.massStar = 0
        self.massADM = 0
        self.pressureStar = 0
        self.radiusStar = 0
        self.phiStar = 0
#Output data
        self.pressure = 0
        self.mass = 0
        self.Phi = 0
        self.Psi = 0
        self.radius = 0
        self.g_tt = 0
        self.g_rr = 0
        self.g_tt_ext = 0
        self.g_rr_ext = 0
        self.r_ext = 0
        self.phi_inf = 0

    def Compute(self):
        if self.log_active:
            print('===========================================================')
            print('SOLVER INSIDE THE STAR')
            print('===========================================================\n')
            print('Initial density: ', self.initDensity, ' MeV/fm^3')
            print('Initial pressure: ', self.initPressure/10**12, ' GPa')
            print('Initial mass: ', self.initMass/massSun, ' solar mass')
            print('Initial phi: ', self.initPhi)
            print('Initial psi: ', self.initPsi)
            print('Number of points: ', self.Npoint)
            print('Radius max: ', self.radiusMax_in/1000, ' km')
        y0 = [1,self.initMass,self.initPhi,self.initPsi]
        if self.log_active:
            print('y0 = ', y0,'\n')
        r = np.linspace(0.00000001,self.radiusMax_in,self.Npoint)
        if self.log_active:
            print('radius min ',0.00000001)
            print('radius max ',self.radiusMax_in)
        sol = solve_ivp(dy_dr, [0.00000001, self.radiusMax_in], y0, t_eval=r ,args=(self.option,self.dilaton_active,self.initPressure))
        # condition for Pressure = 0
        '''
        self.g_rr = b(sol.t, sol.y[1])
        a_dot_a = adota(sol.t, sol.y[0], sol.y[1], sol.y[3], sol.y[2])
        self.g_tt = np.exp(np.concatenate([[0.0], integcum(a_dot_a,sol.t)])-integ(a_dot_a,sol.t))
        plt.plot(self.g_tt/self.g_rr)
        plt.show()
        '''
        if sol.t[-1]<self.radiusMax_in:
            self.pressure = sol.y[0][0:-2]
            self.mass = sol.y[1][0:-2]
            self.Phi = sol.y[2][0:-2]
            self.Psi = sol.y[3][0:-2]
            self.radius = sol.t[0:-2]
            # Value at the radius of star
            self.massStar = sol.y[1][-1]
            self.radiusStar = sol.t[-1]
            self.pressureStar = sol.y[0][-1]
            self.phiStar = sol.y[2][-1]
            n_star = len(self.radius)
            if self.log_active:
                print('Star radius: ', self.radiusStar/1000, ' km')
                print('Star Mass: ', self.massStar, ' solar mass')
                print('Star Mass: ', self.massStar, ' kg')
                print('Star pressure: ', self.pressureStar, ' /P0\n')
                print('===========================================================')
                print('SOLVER OUTSIDE THE STAR')
                print('===========================================================\n')
            y0 = [self.massStar, sol.y[2][-1],sol.y[3][-1]]
            if self.log_active:
                print('y0 = ', y0,'\n')
            r = np.logspace(np.log(self.radiusStar)/np.log(10),np.log(self.radiusMax_out)/np.log(10),self.Npoint)
            if self.log_active:
                print('radius min ',self.radiusStar)
                print('radius max ',self.radiusMax_out)
            sol = solve_ivp(dy_dr_out, [r[0], self.radiusMax_out], y0, t_eval=r, args=(0,self.option,self.dilaton_active,self.initPressure))
            self.pressure = np.concatenate([self.pressure, np.zeros(self.Npoint)])
            self.mass = np.concatenate([self.mass, sol.y[0]])
            self.Phi = np.concatenate([self.Phi, sol.y[1]])
            self.Psi = np.concatenate([self.Psi,  sol.y[2]])
            self.radius = np.concatenate([self.radius, r])
            self.phi_inf = self.Phi[-1]
            if self.log_active:
                print('Phi at infinity ', self.phi_inf)
            # Compute metrics
            self.g_rr = b(self.radius, self.mass*massSun)
            a_dot_a = adota(self.radius, self.pressure*self.initPressure, self.mass*massSun, self.Psi, self.Phi)
            #plt.plot(self.radius, np.concatenate([[0.0], integcum(a_dot_a,self.radius)]))
            #plt.show()
            self.g_tt = np.exp(np.concatenate([[0.0], integcum(a_dot_a,self.radius)])-integ(a_dot_a,self.radius))
            self.massADM = self.mass[-1]
            self.g_tt_ext = np.array(self.g_tt[n_star:-1])
            self.g_rr_ext = np.array(self.g_rr[n_star:-1])
            self.r_ext = np.array(self.radius[n_star:-1])
            self.r_ext[0] = self.radiusStar
            if self.log_active:
                print('Star Mass ADM: ', self.massADM, ' kg')
                print('===========================================================')
                print('END')
                print('===========================================================\n')
        else:
            print('Pressure=0 not reached')


    def ComputeTOV(self):
        """
        ComputeTOV is the function to consider in order to compute "physical" quantities. It takes into account phi_inf->1 r->ininity
        """
        self.Compute()
        if self.dilaton_active:
            self.initPhi = self.initPhi/self.phi_inf
            self.Compute()

    def Plot(self):
        plt.subplot(221)
        plt.plot([x/10**3 for x in self.radius], [x for x in self.pressure])
        plt.xlabel('Radius r (km)')
        plt.title('Pressure P (Pa)', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.subplot(222)
        plt.plot([x/10**3 for x in self.radius], [x for x in self.mass])
        plt.xlabel('Radius r (km)')
        plt.title('Mass $M/M_{\odot}$', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.subplot(223)
        plt.plot([x/10**3 for x in self.radius], self.Phi)
        plt.xlabel('Radius r (km)')
        plt.title('Dilaton field Φ', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.subplot(224)
        plt.plot([x/10**3 for x in self.radius], self.Psi)
        plt.xlabel('Radius r (km)')
        plt.title('Ψ (derivative of Φ)', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.show()

    def PlotMetric(self):
        plt.subplot(121)
        plt.plot([x/10**3 for x in self.radius], self.g_tt)
        plt.xlabel('Radius r (km)')
        plt.title('g_tt', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.subplot(122)
        plt.plot([x/10**3 for x in self.radius], self.g_rr)
        plt.xlabel('Radius r (km)')
        plt.title('g_rr', fontsize=12)
        plt.axvline(x=self.radiusStar/10**3, color='r')

        plt.show()
