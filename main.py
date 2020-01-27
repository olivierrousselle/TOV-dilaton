#!/usr/bin/env python
from TOV import *
import matplotlib 
import matplotlib.pyplot as plt
import scipy.constants as cst

def test():
    rhoInit = 1000*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    radiusStep = 100
    PsiInit = 0
    PhiInit = 1
    radiusInit = 0.000001
    option = 1
    dilaton = True
    tov = TOV(radiusInit, rhoInit, PsiInit, PhiInit, radiusStep, option, dilaton)
    tov.Compute()
    tov.PlotEvolution()
    print(tov.radiusStar)
    print(tov.massStar)
    print(tov.pressureStar)
    print(tov.initPressure)

def plotDensityMass(PsiInit, PhiInit, option, dilaton):
    radiusStep = 100
    rhoMin = 100
    rhoMax = 5000
    radiusInit = 0.000001

    massStar = []
    rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in linspace(rhoMin,rhoMax,100)]
    for iRho in rho:
        rhoInit = iRho
        tov = TOV(radiusInit, rhoInit, PsiInit, PhiInit, radiusStep, option, dilaton)
        tov.Compute()
        massStar.append(tov.massStar)
    if dilaton:
        plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/(1.989*10**30) for x in massStar], label='$\Phi_0=$%.1f' %PhiInit)
    else:
        plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/(1.989*10**30) for x in massStar], label='GR')        
    plt.xlabel('Density $\\rho$ ($Mev/fm^3$)')
    plt.ylabel('Mass $M/M_{\odot}$')
    plt.legend()
        
def plotRadiusMass(PsiInit, PhiInit, option, dilaton):
    radiusStep = 100
    rhoMin = 100
    rhoMax = 5000
    radiusInit = 0.000001
    massStar = []
    radiusStar = []
    rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in linspace(rhoMin,rhoMax,100)]
    for iRho in rho:
        rhoInit = iRho
        tov = TOV(radiusInit, rhoInit, PsiInit, PhiInit, radiusStep, option, dilaton)
        tov.Compute()
        massStar.append(tov.massStar)
        radiusStar.append(tov.radiusStar)
    if dilaton:
        plt.plot([x/1000 for x in radiusStar], [x/(1.989*10**30) for x in massStar], label='$\Phi_0=$%.1f' %PhiInit)
    else:
        plt.plot([x/1000 for x in radiusStar], [x/(1.989*10**30) for x in massStar], label='GR')
    plt.xlabel('Radius $R_*$ (km)')
    plt.ylabel('Mass $M/M_{\odot}$')
    plt.legend()

def plotMass():
    radiusStep = 100
    rhoMin = 100
    rhoMax = 5000
    radiusInit = 0.000001
    PsiInit = 0
    option = 1
    dilaton = True
    Phi = linspace(1,6,50)
    rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in linspace(rhoMin,rhoMax,50)]
    massStar = zeros((len(Phi),len(rho)))    
    for i in range(len(Phi)):
        print(Phi[i])
        for j in range(len(rho)):
            PhiInit = Phi[i]
            rhoInit = rho[j]
            tov = TOV(radiusInit, rhoInit, PsiInit, PhiInit, radiusStep, option, dilaton)
            tov.Compute()
            massStar[i][j] = tov.massStar
    figure()
    imshow(massStar/(1.989*10**30),extent=[min(rho)/(cst.eV*10**6/(cst.c**2*cst.fermi**3)), max(rho)/(cst.eV*10**6/(cst.c**2*cst.fermi**3)), min(Phi), max(Phi)],aspect='auto', origin='lower')
    title("Mass $M/M_{\odot}$")
    ylabel("Dilaton $\Phi_0$")
    xlabel("Density $\\rho$ ($Mev/fm^3$)")    
    colorbar()
    
def plotPhi(PsiInit, PhiInit, option, dilaton):
    rhoInit = 1000*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    radiusStep = 100
    radiusInit = 0.000001
    tov = TOV(radiusInit, rhoInit, PsiInit, PhiInit, radiusStep, option, dilaton)
    tov.Compute()    
    tov.Phi_r()
    plt.xlabel('Radius r (km)')
    plt.title('Dilaton field $\Phi$ - $\Phi_0$', fontsize=12)
    xlim(0,450)

def main():
    
    test()
    
    """figure()
    plotDensityMass(0, 1, 0, False)
    plotDensityMass(0, 1, 1, True)
    plotDensityMass(0, 2, 1, True)
     plotDensityMass(0, 4, 1, True)
    plotDensityMass(0, 6, 1, True)
    plt.show()
    
    figure()
    plotRadiusMass(0, 1, 0, False)
    plotRadiusMass(0, 1, 1, True)
    plotRadiusMass(0, 2, 1, True)
    plotRadiusMass(0, 4, 1, True)
    plotRadiusMass(0, 6, 1, True)
    plt.show()
    
    plotMass()
    
    plt.figure()
    plotPhi(0, 1, 1, True)
    plotPhi(0, 2, 1, True)
    plotPhi(0, 3, 1, True)
    plotPhi(0, 4, 1, True)
    plotPhi(0, 5, 1, True)
    plotPhi(0, 6, 1, True)
    legend(('$\Phi_0=$1', '$\Phi_0=$2','$\Phi_0=$3','$\Phi_0=$4','$\Phi_0=$5','$\Phi_0=$6'))
    """
    
main()
