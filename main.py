#!/usr/bin/env python
from TOV import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as cst

def test():
    rhoInit = 10*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    radiusMax = 10000000
    Npoint = 100000
    PsiInit = 0
    PhiInit = 2/2.0005115957706754/1.0000000175430597
    option = 1
    dilaton = True
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax, Npoint, option)
    tov.Compute()
    tov.PlotEvolution()
    print(tov.Phi_infini())
    print(tov.radiusStar)


def plotDensityMass(PsiInit, PhiInit, option, dilaton):
    radiusMax = 50000
    Npoint = 1000
    rhoMin = 10
    rhoMax = 5000

    massStar = []
    rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in range(rhoMin,rhoMax,100)]
    for iRho in rho:
        rhoInit = iRho
        tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax, Npoint, option)
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
    radiusMax = 100000
    Npoint = 1000
    rhoMin = 10
    rhoMax = 100000

    massStar = []
    radiusStar = []
    rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in range(rhoMin,rhoMax,100)]
    for iRho in rho:
        rhoInit = iRho
        tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax, Npoint, option)
        tov.Compute()
        massStar.append(tov.massStar)
        radiusStar.append(tov.radiusStar)
    if dilaton:
        plt.plot([x/1000 for x in radiusStar], [x/(1.989*10**30) for x in massStar], label='$\Phi_0=$%.1f' %PhiInit)
    else:
        plt.plot([x/1000 for x in radiusStar], [x/(1.989*10**30) for x in massStar], label='GR')
    plt.xlabel('Radius R (km)')
    plt.ylabel('Mass $M/M_{\odot}$')
    plt.legend()

def plotMass():
    radiusMax = 100000
    Npoint = 1000
    rhoMin = 10
    rhoMax = 1000
    PsiInit = 0
    option = 1

    Phi = linspace(0.5,6,20)
    rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in linspace(rhoMin,rhoMax,20)]
    massStar = zeros((len(Phi),len(rho)))
    for i in range(len(Phi)):
        print(Phi[i])
        for j in range(len(rho)):
            PhiInit = Phi[i]
            rhoInit = rho[j]
            tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax, Npoint, option)
            tov.Compute()
            massStar[i][j] = tov.massStar
    figure()
    imshow(massStar/(1.989*10**30),extent=[min(rho)/(cst.eV*10**6/(cst.c**2*cst.fermi**3)), max(rho)/(cst.eV*10**6/(cst.c**2*cst.fermi**3)), max(Phi), min(Phi)],aspect='auto', cmap='inferno', origin='upper')
    title("Mass $M/M_{\odot}$")
    ylabel("Dilaton $\Phi_0$")
    xlabel("Density $\\rho$ ($Mev/fm^3$)")
    colorbar()

def plotPhi_infini():
    radiusStep = 100
    rhoMin = 100
    rhoMax = 5000
    radiusInit = 0.000001
    PsiInit = 0
    option = 1
    Phi = linspace(0.5,6,3)
    rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in linspace(rhoMin,rhoMax,3)]
    Phi_infini = zeros((len(Phi),len(rho)))
    for i in range(len(Phi)):
        print(Phi[i])
        for j in range(len(rho)):
            PhiInit = Phi[i]
            rhoInit = rho[j]
            tov = TOV(radiusInit, rhoInit, PsiInit, PhiInit, radiusStep, option)
            tov.Compute()
            Phi_infini[i][j] = tov.Phi_infini()
    figure()
    imshow(Phi_infini,extent=[min(rho)/(cst.eV*10**6/(cst.c**2*cst.fermi**3)), max(rho)/(cst.eV*10**6/(cst.c**2*cst.fermi**3)), min(Phi), max(Phi)], aspect='auto', origin='lower')
    title("Dilaton $\Phi_\u221e$")
    ylabel("Dilaton $\Phi_0$")
    xlabel("Density $\\rho$ ($Mev/fm^3$)")
    colorbar()

def plotPhi():
    rhoInit = 1000*cst.eV*10**6/(cst.c**2*cst.fermi**3)


    Phi0 = linspace(0.5,6,20)
    M_Phi =  zeros((len(Phi0),1000))
    for i in range(len(Phi0)):
        print(Phi0[i])
        PhiInit = Phi0[i]
        tov = TOV(radiusInit, rhoInit, PsiInit, PhiInit, radiusStep, option)
        tov.Compute()
        M = [x for x in tov.Phi_r()]
        for j in range(len(M)):
            M_Phi[i][j] = M[j]-PhiInit
    figure()
    imshow(M_Phi,extent=[0, 1000, min(Phi0), max(Phi0)], aspect='auto', origin='lower')
    title("Dilaton $\Phi$ - $\Phi_0$")
    ylabel("Dilaton $\Phi_0$")
    xlabel("Radius (km)")
    colorbar()

#test()
def main():
    test()
    #plotRadiusMass(0, 1, 1, True)
    #plt.show()
    #plotMass()
    #plotPhi_infini()
    #plotPhi()

main()
