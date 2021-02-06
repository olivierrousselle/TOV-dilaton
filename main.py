#!/usr/bin/env python
from TOV_2 import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as cst
import numpy as np
import math

def unit_test():
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 40000
    radiusMax_out = 100000
    Npoint = 100000
    log_active = True
    dilaton_active = False
    rhoInit = 100*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active, log_active)
    tov.ComputeTOV()

    tov.Plot()
    tov.PlotMetric()
    plt.axvline(23532/1000, color='r')
    r = tov.radius
    a = tov.g_tt
    b = tov.g_rr
    phi_dot = tov.Psi
    a_dot = (-a[1:-2]+a[2:-1])/(r[2:-1]-r[1:-2])
    b_dot = (-b[1:-2]+b[2:-1])/(r[2:-1]-r[1:-2])

    plt.plot(r[1:-2]/1000,-a_dot*r[1:-2]*r[1:-2]/1000, label='$f_a$')
    plt.plot(r[1:-2]/1000,-b_dot*r[1:-2]*r[1:-2]/1000, label='$f_b$')
    plt.plot(r/1000,-phi_dot*r*r/1000, label='$f_\Phi$')
    plt.legend()
    plt.xlabel('radius [km]')
    plt.ylabel('[km]')
    plt.show()
    plt.plot(a_dot, marker = 'o')
    plt.show()

    p = tov.pressure
    plt.plot(r[1:-2],(-p[1:-2]+p[2:-1])/(r[2:-1]-r[1:-2]),'o')
    plt.axvline(tov.radiusStar)
    plt.show()

def findSameMass(mass):
    """
    input:
    - mass: mass in solar mass
    output:
    - radius
    """
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 50000
    radiusMax_out = 10000000
    Npoint = 50000
    "density in MeV/fm^3"
    rhoMin = 100
    rhoMax = 8200
    log_active = False
    if 0:
        "Log scale"
        Npoint_rho = 50 #<- number of points
        rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in np.logspace(np.log10(rhoMin),np.log10(rhoMax),num=Npoint_rho)]
    else:
        "Linear scale"
        rhoStep = 50 #<- step in MeV/fm^3
        rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in range(rhoMin,rhoMax,rhoStep)]
    massStar_GR = np.array([])
    massStar_ER = np.array([])
    radiusStar_GR = np.array([])
    radiusStar_ER = np.array([])
    for iRho in rho:
        rhoInit = iRho
        tov = TOV(iRho, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, False, log_active)
        tov.ComputeTOV()
        massStar_GR = np.append(massStar_GR,tov.massStar)
        tov = TOV(iRho, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, True, log_active)
        tov.ComputeTOV()
        massStar_ER = np.append(massStar_ER,tov.massStar)
    rho = np.array(rho)/(cst.eV*10**6/(cst.c**2*cst.fermi**3))
    # ER
    massStar_ER = massStar_ER/(1.989*10**30)-mass
    bool_sup_zero_ER = (massStar_ER>0)
    ind = np.where((bool_sup_zero_ER[1:-1] == bool_sup_zero_ER[0:-2]) == False)
    rho_ER = rho[ind]
    sol_acc_ER = 100*massStar_ER[ind]/mass
    if len(rho_ER)==0:
        print('no solution')
    else:
        print('density in [MeV/fm^3]: ', rho_ER)
        print('mass accuracy un %: ', sol_acc_ER)
    # GR
    massStar_GR = massStar_GR/(1.989*10**30)-mass
    bool_sup_zero_GR = (massStar_GR>0)
    ind = np.where((bool_sup_zero_GR[1:-1] == bool_sup_zero_GR[0:-2]) == False)
    rho_GR = rho[ind]
    sol_acc_GR = 100*massStar_GR[ind]/mass
    if len(rho_GR)==0:
        print('no solution')
    else:
        print('density in [MeV/fm^3]: ', rho_GR)
        print('mass accuracy un %: ', sol_acc_GR)

    plt.scatter(rho, massStar_ER+mass)
    for i in range(len(rho_ER)):
        plt.axvline(rho_ER[i])
    plt.axhline(mass)
    plt.show()

def plotRelation():
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 50000
    radiusMax_out = 10000000
    Npoint = 50000
    rhoMin = 100
    rhoMax = 8200
    log_active = False
    massStar_GR = []
    massADM_ER = []
    massStar_ER = []
    radiusStar_GR = []
    radiusStar_ER = []
    rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in range(rhoMin,rhoMax,100)]
    for iRho in rho:
        rhoInit = iRho
        tov = TOV(iRho, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, False, log_active)
        tov.ComputeTOV()
        massStar_GR.append(tov.massStar)
        radiusStar_GR.append(tov.radiusStar)
        tov = TOV(iRho, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, True, log_active)
        tov.ComputeTOV()
        massStar_ER.append(tov.massStar)
        massADM_ER.append(tov.massADM)
        radiusStar_ER.append(tov.radiusStar)
    plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/(1.989*10**30) for x in massStar_GR], label='GR')
    plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/(1.989*10**30) for x in massStar_ER], label='ER')
    plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/(1.989*10**30) for x in massADM_ER], label='ER ADM')
    plt.xlabel('Density $\\rho$ ($Mev/fm^3$)')
    plt.ylabel('Mass $M/M_{\odot}$')
    plt.legend()
    plt.show()
    plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/1000 for x in radiusStar_GR], label='GR')
    plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/1000 for x in radiusStar_ER], label='ER')
    plt.xlabel('Density $\\rho$ ($Mev/fm^3$)')
    plt.ylabel('Radius (km)')
    plt.legend()
    plt.show()
    plt.plot([x/1000 for x in radiusStar_GR], [x/(1.989*10**30) for x in massStar_GR], label='GR')
    plt.plot([x/1000 for x in radiusStar_ER], [x/(1.989*10**30) for x in massStar_ER], label='ER')
    plt.plot([x/1000 for x in radiusStar_ER], [x/(1.989*10**30) for x in massADM_ER], label='ER ADM')
    plt.xlabel('Radius (km)')
    plt.ylabel('Mass $M/M_{\odot}$')
    plt.legend()
    plt.show()

def plotFig1_2():
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 50000
    radiusMax_out = 10000000
    Npoint = 50000
    rhoMin = 100
    rhoMax = 8200
    log_active = False
    massStar_GR = []
    massStar_ER = []
    radiusStar_GR = []
    radiusStar_ER = []
    rho = [x*cst.eV*10**6/(cst.c**2*cst.fermi**3) for x in range(rhoMin,rhoMax,5)]
    for iRho in rho:
        rhoInit = iRho
        tov = TOV(iRho, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, False, log_active)
        tov.ComputeTOV()
        massStar_GR.append(tov.massStar)
        radiusStar_GR.append(tov.radiusStar)
        tov = TOV(iRho, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, True, log_active)
        tov.ComputeTOV()
        massStar_ER.append(tov.massStar)
        radiusStar_ER.append(tov.radiusStar)
    massStar_GR = [x/(1.989*10**30) for x in massStar_GR]
    massStar_ER = [x/(1.989*10**30) for x in massStar_ER]
    rho = [x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho]
    plt.plot(rho, massStar_GR, label='General Relativity $M_{max}$ = '+str(round(max(massStar_GR),1))+': $\\rho_0$ ='+str(rho[massStar_GR.index(max(massStar_GR))]))
    plt.plot(rho, massStar_ER, label='Entangled Relativity $M_{max}$ = '+str(round(max(massStar_ER),1))+': $\\rho_0$ ='+str(rho[massStar_ER.index(max(massStar_ER))]))
    plt.xlabel('Density $\\rho$ ($Mev/fm^3$)')
    plt.ylabel('Mass $M/M_{\odot}$')
    plt.axvline(x=rho[massStar_GR.index(max(massStar_GR))])
    plt.axvline(x=rho[massStar_ER.index(max(massStar_ER))])
    plt.legend()
    plt.show()
    plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/1000 for x in radiusStar_GR], label='GR')
    plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/1000 for x in radiusStar_ER], label='ER')
    plt.xlabel('Density $\\rho$ ($Mev/fm^3$)')
    plt.ylabel('Radius (km)')
    plt.legend()
    plt.show()
    plt.plot([x/1000 for x in radiusStar_GR], [x/(1.989*10**30) for x in massStar_GR], label='GR')
    plt.plot([x/1000 for x in radiusStar_ER], [x/(1.989*10**30) for x in massStar_ER], label='ER')
    plt.xlabel('Radius (km)')
    plt.ylabel('Mass $M/M_{\odot}$')
    plt.legend()
    plt.show()

def plotFig4():
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 50000
    radiusMax_out = 10000000
    Npoint = 50000
    rhoMin = 100
    rhoMax = 8200
    log_active = False
    rho = np.array([606.25, 2023.75, 403.75, 4150.0])
    labels = ['GR (%.2f)' %rho[0], 'GR (%.2f)' %rho[1], 'ER (%.2f)' %rho[2], 'ER (%.2f)' %rho[3]]
    rho = rho*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    dilaton_active = [False, False, True, True]
    colors = ['blue', 'green', 'red', 'orange']
    linestyles = ['-', '-', '--', '--']
    for i in range(4):
        tov = TOV(rho[i], PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active[i], log_active)
        tov.ComputeTOV()
        plt.plot(tov.radius/1000, tov.g_rr*tov.g_tt, color=colors[i], linestyle=linestyles[i], label = labels[i])
        plt.axvline(tov.radiusStar/1000, color=colors[i], linestyle=linestyles[i])
    plt.axis([5, 40, 0.2, 1.2])
    plt.legend()
    plt.show()

def plotFig5():
    PhiInit = 1
    PsiInit = 0
    option = 1
    radiusMax_in = 50000
    radiusMax_out = 10000000
    Npoint = 50000
    rhoMin = 100
    rhoMax = 8200
    log_active = False
    rho = np.array([606.25, 2023.75, 403.75, 4150.0])
    labels = ['GR (%.2f)' %rho[0], 'GR (%.2f)' %rho[1], 'ER (%.2f)' %rho[2], 'ER (%.2f)' %rho[3]]
    rho = rho*cst.eV*10**6/(cst.c**2*cst.fermi**3)
    dilaton_active = [False, False, True, True]
    colors = ['blue', 'green', 'red', 'orange']
    linestyles = ['-', '-', '--', '--']
    for i in range(4):
        tov = TOV(rho[i], PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active[i], log_active)
        tov.ComputeTOV()
        plt.plot(tov.radius/1000, np.sqrt(tov.g_rr/tov.g_tt), color=colors[i], linestyle=linestyles[i], label = labels[i])
        plt.axvline(tov.radiusStar/1000, color=colors[i], linestyle=linestyles[i])
    plt.axis([0, 100, 1.0, 2.75])
    plt.legend()
    plt.show()


def main():
    unit_test()
    #plotFig1_2()
    #plotRelation()
    #findSameMass(1.25)
    #plotFig4()
    #plotFig5()

main()
