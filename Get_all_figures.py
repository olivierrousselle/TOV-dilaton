from TOV import *

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.constants as cst
import numpy as np

import os

dir = os.getcwd()+'/figures'
if not os.path.exists(dir):
    os.makedirs(dir)

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
    return rho_ER, rho_GR




rho_ER, rho_GR = findSameMass(1.25)




PhiInit = 1
PsiInit = 0
option = 1
radiusMax_in = 50000
radiusMax_out = 10000000
Npoint = 50000
rhoMin = 100
rhoMax = 8200
log_active = False
rho = np.array([rho_GR[0], rho_GR[1], rho_ER[0], rho_ER[1]])


labels = ['GR (%.0f)' %rho[0], 'GR (%.0f)' %rho[1], 'ER (%.0f)' %rho[2], 'ER (%.0f)' %rho[3]]
rho = rho*cst.eV*10**6/(cst.c**2*cst.fermi**3)
dilaton_active = [False, False, True, True]
colors = ['blue', 'green', 'red', 'orange']
linestyles = ['-', '-', '--', '--']
for i in range(4):
    tov = TOV(rho[i], PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active[i], log_active)
    tov.ComputeTOV()
    arr = tov.radius/1000
    rad = tov.radiusStar/1000
    idx = min(range(len(arr)), key=lambda l: abs(arr[l]-rad))
    plt.plot(arr[idx:], np.sqrt(tov.g_rr/tov.g_tt)[idx:], color=colors[i], linestyle=linestyles[i], label = labels[i])
    plt.axvline(rad, color=colors[i], linestyle=linestyles[i])
plt.axis([8, 30, 1.0, 1.8])
plt.title('Shapiro potential = $\sqrt{g_{rr}/g_{00}}$')
plt.xlabel('km')
plt.legend()
plt.savefig(f'figures/shapiro_pot.png', dpi = 200)
#plt.show()
plt.close()


for i in range(4):
    tov = TOV(rho[i], PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active[i], log_active)
    tov.ComputeTOV()
    plt.plot(tov.radius/1000, tov.g_rr*tov.g_tt, color=colors[i], linestyle=linestyles[i], label = labels[i])
    plt.axvline(tov.radiusStar/1000, color=colors[i], linestyle=linestyles[i])
plt.axis([5, 40, 0.2, 1.2])
plt.title('$g_{rr}*g_{00}$')
plt.legend()
plt.savefig(f'figures/g00grr.png', dpi = 200)
# plt.show()
plt.close()




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
massADM_ER = []
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




plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/(1.989*10**30) for x in massStar_ER], color = 'tab:blue', label=f'Entangled Relativity, Lm = $-\\rho$\n M_Max = {np.max(massStar_ER)/(1.989*10**30):.2f}, $\\rho_0$ = {rho[massStar_ER.index(np.max(massStar_ER))]/(cst.eV*10**6/(cst.c**2*cst.fermi**3)):.0f} MeV/fm^3')
plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/(1.989*10**30) for x in massStar_GR], color = 'tab:orange', label=f'General Relativity\n M_Max = {np.max(massStar_GR)/(1.989*10**30):.2f}, $\\rho_0$ = {rho[massStar_GR.index(np.max(massStar_GR))]/(cst.eV*10**6/(cst.c**2*cst.fermi**3)):.0f} MeV/fm^3')
plt.axvline(x = rho[massStar_ER.index(np.max(massStar_ER))]/(cst.eV*10**6/(cst.c**2*cst.fermi**3)), color = 'tab:blue', linestyle='dashed')
plt.axvline(x = rho[massStar_GR.index(np.max(massStar_GR))]/(cst.eV*10**6/(cst.c**2*cst.fermi**3)), color = 'tab:orange', linestyle='dashed')
plt.axhline(y = np.max(massStar_ER)/(1.989*10**30), color = 'tab:blue', linestyle='dashed')
plt.axhline(y = np.max(massStar_GR)/(1.989*10**30), color = 'tab:orange', linestyle='dashed')
plt.xlabel('Density $\\rho$ ($Mev/fm^3$)')
plt.ylabel('Mass $M/M_{\odot}$')
plt.legend()
plt.savefig(f'figures/Fig_Mvsrho_GRvsER.png', dpi = 200)
# plt.show()
plt.close()

plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/1000 for x in radiusStar_ER], label='Entangled Relativity, Lm = $-\\rho$')
plt.plot([x/(cst.eV*10**6/(cst.c**2*cst.fermi**3)) for x in rho], [x/1000 for x in radiusStar_GR], label='General Relativity')
plt.xlabel('Density $\\rho$ ($Mev/fm^3$)')
plt.ylabel('Radius (km)')
plt.legend()
plt.savefig(f'figures/Fig_Rvsrho_GRvsER.png', dpi = 200)
# plt.show()
plt.close()

plt.plot([x/1000 for x in radiusStar_ER], [x/(1.989*10**30) for x in massStar_ER], color = 'tab:blue', label=f'Entangled Relativity, Lm = $-\\rho$\n M_Max = {np.max(massStar_ER)/(1.989*10**30):.2f}, R = {radiusStar_ER[massStar_ER.index(np.max(massStar_ER))]/1000:.1f} km')
plt.plot([x/1000 for x in radiusStar_GR], [x/(1.989*10**30) for x in massStar_GR], color = 'tab:orange', label=f'General Relativity\n M_Max = {np.max(massStar_GR)/(1.989*10**30):.2f}, R = {radiusStar_GR[massStar_GR.index(np.max(massStar_GR))]/1000:.1f} km')
plt.axvline(x = radiusStar_ER[massStar_ER.index(np.max(massStar_ER))]/1000, color = 'tab:blue', linestyle='dashed')
plt.axvline(x = radiusStar_GR[massStar_GR.index(np.max(massStar_GR))]/1000, color = 'tab:orange', linestyle='dashed')
plt.axhline(y = np.max(massStar_ER)/(1.989*10**30), color = 'tab:blue', linestyle='dashed')
plt.axhline(y = np.max(massStar_GR)/(1.989*10**30), color = 'tab:orange', linestyle='dashed')
plt.xlabel('Radius (km)')
plt.ylabel('Mass $M/M_{\odot}$')
plt.legend()
plt.savefig(f'figures/Fig_MvsR_GRvsER.png', dpi = 200)
# plt.show()
plt.close()



plt.plot([x/1000 for x in radiusStar_ER], [x/(1.989*10**30) for x in massStar_ER], color = 'tab:blue')
plt.plot([x/1000 for x in radiusStar_ER], [x/(1.989*10**30) for x in massADM_ER], color = 'tab:green')
plt.axvline(x = radiusStar_ER[massStar_ER.index(np.max(massStar_ER))]/1000, color = 'tab:blue', linestyle='dashed')
plt.axvline(x = radiusStar_ER[massADM_ER.index(np.max(massADM_ER))]/1000, color = 'tab:green', linestyle='dashed')

plt.axhline(y = np.max(massStar_ER)/(1.989*10**30), color = 'tab:blue', linestyle='dashed', label=f'M_Max = {np.max(massStar_ER)/(1.989*10**30):.2f}')
plt.axhline(y = np.max(massADM_ER)/(1.989*10**30), color = 'tab:green', linestyle='dashed', label=f'M_ADM_Max = {np.max(massADM_ER)/(1.989*10**30):.2f}')
plt.title('Entangled Relativity, Lm =$-\\rho$')
plt.legend()
plt.savefig(f'figures/Fig_MvsR_ADM.png', dpi = 200)
# plt.show()
plt.close()


PhiInit = 1
PsiInit = 0
option = 1
radiusMax_in = 50000
radiusMax_out = 10000000
Npoint = 50000
rhoMin = 100
rhoMax = 8200
log_active = False
rho = rho_ER

rho = rho*cst.eV*10**6/(cst.c**2*cst.fermi**3)
dilaton_active = [True, True]
for i in range(2):
    tov = TOV(rho[i], PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, option, dilaton_active[i], log_active)
    tov.ComputeTOV()
    fig,ax = plt.subplots()
    phi_ER = tov.Phi
    psi_ER = tov.Psi
    x = tov.radius/1000
    radius = tov.radiusStar/1000
    density_ER = rho[i] /(cst.eV*10**6/(cst.c**2*cst.fermi**3))

    plt.xlabel('km')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlim(0,60)

    ax.plot(x, psi_ER, color = 'tab:blue', label =f'$\Psi$ ER ({density_ER :.0f})')
    ax.set_ylabel('$\Psi$', color = 'tab:blue')
    ax2=ax.twinx()
    ax2.plot(x, phi_ER, color = 'tab:green')
    ax2.set_ylabel('$\Phi$', color = 'tab:green')

    plt.axvline(x = radius, color = 'black', label = f'Radius for $\\rho_0$ = {density_ER:.0f} $MeV/fm^3$')
    plt.legend(loc = 'center right')

    plt.title(f'$\Phi(R_*$={x[np.where(x > radius)[0][0]]:.1f} km) = {phi_ER[np.where(x > radius)[0][0]]:.2f}')

    plt.savefig(f'figures/phi_psi_{i+1}.png', dpi = 200)

    # plt.show()
    plt.close()
