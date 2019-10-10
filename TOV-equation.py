#!/usr/bin/env python
from numpy import *
import matplotlib 
import matplotlib.pyplot as plt
from scipy.integrate import *

#Physical parameter
K = 0.1

#parameter to choose equation of state
EqS = 1

# Initial conditions
Ψ0 = 0
Φ0 = 1
P0 = 1
m0 = 0
ρ0 = 2

#equation of state
def ρ(P,EqS):
    if(EqS == 1):#incompressible
        return ρ0
    elif(EqS == 2):#barotropic
        return ρ0*P**(-1/3)

#Ψ equation
def f1(r, Ψ, Φ, P, m, Eqs):
    Lm = P
    #Lm = P
    if(r==0):
        return K*((1/3)*Φ**(1/2)*(3*P-ρ(P,EqS)-Lm))
    else:
        return K*(1.-m*K/(4.*pi*r))**(-1)*((1/3)*Φ**(1/2)*(3*P-ρ(P,EqS)-Lm)+(1/(8*pi*r**2))*(r*f4(r, Ψ, Φ, P, m, Eqs)-m)*Ψ)

#Φ equation
def f2(r, Ψ, Φ, P, m, Eqs):
    return Ψ

#P equation
def f3(r, Ψ, Φ, P, m, EqS):
    Lm = P
    #Lm = P
    if(r==0):
        return 0
    else:
        return -K/(8.*pi)*r**(-2)*((P+ρ(P,EqS))*((P/Φ**(1/2))*4.*pi*r**3+m)-(1/3)*(Lm+ρ(P,EqS)-3*P)*((4*pi*P*r**3*Φ**(-1/2))+16*pi*(r*Φ**(1/2))-3*m))*(1.-m*K/(4.*pi*r))**(-1)

#mass equation
def f4(r, Ψ, Φ, P, m, EqS):
    Lm = P
    #Lm = P
    return 4*pi*Φ**(-1/2)*(r**2*ρ(P,EqS)-K*(1/3)*(Lm+ρ(P,EqS)-3*P))

#P equation
def f5(r, P, m, EqS):
    if(r==0):
        return 0
    else:
        return -K/(8.*pi)*r**(-2)*(P+ρ(P,EqS))*(P*4.*pi*r**3+m)*(1.-m*K/(4.*pi*r))**(-1)

#mass equation
def f6(r, P, m, EqS):
    return 4*pi*ρ(P,EqS)*r**2

#Integration-------------------------------------------------------
#Integration parameter
n = 0
#initial radius
r0 = 0
#radius step 
dr = 0.001
#at the origin
Ψ = [Ψ0]
Φ = [Φ0]
P = [P0]
m = [m0]

rIntegration = [r0]

while(P[n]>0.):
    if(n == 5000):
        break
        print('end')
    else:
        Ψ.append( Ψ[n] + dr*f1(rIntegration[n], Ψ[n], Φ[n], P[n], m[n], EqS))
        Φ.append( Φ[n] + dr*f2(rIntegration[n], Ψ[n], Φ[n], P[n], m[n], EqS))
        P.append( P[n] + dr*f3(rIntegration[n], Ψ[n], Φ[n], P[n], m[n], EqS))
        m.append( m[n] + dr*f4(rIntegration[n], Ψ[n], Φ[n], P[n], m[n], EqS))
        n = n+1
        rIntegration.append(r0+n*dr)
Ψ.remove(Ψ[-1])
Φ.remove(Φ[-1])
P.remove(P[-1])
m.remove(m[-1])
rIntegration.remove(rIntegration[-1])

#Result--------------------------------------------------------------------
mass = m[-1]
R = rIntegration[-1]

#Integtation without Dilaton-------------------------------------------------------
#Integration parameter
i = 0
Pwd = [P0]
mwd = [m0]
rIntegration2 = [r0]

while(Pwd[i]>0):
    Pwd.append( Pwd[i] + dr*f5(rIntegration2[i], Pwd[i], mwd[i], EqS))
    mwd.append( mwd[i] + dr*f6(rIntegration2[i], Pwd[i], mwd[i], EqS))
    i = i+1
    rIntegration2.append(r0+i*dr)
    print(i)
    print(rIntegration2[i])
    print(Pwd[i])
Pwd.remove(Pwd[-1])
mwd.remove(mwd[-1])
rIntegration2.remove(rIntegration2[-1])


masswd = mwd[-1]  

#print---------------------------------------------------------------------

print(mass)
print(masswd)
print(P[-1])
print(R)

fig, ax = plt.subplots()

ax1 = plt.subplot(2,2,1)
plt.plot(rIntegration, Ψ)
plt.xlabel('radius r')
plt.ylabel('$Ψ$')
plt.title("K={},$Ψ_0$={},$Φ_0$={},$P_0$={},$m_0$={}".format(K, Ψ0, Φ0, P0, m0))

ax2 = plt.subplot(2,2,2)
plt.plot(rIntegration, Φ)
plt.xlabel('radius r')
plt.ylabel('$Φ$')

ax3 = plt.subplot(2,2,3)
plot3, = plt.plot(rIntegration, P)
plot4, = plt.plot(rIntegration2, Pwd)
plt.xlabel('radius r')
plt.ylabel('P')

ax4 = plt.subplot(2,2,4)
plot1, = plt.plot(rIntegration, m)
plot2, = plt.plot(rIntegration2, mwd)
plt.xlabel('radius r')
plt.ylabel('m')
plt.legend([plot1,plot2],["with Dilaton","without Dilaton"])

plt.show()