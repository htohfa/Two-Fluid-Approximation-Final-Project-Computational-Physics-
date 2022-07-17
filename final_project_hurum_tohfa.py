#Author: Hurum Maksora Tohfa
#importing necessary libraries
import numpy as np
from scipy.misc import derivative
import matplotlib. pyplot as plt
from scipy import integrate
#defining constants from planck 2018
omega_m =.3153
omega_r = 9.236e-5
omega_b = .04884
H0 = 67.36 #kms^-1Mpc^-1
a_eq =0.0003029385

#defining integrand for comoving distance
def deta(a):
    Om_m=.3153
    Om_r= 9.236e-5
    Om_l= .679
    Om_k =1- Om_m- Om_r- Om_l
    H0 = 67.36
    asquareH=H0*np.sqrt((Om_m*a)+Om_r+Om_l*a**4+Om_k*a**2)
    deta=299792.458/asquareH #Mpc
    return deta
#defining conformal time
def conformal_time(a):
    #In Mpc
    eta=integrate.romberg(deta,0,a,rtol=1e-2)
    return eta

a_rec= 1/1100
a_eq = 1/3300
tau_r = (omega_m/a_rec)**(1/2)/2
print(a_eq)
print(a_rec)
print(conformal_time(a_rec))
print(conformal_time(a_eq))
print((conformal_time(a_rec)/tau_r)/68)
print(conformal_time(1)/conformal_time(a_rec))


#writing down the coupled differential equations to solve
def DEtoSolve(initial_condition, x,k):

    kap= k*278.43522610303745

    delta_c, delta_gamma, nu_c,nu_gamma, phi= initial_condition
    alpha =np.sqrt(0.0009090909090909091/0.00030303030303030303)

    y= (alpha*x)**2+2*alpha*x
    yc = (1 - ((.04884)/.3153)/1.68)*y
    yb = 1.68*(.04884/.3153)*y
    delta = (delta_gamma*(1+4/3*(y-yc))+yc*delta_c)/(1+y)
    nu= (nu_gamma*(4/3+y- yc)+ yc*nu_c)/(1+y)
    eta = 2*alpha*(alpha*x +1)/(alpha**2*x**2+2*alpha*x)

    phi_prime= -eta*phi+3*eta**2*nu/(2*kap)



    dot_delta_c = -kap*nu_c+ 3*phi_prime
    dot_delta_gamma = -4/3*kap*nu_gamma+ 4*phi_prime
    dot_nu_c = -eta*nu_c+kap*phi
    dot_nu_gamma = ((4/3+ yb)**(-1))*(-eta*yb*nu_gamma+kap*delta_gamma/3+kap*phi*(4/3+yb))
    return np.array([dot_delta_c,dot_delta_gamma,dot_nu_c,dot_nu_gamma,phi_prime])

#defining the runge-kutta method
def RK4(func, X0, x,k):
    dx = x[1] - x[0]
    nt = len(x)
    X  = np.zeros([nt, len(X0)])
    X[0] = X0
    for i in range(nt-1):
        k1 = func(X[i], x[i],k)
        k2 = func(X[i] + dx/2. * k1, x[i] + dx/2.,k)
        k3 = func(X[i] + dx/2. * k2, x[i] + dx/2.,k)
        k4 = func(X[i] + dx    * k3, x[i] + dx,k)
        X[i+1] = X[i] + dx / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X

#defining initial conditons
phi= 1
xmin = 0.0016413968095176856
k= 0.2
kap= k*conformal_time(a_rec)
alpha =np.sqrt(0.0009090909090909091/0.00030303030303030303)
eta0 = 2*alpha*(alpha*xmin +1)/(alpha**2*xmin**2+2*alpha*xmin)
y0= (alpha*xmin)**2+2*alpha*xmin
delta_gamma= -2*phi*(1+3*y0/16)
delta_c= (3/4)*delta_gamma
nu_gamma= -kap/eta0*(delta_gamma/4+ (2*kap**2*(1+y0)*phi)/(9*eta0**2*(4/3+y0)))
nu_c= nu_gamma

inital = [delta_c, delta_gamma, nu_c,nu_gamma,phi]

#Implementing the rk4 method to solve for sources
N = 100000 #Number of steps

xmax= 50.918255565028645
x = np.linspace(xmin,xmax, N)
X0 = [delta_c, delta_gamma, nu_c,nu_gamma,phi]
Xrk4 = RK4(DEtoSolve, X0, x,k)

#Plotting the sources for k=.2 to compare with fermilab's result
plt.plot(x, np.abs(Xrk4[:,0]), label= '$\delta_{c}$')
plt.plot(x, np.abs(Xrk4[:,1]), label= '$\delta_{\gamma}$')
plt.plot(x, np.abs(Xrk4[:,2]), label= '$\delta_{c}$')
plt.plot(x, np.abs(Xrk4[:,3]), label= '$\delta_{\gamma}$')
plt.axvline(x=0.4397308486753022, color='r', linestyle='--', label= 'recombination')

plt.xlabel('x')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="upper right")
plt.savefig('k23.pdf')
plt.show(block=True)

#storing the sources in different lists for a range of k values
k = np.linspace(1e-3,1e3, 1500)
xmin = 0.0016413968095176856
xmax =0.4397308486753022
x = np.linspace(xmin,xmax, N)
delta_cc =[]
delta_gam =[]
nu_cc =[]
nu_gam =[]
phi_ =[]

for i in range(0,len(k)):
    Xrk4 = RK4(DEtoSolve, X0, x,k[i])
    delta_cc.append(Xrk4[:,0][-1])
    delta_gam.append(Xrk4[:,1][-1])
    nu_cc.append(Xrk4[:,2][-1])
    nu_gam.append(Xrk4[:,3][-1])
    phi_.append(Xrk4[:,4][-1])

#interpolating the source terms with k values
from scipy.interpolate import interp1d
delta_c_list = interp1d(k,delta_cc, fill_value='extrapolate' )
delta_gamma_list = interp1d(k,delta_gam, fill_value='extrapolate' )
nu_c_list = interp1d(k,nu_cc, fill_value='extrapolate' )
nu_gamma_list = interp1d(k, nu_gam, fill_value='extrapolate' )
phi_list =  interp1d(k, phi_, fill_value='extrapolate' )

def y(x):
    alpha =np.sqrt(0.0009090909090909091/0.00030303030303030303)
    return (alpha*x)**2+2*alpha*x

#defining functions for damping term
from scipy.special import spherical_jn
l = np.linspace(0, 1500,500)
sigma = .03
xrec= ((alpha**2+1)**(1/2)-1)/alpha
delta_phi= (2-8/y(xrec)+16*xrec/y(xrec)**3)/10*y(xrec)

xs = .6*.3153**(1/4)*.04884**(-1/2)*a_rec**(3/4)*.673**(-1/2)
k = np.linspace(-3, 3, 10000)
cl_list=[]
#calculating the integrand
for i in range(0, len(l)):
    new_k = np.exp(k)
    a= int(l[i])
    T= np.exp(-new_k**2*(2*xs**2+sigma**2*xrec**2))
    cl_k = 4*np.pi*T*new_k*((spherical_jn(a, new_k*50.91825556502865, derivative =False)*np.abs(delta_gamma_list(new_k)/4+phi_list(new_k)+2*delta_phi)+nu_gamma_list(new_k)*spherical_jn(a, new_k*50.91825556502865, derivative =True))**2)
    area =0
    #integrating using rk4
    for j in range(0,len(k)-1):
        area =area+ (cl_k[j+1]+ cl_k[j])*(k[j+1]-k[j])/(2)
    cl_list.append(area)
#calculating l(l+1)cl
cl_= cl_list*l*(l+1)/(2*np.pi)
#plotting
plt.plot(l[40:len(l)], cl_[40:len(l)])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_{\ell}$')
