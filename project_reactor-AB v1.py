# Microreactor simulation code

# Pseudo-code
# Get beginning of life cross sections
# Define beginning of life  spacial distribution
# Run power iteration
# Calculate new cross sections from burnup
# next power iteration

import numpy as np
import matplotlib.pyplot as plt

# BOL
# fuelmix_xc_BOL  = {
#     'sigTr': 0.317, 
#     'sigA': 7.38E-2, 
#     'nusigF': 8.39E-2, 
#     'kapsigF': 1.19E-12
# }

fuelmix_fast_BOL= {
    'sigTr': 0.278,
    'sigA': 0.01,
    'nusigF': 0.005,
    'kapsigF': 6.4e-14,
    'sigS12':0.015}

fuelmix_th_BOL={
    'sigTr':0.833,
    'sigA': 0.1,
    'nusigF': 0.15,
    'kapsigF': 1.9e-12,
    'sigS12': 0}

# reflector_xs_BOL = {
#     'sigTr': 0.222,
#     'sigA': 0.002,
#     'nusigF': 0.0,
#     'kapsigF': 0.0}

reflector_fast_BOL = {
    'sigTr': 0.303,
    'sigA': 0.001,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12': 0.04}

reflector_th_BOL = {
    'sigTr': 1.667,
    'sigA': 0.02,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12': 0}

# shield_xs_BOL = {
#     'sigTr': 0.417,
#     'sigA': 0.050,
#     'nusigF': 0.0,
#     'kapsigF': 0.0}

shield_fast_BOL = {
    'sigTr':0.37,
    'sigA': 0.05,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12': 0.005,}

shield_th_BOL = {
    'sigTr': 1.111,
    'sigA':0.5,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12': 0}

# EOL
# fuelmix_xc_EOL = {
#     'sigTr':   0.309,
#     'sigA':    0.0684,   
#     'nusigF':  0.0770,   
#     'kapsigF': 5.270e-13
# }

fuelmix_fast_EOL={
    'sigTr': 0.282,
    'sigA':0.012,
    'nusigF': 0.004,
    'kapsigF': 5.1e-14,
    'sigS12': 0.015}

fuelmix_th_EOL = {
    'sigTr': 0.850,
    'sigA': 0.130,
    'nusigF': 0.11,
    'kapsigF': 1.4e-12,
    'sigS12': 0}

# reflector_xs_EOL = {
#     'sigTr': 0.222,
#     'sigA': 0.002,
#     'nusigF': 0.0,
#     'kapsigF': 0.0}

reflector_fast_EOL = {
    'sigTr': 0.303,
    'sigA': 0.001,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12': 0.04}

reflector_th_EOL = {
    'sigTr': 1.667,
    'sigA': 0.02,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12': 0}

# shield_xs_EOL = {
#     'sigTr': 0.417,
#     'sigA': 0.050,
#     'nusigF': 0.0,
#     'kapsigF': 0.0}

sheild_fast_EOL = {
    'sigTr':0.37,
    'sigA': 0.05,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12': 0.005,}

shield_th_EOL = {
    'sigTr': 1.111,
    'sigA':0.5,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12': 0}

# Assume only fuel is impacted by burnup
def burnup(B_int,prop,speed):

    if speed=="fast":
        Sig_B=fuelmix_fast_BOL[prop]
        Sig_E=fuelmix_fast_EOL[prop]
    else:
        Sig_B=fuelmix_th_BOL[prop]
        Sig_E=fuelmix_th_EOL[prop]

    Sig_int=Sig_B+(Sig_E-Sig_B)/(B_EOL-B_BOL)*(B_int-B_BOL)
    return Sig_int

class Reactor1D:
    # Class inputs:
        # H: vector of lengths for each material, assuming symmetric reactor
        #    with fuel in the middle i.e. [50,30,40]
        # dx: length mesh size (Same units as H)
        # P: Total power output [W]
        # rho_ihm: Density of initial heavy metal [MWD/MTU]
        # EOL: End of Life Burnup value []
        # BOL: Beginning of Life Burnup value []
    def __init__(self, H, dx, P, rho_ihm,EOL,BOL):
        self.H = np.sum(H)
        self.dx = dx
        self.Nx = int(self.H/dx)+1
        self.P = P
        self.rho_ihm = rho_ihm
        self.x=np.linspace(0,self.H,self.Nx)
        self.EOL=EOL
        self.BOL=BOL
        
        # State variables
        self.Bt = np.zeros(self.Nx)
        self.Dt_f = np.zeros(self.Nx)
        self.Dt_th = np.zeros(self.Nx)
        self.phi_f = np.ones(self.Nx)
        self.phi_th = np.ones(self.Nx)
        self.k = 1.0
        self.history={'k': [], 'phi_f': [], 'phi_th': [], 'Bt': []}
        
        # FAST cross section vectors
        self.sigA_f=np.zeros(self.Nx)
        self.sigTr_f=np.zeros(self.Nx)
        self.nusigF_f=np.zeros(self.Nx)
        self.kapsigF_f=np.zeros(self.Nx)
        self.sigS12_f=np.zeros(self.Nx)
        self.sig_R_1=np.zeros(self.Nx)
        
        # THERMAL cross section vectors
        self.sigA_th=np.zeros(self.Nx)
        self.sigTr_th=np.zeros(self.Nx)
        self.nusigF_th=np.zeros(self.Nx)
        self.kapsigF_th=np.zeros(self.Nx)
        
        # FAST Fuel Region
        self.sigA_f[0:H[0]]=fuelmix_fast_BOL['sigA']
        self.sigTr_f[0:H[0]]=fuelmix_fast_BOL['sigTr']
        self.nusigF_f[0:H[0]]=fuelmix_fast_BOL['nusigF']
        self.kapsigF_f[0:H[0]]=fuelmix_fast_BOL['kapsigF']
        self.sigS12_f[0:H[0]]=fuelmix_fast_BOL['sigS12']
        
        # THERMAL Fuel Region
        self.sigA_th[0:H[0]]=fuelmix_th_BOL['sigA']
        self.sigTr_th[0:H[0]]=fuelmix_th_BOL['sigTr']
        self.nusigF_th[0:H[0]]=fuelmix_th_BOL['nusigF']
        self.kapsigF_th[0:H[0]]=fuelmix_th_BOL['kapsigF']
        
        # FAST Reflector Region
        self.sigA_f[H[0]:H[0]+H[1]]=reflector_fast_BOL['sigA']
        self.sigTr_f[H[0]:H[0]+H[1]]=reflector_fast_BOL['sigTr']
        self.nusigF_f[H[0]:H[0]+H[1]]=reflector_fast_BOL['nusigF']
        self.kapsigF_f[H[0]:H[0]+H[1]]=reflector_fast_BOL['kapsigF']
        self.sigS12_f[H[0]:H[0]+H[1]]=reflector_fast_BOL['sigS12']
        
        # THERMAL Reflector Region
        self.sigA_th[H[0]:H[0]+H[1]]=reflector_th_BOL['sigA']
        self.sigTr_th[H[0]:H[0]+H[1]]=reflector_th_BOL['sigTr']
        self.nusigF_th[H[0]:H[0]+H[1]]=reflector_th_BOL['nusigF']
        self.kapsigF_th[H[0]:H[0]+H[1]]=reflector_th_BOL['kapsigF']
        
        # FAST Shield Region
        self.sigA_f[H[0]+H[1]:]=shield_fast_BOL['sigA']
        self.sigTr_f[H[0]+H[1]:]=shield_fast_BOL['sigTr']
        self.nusigF_f[H[0]+H[1]:]=shield_fast_BOL['nusigF']
        self.kapsigF_f[H[0]+H[1]:]=shield_fast_BOL['kapsigF']
        self.sigS12_f[H[0]+H[1]:]=shield_fast_BOL['sigS12']
        
        # THERMAL Shield Region
        self.sigA_th[H[0]+H[1]:]=shield_th_BOL['sigA']
        self.sigTr_th[H[0]+H[1]:]=shield_th_BOL['sigTr']
        self.nusigF_th[H[0]+H[1]:]=shield_th_BOL['nusigF']
        self.kapsigF_th[H[0]+H[1]:]=shield_th_BOL['kapsigF']
        
    def get_xs(self, H):
        self.sigA_f[:H[0]] = np.array([burnup(self.Bt[i], 'sigA', 'fast') for i in range(int(H[0]/self.dx))])
        self.sigTr_f[:H[0]] = np.array([burnup(self.Bt[i], 'sigTr', 'fast') for i in range(int(H[0]/self.dx))])
        self.nusigF_f[:H[0]] = np.array([burnup(self.Bt[i], 'nusigF', 'fast') for i in range(int(H[0]/self.dx))])
        self.kapsigF_f[:H[0]] = np.array([burnup(self.Bt[i], 'kapsigF', 'fast') for i in range(int(H[0]/self.dx))])
        self.sigS12_f[:H[0]] = np.array([burnup(self.Bt[i], 'sigS12', 'fast') for i in range(int(H[0]/self.dx))])

        self.Dt_f = 1 / (3 * self.sigTr_f)
        
        self.sigA_th[:H[0]] = np.array([burnup(self.Bt[i], 'sigA', 'thermal') for i in range(int(H[0]/self.dx))])
        self.sigTr_th[:H[0]] = np.array([burnup(self.Bt[i], 'sigTr', 'thermal') for i in range(int(H[0]/self.dx))])
        self.nusigF_th[:H[0]] = np.array([burnup(self.Bt[i], 'nusigF', 'thermal') for i in range(int(H[0]/self.dx))])
        self.kapsigF_th[:H[0]] = np.array([burnup(self.Bt[i], 'kapsigF', 'thermal') for i in range(int(H[0]/self.dx))])

        self.Dt_th = 1 / (3 * self.sigTr_th)
        
    def step(self, H, dt_days,tol=1E-6):
        self.get_xs(H)
        self.sig_R_1=self.sigA_f+self.sigS12_f
        self.k,self.phi_f,self.phi_th=power_iteration(self.H, self.dx, 
                                                      self.Dt_f,self.Dt_th, 
                                                      self.sig_R_1,self.sigA_th,
                                                      self.nusigF_f,self.nusigF_th,
                                                      self.kapsigF_f,self.kapsigF_th, self.sigS12_f, 
                                                      tol,self.P/2, ['reflective','vacuum'])
        
        Pi=(self.kapsigF_f*self.phi_f*self.dx+self.kapsigF_th*self.phi_th*self.dx)/1e6
        self.Bt+=(Pi*dt_days)/(self.rho_ihm*self.dx)
        
        self.history['k'].append(self.k)
        self.history['phi_f'].append(self.phi_f.copy())
        self.history['phi_th'].append(self.phi_th.copy())
        self.history['Bt'].append(self.Bt.copy())
        pass
    
    def full_core(self,var,time_index=-1):
        half=self.history[var][time_index]
        full=np.concatenate((np.flip(half[1:]), half))
        return full


def power_iteration(H,dx,D_f,D_th,sig_R_1,sig_a_th,nu_sig_f,nu_sig_th,
                    kap_sig_f,kap_sig_th,sig_S12,tol,P,boundary):
    if isinstance(H,(list,tuple,np.ndarray)):
        H_fuel,H_ref=H[0], H[1]
        H_tot=sum(H)
        N=int(H_tot/dx+1)
        x=np.linspace(0,H_tot,N)
        ifuel=H_fuel/dx

    H_tot=H
    N=int(H/dx+1)
    x=np.linspace(0,H,N)
    d_f=0.7104*3*D_f[-1]
    d_th=0.7104*3*D_th[-1]
    
    # Build A_f
    A_f=np.zeros((N,N))
    for i in range(1,N-1):
        # Dleft=(D_f[i]+D_f[i-1])/2
        # Dright=(D_f[i]+D_f[i+1])/2
        # Harmonic Mean
        Dleft=(2*D_f[i]*D_f[i-1])/(D_f[i]+D_f[i-1])
        Dright=(2*D_f[i]*D_f[i+1])/(D_f[i]+D_f[i+1])
        
        A_f[i,i-1]=-Dleft/dx**2 # for homogenous cell widths
        A_f[i,i]=(Dleft+Dright)/dx**2 + sig_R_1[i]
        A_f[i,i+1]=-Dright/dx**2

    # Boundary Conditions
    # Left Boundary
    if boundary[0].lower()=='vacuum':
        A_f[0,0]=1.0
    elif boundary[0].lower()=='reflective':
        A_f[0,0]=1.0
        A_f[0,1]=-1.0

    # Right Boundary
    if boundary[1].lower()=='vacuum':
        A_f[-1, -1] = 1 + (2 * D_f[-1] / d_f)  
        A_f[-1, -2] = -(2 * D_f[-1] / d_f)      # If using d_phi/dx approx
        #b[-1] = 0
    elif boundary[1].lower()=='reflective':
        A_f[-1,-1]=1.0
        A_f[-1,-2]=-1.0
        
    # Build A_th
    A_th=np.zeros((N,N))
    for i in range(1,N-1):
        # Dleft=(D_th[i]+D_th[i-1])/2
        # Dright=(D_th[i]+D_th[i+1])/2
        Dleft=(2*D_th[i]*D_th[i-1])/(D_th[i]+D_th[i-1])
        Dright=(2*D_th[i]*D_th[i+1])/(D_th[i]+D_th[i+1])
            
        A_th[i,i-1]=-Dleft/dx**2 # for homogenous cell widths
        A_th[i,i]=(Dleft+Dright)/dx**2 + sig_a_th[i]
        A_th[i,i+1]=-Dright/dx**2

    # Boundary Conditions
    # Left Boundary
    if boundary[0].lower()=='vacuum':
        A_th[0,0]=1.0
    elif boundary[0].lower()=='reflective':
        A_th[0,0]=1.0
        A_th[0,1]=-1.0

    # Right Boundary
    if boundary[1].lower()=='vacuum':
        A_th[-1, -1] = 1 + (2 * D_th[-1] / d_th)  
        A_th[-1, -2] = -(2 * D_th[-1] / d_th)      # If using d_phi/dx approx
        #b[-1] = 0
    elif boundary[1].lower()=='reflective':
        A_th[-1,-1]=1.0
        A_th[-1,-2]=-1.0

    phi_f=np.sin(np.pi*x/(2*H_tot)) if boundary[0]=='vacuum' else np.ones(N)
    phi_th=np.sin(np.pi*x/(2*H_tot)) if boundary[0]=='vacuum' else np.ones(N)
    
    # Build A_tot matrix, phi_tot vector, nusigF matrix
    sig_s=np.eye(N)*(-sig_S12)
    sig_s[0,0]=0
    sig_s[N-1,N-1]=0
    Zero=np.zeros((N,N))
    A_tot=np.block([[A_f, Zero],[sig_s, A_th]])
    
    phi_tot=np.concatenate([phi_f,phi_th])
    
    nusigF1=np.diag(nu_sig_f)
    nusigF2=np.diag(nu_sig_th)
    nusigF_mat=np.block([[nusigF1,nusigF2],[Zero,Zero]])
    nusigF_mat[0,:]=0 
    nusigF_mat[N-1,:]=0
    nusigF_mat[N,:]=0
    nusigF_mat[2*N-1,:]=0
    
    k_old,error,i=1,1,0

    while error>tol and i<1000:
        
        source=np.dot(nusigF_mat,phi_tot)
        #Solve for new phi
        phi_new=np.linalg.solve(A_tot,source/k_old)

        # Find new k
        F_old=np.sum(np.dot(nusigF_mat,phi_tot))
        F_new=np.sum(np.dot(nusigF_mat,phi_new))
        k_new=k_old*(F_new/F_old)

        # Check convergence
        error=abs(k_new-k_old)

        # Set as final value/reinitialize for next iteration
        phi_tot=phi_new/np.max(phi_new)
        k_old=k_new
        i+=1

    phi_f=phi_tot[:N]
    phi_th=phi_tot[N:]    
    norm_power_f=np.sum(kap_sig_f*phi_f*dx)
    norm_power_th=np.sum(kap_sig_th*phi_th*dx)
    norm_power=norm_power_f+norm_power_th
    scale_factor=P/norm_power
    phi_tot=phi_tot*scale_factor
    return k_old,phi_f,phi_th

# 1D slab geometry:
    # vacuum [shield | reflector | core | reflector | shield] vacuum
    #          40       30         100 

# Sizes for half the reactor
shield=40
reflector=30
fuel=50
L=240 # Length of the reactor (240 Max)
W=240 # Width of the reactor (240 Max)
P_tot=1e6
P_1D=P_tot/(L*W)
Size=([fuel,reflector,shield])
size_tot=sum(Size)
dx=0.2
B_EOL=36500 # MWD/MTU
B_BOL=0
hw_reactor=Reactor1D(H=Size,dx=1,P=P_1D,rho_ihm=9000,EOL=36500,BOL=0)

T=15 # years
dt=1 # year

Nt=int(T/dt)
t_conv=365
dt=dt*t_conv

for step in range(Nt):
    hw_reactor.step(Size,dt_days=dt)
    if step%10==0:
        print(f"Step {step}: k= {hw_reactor.k:.5f}, MaxBurnup={np.max(hw_reactor.Bt):.1f}")

phi_BOL_th=hw_reactor.full_core('phi_th',0)
phi_EOL_th=hw_reactor.full_core('phi_th',-1)
phi_BOL_f=hw_reactor.full_core('phi_f',0)
phi_EOL_f=hw_reactor.full_core('phi_f',-1)
x=np.linspace(-size_tot,size_tot,len(phi_EOL_f))

# plt.plot(x,phi_BOL_th/np.max(phi_BOL_th),'r--',label="Beginning of Life Flux (Thermal)")
# plt.plot(x,phi_BOL_f/np.max(phi_BOL_f),'k-.',label="Beginning of Life Flux (Fast)")
# plt.plot(x,phi_EOL_th/np.max(phi_EOL_th),'-.',label="End of Life Flux (Thermal)")
# plt.plot(x,phi_EOL_f/np.max(phi_EOL_f),'k-.',label="End of Life Flux (Fast)")
plt.plot(x,phi_BOL_f,'r--',label="Beginning of Life (Fast)")
plt.plot(x,phi_BOL_th,'k-.',label="Beginning of Life (Thermal)")
plt.axvline(x=fuel, color='gray',linestyle='--', label='Fuel Boundary')
plt.axvline(x=fuel+reflector, color='gray', linestyle=':',label='Reflector Boundary')
plt.axvline(x=-fuel, color='gray',linestyle='--')
plt.axvline(x=-(fuel+reflector),color='gray',linestyle=':')
plt.legend()
plt.ylabel('Normalized Flux')
plt.xlabel('Reactor Position [cm]')
plt.show()
