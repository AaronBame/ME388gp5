# Microreactor simulation code

# Pseudo-code
# Get beginning of life cross sections
# Define beginning of life  spacial distribution
# Run power iteration
# Calculate new cross sections from burnup
# next power iteration

import numpy as np
import matplotlib.pyplot as plt
import json

# BOL
# fuelmix_xc_BOL  = {
#     'sigTr': 0.317, 
#     'sigA': 7.38E-2, 
#     'nusigF': 8.39E-2, 
#     'kapsigF': 1.19E-12
# }

with open('project_2group_xs_BOL.json','r') as f:
    data=json.load(f)
    
fuel=data['fuel']
refl=data['reflector']
shld=data['shield']
geo=data['geometry_cm_half_core']

# EOL
fuelmix_fast_EOL={
    'sigTr': 0.84581,
    'sigA':0.01249,
    'nusigF': 0.00408,
    'kapsigF': 0.31223,
    'sigS12_fast_to_thermal': 0.0338324717451761}

fuelmix_th_EOL = {
    'sigTr': 2.04194,
    'sigA': 0.08608,
    'nusigF': 0.14107,
    'kapsigF': 11.28578,
    'sigS12_thermal_to_fast': 0}

reflector_fast_EOL = {
    'sigTr': 0.31870373749771286,
    'sigA': 0.00,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12_fast_to_thermal': 0.0029790744275745673}

reflector_th_EOL = {
    'sigTr': 0.3846073042512662,
    'sigA': 0.0,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12_thermal_to_fast': 0}

sheild_fast_EOL = {
    'sigTr':0.6410438594515198,
    'sigA': 0.05,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12_fast_to_thermal': 0.005,}

shield_th_EOL = {
    'sigTr': 1.111,
    'sigA':0.0002445175225956759,
    'nusigF': 0,
    'kapsigF': 0,
    'sigS12_thermal_to_fast': 0.0021529822846500764}

# Assume only fuel is impacted by burnup
def burnup(B_int,prop,speed):

    if speed=="fast":
        if prop=='sigS12_fast_to_thermal':
            Sig_B=fuel[prop]
            Sig_E=fuelmix_fast_EOL[prop]
        else:
            Sig_B=fuel[prop][0]
            Sig_E=fuelmix_fast_EOL[prop]
    else:
        Sig_B=fuel[prop][1]
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
        
        ifuel=int(geo['fuel']/self.dx)
        irefl=int((geo['fuel']+geo['reflector'])/self.dx)
        
        # State variables
        self.Bt = np.zeros(self.Nx)
        self.Dt_f = np.zeros(self.Nx)
        self.Dt_th = np.zeros(self.Nx)
        self.phi_f = np.ones(self.Nx)
        self.phi_th = np.ones(self.Nx)
        self.phi_phys_f=np.ones(self.Nx)
        self.phi_phys_th=np.ones(self.Nx)
        self.k = 1.0
        self.history={'k': [], 'phi_f': [], 'phi_th': [], 'Bt': [], 'phi_phys_f': [], 'phi_phys_th': []}
        
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
        self.sigA_f[0:ifuel]=fuel['sigA'][0]
        self.sigTr_f[0:ifuel]=fuel['sigTr'][0]
        self.nusigF_f[0:ifuel]=fuel['nusigF'][0]
        self.kapsigF_f[0:ifuel]=fuel['kapsigF'][0]
        self.sigS12_f[0:ifuel]=fuel['sigS12_fast_to_thermal']
        
        # THERMAL Fuel Region
        self.sigA_th[0:ifuel]=fuel['sigA'][1]
        self.sigTr_th[0:ifuel]=fuel['sigTr'][1]
        self.nusigF_th[0:ifuel]=fuel['nusigF'][1]
        self.kapsigF_th[0:ifuel]=fuel['kapsigF'][1]
        
        # FAST Reflector Region
        self.sigA_f[ifuel:irefl]=refl['sigA'][0]
        self.sigTr_f[ifuel:irefl]=refl['sigTr'][0]
        self.nusigF_f[ifuel:irefl]=refl['nusigF'][0]
        self.kapsigF_f[ifuel:irefl]=refl['kapsigF'][0]
        self.sigS12_f[ifuel:irefl]=refl['sigS12_fast_to_thermal']
        
        # THERMAL Reflector Region
        self.sigA_th[ifuel:irefl]=refl['sigA'][1]
        self.sigTr_th[ifuel:irefl]=refl['sigTr'][1]
        self.nusigF_th[ifuel:irefl]=refl['nusigF'][1]
        self.kapsigF_th[ifuel:irefl]=refl['kapsigF'][1]
        
        # FAST Shield Region
        self.sigA_f[irefl:]=shld['sigA'][0]
        self.sigTr_f[irefl:]=shld['sigTr'][0]
        self.nusigF_f[irefl:]=shld['nusigF'][0]
        self.kapsigF_f[irefl:]=shld['kapsigF'][0]
        self.sigS12_f[irefl:]=shld['sigS12_fast_to_thermal']
        
        # THERMAL Shield Region
        self.sigA_th[irefl:]=shld['sigA'][1]
        self.sigTr_th[irefl:]=shld['sigTr'][1]
        self.nusigF_th[irefl:]=shld['nusigF'][1]
        self.kapsigF_th[irefl:]=shld['kapsigF'][1]
        
    def get_xs(self, H):
        self.sigA_f[:H[0]] = np.array([burnup(self.Bt[i], 'sigA', 'fast') for i in range(int(H[0]/self.dx))])
        self.sigTr_f[:H[0]] = np.array([burnup(self.Bt[i], 'sigTr', 'fast') for i in range(int(H[0]/self.dx))])
        self.nusigF_f[:H[0]] = np.array([burnup(self.Bt[i], 'nusigF', 'fast') for i in range(int(H[0]/self.dx))])
        self.kapsigF_f[:H[0]] = np.array([burnup(self.Bt[i], 'kapsigF', 'fast') for i in range(int(H[0]/self.dx))])
        self.sigS12_f[:H[0]] = np.array([burnup(self.Bt[i], 'sigS12_fast_to_thermal', 'fast') for i in range(int(H[0]/self.dx))])

        self.Dt_f = 1 / (3 * self.sigTr_f)
        
        self.sigA_th[:H[0]] = np.array([burnup(self.Bt[i], 'sigA', 'thermal') for i in range(int(H[0]/self.dx))])
        self.sigTr_th[:H[0]] = np.array([burnup(self.Bt[i], 'sigTr', 'thermal') for i in range(int(H[0]/self.dx))])
        self.nusigF_th[:H[0]] = np.array([burnup(self.Bt[i], 'nusigF', 'thermal') for i in range(int(H[0]/self.dx))])
        self.kapsigF_th[:H[0]] = np.array([burnup(self.Bt[i], 'kapsigF', 'thermal') for i in range(int(H[0]/self.dx))])

        self.Dt_th = 1 / (3 * self.sigTr_th)



        ifuel = int(geo['fuel'] / self.dx)

        if MODE == "shutdown":
              # Fully inserted controls
         control_abs_th = 0.06 

        elif MODE == "operating":
         control_abs_BOL = 0.0435
         control_abs_EOL = 0.0265

         burn_frac = min(np.max(self.Bt) / B_EOL, 1.0)
         control_abs_th = control_abs_BOL * (1 - burn_frac) + control_abs_EOL * burn_frac

    
         
        self.sigA_th[0:ifuel] += control_abs_th

          

        print("max sigA_th fuel =", np.max(self.sigA_th[0:ifuel]))
        
    def step(self, H, dt_days,tol=1E-6):
        self.get_xs(H)
        self.sig_R_1=self.sigA_f+self.sigS12_f
        self.k,self.phi_f,self.phi_th,self.phi_phys_f,self.phi_phys_th=power_iteration(self.H, self.dx, 
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
        self.history['phi_phys_f'].append(self.phi_phys_f.copy())
        self.history['phi_phys_th'].append(self.phi_phys_th.copy())
        pass
    
    def full_core(self,var,time_index=-1):
        half=self.history[var][time_index]
        full=np.concatenate((np.flip(half[1:]), half))
        return full


def power_iteration(H,dx,D_f,D_th,sig_R_1,sig_a_th,nu_sig_f,nu_sig_th,
                    kap_sig_f,kap_sig_th,sig_S12,tol,P,boundary):

    H_tot=H
    N=int(H/dx+1)
    x=np.linspace(0,H,N)
    d_f=0.7104*3*D_f[-1]
    d_th=0.7104*3*D_th[-1]
    
    # Build A_f
    A_f=np.zeros((N,N))
    for i in range(1,N-1):

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
    sig_s=np.diag(-sig_S12)
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
    conv_kapsig=3.2e-11
    norm_power_f=np.sum(kap_sig_f*phi_f*dx)
    norm_power_th=np.sum(kap_sig_th*phi_th*dx)
    norm_power=(norm_power_f+norm_power_th)*conv_kapsig
    scale_factor=P/norm_power
    phi_tot=phi_tot*scale_factor
    phi_phys_f=phi_tot[:N]
    phi_phys_th = phi_tot[N:]
    return k_old,phi_f,phi_th,phi_phys_f,phi_phys_th

# 1D slab geometry:
    # vacuum [shield | reflector | core | reflector | shield] vacuum
    #          40       30         100 

# Sizes for half the reactor
L=240 # Length of the reactor (240 Max)
W=240 # Width of the reactor (240 Max)
P_tot=1e6
#P_1D=P_tot/(L*W) # No longer need this because one slice has to produce 1 MW
P_1D=P_tot*10**(-4)
Size=[int(geo['fuel']),int(geo['reflector']),int(geo['shield'])]
size_tot=sum(Size)
dx=0.2
B_EOL=36500 # MWD/MTU
B_BOL=0
B_rho=4.9725e-7 # MT/cm^3
hw_reactor=Reactor1D(H=Size,dx=1,P=P_1D,rho_ihm=B_rho,EOL=B_EOL,BOL=0)
MODE = "operating"


T=15 # years
dt=1 # year

Nt=int(T/dt)
t_conv=365
dt=dt*t_conv

for step in range(Nt):
    hw_reactor.step(Size,dt_days=dt)
    if step%14==0:
        print(f"Step {step}: k= {hw_reactor.k:.5f}, MaxBurnup={np.max(hw_reactor.Bt):.1f}")

phi_BOL_th=hw_reactor.full_core('phi_phys_th',0)
phi_EOL_th=hw_reactor.full_core('phi_phys_th',-1)
phi_BOL_f=hw_reactor.full_core('phi_phys_f',0)
phi_EOL_f=hw_reactor.full_core('phi_phys_f',-1)
x=np.linspace(-size_tot,size_tot,len(phi_EOL_f))

# BOL Peak-to-Average
P_avg_f=np.mean(hw_reactor.kapsigF_f*hw_reactor.phi_phys_f*dx)
P_max_f=np.max(hw_reactor.kapsigF_f*hw_reactor.phi_phys_f*dx)
P_avg_th=np.mean(hw_reactor.kapsigF_th*hw_reactor.phi_phys_th*dx)
P_max_th=np.max(hw_reactor.kapsigF_th*hw_reactor.phi_phys_th*dx)

FQ_f=P_max_f/P_avg_f
FQ_th=P_max_th/P_avg_th

print(f"Peak-to-Average (Fast)= {FQ_f:.2f}, Peak-to-Average (Thermal)= {FQ_th:.2f}, External flux (fast)= {hw_reactor.history['phi_phys_f'][-1][-1]:.2f}, External flux (thermal)= {hw_reactor.history['phi_phys_th'][-1][-1]:.2f}")

# plt.plot(x,phi_BOL_th/np.max(phi_BOL_th),'r--',label="Beginning of Life Flux (Thermal)")
# plt.plot(x,phi_BOL_f/np.max(phi_BOL_f),'k-.',label="Beginning of Life Flux (Fast)")
# plt.plot(x,phi_EOL_th/np.max(phi_EOL_th),'-.',label="End of Life Flux (Thermal)")
# plt.plot(x,phi_EOL_f/np.max(phi_EOL_f),'k-.',label="End of Life Flux (Fast)")
plt.plot(x,phi_BOL_f,'r--',label="Beginning of Life (Fast)")
plt.plot(x,phi_BOL_th,'k-.',label="Beginning of Life (Thermal)")
plt.axvline(x=Size[0], color='gray',linestyle='--', label='Fuel Boundary')
plt.axvline(x=Size[0]+Size[1], color='gray', linestyle=':',label='Reflector Boundary')
plt.axvline(x=-Size[0], color='gray',linestyle='--')
plt.axvline(x=-(Size[0]+Size[1]),color='gray',linestyle=':')
plt.legend()
plt.ylabel('Neutron Flux')
plt.xlabel('Reactor Position [cm]')
plt.show()
