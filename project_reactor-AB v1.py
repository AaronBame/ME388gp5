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
fuelmix_xc_BOL  = {
    'sigTr': 0.317, 
    'sigA': 7.38E-2, 
    'nusigF': 8.39E-2, 
    'kapsigF': 1.19E-12
}

reflector_xs_BOL = {
    'sigTr': 0.222,
    'sigA': 0.002,
    'nusigF': 0.0,
    'kapsigF': 0.0}

shield_xs_BOL = {
    'sigTr': 0.417,
    'sigA': 0.050,
    'nusigF': 0.0,
    'kapsigF': 0.0}

# EOL
fuelmix_xc_EOL = {
    'sigTr':   0.309,
    'sigA':    0.0684,   
    'nusigF':  0.0770,   
    'kapsigF': 5.270e-13
}

reflector_xs_EOL = {
    'sigTr': 0.222,
    'sigA': 0.002,
    'nusigF': 0.0,
    'kapsigF': 0.0}

shield_xs_EOL = {
    'sigTr': 0.417,
    'sigA': 0.050,
    'nusigF': 0.0,
    'kapsigF': 0.0}

# Assume only fuel is impacted by burnup
def burnup(B_int,prop):

    Sig_B=fuelmix_xc_BOL[prop]
    Sig_E=fuelmix_xc_EOL[prop]

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
        self.Dt = np.zeros(self.Nx)
        self.phi = np.ones(self.Nx)
        self.k = 1.0
        self.history={'k': [], 'phi': [], 'Bt': []}
        
        # cross section vectors
        self.sigA=np.zeros(self.Nx)
        self.sigTr=np.zeros(self.Nx)
        self.nusigF=np.zeros(self.Nx)
        self.kapsigF=np.zeros(self.Nx)
        
        self.sigA[0:H[0]]=fuelmix_xc_BOL['sigA']
        self.sigTr[0:H[0]]=fuelmix_xc_BOL['sigTr']
        self.nusigF[0:H[0]]=fuelmix_xc_BOL['nusigF']
        self.kapsigF[0:H[0]]=fuelmix_xc_BOL['kapsigF']
        
        self.sigA[H[0]:H[0]+H[1]]=reflector_xs_BOL['sigA']
        self.sigTr[H[0]:H[0]+H[1]]=reflector_xs_BOL['sigTr']
        self.nusigF[H[0]:H[0]+H[1]]=reflector_xs_BOL['nusigF']
        self.kapsigF[H[0]:H[0]+H[1]]=reflector_xs_BOL['kapsigF']
        
        self.sigA[H[0]+H[1]:]=shield_xs_BOL['sigA']
        self.sigTr[H[0]+H[1]:]=shield_xs_BOL['sigTr']
        self.nusigF[H[0]+H[1]:]=shield_xs_BOL['nusigF']
        self.kapsigF[H[0]+H[1]:]=shield_xs_BOL['kapsigF']
        
    def get_xs(self, H):
        self.sigA[:H[0]] = np.array([burnup(self.Bt[i], 'sigA') for i in range(H[0])])
        self.sigTr[:H[0]] = np.array([burnup(self.Bt[i], 'sigTr') for i in range(H[0])])
        self.nusigF[:H[0]] = np.array([burnup(self.Bt[i], 'nusigF') for i in range(H[0])])
        self.kapsigF[:H[0]] = np.array([burnup(self.Bt[i], 'kapsigF') for i in range(H[0])])


        self.Dt = 1 / (3 * self.sigTr)
        
    def step(self, H, dt_days,tol=1E-6):
        self.get_xs(H)
        self.k,self.phi=power_iteration(self.H, self.dx, self.Dt, self.sigA,
                                        self.nusigF, self.kapsigF, tol, 
                                        self.P/2, ['reflective','vacuum'])
        
        Pi=(self.kapsigF*self.phi*self.dx)/1e6
        self.Bt+=(Pi*dt_days)/(self.rho_ihm*self.dx)
        
        self.history['k'].append(self.k)
        self.history['phi'].append(self.phi.copy())
        self.history['Bt'].append(self.Bt.copy())
        pass
    
    def full_core(self,var,time_index=-1):
        half=self.history[var][time_index]
        full=np.concatenate((np.flip(half[1:]), half))
        return full


def power_iteration(H,dx,D,sig_a,nu_sig_f,kap_sig_f,tol,P,boundary):
    if isinstance(H,(list,tuple,np.ndarray)):
        H_fuel,H_ref=H[0], H[1]
        H_tot=sum(H)
        N=int(H_tot/dx+1)
        x=np.linspace(0,H_tot,N)
        ifuel=H_fuel/dx

        # This block was for a homogeneous mixture assumption
        #Dvec=np.zeros(N)
        #sigavec,nusigFvec,kapsigFvec=Dvec.copy(),Dvec.copy(),Dvec.copy()
        
        #Dvec[0:int(ifuel)]=D[0]
        #Dvec[int(ifuel):]=D[1]
        #sigavec[0:int(ifuel)]=sig_a[0]
        #sigavec[int(ifuel):]=sig_a[1]
        #nusigFvec[0:int(ifuel)]=nu_sig_f[0]
        #nusigFvec[int(ifuel):]=nu_sig_f[1]
        #kapsigFvec[0:int(ifuel)]=kap_sig_f[0]
        #kapsigFvec[int(ifuel):]=kap_sig_f[1]
        
        Dvec=D
        sigavec=sig_a
        nusigFvec=nu_sig_f
        kapsigFvec=kap_sig_f
    else:
        H_tot=H
        N=int(H_tot/dx+1)
        x=np.linspace(0,H,N)
        Dvec=np.full(N,D)
        sigavec=np.full(N,sig_a)
        nusigFvec=np.full(N,nu_sig_f)
        kapsigFvec=np.full(N,kap_sig_f)

    d=0.7104*3*Dvec[-1]
    # Build A
    A=np.zeros((N,N))
    for i in range(1,N-1):
        Dleft=(Dvec[i]+Dvec[i-1])/2
        Dright=(Dvec[i]+Dvec[i+1])/2
        A[i,i-1]=-Dleft/dx**2 # for homogenous cell widths
        A[i,i]=(Dleft+Dright)/dx**2 + sigavec[i]
        A[i,i+1]=-Dright/dx**2

    # Boundary Conditions
    # Left Boundary
    if boundary[0].lower()=='vacuum':
        A[0,0]=1.0
    elif boundary[0].lower()=='reflective':
        A[0,0]=1.0
        A[0,1]=-1.0

    # Right Boundary
    if boundary[1].lower()=='vacuum':
        A[-1, -1] = 1 + (2 * Dvec[-1] / d)  
        A[-1, -2] = -(2 * Dvec[-1] / d)      # If using d_phi/dx approx
        #b[-1] = 0
    elif boundary[1].lower()=='reflective':
        A[-1,-1]=1.0
        A[-1,-2]=-1.0

    phi=np.sin(np.pi*x/(2*H_tot)) if boundary[0]=='vacuum' else np.ones(N)
    k_old,error,i=1,1,0

    while error>tol and i<1000:
        # Compute source term
        source=(nusigFvec*phi)/k_old

        source[0]=0.0 # Phi(BC)=0
        if boundary[1].lower()=='vacuum':
            source[-1]=0

        #Solve for new phi
        phi_new=np.linalg.solve(A,source)

        # Find new k
        F_old=np.sum(nusigFvec*phi)
        F_new=np.sum(nusigFvec*phi_new)
        k_new=k_old*(F_new/F_old)

        # Check convergence
        error=abs(k_new-k_old)

        # Set as final value/reinitialize for next iteration
        phi=phi_new/np.max(phi_new)
        k_old=k_new
        i+=1

    norm_power=np.sum(kapsigFvec*phi*dx)
    scale_power=P/norm_power
    phi=phi*scale_power
    return k_old,phi

# 1D slab geometry:
    # vacuum [shield | reflector | core | reflector | shield] vacuum
    #          40       30         100 

# Sizes for half the reactor
shield=40
reflector=30
fuel=50
Size=([fuel,reflector,shield])
size_tot=sum(Size)
dx=1
B_EOL=36500 # MWD/MTU
B_BOL=0
hw_reactor=Reactor1D(H=Size,dx=1,P=1.8e7,rho_ihm=9000,EOL=36500e-6,BOL=0)

T=10 # years
dt=1 # year

Nt=int(T/dt)
t_conv=365
dt=dt*t_conv

for step in range(Nt):
    hw_reactor.step(Size,dt_days=dt)
    if step%10==0:
        print(f"Step {step}: k= {hw_reactor.k:.5f}, MaxBurnup={np.max(hw_reactor.Bt):.1f}")

phi_BOL=hw_reactor.full_core('phi',0)
phi_EOL=hw_reactor.full_core('phi',-1)
x=np.linspace(-size_tot,size_tot,len(phi_EOL))

phi_EOL = hw_reactor.full_core('phi', -1)
print("Flux at outer boundary =", phi_EOL[-1])

plt.plot(x,phi_BOL/np.max(phi_BOL),'r--',label="Beginning of Life Flux")
plt.plot(x,phi_EOL/np.max(phi_EOL),'k-.',label="End of Life Flux")
plt.axvline(x=fuel, color='gray',linestyle='--', label='Fuel Boundary')
plt.axvline(x=fuel+reflector, color='gray', linestyle=':',label='Reflector Boundary')
plt.axvline(x=-fuel, color='gray',linestyle='--')
plt.axvline(x=-(fuel+reflector),color='gray',linestyle=':')
plt.legend()
plt.ylabel('Normalized Flux')
plt.xlabel('Reactor Position [cm]')
plt.show()
