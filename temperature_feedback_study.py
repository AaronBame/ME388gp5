import numpy as np
import matplotlib.pyplot as plt
import json

with open('project_2group_xs_BOL.json','r') as f:
    data = json.load(f)

fuel = data['fuel']
refl = data['reflector']
shld = data['shield']
geo = data['geometry_cm_half_core']


def power_iteration(H, dx, D_f, D_th, sig_R_1, sig_a_th, nu_sig_f, nu_sig_th,
                    kap_sig_f, kap_sig_th, sig_S12, tol, P, boundary):

    N = int(H/dx + 1)
    x = np.linspace(0, H, N)

    d_f = 0.7104 * 3 * D_f[-1]
    d_th = 0.7104 * 3 * D_th[-1]

    A_f = np.zeros((N, N))

    for i in range(1, N-1):

        Dleft = (2 * D_f[i] * D_f[i-1]) / (D_f[i] + D_f[i-1])
        Dright = (2 * D_f[i] * D_f[i+1]) / (D_f[i] + D_f[i+1])

        A_f[i, i-1] = -Dleft / dx**2
        A_f[i, i] = (Dleft + Dright) / dx**2 + sig_R_1[i]
        A_f[i, i+1] = -Dright / dx**2

    A_f[0,0] = 1.0
    A_f[0,1] = -1.0

    A_f[-1,-1] = 1 + (2 * D_f[-1] / d_f)
    A_f[-1,-2] = -(2 * D_f[-1] / d_f)

    A_th = np.zeros((N, N))

    for i in range(1, N-1):

        Dleft = (2 * D_th[i] * D_th[i-1]) / (D_th[i] + D_th[i-1])
        Dright = (2 * D_th[i] * D_th[i+1]) / (D_th[i] + D_th[i+1])

        A_th[i, i-1] = -Dleft / dx**2
        A_th[i, i] = (Dleft + Dright) / dx**2 + sig_a_th[i]
        A_th[i, i+1] = -Dright / dx**2

    A_th[0,0] = 1.0
    A_th[0,1] = -1.0

    A_th[-1,-1] = 1 + (2 * D_th[-1] / d_th)
    A_th[-1,-2] = -(2 * D_th[-1] / d_th)

    phi_f = np.ones(N)
    phi_th = np.ones(N)

    sig_s = np.diag(-sig_S12)
    sig_s[0,0] = 0
    sig_s[-1,-1] = 0

    Zero = np.zeros((N, N))

    A_tot = np.block([
        [A_f, Zero],
        [sig_s, A_th]
    ])

    phi_tot = np.concatenate([phi_f, phi_th])

    nusigF1 = np.diag(nu_sig_f)
    nusigF2 = np.diag(nu_sig_th)

    nusigF_mat = np.block([
        [nusigF1, nusigF2],
        [Zero, Zero]
    ])

    k_old = 1.0
    error = 1.0

    while error > tol:

        source = np.dot(nusigF_mat, phi_tot)

        phi_new = np.linalg.solve(A_tot, source / k_old)

        F_old = np.sum(np.dot(nusigF_mat, phi_tot))
        F_new = np.sum(np.dot(nusigF_mat, phi_new))

        k_new = k_old * (F_new / F_old)

        error = abs(k_new - k_old)

        phi_tot = phi_new / np.max(phi_new)

        k_old = k_new

    return k_old


class Reactor1D:

    def __init__(self, H, dx):

        self.H_regions = H
        self.H = np.sum(H)

        self.dx = dx

        self.Nx = int(self.H / dx) + 1

        ifuel = int(geo['fuel'] / dx)
        irefl = int((geo['fuel'] + geo['reflector']) / dx)

        self.sigA_f = np.zeros(self.Nx)
        self.sigTr_f = np.zeros(self.Nx)
        self.nusigF_f = np.zeros(self.Nx)
        self.kapsigF_f = np.zeros(self.Nx)
        self.sigS12_f = np.zeros(self.Nx)

        self.sigA_th = np.zeros(self.Nx)
        self.sigTr_th = np.zeros(self.Nx)
        self.nusigF_th = np.zeros(self.Nx)
        self.kapsigF_th = np.zeros(self.Nx)

        self.sigA_f[0:ifuel] = fuel['sigA'][0]
        self.sigTr_f[0:ifuel] = fuel['sigTr'][0]
        self.nusigF_f[0:ifuel] = fuel['nusigF'][0]
        self.kapsigF_f[0:ifuel] = fuel['kapsigF'][0]
        self.sigS12_f[0:ifuel] = fuel['sigS12_fast_to_thermal']

        self.sigA_th[0:ifuel] = fuel['sigA'][1]
        self.sigTr_th[0:ifuel] = fuel['sigTr'][1]
        self.nusigF_th[0:ifuel] = fuel['nusigF'][1]
        self.kapsigF_th[0:ifuel] = fuel['kapsigF'][1]

        self.sigA_f[ifuel:irefl] = refl['sigA'][0]
        self.sigTr_f[ifuel:irefl] = refl['sigTr'][0]
        self.nusigF_f[ifuel:irefl] = refl['nusigF'][0]
        self.kapsigF_f[ifuel:irefl] = refl['kapsigF'][0]
        self.sigS12_f[ifuel:irefl] = refl['sigS12_fast_to_thermal']

        self.sigA_th[ifuel:irefl] = refl['sigA'][1]
        self.sigTr_th[ifuel:irefl] = refl['sigTr'][1]
        self.nusigF_th[ifuel:irefl] = refl['nusigF'][1]
        self.kapsigF_th[ifuel:irefl] = refl['kapsigF'][1]

        self.sigA_f[irefl:] = shld['sigA'][0]
        self.sigTr_f[irefl:] = shld['sigTr'][0]
        self.nusigF_f[irefl:] = shld['nusigF'][0]
        self.kapsigF_f[irefl:] = shld['kapsigF'][0]
        self.sigS12_f[irefl:] = shld['sigS12_fast_to_thermal']

        self.sigA_th[irefl:] = shld['sigA'][1]
        self.sigTr_th[irefl:] = shld['sigTr'][1]
        self.nusigF_th[irefl:] = shld['nusigF'][1]
        self.kapsigF_th[irefl:] = shld['kapsigF'][1]

        self.Dt_f = 1 / (3 * self.sigTr_f)
        self.Dt_th = 1 / (3 * self.sigTr_th)


def k_with_fuel_temperature(T_fuel, reactor):

    T_ref = 600.0
    beta_mod = 2.0e-4

    sigS12_temp = np.copy(reactor.sigS12_f)

    ifuel = int(geo['fuel'] / reactor.dx)

    temp_factor = 1.0 - beta_mod * (T_fuel - T_ref)
    temp_factor = max(temp_factor, 0.50)

    sigS12_temp[0:ifuel] *= temp_factor

    sig_R_1_temp = reactor.sigA_f + sigS12_temp

    return power_iteration(
        reactor.H,
        reactor.dx,
        reactor.Dt_f,
        reactor.Dt_th,
        sig_R_1_temp,
        reactor.sigA_th,
        reactor.nusigF_f,
        reactor.nusigF_th,
        reactor.kapsigF_f,
        reactor.kapsigF_th,
        sigS12_temp,
        1e-6,
        100.0,
        ['reflective', 'vacuum']
    )


Size = [
    int(geo['fuel']),
    int(geo['reflector']),
    int(geo['shield'])
]

hw_reactor = Reactor1D(H=Size, dx=1)

T_values = np.array([600, 700, 800, 900, 1000, 1100, 1200])

k_values = np.array([
    k_with_fuel_temperature(T, hw_reactor)
    for T in T_values
])

rho_values = (k_values - 1.0) / k_values

alpha_pcm_per_K = (
    (rho_values[-1] - rho_values[0])
    / (T_values[-1] - T_values[0])
    * 1e5
)

print("\nFuel temperature feedback study")
print("--------------------------------")

for T, k, rho in zip(T_values, k_values, rho_values):
    print(f"T = {T:.0f} K, k = {k:.5f}, rho = {rho*1e5:.1f} pcm")

print(f"\nEstimated fuel temperature coefficient = {alpha_pcm_per_K:.3f} pcm/K")
print(f"Reactivity inserted per 100 K = {alpha_pcm_per_K*100:.1f} pcm")

plt.figure()
plt.plot(T_values, k_values, marker='o')
plt.xlabel("Fuel Temperature [K]")
plt.ylabel("k-effective")
plt.title("Fuel Temperature Feedback Sensitivity")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(T_values, rho_values * 1e5, marker='o')
plt.xlabel("Fuel Temperature [K]")
plt.ylabel("Reactivity [pcm]")
plt.title("Fuel Temperature Reactivity Feedback")
plt.grid(True)
plt.show()
