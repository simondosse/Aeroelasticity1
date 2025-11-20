#%%
''' Import the necessary libraries '''
from ast import mod
from cProfile import label
from fileinput import filename
import os
from pyexpat import model
from turtle import mode
from matplotlib.pylab import eig
import numpy as np
import matplotlib.pyplot as plt
from input import ModelParameters
import ROM
import plotter
import NACA
from _functions_AS_optim import *

from data_manager import save_modal_data, _load_npz
from scipy.linalg import eigh

# modifie tous les parametres de matplotlib
# --- Configuration globale Matplotlib ---
# plt.rcParams.update({
#     'figure.figsize': [7.17, 4.34],       # Taille des figures (format article)
#     # 'text.usetex': True,                  # Active le rendu LaTeX
#     # 'text.latex.preamble': r'\usepackage{mathptmx}',  # Police Times pour le texte et les maths
#     'font.family': 'serif',               # Police avec empattements
#     'font.size': 11,                      # Taille globale du texte
#     'axes.labelsize': 11,                 # Taille des labels d'axes
# })

#%% Set and run models, save data

''' Structural parameters '''
s = 2 #half span
c = 0.2 #chord
x_ea = c/3 # elastic axis location from leading edge
x_cg = 0.379*c # center of gravity location from leading edge
m = 2.4 # mass per unit span

EIx = 366 # bending stiffness
GJ = 78 # torsional stiffness
eta_w = 0.005 # structural damping ratio in bending
eta_alpha = 0.005 # structural damping ratio in torsion, damping ratio are arbitrary choosen here



#%%_________________CAS IAT________________________________



''' Wingtip parameters '''

wingtip_mass_study = True
if wingtip_mass_study:
    Mt = 362e-3   # mass of the tip body
    I_alpha_t = 6.11e-4 # mass moment of inertia of the tip body
                    # I_alpha_t must depends on x_t right ??
    x_t = 0.007  # location of the tip body from leading edge (from the elastic axis isnt it ?)
else:
    Mt = None
    I_alpha_t = None
    x_t = None


s=1.5
Uc=70

# factor_GJ = np.random.uniform(0.8, 1.2)
# factor_EIx = np.random.uniform(0.8, 1.2)
# GJ = GJ * factor_GJ
# EIx = EIx * factor_EIx

GJ = 71.5
EIx= 264.1

model_maxime = ModelParameters(s, c, x_ea, x_cg, m, EIx, GJ, eta_w, eta_alpha, Mt, I_alpha_t, x_t, model_aero='Theodorsen')
model_maxime.Umax=50
model_maxime.Ustep=100


f, damping,eigvecs_U, f_modes_U, *_ = ROM.ModalParamDyn(model_maxime, tracked_idx=(0,1,2), track_using=None)
Vq_U = eigvecs_U[:,:model_maxime.Nq, :]
plotter.plot_modal_data_single(f,damping,model_maxime, suptitle='Modal data for cross-validation model')
Uc, _ , status = ROM.obj_evaluation(U = model_maxime.U, damping = damping[:,1], return_status=True)



plotter.plot_vi_grid(Vq=Vq_U[0,:,:], Nw=model_maxime.Nw, Nalpha=model_maxime.Nalpha, freqs_hz=f[0,:], kind='abs', normalize='l2', sharey=True, suptitle='Modal coefficients per mode',mode_indices=(0,1,2))

plotter.plot_vi_grid_over_U(U=model_maxime.U,
                            Vq_U=Vq_U,
                            Nw=model_maxime.Nw,
                            Nalpha=model_maxime.Nalpha,
                            f_modes_U=f_modes_U,
                            normalize = 'l2',
                            mode_indices=(0,1,2))



#%% __________________________ULiege wing configuration__________________
model_liege = ModelParameters(s=1.2, c=0.16, x_ea= c/4, x_cg=0.41*c, m=1.106, EIx=18.19, GJ=21.27, eta_w=0.005, eta_alpha=0.005, model_aero='Theodorsen')
model_liege.Umax=44.5
model_liege.Ustep=100
f, damping,eigvecs_U, f_modes_U, *_ = ROM.ModalParamDyn(model_liege, tracked_idx=(0,1,2), track_using=None)
Vq_U = eigvecs_U[:,:model_liege.Nq, :]
Uc, _ , status = ROM.obj_evaluation(U = model_liege.U, damping = damping[:,1], return_status=True)
plotter.plot_modal_data_single(f,damping,model_liege, Uc = Uc,colors = ['tab:blue','tab:green','tab:orange'], save = True, filename='model_liege_ref') #suptitle='Existing wing model - Frequencies and damping evolution'

plotter.plot_vi_grid(Vq=Vq_U[0,:,:], Nw=model_liege.Nw, Nalpha=model_liege.Nalpha, freqs_hz=f[0,:], kind='abs', normalize='l2', sharey=True, suptitle='Modal coefficients per mode',mode_indices=(0,1,2))
plotter.plot_vi_grid_over_U(U=model_liege.U,
                            Vq_U=Vq_U,
                            Nw=model_liege.Nw,
                            Nalpha=model_liege.Nalpha,
                            f_modes_U=f_modes_U,
                            normalize = 'l2',
                            mode_indices=(0,1,2))
#%%___________________________________________OPTIMAL WING____________________________________

algorithm_name = "DE"
target_mode_idx=1
save = True

data = np.load(f'data/res_target_{target_mode_idx}_'+algorithm_name+'.npz')
X_opt = map_to_physical(data['resX'])
s, c = 2.0, 0.2
m = 2.4
eta_w = 0.005
eta_alpha = 0.005
XX = [X_opt[0]*c,X_opt[1]*c,X_opt[2],X_opt[3]]
# x_ea = 0.05
# x_cg = 0.12
# EIx = 447
# GJ = 41
x_ea = XX[0]
x_cg = XX[1]
EIx = XX[2]
GJ = XX[3]
# x_cg=0.12
# EIx=300
model = ModelParameters(s, c, x_ea=x_ea, x_cg=x_cg, m=m, EIx=EIx, GJ=GJ, eta_w=eta_w, eta_alpha=eta_alpha,model_aero= 'Theodorsen')
model.airfoil.plot_naca00xx_section(save= True, filename=f'model_opt_naca00{int(model.airfoil.t_c*100)}_section_{target_mode_idx}')

f0, zeta0, eigvals0, eigvecs0, w_modes, alpha_modes, energy_dict = ROM.ModalParamAtRest(model) # normalize = 'per_field' or 'per_mode'
Vq = eigvecs0[:model.Nq, :]

phase0 = ROM._rel_phase_from_eigvec(model, Vq[:,0])
phase1 = ROM._rel_phase_from_eigvec(model, Vq[:,1])
phase2 = ROM._rel_phase_from_eigvec(model, Vq[:,2])
print(f'Phase à U = 0 > mode 0 : {phase0:.4f} rad, mode 1 : {phase1:.4f} rad, mode 2 : {phase2:.4f} rad')

model.Umax = 25
model.Ustep = 100

f, damping, eigvecs_U, f_modes_U, *_ = ROM.ModalParamDyn(model, tracked_idx=(0,1,2))
Vq_U = eigvecs_U[:,:model.Nq, :]
Uc, _ , status = ROM.obj_evaluation(U = model.U, damping = damping[:,target_mode_idx], return_status=True)
# plotter.plot_modal_data_single(f, damping, model, suptitle=fr"$EI_x$ = {model.EIx:.1f}, $GJ$ = {model.GJ:.1f}, $x_{{ea}}$ = {model.airfoil.x_ea:.3f}, $x_{{cg}}$ = {model.airfoil.x_cg:.3f}, $U_c$ = {Uc:.1f}",
#                                save = save, filename = f'modal_data_{target_mode_idx}')
plotter.plot_modal_data_single(f, damping, model, Uc = Uc,
                               save = save, filename = f'modal_data_{target_mode_idx}')

mask = (model.U >= Uc-5) & (model.U <= Uc+5)
phase_w_a_U_0 = ROM._rel_phase_from_eigvecs_over_U(model, Vq_U, target_mode_idx=0)
phase_w_a_U_1 = ROM._rel_phase_from_eigvecs_over_U(model, Vq_U, target_mode_idx=1)
plotter.plot_vi_wa_phase_over_U(model, model.U[mask], [phase_w_a_U_0[mask], phase_w_a_U_1[mask]], idx_modes=[0, 1],
                                save = save, filename = f'phase_w_a_over_U_{target_mode_idx}')
plotter.plot_vi_contribution_over_U(U = model.U,Vq_U = Vq_U, Nw=model.Nw, Nalpha=model.Nalpha,mode_index=0,
                                    save = save, filename = f'vi_contribution_over_U_{0}')

plotter.plot_vi_grid(Vq=Vq, Nw=model.Nw, Nalpha=model.Nalpha, freqs_hz=f0, kind='abs', normalize='l2', sharey=True, suptitle='Modal coefficients per mode',mode_indices=(0,1,2))
plotter.plot_vi_grid_over_U(U=model.U,
                            Vq_U=Vq_U,
                            Nw=model.Nw,
                            Nalpha=model.Nalpha,
                            f_modes_U=f_modes_U,
                            normalize = 'l2',
                            mode_indices=(0,1,2),
                            save = save, filename = f'vi_grid_over_U_{target_mode_idx}')

# idx_U = np.where(model.U==Uc)[0][0]
# phase0_U = ROM._rel_phase_from_eigvec(model, Vq_U[idx_U,:,0])
# phase1_U = ROM._rel_phase_from_eigvec(model, Vq_U[idx_U,:,1])
# phase2_U = ROM._rel_phase_from_eigvec(model, Vq_U[idx_U,:,2])
# print(f'Phase à U = {model.U[idx_U]:.1f} m/s > mode 0 : {phase0_U:.4f} rad, mode 1 : {phase1_U:.4f} rad, mode 2 : {phase2_U:.4f} rad')

# phi1 = np.angle(Vq_U[idx_U,0,1])
# phi2 = np.angle(Vq_U[idx_U,3,1])
# dphi = phi2 - phi1





#%% ___________Temporal simulation and Work over the span______________________
# model = model_maxime
t0 = 0
tf = 2
dt = 0.001
t = np.arange(t0, tf+dt, dt)

X0 = ROM.build_state_q_from_real(
    par=model,
    w_tip=0.01,        # m
    alpha_tip=0.01,     # rad
    wdot_tip=0.0,
    alphadot_tip=0.0
) # même si on veut simuler une réponse temporelle avec un U!=0 faut mettre un petit w0 ou aplha0 sinon tous les efforts symétriques s'annulent parfaitement

'''
thanks to range kutta we get the temporal solutions of a initial state X0 and a freestream speed U
then we plot w(y,t) alpha(y,t)
then we plot w(y=s,t) alpha(y=s,t) + FFT
'''

f, damping, eigvecs_U, f_modes_U, *_ = ROM.ModalParamDyn(model, tracked_idx= (0,1,2,3,4,5))
coupled_mode_idx = np.array([0,1])

U = 26
# we get the freq at U
idx = np.where(model.U >= U)[0][0]
f_at_U = f[idx,:]
omega_ref = 2*np.pi*(f_at_U[coupled_mode_idx[0]]+f_at_U[coupled_mode_idx[1]])*0.5

t,X,A = ROM.integrate_state_rk(par = model, U = U , t=t, x0 = X0, rk_order=4)
# ROM.plot_tip_time_and_fft(par = model, t=t,X=X, U=U, detrend=True)



# we got back to physical space for plotting
'''
q and qdot obtained from X
then w(y,t) and alpha(y,t) from q(t) and qdot(t)
qddot(t) from A @ X.T
then Q_aero(y,t) from q(t), qdot(t) and qddot(t)
'''
q = X[:, :model.Nq]
qdot = X[:,model.Nq:]
qdot_w = X[:,model.Nq:model.Nq+model.Nw]
qdot_a = X[:,model.Nq+model.Nw:model.Nq+model.Nw+model.Nalpha]
qddot = (A @ X.T).T[:,model.Nq:model.Nq+model.Nw+model.Nalpha]

wdot_map, alphadot_map, Phi_w, Phi_a = ROM._modal_to_physical_fields(model, qdot_w, qdot_a, return_shapes=True)
Ka, Ca, Ma = ROM.TheodoresenAeroModel(par=model, U=U, omega=omega_ref)

Q_aero = -(Ma @ qddot.T + Ca @ qdot.T + Ka @ q.T).T # aero forces in q space

f_w = Q_aero[:, :model.Nw] @ Phi_w # bending forces along the span : f_w(y) = Σ_i Q_aero_w[i] * phi_w_i(y)
m_a = Q_aero[:, model.Nw:model.Nw+model.Nalpha] @ Phi_a  # torsional forces along the span
p_w = f_w * wdot_map  # power from bending forces : p_w(y,t) = f_w(y,t) * wdot(y,t)
p_a = m_a * alphadot_map  # power from torsional forces : p_a(y,t) = m_a(y,t) * alphadot(y,t)
p = p_w + p_a

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(t, p_w[:,-1], lw=1,label='bending contribution')
ax.plot(t, p_a[:,-1], lw=1,label='torsion contribution')
ax.plot(t, p[:,-1], lw=1,label='total')
ax.legend()
ax.set_title(f'Power at wingtip at U={U} m/s, all modes considered')
plt.show()
# we calcule the energy over a period :
W_w = np.zeros(model.y.shape)
W_a = np.zeros(model.y.shape)
W = np.zeros(model.y.shape)
T_period = 1/ (omega_ref/(2*np.pi))
t0=0.5 # to avoid initial transient
for i in range(len(model.y)):
    yi = model.y[i]
    p_a_yi = p_a[:,i]
    p_w_yi = p_w[:,i]
    p_yi = p[:,i]
    # we integrate over a period
    mask_time = (t >= t0) & (t <= t0 + T_period)
    W_w[i] = np.trapezoid(p_w_yi[mask_time], t[mask_time])
    W_a[i] = np.trapezoid(p_a_yi[mask_time], t[mask_time])
    W[i] = np.trapezoid(p_yi[mask_time], t[mask_time])

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(W_w, model.y, lw=1, label='bending contribution')
ax.plot(W_a, model.y, lw=1, label='torsion contribution')
ax.plot(W, model.y, lw=1, label='total')
ax.legend()
ax.vlines(0, ymin=0, ymax=model.s, colors='k', linestyles='--', lw=0.8)

ax.set_xlim((-max(np.abs(W)*1.1),max(np.abs(W)*1.1)))
ax.set_xlabel('Forces work on a period [J]')
ax.set_ylabel('Spanwise location y [m]')
ax.set_title(f'Energy distribution along the span at U = {U} m/s')
ax.grid(True, linewidth=0.3, alpha=0.5)

'''
the energy (positive or negative) is mostly echanged at the tip bc of the larger amplitudes of the DOFs there.
even for the unstable set of parameters, the energy is <0, because of the stable mode that dissipate energy more than
the unstable mode steals energy. The forces actings on stable mode are larger than those on unstable mode.

we must look mode per mode to see > and < 0 regions

'''
#%% same but mode per mode
k=0
W_tot = np.zeros(model.y.shape)
# for k in [0,1,2,3,4,5]:
idx_U=-1
v_k = eigvecs_U[idx_U,:model.Nq, k]
eta = np.linalg.pinv(eigvecs_U[idx_U,:model.Nq,:]) @ q.T # eta = V^-1 @ q (enfin c'est qu'une partie de eigvecs_U, la supérieure))
# eta de taille nmodes x ntimes, on prend la k-ieme ligne pour avoir eta_k(t)
eta_dot = np.linalg.pinv(eigvecs_U[idx_U,:model.Nq,:]) @ qdot.T
eta_ddot = np.linalg.pinv(eigvecs_U[idx_U,:model.Nq,:]) @ qddot.T


q_k = np.outer(eta[k,:], v_k).real #equivalent to et .* on matlab
qdot_k  = np.outer(eta_dot[k,:],  v_k).real
qddot_k = np.outer(eta_ddot[k,:], v_k).real

q_k_w = q_k[:, :model.Nw]
q_k_a = q_k[:, model.Nw:model.Nw+model.Nalpha]
qdot_k_w = qdot_k[:, :model.Nw]
qdot_k_a = qdot_k[:, model.Nw:model.Nw+model.Nw+model.Nalpha]
qddot_k_w = qddot_k[:, :model.Nw]
qddot_k_a = qddot_k[:, model.Nw:model.Nw+model.Nw+model.Nalpha]

Q_aero_k = -(Ma @ qddot_k.T + Ca @ qdot_k.T + Ka @ q_k.T).T  # (nt, Nq), on est encore en coordonnées de Ritz
f_w_k = Q_aero_k[:, :model.Nw] @ Phi_w #on repasse en espace physique
m_a_k = Q_aero_k[:, model.Nw:model.Nw+model.Nalpha] @ Phi_a

wdot_map, alphadot_map, _ , _ = ROM._modal_to_physical_fields(model, qdot_k_w, qdot_k_a, return_shapes=True)
p_k_w = f_w_k* wdot_map
p_k_a = + m_a_k * alphadot_map
p_k = p_k_w + p_k_a
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(t, p_k_w[:,-1], lw=1,label='bending contribution')
ax.plot(t, p_k_a[:,-1], lw=1,label='torsion contribution')
ax.plot(t, p_k[:,-1], lw=1,label='total')
ax.legend()
ax.set_title(f'Power at wingtip for mode {k} at U={U} m/s')
plt.show()

W_w = np.zeros(model.y.shape)
W_a = np.zeros(model.y.shape)
W = np.zeros(model.y.shape)

T_period = 1/ (omega_ref/(2*np.pi))
t0=0.5 # to avoid initial transient
for i in range(len(model.y)):
    yi = model.y[i]
    p_w_k_yi = p_k_w[:,i]
    p_a_k_yi = p_k_a[:,i]
    p_yi = p_k[:,i]
    # we integrate over a period
    mask_time = (t >= t0) & (t <= t0 + T_period)
    W_w[i] = np.trapezoid(p_w_k_yi[mask_time], t[mask_time])
    W_a[i] = np.trapezoid(p_a_k_yi[mask_time], t[mask_time])
    W[i] = np.trapezoid(p_yi[mask_time], t[mask_time])
# W_tot[i] += W[i]

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(W_w, model.y, lw=1, label='bending contribution')
ax.plot(W_a, model.y, lw=1, label='torsion contribution')
ax.plot(W, model.y, lw=1, label='total')
ax.vlines(0, ymin=0, ymax=model.s, colors='k', linestyles='--', lw=0.8)

ax.set_xlim((-max(np.abs(W)*1.1),max(np.abs(W)*1.1)))
ax.set_xlabel('Forces work on a period [J]')
ax.set_ylabel('Spanwise location y [m]')
ax.set_title(f'Energy distribution along the span at U = {U} m/s' + f' for mode {k}')
ax.legend()
ax.grid(True, linewidth=0.3, alpha=0.5)

# fig, ax = plt.subplots(constrained_layout=True)
# ax.plot(W_tot, model.y, lw=1, label='total')
# ax.vlines(0, ymin=0, ymax=model.s, colors='k', linestyles='--', lw=0.8)

# ax.set_xlim((-max(np.abs(W_tot)*1.1),max(np.abs(W)*1.1)))
# ax.set_xlabel('Forces work on a period [J]')
# ax.set_ylabel('Spanwise location y [m]')
# ax.set_title(f'Energy distribution along the span at U = {U} m/s' + f' sum modes')
# ax.grid(True, linewidth=0.3, alpha=0.5)


#%%______________plot animation

fig,ani = plotter.animate_beam(par=model_opt, t=t, X=X, U=U, n_stations=12, scale_w=1.0, scale_alpha=1.0, scale_chord=1.0, show_airfoil=False,save_path='animations/model_opt.gif')




# %%______''' Test de la fonction _phase_align_column'''____________________

# from ROM import _phase_align_column
# import numpy as np
# import matplotlib.pyplot as plt
# avant = np.array([
#     -2.93409564e-04 - 3.14411317e-02j,
#      1.13616867e-07 - 4.13653348e-05j,
#      3.37975911e-09 - 9.35489643e-07j,
#     -3.67894319e-04 - 8.58101896e-02j,
#      4.78765260e-06 + 2.21947399e-03j,
#     -2.06706323e-07 - 1.20589727e-04j
# ])
# apres,k0,arg = _phase_align_column(avant) # sortie de ta fonction

# print(f"Composante de ref k0 = {k0}, déphasage appliqué (rad) = {arg:.6f}")

# # --- Magnitude --- le module n'est évidemment pas changé
# plt.figure()
# plt.plot(np.abs(avant), label='Avant (|Vq[:, i]|)')
# plt.plot(np.abs(apres), linestyle='--', label='Après (|vi|)')
# plt.xlabel('Indice k')
# plt.ylabel('Amplitude')
# plt.title('Amplitude avant / après alignement de phase')
# plt.legend()
# plt.grid(True)

# # --- Phase (déroulée) ---
# plt.figure()
# phase_avant = np.unwrap(np.angle(avant))
# phase_apres = np.unwrap(np.angle(apres))
# plt.plot(phase_avant, label='Avant (phase déroulée)')
# plt.plot(phase_apres, linestyle='--', label='Après (phase déroulée)')
# plt.xlabel('Indice k')
# plt.ylabel('Phase [rad]')
# plt.title('Phase avant / après alignement')
# plt.legend()
# plt.grid(True)

# # --- Plan complexe (nuage de points) ---
# plt.figure()
# plt.plot(avant.real, avant.imag, 'o', label='Avant')
# plt.plot(apres.real, apres.imag, 'x', label='Après')
# plt.xlabel('Réel')
# plt.ylabel('Imaginaire')
# plt.axis('equal')
# plt.title('Plan complexe : avant vs après')
# plt.legend()
# plt.grid(True)

# plt.show()

