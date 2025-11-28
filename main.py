#%%
''' Import the necessary libraries '''
from ast import mod
from cProfile import label
from fileinput import filename
import os
from pyexpat import model
import time
from turtle import mode
from matplotlib.pylab import eig
import numpy as np
import matplotlib.pyplot as plt
from input import ModelParameters
import ROM
import plotter
import NACA
from _functions_AS_optim import *

import copy

from data_manager import save_modal_data, _load_npz
from scipy.linalg import eigh


#%%___________________________IAT WING__________________________

''' Structural parameters '''
s=1.5 # span
c = 0.2 #chord
x_ea = c/3 # elastic axis location from leading edge
x_cg = 0.379*c # center of gravity location from leading edge
m = 2.4 # mass per unit span

EIx = 366 # bending stiffness
GJ = 78 # torsional stiffness
eta_w = 0.005 # structural damping ratio in bending
eta_alpha = 0.005 # structural damping ratio in torsion, damping ratio are arbitrary choosen here

dCL = 3.8999
dCM = 0.4693

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

# factor_GJ = np.random.uniform(0.8, 1.2)
# factor_EIx = np.random.uniform(0.8, 1.2)
# GJ = GJ * factor_GJ
# EIx = EIx * factor_EIx

# GJ = 71.5
# EIx= 264.1

model_iat = ModelParameters(s, c, x_ea, x_cg, m, EIx, GJ, eta_w, eta_alpha,
                            dCL = dCL, dCM = dCM,
                            Mt = Mt , I_alpha_t = I_alpha_t , x_t = x_t, model_aero='Theodorsen')
model_iat.airfoil.plot_naca00xx_section()

model_iat.Umax=35
model_iat.Ustep=100

# plotter.plot_params_table(model_iat, save = True, filename='model_iat_params_table')
f, damping,eigvecs_U, f_modes_U, *_ = ROM.ModalParamDyn(model_iat, tracked_idx=(0,1,2), track_using=None)
Vq_U = eigvecs_U[:,:model_iat.Nq, :]
Uc, _ , status = ROM.obj_evaluation(U = model_iat.U, damping = damping[:,1], return_status=True)

model_iat.Vq_U = Vq_U
model_iat.f = f
model_iat.damping = damping
model_iat.Uc = Uc

plotter.plot_modal_data_single(f,damping,model_iat, suptitle='IAT wing model - Frequencies and damping evolution', figsize=(6,4))

plotter.plot_vi_grid(Vq=Vq_U[0,:,:],
                     Nw=model_iat.Nw, Nalpha=model_iat.Nalpha,
                     freqs_hz=f[0,:], kind='abs', normalize='l2',
                     sharey=True, suptitle='Modal coefficients per mode',mode_indices=(0,1,2))
plotter.plot_vi_grid(Vq=Vq_U[-1,:,:],
                     Nw=model_iat.Nw, Nalpha=model_iat.Nalpha,
                     freqs_hz=f[0,:], kind='abs', normalize='l2',
                     sharey=True, suptitle='Modal coefficients per mode',mode_indices=(0,1,2))
plotter.plot_vi_grid_over_U(U=model_iat.U,
                            Vq_U=Vq_U,
                            Nw=model_iat.Nw,
                            Nalpha=model_iat.Nalpha,
                            f_modes_U=f_modes_U,
                            normalize = 'l2',
                            mode_indices=(0,1,2))



#%% __________________________ULiege WING_______________________
save=False

Mt = None
I_alpha_t = None
x_t = None

s=1.2
c=0.16
x_ea = c/4
x_cg = 0.41*c
m=1.106
EIx=18.9
GJ=21.27
eta_w=0.005
eta_alpha=0.005

model_liege = ModelParameters(s=s, c=c, x_ea= x_ea, x_cg=x_cg, m=m, EIx=EIx, GJ=GJ, eta_w=eta_w, eta_alpha=eta_alpha, model_aero='Theodorsen')
model_liege.airfoil.plot_naca00xx_section()
model_liege.Umax=50
model_liege.Ustep=150
# model_liege.airfoil.Ialpha_EA = 2.587e-3
model_liege.dCL=4   # 4 to get the same results as Maxime's paper
model_liege.dCM=0.3 # 0.3 to get the same results as Maxime's paper
# plotter.plot_params_table(model_liege, save = True, filename='model_liege_params_table')

f, damping,eigvecs_U, f_modes_U, *_ = ROM.ModalParamDyn(model_liege, tracked_idx=(0,1,2), track_using=None)
Vq_U = eigvecs_U[:,:model_liege.Nq, :]
Uc, _ , status = ROM.obj_evaluation(U = model_liege.U, damping = damping[:,1], return_status=True)

model_liege.Vq_U = Vq_U
model_liege.f = f
model_liege.damping = damping
model_liege.Uc = Uc

plotter.plot_modal_data_single(f,damping,model_liege,colors = ['tab:blue','tab:green','tab:orange'],#suptitle='Liège wing model - Frequencies and damping evolution',
                            #figsize=(6,4),
                                Uc = Uc,
                               save = save, filename='model_liege_ref') #

plotter.plot_vi_grid(Vq=Vq_U[0,:,:], Nw=model_liege.Nw, Nalpha=model_liege.Nalpha, freqs_hz=f[0,:], kind='abs', normalize='l2', sharey=True, suptitle='Modal coefficients per mode',mode_indices=(0,1,2))
plotter.plot_vi_grid_over_U(U=model_liege.U,
                            Vq_U=Vq_U,
                            Nw=model_liege.Nw,
                            Nalpha=model_liege.Nalpha,
                            f_modes_U=f_modes_U,
                            normalize = 'l2',
                            mode_indices=(0,1,2))
#%%___________________________OPTIMAL WING______________________

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

x_ea = XX[0]
x_cg = XX[1]
EIx = XX[2]
GJ = XX[3]
# x_cg=0.12
# EIx=300
model = ModelParameters(s, c, x_ea=x_ea, x_cg=x_cg, m=m, EIx=EIx, GJ=GJ, eta_w=eta_w, eta_alpha=eta_alpha,model_aero= 'Theodorsen')
model.Umax = 25
model.Ustep = 100
model.airfoil.plot_naca00xx_section(save= save, filename=f'model_opt_naca00{int(model.airfoil.t_c*100)}_section_{target_mode_idx}')
# plotter.plot_params_table(model, save = True, filename=f'model_opt_{target_mode_idx}_params_table')

f0, zeta0, eigvals0, eigvecs0, w_modes, alpha_modes, energy_dict = ROM.ModalParamAtRest(model) # normalize = 'per_field' or 'per_mode'
Vq = eigvecs0[:model.Nq, :]

# phase0 = ROM._rel_phase_from_eigvec(model, Vq[:,0])
# phase1 = ROM._rel_phase_from_eigvec(model, Vq[:,1])
# phase2 = ROM._rel_phase_from_eigvec(model, Vq[:,2])
# print(f'Phase à U = 0 > mode 0 : {phase0:.4f} rad, mode 1 : {phase1:.4f} rad, mode 2 : {phase2:.4f} rad')

f, damping, eigvecs_U, f_modes_U, *_ = ROM.ModalParamDyn(model, tracked_idx=(0,1,2))
Vq_U = eigvecs_U[:,:model.Nq, :]
Uc, _ , status = ROM.obj_evaluation(U = model.U, damping = damping[:,target_mode_idx], return_status=True)
model.Vq_U = Vq_U
model.f = f
model.damping = damping
model.Uc = Uc



# plotter.plot_modal_data_single(f, damping, model, suptitle=fr"$EI_x$ = {model.EIx:.1f}, $GJ$ = {model.GJ:.1f}, $x_{{ea}}$ = {model.airfoil.x_ea:.3f}, $x_{{cg}}$ = {model.airfoil.x_cg:.3f}, $U_c$ = {Uc:.1f}",
#                                save = save, filename = f'modal_data_{target_mode_idx}')
# plotter.plot_modal_data_single(f, damping, model, Uc = Uc, suptitle = 'Optimal wing model - Frequencies and damping evolution',
#                                figsize=(6,4),
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


# # plot des contributions en w et alpha pour le mode instable à U = 0
# plotter.plot_vi_grid(Vq=Vq, Nw=model.Nw, Nalpha=model.Nalpha, freqs_hz=f0, kind='abs', normalize='l2', sharey=True, suptitle=r"Modal coefficients per mode - $U = 0 m/s$",mode_indices=(0,1,2))

# plot des contributions en w et alpha pour le mode instable à U = U[-1]
plotter.plot_vi_grid(Vq=Vq_U[-1,:,:], Nw=model.Nw, Nalpha=model.Nalpha, freqs_hz=f0, kind='abs', normalize='l2', sharey=True,mode_indices=(0,1), # suptitle=rf"$U = {model.U[-1]} m/s$",
                     save = save, filename = f'vi_grid_Uc_{target_mode_idx}')



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
# model = model_iat
t0, tf, dt = (0, 2, 0.001)
t = np.arange(t0, tf+dt, dt)

X0 = ROM.build_state_X_from_real(
    par=model,
    w_tip=0.05,        # m
    alpha_tip=0.2,     # rad
    wdot_tip=0.0,
    alphadot_tip=0.0,
    q_content={'bending':1, 'torsion':0}
) # même si on veut simuler une réponse temporelle avec un U!=0 faut mettre un petit w0 ou aplha0 sinon tous les efforts symétriques s'annulent parfaitement
'''
the way we build X0 is very important, if we do X [q; qdot] such that q = pinv(phi_w1(y=s))*w0, q will be computed to minimize the norm of q,
then it will excite all the modes that have a contribution in w at the tip, even the very high freq ones with low damping that we dont want to excite

to just excite B1/T1, B2/T2 modes, we can build X0 with ROM.build_state_X_from_real(......., q_content={'bending':1, 'torsion':1} )
'''

'''
thanks to range kutta we get the temporal solutions of a initial state X0 and a freestream speed U
then we plot w(y,t) alpha(y,t)
then we plot w(y=s,t) alpha(y=s,t) + FFT
'''

f, damping, eigvecs_U, f_modes_U, *_ = ROM.ModalParamDyn(model, tracked_idx= (0,1,2,3,4,5))
coupled_mode_idx = np.array([0,1])

U = 24.9
# we get the freq at U, we will only look at that speed
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
q_w = q[:, :model.Nw]
q_a = q[:, model.Nw:model.Nw+model.Nalpha]
qdot = X[:,model.Nq:]
qdot_w = X[:,model.Nq:model.Nq+model.Nw]
qdot_a = X[:,model.Nq+model.Nw:model.Nq+model.Nw+model.Nalpha]
qddot = (A @ X.T).T[:,model.Nq:model.Nq+model.Nw+model.Nalpha]
qddot_w = qddot[:, :model.Nw]
qddot_a = qddot[:, model.Nw:model.Nw+model.Nalpha]

# we compute w, alpha, wdot, alphadot, wdotdot, alphadotdot from q, qdot, qddot w = Phi_w @ q_w, alpha = Phi_a @ q_a
# it's map (t,y)

w_map, alpha_map, wdot_map, alphadot_map, wdotdot_map, alphadotdot_map = ROM.gen_coord_to_physical_all(model, q_w, q_a, qdot_w, qdot_a, qddot_w, qddot_a)


#%% __________Power and energy along the span - from F_A = -(Mddq + Cdq + Kq) __________________

Ka, Ca, Ma = ROM.TheodoresenAeroModel(par=model, U=U, omega=omega_ref)

Phi_w, _, _ = ROM.bendingModeShapes(model)
Phi_a, _, _ = ROM.torsionModeShapes(model)

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
k=1
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

wdot_map, alphadot_map, _ , _ = ROM.gen_coord_to_physical(model, qdot_k_w, qdot_k_a, return_shapes=True)
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






#%% _____Work L M W-C notation, forced motion at the tip E(y=s,\theta)_____

''' 
Work ONLY AT the tip for diff shifts between w_tip and alpha_tip
we impose a motion at the tip : w_tip(t), alpha_tip(t)
then we compute L(t), M(t) at the tip with ROM.computeLiftMoment
we compute p_w(t) = L(t)*wdot_tip(t) and p_a(t) = M(t)*alphadot_tip(t)
then we integrate over a period to get E_w and E_a at the tip
we repeat for different shifts between w_tip and alpha_tip
'''
w0 = 0.1  # m
alpha0 = 0.6  # rad
omega = 2 * np.pi * model.f[idx_U, mode]             # rad/s
T_period = 2 * np.pi / omega

nb_shifts = 60
shifts = np.linspace(-np.pi, np.pi, nb_shifts)

E_w = np.zeros(nb_shifts)
E_a = np.zeros(nb_shifts)
for i, shift in enumerate(shifts):
    w_tip = w0*np.sin(omega*t+shift)  # m
    wdot_tip = w0*omega*np.cos(omega*t+shift)  # m/s
    wdotdot_tip = -w0*omega**2*np.sin(omega*t+shift)  # m/s2
    alpha_tip = alpha0*np.sin(omega*t)      # rad
    alphadot_tip = alpha0*omega*np.cos(omega*t)      # rad/s
    alphadotdot_tip = -alpha0*omega**2*np.sin(omega*t)      # rad/s2
    L, M = ROM.computeLiftMoment(par = model,U = 25,w=w_tip,alpha=alpha_tip,wdot=wdot_tip,alphadot=alphadot_tip,wdotdot=wdotdot_tip,alphadotdot=alphadotdot_tip,omega=omega)
    p_w = -L*wdot_tip
    p_a = M*alphadot_tip
    mask_time = (t >= t0) & (t <= t0 + T_period)
    E_w[i] = np.trapezoid(p_w[mask_time], t[mask_time])
    E_a[i] = np.trapezoid(p_a[mask_time], t[mask_time])

fig,ax = plt.subplots(constrained_layout=True)
ax.plot(shifts, E_w, label='bending contribution')
ax.plot(shifts, E_a, label='torsion contribution')
ax.plot(shifts, E_w+E_a, label='total')
ax.set_xlabel('Phase shift between w and alpha at the tip [rad]')
ax.set_ylabel('Forces work over a period [J]')
ax.set_title('Forces work over a period vs phase shift at U=25 m/s')
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.axvline(-np.pi/2, color='k', lw=1.1, ls=':', alpha=0.7)
ax.axvline(np.pi/2, color='k', lw=1.1, ls=':', alpha=0.7)
ax.legend()
plt.show()

#%%________________Work ALONG the span for a given shift____________________
'''
we impose the motion at the tip : w_tip(t), alpha_tip(t)
then we compute the state X_forced(t) along the span with ROM.build_state_X_from_real
imposing a q form to force either B1/T1 or B2/T1

we compute a q at each time step 
'''
save = True

model = model
mode = 1                     # ex : mode 2 (index 1)
idx_U = -1                   # ex : U = max
U = model.U[idx_U]
omega = 2 * np.pi * model.f[idx_U, mode]  
T_period = (2*np.pi)/omega

t0, tf, dt = (0, 2, 0.001)
t = np.arange(t0, tf+dt, dt)

# arbitrary imposed amplitudes
w0 = 0.1  # m
alpha0 = 0.6  # rad

shift = np.pi/2 # shift between w_tip and alpha_tip (positif = w leads alpha)

w_tip = w0*np.sin(omega*t+shift)  # m
wdot_tip = w0*omega*np.cos(omega*t+shift)  # m/s
wdotdot_tip = -w0*omega**2*np.sin(omega*t+shift)  # m/s2
alpha_tip = alpha0*np.sin(omega*t)      # rad
alphadot_tip = alpha0*omega*np.cos(omega*t)      # rad/s
alphadotdot_tip = -alpha0*omega**2*np.sin(omega*t)      # rad/s2

# L_i, M_i = ROM.computeLiftMoment(par = model,U = 25,w=w_tip[10],alpha=alpha_tip[10],wdot=wdot_tip[10],alphadot=alphadot_tip[10],wdotdot=wdotdot_tip[10],alphadotdot=alphadotdot_tip[10],omega=omega_ref)
U=25
for k in [0,1]:
    q_content = {
        'bending' :k,
        'torsion' :0
    }

    X_forced = np.zeros((len(t), model.Nq*2))
    qddot_forced = np.zeros((len(t), model.Nq))
    for i in range(len(t)):
        X_forced[i,:], qddot_forced[i,:] = ROM.build_state_X_from_real(par=model,w_tip=w_tip[i], alpha_tip=alpha_tip[i],
                                                                wdot_tip=wdot_tip[i], alphadot_tip=alphadot_tip[i],
                                                                wdotdot_tip=wdotdot_tip[i], alphadotdot_tip=alphadotdot_tip[i],
                                                                accel = True,
                                                                q_content = q_content)

    ROM.plot_w_alpha_fields_from_X(model, t, X_forced, U = U, times_to_plot = np.linspace(0, t[int(len(t)/20)], 6))

    A = ROM.stateMatrixAero(par=model, U=U, omega=omega)
    q_forced = X_forced[:, :model.Nq]
    q_w_forced = q_forced[:, :model.Nw]
    q_a_forced = q_forced[:, model.Nw:model.Nw+model.Nalpha]
    qdot_forced = X_forced[:,model.Nq:]
    qdot_w_forced = qdot_forced[:, :model.Nw]
    qdot_a_forced = qdot_forced[:, model.Nw:model.Nw+model.Nalpha]
    qddot_forced = (A @ X_forced.T).T[:,model.Nq:model.Nq+model.Nw+model.Nalpha]
    qddot_w_forced = qddot_forced[:, :model.Nw]
    qddot_a_forced = qddot_forced[:, model.Nw:model.Nw+model.Nalpha]

    # we compute w_*_map, alpha_*_map from q_w_*_forced, q_a_*_forced
    w_map_forced, alpha_map_forced, wdot_map_forced, alphadot_map_forced, wdotdot_map_forced, alphadotdot_map_forced = ROM.gen_coord_to_physical_all(model, q_w_forced, q_a_forced, qdot_w_forced, qdot_a_forced, qddot_w_forced, qddot_a_forced)

    # we plot the wing shape at t=0 to see which modes composed it
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4))
    ax.plot(model.y, w_map_forced[0,:], label='w tip from simulation')
    ax.set_xlim((0, model.s))
    ax.set_xlabel('Spanwise location y [m]')
    ax.set_ylim((-0.2, 0.2))
    ax.set_ylabel('Vertical displacement w [m]')
    ax.set_aspect('equal', adjustable='box')

    # we compute the forces L and M (t,y)
    L, M = ROM.computeLiftMoment_along_span(par = model, U = U,
                                            w_map = w_map_forced, alpha_map = alpha_map_forced, 
                                            wdot_map = wdot_map_forced, alphadot_map = alphadot_map_forced,
                                            wdotdot_map = wdotdot_map_forced, alphadotdot_map = alphadotdot_map_forced,
                                            t=t, omega = omega) 
    
    # we compute the power and energy distributions at each spanwise location
    p_w, p_a, p, E_w, E_a, E = ROM.power_work_computations(par=model,L=L,M=M,wdot_map=wdot_map_forced,alphadot_map=alphadot_map_forced,t=t,omega=omega)
    plotter.plot_aero_work_distribution(model,U=U, E_w=E_w, E_a=E_a, E=E,
                                        save = save, filename=f'aero_work_imposed_motion_B{q_content['bending']+1}_T{q_content['torsion']+1}')

    # we can finally sum the energy over the span to get total energy
    E_w_total = np.trapezoid(E_w, model.y)
    E_a_total = np.trapezoid(E_a, model.y)
    E_total = np.trapezoid(E, model.y)
    print(f'Total energy over a period at U={U} m/s : bending = {E_w_total:.6f} J, torsion = {E_a_total:.6f} J, total = {E_total:.6f} J')

    # we store the results for B1T1 and B2T1 for later comparison
    if k ==0:
        p_w1, p_a1, p1, E_w1, E_a1, E1 = p_w, p_a, p, E_w, E_a, E
    elif k==1:
        p_w2, p_a2, p2, E_w2, E_a2, E2 = p_w, p_a, p, E_w, E_a, E

plotter.plot_two_aero_work_distributions(
    par = model,
    U1=U, E_w1=E_w1, E_a1=E_a1, E1=E1,
    U2=U, E_w2=E_w2, E_a2=E_a2, E2=E2,
    save=save
)



#%%_________________Construct harmonic motion from eigenvector at U______________________

model = copy.deepcopy(model)
# model = copy.deepcopy(model_iat)
# model = copy.deepcopy(model_liege)

# mode shape rayleigh-ritz
Phi_w, *_ = ROM.bendingModeShapes(model)
Phi_a, *_ = ROM.torsionModeShapes(model)


mode = 1                     # le mode que l'on veut recréer, souvent celui instable, ex : mode 2 (index 1)
idx_U = -1                   # ex : U = max
U = model.U[idx_U]
v = Vq_U[idx_U, :, mode]     # (Nq,) complexe

# 1) Mettre tout dans la même phase, on choisit T1 comme la ref de phase et d'amplitude
k_ref = 3  # indice de la composante de référence (T1)
v_aligned = v * np.exp(-1j * np.angle(v[k_ref]))

# aeroelastic mode shapes psi_w(y) et psi_a(y)
psi_w = Phi_w.T @ v_aligned[:model.Nw]                  # (Ny,) complexe
psi_a = Phi_a.T @ v_aligned[model.Nw:model.Nq]      # (Ny,) complexe

# 2) On calcule le mode aeroelastique psi(y) pour w et alpha en tant que CL des modes propres des déformations de Rayleigh-Ritz Phi_w Phi_a

'''
we note that here psi_w and psi_a are juste for the considered mode v^i (eigenvector of A(U))
'''

# 3) Construire le mouvement harmonique
omega = 2 * np.pi * model.f[idx_U, mode]             # rad/s
T_period = 2 * np.pi / omega

t0, tf, dt = (0, 2, 0.001)
t = np.arange(t0, tf+dt, dt)
phasor = np.exp(1j * omega * t)[:, None]

# we build w(t,y) and alpha(t,y) as real parts of the whole stuff
w = np.real(phasor * psi_w[None, :])               
wdot = np.real(1j * omega * phasor * psi_w[None, :])
wdotdot = np.real(-omega**2 * phasor * psi_w[None, :])
alpha = np.real(phasor * psi_a[None, :])       
alphadot = np.real(1j * omega * phasor * psi_a[None, :])
alphadotdot = np.real(-omega**2 * phasor * psi_a[None, :])

plotter.plot_w_alpha_fields(par=model, t=t,
                            w_map = w, a_map = alpha,
                            times_to_plot = np.linspace(0, T_period, 10),
                            U = U)

L, M = ROM.computeLiftMoment_along_span(par = model, U = U,
                                        w_map = w, alpha_map = alpha, 
                                        wdot_map = wdot, alphadot_map = alphadot,
                                        wdotdot_map = wdotdot, alphadotdot_map = alphadotdot,
                                        t=t, omega = omega) 

# we compute the power and energy distributions at each spanwise location
p_w, p_a, p, E_w, E_a, E = ROM.power_work_computations(par=model,L=L,M=M,wdot_map=wdot,alphadot_map=alphadot,t=t,omega=omega)
plotter.plot_aero_work_distribution(model,U=U, E_w=E_w, E_a=E_a, E=E,
                                    save = False, filename=f'aero_work_eigenmode_mode{mode+1}_U{model.U[idx_U]:.1f}')




# Option : fixer l’amplitude au tip
# tip_target = 0.2  # rad, par exemple
# scale = tip_target / np.max(np.abs(psi_w[-1]))
# w *= scale
# alpha *= scale


#%% plot deformation mode shapes
phi_normalized, phi_dot_normalized, phi_dotdot_normalized = ROM.bendingModeShapes(model)
fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4))
ax.plot(model.y, phi_normalized[0,:], label='Mode shape bending 1')
ax.plot(model.y, phi_normalized[1,:], label='Mode shape bending 2')
ax.plot(model.y, phi_normalized[2,:], label='Mode shape bending 3')
ax.set_xlim((0, model.s))
ax.set_xlabel('Spanwise location y [m]')
ax.set_ylabel('Normalized mode shape')
ax.set_title('First three bending mode shapes')
ax.legend()
plt.show()

phi_normalized, phi_dot_normalized, phi_dotdot_normalized = ROM.torsionModeShapes(model)
fig, ax = plt.subplots(constrained_layout=True, figsize=(6,4))
ax.plot(model.y, phi_normalized[0,:], label='Mode shape torsion 1')
ax.plot(model.y, phi_normalized[1,:], label='Mode shape torsion 2')
ax.plot(model.y, phi_normalized[2,:], label='Mode shape torsion 3')
ax.set_xlim((0, model.s))
ax.set_xlabel('Spanwise location y [m]')
ax.set_ylabel('Normalized mode shape')
ax.set_title('First three torsion mode shapes')
ax.legend()
plt.show()






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

