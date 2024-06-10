import numpy as np 
import matplotlib.pyplot as plt 
import pycwt as wavelet
from scipy.integrate import solve_ivp

## Simple Sinusoidal system 

dt=0.01
tvec=np.arange(0,100,dt)

T1,T2,T3=10,20,40
sig=15*np.sin((2*np.pi/T1)*tvec)+5*np.sin((2*np.pi/T2)*tvec)+7*np.sin((2*np.pi/T3)*tvec)

plt.figure()
plt.plot(tvec,sig)

mother = wavelet.DOG(12)                                          
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(sig, dt,s0=2*dt,dj=1/6, wavelet=mother)
power = (np.abs(wave)) ** 2
fft_power = np.abs(fft) ** 2
period = 1/freqs

t=tvec
fig,ax=plt.subplots()
lv=(np.arange(-4,4.5,0.5))
mm=ax.contourf(tvec, (period), np.log2(power),levels=np.arange(10,21),
               extend="both",cmap='jet')
plt.colorbar(mm)
ax.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                           t[:1] - dt, t[:1] - dt]),
        np.concatenate([(coi), [1e-9], (period[-1:]),
                           (period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
ax.set_ylim(5,50)

plt.figure()
plt.plot(1/fftfreqs,fft_power)

#=====================================
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0
initial_state = [1.0, 1.0, 1.0]
t_start = 0.0
t_end = 100.0
dt=0.01
t_transient = 50.0  # Time to discard as transient
t_eval = np.arange(t_start, t_end, dt)

# Solve the Lorenz system
sol = solve_ivp(lorenz, [t_start, t_end], initial_state, args=(sigma, rho, beta), t_eval=t_eval)

# Discard the transient period
transient_index = np.searchsorted(t_eval, t_transient)
t_post_transient = t_eval[transient_index:]
x_post_transient = sol.y[0, transient_index:]
y_post_transient = sol.y[1, transient_index:]
z_post_transient = sol.y[2, transient_index:]

# Plot the results
fig = plt.figure(figsize=(15, 5))

# Plot x vs t
ax1 = fig.add_subplot(131)
ax1.plot(t_post_transient, x_post_transient)
ax1.set_title('x vs Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('x')



t=t_post_transient
sig=x_post_transient
mother = wavelet.DOG(12)                                          
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(sig, dt,s0=2*dt,dj=1/6, wavelet=mother)
power = (np.abs(wave)) ** 2
fft_power = np.abs(fft) ** 2
period = 1/freqs


fig,ax=plt.subplots()
lv=(np.arange(-4,4.5,0.5))
mm=ax.contourf(t, (period), np.log2(power),np.arange(2,22,2),
               extend="both",cmap='jet')
plt.colorbar(mm)
ax.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                           t[:1] - dt, t[:1] - dt]),
        np.concatenate([(coi), [1e-9], (period[-1:]),
                           (period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
#ax.set_ylim(5,50)

plt.figure()
plt.plot(1/fftfreqs,fft_power)
