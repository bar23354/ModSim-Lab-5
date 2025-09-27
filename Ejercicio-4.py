import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# EDO: dy/dx = (x - 3y - 3(x² - y²) + 3xy) / (2x - y + 3(x² - y²) + 2xy)

def f(x, y):
    """Función f(x,y) de la EDO"""
    x = np.asarray(x)
    y = np.asarray(y)
    num = x - 3*y - 3*(x**2 - y**2) + 3*x*y
    den = 2*x - y + 3*(x**2 - y**2) + 2*x*y
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(den) > 1e-10, num/den, np.nan)

# ---- Región y malla ----
x_min, x_max = -2, 3
y_min, y_max = -2, 3
paso = 0.15     # malla más fina
x = np.arange(x_min, x_max+paso, paso)
y = np.arange(y_min, y_max+paso, paso)
X, Y = np.meshgrid(x, y)

# ---- Campo vectorial ----
F = f(X, Y)                       # pendiente
U = np.ones_like(F) / np.sqrt(1 + F**2)    # componente x
V = F / np.sqrt(1 + F**2)                  # componente y
U[~np.isfinite(F)] = 0
V[~np.isfinite(F)] = 0

# ---- Gráficas base ----
fig, axes = plt.subplots(1, 3, figsize=(18,6))

# Contorno del denominador = 0 (singularidades)
den = 2*X - Y + 3*(X**2 - Y**2) + 2*X*Y
for ax in axes:
    ax.contour(X, Y, den, [0], colors='k', linestyles='dashed', linewidths=1)

# ---- 1. Solo campo ----
axes[0].quiver(X, Y, U, V, color='blue', angles='xy', scale_units='xy')
axes[0].set_xlim(x_min, x_max)
axes[0].set_ylim(y_min, y_max)
axes[0].set_aspect('equal')
axes[0].set_title('Campo de Direcciones')
axes[0].grid(alpha=0.3)

# ---- 2. Resolver PVI ----
def edo_sistema(t, z):
    return [f(t, z[0])]

x0, y0 = 1.5, 0
sol_fwd = solve_ivp(edo_sistema, [x0, 2.5], [y0], t_eval=np.linspace(x0,2.5,50))
sol_bwd = solve_ivp(edo_sistema, [x0, 0.5], [y0], t_eval=np.linspace(x0,0.5,50))

axes[1].quiver(X, Y, U, V, color='blue', alpha=0.6, angles='xy', scale_units='xy')
axes[1].plot(sol_fwd.t, sol_fwd.y[0], 'r-', lw=3, label='Solución')
axes[1].plot(sol_bwd.t, sol_bwd.y[0], 'r-', lw=3)
axes[1].plot(x0, y0, 'ro', ms=8, label='Condición inicial')
axes[1].legend()
axes[1].set_xlim(x_min, x_max)
axes[1].set_ylim(y_min, y_max)
axes[1].set_aspect('equal')
axes[1].set_title('Campo + Solución PVI')
axes[1].grid(alpha=0.3)

# ---- 3. Equilibrios ----
def sistema_equilibrio(v):
    x, y = v
    return [
        x - 3*y - 3*(x**2 - y**2) + 3*x*y,
        2*x - y + 3*(x**2 - y**2) + 2*x*y
    ]

puntos_ini = [(0,0), (1,0), (0,1), (-1,0), (1,1)]
equilibrios = []
tol = 1e-4
for p0 in puntos_ini:
    sol = fsolve(sistema_equilibrio, p0)
    if np.allclose(sistema_equilibrio(sol), (0,0), atol=tol):
        if not any(np.allclose(sol, e, atol=tol) for e in equilibrios):
            if x_min <= sol[0] <= x_max and y_min <= sol[1] <= y_max:
                equilibrios.append(sol)

axes[2].quiver(X, Y, U, V, color='blue', alpha=0.6, angles='xy', scale_units='xy')
axes[2].plot(sol_fwd.t, sol_fwd.y[0], 'r-', lw=3, label='Solución')
axes[2].plot(sol_bwd.t, sol_bwd.y[0], 'r-', lw=3)
axes[2].plot(x0, y0, 'ro', ms=8, label='Condición inicial')
for i,(xe,ye) in enumerate(equilibrios):
    axes[2].plot(xe, ye, 'ko', ms=9, markeredgecolor='w', label='Equilibrio' if i==0 else '')
axes[2].legend()
axes[2].set_xlim(x_min, x_max)
axes[2].set_ylim(y_min, y_max)
axes[2].set_aspect('equal')
axes[2].set_title('Campo + Solución + Equilibrios')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
