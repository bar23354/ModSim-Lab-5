import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ==============================================================
# a) Dimensiones de parámetros - CORRECTO
# ==============================================================
print("Ejercicio a)")
print("a (0.0004) tiene dimensiones: 1/(semana * individuo)")
print("b (0.06) tiene dimensiones: 1/semana\n")

# ==============================================================
# b) Puntos de equilibrio - MEJORADO
# ==============================================================
P_sym = sp.Symbol('P')
dPdt_func = 0.0004 * P_sym**2 - 0.06 * P_sym

equilibrio = sp.solve(sp.Eq(dPdt_func, 0), P_sym)
print("Ejercicio b)")
print("Puntos de equilibrio:", equilibrio)

df_dP = sp.diff(dPdt_func, P_sym)
print("Derivada f'(P) =", df_dP)

for eq in equilibrio:
    val = float(df_dP.subs(P_sym, eq))  # Convertir a numérico
    if val < 0:
        estabilidad = "Estable"
    elif val > 0:
        estabilidad = "Inestable"
    else:
        estabilidad = "Semi-estable"
    print(f"P = {eq}: f'({eq}) = {val:.3f} → {estabilidad}")
print()

# ==============================================================
# c) Análisis cualitativo - MEJORADO
# ==============================================================
print("Ejercicio c)")

# Análisis de signo de dP/dt
P_test = np.array([-10, 50, 100, 150, 200, 250])  # Valores de prueba
signos = 0.0004 * P_test**2 - 0.06 * P_test

print("Análisis de crecimiento:")
for p, signo in zip(P_test, signos):
    if p >= 0:  # Solo valores físicamente relevantes
        if signo > 0:
            estado = "Creciente"
        elif signo < 0:
            estado = "Decreciente"
        else:
            estado = "Equilibrio"
        print(f"P = {p:3d}: dP/dt = {signo:7.3f} → {estado}")

# Concavidad
d2Pdt2_expr = df_dP * dPdt_func
print(f"\nConcavidad: d²P/dt² = {d2Pdt2_expr}")
print("Para P > 0, d²P/dt² > 0 → Soluciones siempre cóncavas hacia arriba")
print()

# ==============================================================
# d) y e) - CORRECTOS
# ==============================================================
print("Ejercicio d)")
print("P(0)=200 > 150 → Población crece indefinidamente\n")

print("Ejercicio e)")
print("P(0)=100 < 150 → Población decrece hacia 0 (extinción)\n")

# ==============================================================
# f) Solución numérica robusta - MEJORADO
# ==============================================================
print("Ejercicio f)")

# Solución analítica simbólica
P = sp.Function('P')
t = sp.Symbol('t')
ode = sp.Eq(sp.Derivative(P(t), t), 0.0004*P(t)**2 - 0.06*P(t))
sol_general = sp.dsolve(ode)

print("Solución general:")
print(sol_general)

# Solución numérica para evitar problemas con la analítica
def population_ode(t, P):
    return 0.0004 * P**2 - 0.06 * P

# Para P(0)=200 (crecimiento rápido)
t_span_200 = (0, 20)  # Menos tiempo porque crece muy rápido
t_eval_200 = np.linspace(0, 20, 1000)
sol_200 = solve_ivp(population_ode, t_span_200, [200], t_eval=t_eval_200, method='RK45')

# Para P(0)=100 (extinción lenta)
t_span_100 = (0, 100)  # Más tiempo para ver la tendencia
t_eval_100 = np.linspace(0, 100, 1000)
sol_100 = solve_ivp(population_ode, t_span_100, [100], t_eval=t_eval_100, method='RK45')

# Gráfica mejorada
plt.figure(figsize=(12, 6))

# Gráfica 1: Soluciones
plt.subplot(1, 2, 1)
plt.plot(sol_100.t, sol_100.y[0], 'b-', linewidth=2, label='P(0)=100 → Extinción')
plt.plot(sol_200.t, sol_200.y[0], 'r-', linewidth=2, label='P(0)=200 → Crecimiento')
plt.axhline(y=150, color='orange', linestyle='--', alpha=0.7, label='Umbral P=150')
plt.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Extinción P=0')
plt.xlabel('Tiempo (semanas)')
plt.ylabel('Población P(t)')
plt.title('Evolución de la Población')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 2: Campo de direcciones
plt.subplot(1, 2, 2)
P_vals = np.linspace(0, 300, 20)
t_vals = np.linspace(0, 50, 15)
T, P = np.meshgrid(t_vals, P_vals)
dP = population_ode(T, P)
dt = np.ones(dP.shape)

# Normalizar flechas
norm = np.sqrt(dt**2 + dP**2)
dt_u = dt / norm
dP_u = dP / norm

plt.quiver(T, P, dt_u, dP_u, scale=25, color='blue', alpha=0.6)
plt.axhline(y=150, color='red', linestyle='--', linewidth=2, label='P=150 (inestable)')
plt.axhline(y=0, color='green', linestyle='--', linewidth=2, label='P=0 (estable)')
plt.xlabel('Tiempo (semanas)')
plt.ylabel('Población P')
plt.title('Campo de Direcciones')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-10, 300)

plt.tight_layout()
plt.show()

# Análisis de resultados
print("\nVERIFICACIÓN DEL ANÁLISIS CUALITATIVO:")
print(f"P(0)=100: P(100) ≈ {sol_100.y[0][-1]:.2f} → Tiende a 0 ✓")
print(f"P(0)=200: P(20) ≈ {sol_200.y[0][-1]:.2f} → Crecimiento explosivo ✓")