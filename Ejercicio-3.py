import numpy as np
import matplotlib.pyplot as plt

# Resolver el problema xy'' + 2y' = 6x

print("Problema 3: xy'' + 2y' = 6x")
print("=" * 40)

# Paso 1: Convertir a EDO de primer orden
print("\n1. Sustitución:")
print("Sea v = y', entonces v' = y''")
print("La ecuación se convierte en: xv' + 2v = 6x")
print("Dividiendo por x: v' + (2/x)v = 6")

# Paso 2: Resolver la EDO de primer orden
print("\n2. Resolución:")
print("Factor integrante: μ(x) = x²")
print("Multiplicando: x²v' + 2xv = 6x²")
print("(x²v)' = 6x²")
print("Integrando: x²v = 2x³ + C₁")
print("Por tanto: v = 2x + C₁/x²")

print("\nComo v = y':")
print("y' = 2x + C₁/x²")
print("Integrando: y = x² - C₁/x + C₂")

print("\nSolución general: y(x) = x² - C₁/x + C₂")

# Paso 3: Problemas de valor inicial (C₂ = 0)
print("\n3. Problemas de valor inicial (C₂ = 0):")
print("y(x) = x² - C₁/x")

def calcular_C1(x0, y0):
    if x0 == 0:
        return None
    return x0**3 - x0*y0

def solucion(x, C1):
    return x**2 - C1/x

problemas = [
    (1, 2, "y(1) = 2"),
    (1, -2, "y(1) = -2"),
    (1, 1, "y(1) = 1"),
    (0, -3, "y(0) = -3")
]

soluciones = []

for x0, y0, desc in problemas:
    print(f"\n{desc}:")
    if x0 == 0:
        print("  No se puede resolver (x = 0 es singular)")
    else:
        C1 = calcular_C1(x0, y0)
        print(f"  C₁ = {C1}")
        print(f"  y(x) = x² - {C1}/x")
        soluciones.append((C1, desc, x0, y0))

# Paso 4: Gráficas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Primera gráfica - Intervalos maximales
x_neg = np.linspace(-3, -0.1, 200)  # Más separado de x=0
x_pos = np.linspace(0.1, 3, 200)    # Más separado de x=0

C1_vals = [-2, 1, 3, 5]  # Cambié el 0 por 1 para evitar confusión visual
colores = ['darkgreen', 'blue', 'purple', 'orange']

for i, C1 in enumerate(C1_vals):
    y_neg = solucion(x_neg, C1)
    y_pos = solucion(x_pos, C1)
    
    # Dibujar con un pequeño gap cerca de x=0 para mostrar la discontinuidad
    ax1.plot(x_neg, y_neg, color=colores[i], linewidth=2.5, alpha=0.9)
    ax1.plot(x_pos, y_pos, color=colores[i], linewidth=2.5, alpha=0.9,
             label=f'C₁ = {C1}' if i == 0 else "")

# Líneas de referencia
ax1.axvline(x=0, color='red', linewidth=5, alpha=0.8, label='x = 0 (SINGULAR)')
ax1.axvline(x=-3, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax1.axvline(x=3, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Agregar puntos que muestren la discontinuidad
for C1 in C1_vals:
    # Punto límite desde la izquierda (x → 0⁻)
    y_limit_left = np.inf if C1 != 0 else 0
    # Punto límite desde la derecha (x → 0⁺)  
    y_limit_right = np.inf if C1 != 0 else 0
    
    if C1 != 0:  # Para C1 ≠ 0, las curvas van a infinito
        ax1.plot([-0.1], [solucion(-0.1, C1)], 'o', color='red', markersize=4, alpha=0.7)
        ax1.plot([0.1], [solucion(0.1, C1)], 'o', color='red', markersize=4, alpha=0.7)

ax1.set_xlim(-3, 3)
ax1.set_ylim(-15, 15)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Intervalos Maximales')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Segunda gráfica - Soluciones específicas
x_pos_graf2 = np.linspace(0.1, 3, 200)  # También más separado
x_neg_graf2 = np.linspace(-3, -0.1, 200)

colores2 = ['blue', 'red', 'green']

for i, (C1, desc, x0, y0) in enumerate(soluciones):
    color = colores2[i]
    
    y_pos = solucion(x_pos_graf2, C1)
    y_neg = solucion(x_neg_graf2, C1)
    
    ax2.plot(x_pos_graf2, y_pos, color=color, linewidth=3, label=f'{desc}, C₁={C1}')
    ax2.plot(x_neg_graf2, y_neg, color=color, linewidth=3, linestyle='--', alpha=0.7)
    ax2.plot(x0, y0, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1.5)

ax2.axvline(x=0, color='red', linewidth=3, linestyle=':', alpha=0.8, label='x = 0')
ax2.set_xlim(-3, 3)
ax2.set_ylim(-15, 15)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Soluciones: y = x² - C₁/x')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

print("\n4. Conclusiones:")
print("- x = 0 es un punto singular")
print("- Las soluciones existen en (-∞, 0) o (0, ∞) pero no en ambos")
print("- No existe solución continua que pase por x = 0")
print(f"- Se resolvieron {len(soluciones)} de los 4 problemas")