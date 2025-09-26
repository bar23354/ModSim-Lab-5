"""
Universidad del Valle de Guatemala
Modelación y Simulación
Laboratorio 5 - Ejercicio 2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
from sympy import symbols, Function, exp

plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def crear_campo_direccional(func_dy_dx, x_range, y_range, num_points=20, ax=None, title="Campo Direccional"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    DY = func_dy_dx(X, Y)
    DX = np.ones_like(DY)
    M = np.sqrt(DX**2 + DY**2)
    M[M == 0] = 1
    DX_norm = DX/M
    DY_norm = DY/M
    
    ax.quiver(X, Y, DX_norm, DY_norm, M, scale=30, cmap='viridis', alpha=0.6)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax

def resolver_numericamente(func_dy_dx, y0, x_span, num_points=1000):
    x = np.linspace(x_span[0], x_span[1], num_points)
    
    def dydt(y, t):
        return func_dy_dx(t, y)
    
    y = odeint(dydt, y0, x)
    return x, y.flatten()

#i) y' = -xy
def edo_1(x, y):
    return -x * y

#ii) y' = xy  
def edo_2(x, y):
    return x * y

#iii) xdx + ydy = 0 → dy/dx = -x/y
def edo_3(x, y):
    return np.where(y != 0, -x/y, 0)

#iv) ydx + xdy = 0 → dy/dx = -y/x
def edo_4(x, y):
    return np.where(x != 0, -y/x, 0)

#v) dy/dx = y² - y
def edo_5(x, y):
    return y**2 - y
# ==================

def soluciones_analiticas():
    """
    soluciones analíticas documentadas a mano
    """
    return {
        'edo_1': "y = A*exp(-x**2/2)",
        'edo_2': "y = A*exp(x**2/2)", 
        'edo_3': "x**2 + y**2 = K",
        'edo_4': "x*y = C",
        'edo_5': "y = 1/(1 - A*exp(x))"
    }

def evaluar_solucion_analitica(ecuacion, A_val, x_vals):
    """
    Evalúa una solución analítica para valores específicos
    """
    if ecuacion == 1:
        return A_val * np.exp(-x_vals**2 / 2)
    elif ecuacion == 2:
        return A_val * np.exp(x_vals**2 / 2)
    elif ecuacion == 3:
        K = A_val
        y_pos = np.sqrt(np.maximum(0, K - x_vals**2))
        y_neg = -np.sqrt(np.maximum(0, K - x_vals**2))
        return y_pos, y_neg
    elif ecuacion == 4:
        return np.where(x_vals != 0, A_val / x_vals, np.nan)
    elif ecuacion == 5:
        denominador = 1 - A_val * np.exp(x_vals)
        return np.where(denominador != 0, 1 / denominador, np.nan)

# ==================

def crear_visualizacion_completa():
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis Cualitativo vs Analítico de EDO', fontsize=16, fontweight='bold')
    
    configuraciones = [
        {
            'func': edo_1,
            'titulo': "i) y' = -xy",
            'x_range': (-3, 3),
            'y_range': (-3, 3),
            'ecuacion': 1,
            'A_values': [0.5, 1, 2]
        },
        {
            'func': edo_2,
            'titulo': "ii) y' = xy",
            'x_range': (-2, 2),
            'y_range': (-3, 3),
            'ecuacion': 2,
            'A_values': [0.1, 0.5, 1]
        },
        {
            'func': edo_3,
            'titulo': "iii) xdx + ydy = 0",
            'x_range': (-3, 3),
            'y_range': (-3, 3),
            'ecuacion': 3,
            'A_values': [1, 4, 9]
        },
        {
            'func': edo_4,
            'titulo': "iv) ydx + xdy = 0",
            'x_range': (-3, 3),
            'y_range': (-3, 3),
            'ecuacion': 4,
            'A_values': [1, 2, -1]
        },
        {
            'func': edo_5,
            'titulo': "v) dy/dx = y² - y",
            'x_range': (-2, 2),
            'y_range': (-1, 3),
            'ecuacion': 5,
            'A_values': [0.1, 0.5, 2]
        }
    ]
    
    #graphs EDO
    for i, config in enumerate(configuraciones):
        if i < 5:  #solo tenemos 5 EDO
            ax = axes[i//3, i%3] if i < 3 else axes[1, i-3]
            
            crear_campo_direccional(
                config['func'], 
                config['x_range'], 
                config['y_range'], 
                ax=ax,
                title=config['titulo']
            )
            
            #add sol analíticas
            x_vals = np.linspace(config['x_range'][0], config['x_range'][1], 200)
            
            for j, A in enumerate(config['A_values']):
                try:
                    if config['ecuacion'] == 3:
                        y_pos, y_neg = evaluar_solucion_analitica(config['ecuacion'], A, x_vals)
                        ax.plot(x_vals, y_pos, 'r-', linewidth=2, alpha=0.8, 
                               label=f'Analítica K={A}' if j == 0 else "")
                        ax.plot(x_vals, y_neg, 'r-', linewidth=2, alpha=0.8)
                    else:
                        y_anal = evaluar_solucion_analitica(config['ecuacion'], A, x_vals)
                        mask = np.isfinite(y_anal) & (np.abs(y_anal) < 10)
                        ax.plot(x_vals[mask], y_anal[mask], 'r-', linewidth=2, alpha=0.8,
                               label=f'Analítica A={A}' if j == 0 else "")
                except:
                    continue
            
            ax.legend(loc='best', fontsize=8)
    
    axes[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('Ej-2-Soluciones/comparacion_edo_todas.png', dpi=300, bbox_inches='tight')
    plt.show()

def analisis_puntos_equilibrio():
    """
    Analiza los puntos de equilibrio y estabilidad
    """
    print("\n" + "="*80)
    print("ABOUT - PUNTOS DE EQUILIBRIO Y ESTABILIDAD")
    print("="*80)
# ==================  
    print("\ni) y' = -xy")
    print("Pts de equilibrio: y = 0 para cualquier x")
    print("Nota: Las soluciones son gaussianas que decaen hacia 0")
# ==================    
    print("\nii) y' = xy") 
    print("Pts de equilibrio: y = 0 para cualquier x")
    print("Nota: Las soluciones crecen exponencialmente alejándose de 0")
# ==================
    print("\niii) xdx + ydy = 0")
    print("Esta EDO representa circunferencias")
    print("No hay pts de equilibrio")
# ==================   
    print("\niv) ydx + xdy = 0") 
    print("Esta EDO representa hipérbolas")
    print("Singularidad en (0,0)")
# ==================  
    print("\nv) dy/dx = y² - y = y(y-1)")
    print("Pts de equilibrio: y = 0 y y = 1")
    print("y = 0: inestable (las soluciones se alejan)")
    print("y = 1: estable (las soluciones se acercan)")
    print("Para y > 1: las soluciones crecen indefinidamente")
    print("Para 0 < y < 1: las soluciones tienden a y = 1")
    print("Para y < 0: las soluciones tienden a y = 0")

def discusion_comparativa():
    """
    Discusión simplificada sobre la comparación de métodos
    """
    print("\n" + "="*60)
    print("DISCUSIÓN Y COMPARACIÓN DE MÉTODOS")
    print("="*60)
    
    print("\nCONCORDANCIA:")
    print("Los campos direccionales coinciden perfectamente con las solucones analíticas")
    print("Las curvas siguen exacvtamente las direcciones mostradas por las flechas")
    
    print("\nÉTODOS CUALITATIVOS:")
    print("- Intuición geometrica inmediata")
    print("- Comportamiento global visible")
    print("- Identificación rápida de puntos críticos")
    
    print("\nMÉTODOS ANALÍTICOS:")
    print("- Expresiones matemáticas exactas")
    print("- Cáalculos precisos")
    print("- Análisis riguroso")
    
    print("\nPOR EDO:")
    print("  i-ii)Comportamiento gaussiano en ambos métodos")
    print("  iii)Circumferencias evidentes geométrica y analíticamente")
    print("  iv)Hipérbolas y singularidad en (0,0) consistentes")
    print("  v)Puntos de equilibrio y = 0, y = 1 confirmados")
    
    print("\nCONCLUSIÓN:")
    print("Ambos métodos se complementan y validan ambos mutuamente")
    print("Dan la misma información sobre el comportamiento de las EDO")  

if __name__ == "__main__":
    print("LAB 05 - EJ 2")
    print("="*80)

    soluciones = soluciones_analiticas()
    
    crear_visualizacion_completa()
    analisis_puntos_equilibrio()
    discusion_comparativa()
    
    print("\n" + "="*80)
    print("gráfica guardada en: Ej-2-Soluciones/comparacion_edo_todas.png")
    print("="*80)