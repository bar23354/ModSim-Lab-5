"""
Universidad del Valle de Guatemala
Modelación y Simulación
Laboratorio 5 - Ejercicio 1
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_direction_field(f, xmin, xmax, ymin, ymax, xstep, ystep, 
                        unit_field=False, streamlines=False, title="Campo de Direcciones",
                        save_path=None, quiver_scale=None, separate_plots=False):
    """
    función principal para graficar campos de direcciones de EDOs
    """
    
    x = np.arange(xmin, xmax + xstep, xstep)
    y = np.arange(ymin, ymax + ystep, ystep)
    X, Y = np.meshgrid(x, y)
    DX = np.ones_like(X)
    DY = f(X, Y)
    DY = np.clip(DY, -1e6, 1e6)
    
    if unit_field:
        magnitude = np.sqrt(DX**2 + DY**2)
        magnitude[magnitude == 0] = 1
        DX = DX / magnitude
        DY = DY / magnitude
        field_type = "Campo Unitario"
    else:
        field_type = "Campo F"
    
    if quiver_scale is None:
        if unit_field:
            quiver_scale = 3
        else:
            avg_magnitude = np.mean(np.sqrt(DX**2 + DY**2))
            quiver_scale = avg_magnitude * 15
    
    if separate_plots and streamlines:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.quiver(X, Y, DX, DY, angles='xy', scale_units='xy', scale=quiver_scale, 
                  alpha=0.8, width=0.003, color='blue')
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'{title} - {field_type} (Vectores)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        ax2.streamplot(X, Y, DX, DY, color='red', density=1, linewidth=1.5)
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'{title} - {field_type} (Líneas de Flujo)')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        ax = (ax1, ax2)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        alpha_val = 0.5 if streamlines else 0.8
        ax.quiver(X, Y, DX, DY, angles='xy', scale_units='xy', scale=quiver_scale, 
                  alpha=alpha_val, width=0.003, color='blue')
        
        if streamlines:
            ax.streamplot(X, Y, DX, DY, color='red', density=0.8, linewidth=1.2)
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title} - {field_type}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"grafica guardada en: {save_path}")
    
    return fig, ax

def F(x, y, case=1):
    """
    función auxiliar de PRUEBA que define diferentes ecuaciones diferenciales
    """
    if case == 1:
        return x + y
    elif case == 2:
        return x * y
    elif case == 3:
        return np.sin(x) + np.cos(y)
    elif case == 4:
        #singularidad en x=0
        return np.where(np.abs(x) < 1e-8, 0, -y / x)

if __name__ == "__main__":
    save_dir = "Ej-1-Graficas"
    
    print("Graph 1: dy/dx = x + y (Campo F con streamlines)")
    fig1, ax1 = plot_direction_field(
        lambda x, y: F(x, y, case=1),
        xmin=-3, xmax=3, ymin=-3, ymax=3,
        xstep=0.3, ystep=0.3,
        unit_field=False, streamlines=True,
        title="Caso 1: dy/dx = x + y",
        save_path=f"{save_dir}/Graph-1-case1-campo-F.png",
        separate_plots=True
    )
    plt.show()
    
    print("Graph 1b: dy/dx = x + y (Campo Unitario)")
    fig1b, ax1b = plot_direction_field(
        lambda x, y: F(x, y, case=1),
        xmin=-3, xmax=3, ymin=-3, ymax=3,
        xstep=0.3, ystep=0.3,
        unit_field=True, streamlines=False,
        title="Caso 1: dy/dx = x + y",
        save_path=f"{save_dir}/Graph-1b-case1-unitario.png"
    )
    plt.show()
    
    print("Graph 2: dy/dx = x*y")
    fig2, ax2 = plot_direction_field(
        lambda x, y: F(x, y, case=2),
        xmin=-2, xmax=2, ymin=-2, ymax=2,
        xstep=0.2, ystep=0.2,
        unit_field=False, streamlines=True,
        title="Caso 2: dy/dx = x*y",
        save_path=f"{save_dir}/Graph-2-case2-campo-F.png"
    )
    plt.show()

    print("Graph 3: dy/dx = sin(x) + cos(y)")
    fig3, ax3 = plot_direction_field(
        lambda x, y: F(x, y, case=3),
        xmin=-2*np.pi, xmax=2*np.pi, ymin=-2*np.pi, ymax=2*np.pi,
        xstep=0.5, ystep=0.5,
        unit_field=True, streamlines=True,
        title="Caso 3: dy/dx = sin(x) + cos(y)",
        save_path=f"{save_dir}/Graph-3-case3-unitario.png"
    )
    plt.show()
    
    print("Graph 4: dy/dx = -y/x (singularidad)")
    fig4, ax4 = plot_direction_field(
        lambda x, y: F(x, y, case=4),
        xmin=-3, xmax=3, ymin=-3, ymax=3,
        xstep=0.3, ystep=0.3,
        unit_field=True, streamlines=True,
        title="Caso 4: dy/dx = -y/x",
        save_path=f"{save_dir}/Graph-4-case4-singularidad.png"
    )
    plt.show()

    print("\ngráficas guardadas en la carpeta 'Ej-1-Graficas'")