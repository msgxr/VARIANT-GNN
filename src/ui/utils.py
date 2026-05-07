
import matplotlib.pyplot as plt


def plot_dark(fig: plt.Figure, ax: plt.Axes) -> None:
    """Matplotlib grafiklerini koyu temaya (UI ile uyumlu) dönüştürür."""
    fig.patch.set_facecolor('#1a2744')
    ax.set_facecolor('#1a2744')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#e2e8f0')
    for spine in ax.spines.values():
        spine.set_edgecolor((0.388, 0.702, 0.929, 0.15))
    ax.grid(True, color=(1.0, 1.0, 1.0, 0.05), linewidth=0.5)

def hex_to_rgb(hex_color: str) -> str:
    """'#fc8181' -> '252,129,129' formatına dönüştürür (CSS rgba için)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
