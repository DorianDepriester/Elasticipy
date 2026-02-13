from matplotlib.colors import Normalize
from matplotlib import pyplot as plt, cm
import plotly.graph_objects as go

def _plot3D_plotly(u, r, fig=None, **kwargs):
    if fig is None:
        fig = go.Figure()
    xyz = (u.T * r.T).T
    fig.add_trace(go.Surface(
        x=xyz[:, :, 0],
        y=xyz[:, :, 1],
        z=xyz[:, :, 2],
        surfacecolor=r,
        colorscale='Viridis',
        cmin=r.min(),
        cmax=r.max(),
        **kwargs
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        coloraxis=dict(
            colorscale='Viridis',
            cmin=r.min(),
            cmax=r.max(),
            colorbar=dict(title='Valeurs')
        ),
    )
    return fig

def _plot3D_matplotlib(u, r, fig=None, **kwargs):
    if fig is None:
        fig = plt.figure()
    norm = Normalize(vmin=r.min(), vmax=r.max())
    colors = cm.viridis(norm(r))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    xyz = (u.T * r.T).T
    ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], facecolors=colors, rstride=1, cstride=1,
                    antialiased=False, **kwargs)
    mappable = cm.ScalarMappable(cmap='viridis', norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax