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

def _draw_plotly_arrow(fig, point, dir, length=1., color='black', cone_scale=2., label=None):
    x_line = [point[0], point[0] + length * dir[0]]
    y_line = [point[1], point[1] + length * dir[1]]
    z_line = [point[2], point[2] + length * dir[2]]

    fig.add_trace(go.Scatter3d(
        x=x_line,
        y=y_line,
        z=z_line,
        mode='lines',
        line=dict(color=color, width=4),
        name=label
    ))

    x_cone = point[0] + length * dir[0]
    y_cone = point[1] + length * dir[1]
    z_cone = point[2] + length * dir[2]

    fig.add_trace(go.Cone(
        x=[x_cone],
        y=[y_cone],
        z=[z_cone],
        u=[cone_scale * dir[0]],
        v=[cone_scale * dir[1]],
        w=[cone_scale * dir[2]],
        colorscale=[[0, color], [1, color]],
        showscale=False,
        sizemode='absolute',
        sizeref=1.0
    ))

    return fig

def _draw_plotly_isosurface(fig, x, y, z, f, label=None, opacity=1.0, color='black'):
    fig.add_trace(go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=f.flatten(),
        isomin=0,
        isomax=0,
        caps=dict(x_show=False, y_show=False),
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=opacity,
        name=label,
    ))
    fig.update_layout(scene=dict(
        xaxis=dict(
            title=dict(
                text=r'σ₁'
            )
        ),
        yaxis=dict(
            title=dict(
                text=r'σ₂'
            )
        ),
        zaxis=dict(
            title=dict(
                text=r'σ₃'
            )
        ),
    ), )
    return fig