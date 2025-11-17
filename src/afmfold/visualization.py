import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import plotly.graph_objects as go
from scipy.ndimage import center_of_mass
import math
import mdtraj as md
import itertools
from scipy import stats

from afmfold.domain import compute_domain_distance, get_domain_pairs, get_domain_pair_names

def get_color(i, n, cmap_name='tab10'):
    cmap = plt.get_cmap(cmap_name)
    return cmap(i % n / max(n-1, 1))

def plot_afm(
    *images, 
    resolution_nm=1.0,
    x_range=None, 
    y_range=None, 
    save_path=None, 
    title="AFM image", 
    subplots=None, 
    vmin=None, 
    vmax=None, 
    unit="nm", 
    fontsize=14, 
    max_figures=10,
    axsize=4.0,
    padding=2.0,
    text=None,
    text_dict={},
    traj=None,
    traj_dict={},
    **kwargs
    ):
    """
    Display or save AFM images, tip shapes, and atomic structure scatter plots in a 2x2 subplot layout.

    Parameters:
        images (list or array): AFM images. Array of shape [..., H, W]
        x_range (tuple): x-axis values
        y_range (tuple): y-axis values
        save_path (str): Path to save the file (if None, only display)

    Returns:
        fig, axes: matplotlib Figure and Axes objects
    """
    # Initial setup
    image_list =[]
    for image in images:
        if isinstance(image, torch.Tensor):
            image_np = image.detach().cpu().to(torch.float32).numpy()
        elif isinstance(image, np.ndarray):
            image_np = image
        else:
            raise NotImplementedError()
        image_list.append(image_np.reshape(-1,*image.shape[-2:]))
    images = np.concatenate(image_list, axis=0)
        
    if isinstance(x_range, torch.Tensor):
        x_range = x_range.detach().cpu()
    
    if isinstance(y_range, torch.Tensor):
        y_range = y_range.detach().cpu()
    
    H, W = images.shape[-2:]
    images = images.reshape((-1, H, W))
    B = images.shape[0]
    if B > max_figures:
        raise ValueError(f"Too many figures: {B} > {max_figures}")
    
    if x_range is None:
        x_range = resolution_nm * np.arange(W)
        x_range = np.tile(x_range[None,:], (B, 1))
    else:
        _W = x_range.shape[-1]
        x_range = x_range.reshape((-1, _W))
        if x_range.shape[-1] == W + 1:
            x_range = 0.5 * (x_range[:,:-1] + x_range[:,1:])
        if x_range.shape[0] == 1:
            x_range = np.tile(x_range[None,:], (B, 1))
    assert x_range.shape == (B, W)
    
    if y_range is None:
        y_range = resolution_nm * np.arange(H)
        y_range = np.tile(y_range[None,:], (B, 1))
    else:
        _H = y_range.shape[-1]
        y_range = y_range.reshape((-1, _H))
        if y_range.shape[-1] == H + 1:
            y_range = 0.5 * (y_range[:,:-1] + y_range[:,1:])
        if y_range.shape[0] == 1:
            y_range = np.tile(y_range[None,:], (B, 1))
    assert y_range.shape == (B, H)
    
    if isinstance(title, (list, tuple)):
        assert len(title) == B
        title_list = title
    else:
        title_list = [title for _ in range(B)]
    
    # Create subplots (2 rows x 2 columns)
    if subplots is not None:
        fig, axes = subplots
    else:
        fig, axes = plt.subplots(1, B, figsize=(B*(axsize+padding), axsize))
    
    if B == 1:
        axes = [axes]
            
    for i in range(B):
        ax = axes[i]
        sub_title = title_list[i]
        
        # Automatically determine colorbar range
        _vmin = vmin if vmin is not None else np.min(images[i])
        _vmax = vmax if vmax is not None else np.max(images[i])
        
        # AFM image
        extent = [x_range[i,0], x_range[i,-1], y_range[i,-1], y_range[i,0]]  # Flip Y-axis
        im = ax.imshow(
            images[i], interpolation='none', 
            origin='upper', cmap="afmhot", aspect="equal", 
            extent=extent, vmin=_vmin, vmax=_vmax
            )
        
        if unit == "nm" or unit == "nanometer":
            ax.set_xlabel("X [nm]", fontsize=fontsize)
            ax.set_ylabel("Y [nm]", fontsize=fontsize)
        elif unit == "A" or unit == "Å" or unit == "angstrom":
            ax.set_xlabel("X [Å]", fontsize=fontsize)
            ax.set_ylabel("Y [Å]", fontsize=fontsize)
        else:
            raise ValueError(f"Invalid unit: {unit} not in [ 'nm', 'nanometer', 'A', 'Å', 'angstrom']")
        
        ax.set_title(sub_title, {"fontsize": 1.2*fontsize})
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if unit == "nm" or unit == "nanometer":
            cbar.set_label("[nm]", fontdict={"fontsize": fontsize})
        elif unit == "A" or unit == "Å" or unit == "angstrom":
            cbar.set_label("[nm]", fontdict={"fontsize": fontsize})
    
    if text is not None:
        _ = ax.text(
            0.95, 0.95, text, 
            transform=ax.transAxes, fontsize=fontsize, 
            color=text_dict.get("color", "white"), va=text_dict.get("va", "top"), ha=text_dict.get("ha", "right"),
            )
    
    if traj is not None:
        _ = plot_pdb_z_projection(traj, subplots=(fig, ax), unit=unit, r=traj_dict.get("r", 1.0), fontsize=fontsize, is_tqdm=traj_dict.get("is_tqdm", True))
        
    # Save or display
    if save_path is not None:
        fig.savefig(save_path, dpi=600)
        plt.close(fig)
    
    plt.tight_layout()

    return fig, axes

def plot_loss(
    *args,
    subplots=None,
    axsize=(5.0, 3.0),
    smoothness=100,
    xlabel=None,
    ylabel=None,
    title=None,
    grid=True,
    ):
    # Initialize
    if subplots is None:
        fig, ax = plt.subplots(figsize=axsize)
    else:
        fig, ax = subplots
    
    if len(args) == 1:
        loss = args[0].ravel()
        epochs = np.arange(len(loss))
    elif len(args) == 2:
        epochs = args[0].ravel()
        loss = args[1].ravel()
    else:
        raise NotImplementedError(args)
    
    # Smooth
    len_loss = len(loss.ravel()) // smoothness
    target_loss = loss.ravel()[:len_loss*smoothness].reshape((len_loss, smoothness))
    target_epochs = epochs[:len_loss*smoothness].reshape((len_loss, smoothness))
    smoothed_loss = np.mean(target_loss, axis=1)
    smoothed_epochs = np.median(target_epochs, axis=1)
    
    # Set xticks
    xmax = np.max(smoothed_epochs)
    if xmax <= 1000:
        ax.set_xticks(np.linspace(0, xmax - 1, 10, dtype=int))
    else:
        xbin = int((xmax / 10) // 50 * 50)
        xdecimal = 10 ** math.floor(math.log10(xmax))
        xticks = np.arange(0, xmax, xbin)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x / xdecimal:.1f}" for x in xticks])
        # Add [xdecimal] annotation to the right side
        ax.text(
            1.02, -0.09, f"[{xdecimal:.0e}]",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom'
        )
    
    # Plot
    ax.plot(smoothed_epochs, smoothed_loss)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=grid)
    ax.set_title(title)
    return fig, ax

def plot_3d_surface(surface, title=None):
    fig_tip = go.Figure(data=[go.Surface(z=surface)])
    fig_tip.update_traces(contours_z=dict(show=True, usecolormap=True,
                                         highlightcolor="limegreen", project_z=True))
    fig_tip.update_layout(title=title, autosize=False,
                         width=600, height=500,
                         margin=dict(l=65, r=50, b=65, t=50))
    fig_tip.show()

def get_noise_robustness_axes():
    # Create Figure and GridSpec
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(2, 6, figure=fig)

    # Left side (4 small plots in a 2×2 grid)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Right side (two large plots, each spanning 2×2)
    ax5 = fig.add_subplot(gs[:, 2:4])
    ax6 = fig.add_subplot(gs[:, 4:6])
    
    axes = np.array([ax1, ax2, ax3, ax4, ax5, ax6])
    
    return fig, axes
    
def plot_explicit_heatmap(
    X, Y, Z,
    cmap="coolwarm",
    xlabel="X", ylabel="Y", zlabel="Z",
    decimals=1,
    subplots=None,
    fontsize=None,
    invert_yaxis=True,
    xticks_size=None,
    yticks_size=None,
    xticks_skip=1,
    yticks_skip=1,
    **kwargs
):
    """
    Args:
        X, Y: (N,) array.
        Z: (N, N) array.
        cmap: str = "coolwarm"
        xlabel, ylabel, zlabel: axis and colorbar labels
        decimals: number of decimal places for axis ticks
        subplots: (fig, ax) or None
    """
    # Prepare subplot
    if subplots is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = subplots
    
    # Define color mapping
    norm = mcolors.Normalize(vmin=np.min(Z), vmax=np.max(Z))
    colormap = plt.get_cmap(cmap)

    # Fill one rectangle at a time
    for i in range(len(X)):
        for j in range(len(Y)):
            color = colormap(norm(Z[i, j]))
            ax.add_patch(
                plt.Rectangle((i, j), 1, 1, facecolor=color, edgecolor="none")
            )

    # Tick skipping
    X_indices = np.arange(len(X))
    Y_indices = np.arange(len(Y))
    X_ticks = X_indices[::xticks_skip]
    Y_ticks = Y_indices[::yticks_skip]

    # Set axis ticks
    ax.set_xticks(X_ticks + 0.5)
    ax.set_xticklabels(
        [f"{X[i]:.{decimals}f}" for i in X_ticks],
        fontsize=xticks_size
    )

    ax.set_yticks(Y_ticks + 0.5)
    ax.set_yticklabels(
        [f"{Y[i]:.{decimals}f}" for i in Y_ticks],
        fontsize=yticks_size
    )

    # Axis labels
    ax.set_xlabel(xlabel, fontdict={"fontsize": fontsize})
    ax.set_ylabel(ylabel, fontdict={"fontsize": fontsize})

    # Axis limits and style
    ax.set_xlim(0, len(X))
    ax.set_ylim(0, len(Y))
    ax.set_aspect("equal")
    if invert_yaxis:
        ax.invert_yaxis()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(zlabel, fontdict={"fontsize": fontsize})

    return fig, ax

# Dictionary of atomic radii (unit: Ångström)
ATOMIC_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'Na': 1.86, 'K': 2.27, 'Ca': 1.97, 'Fe': 1.26, 'Mg': 1.60,
    # Add more elements as needed
}

def get_atomic_radius(element):
    """Get atomic radius. Returns default value if not found in dictionary"""
    return ATOMIC_RADII.get(element.capitalize(), 1.0)

def plot_pdb_z_projection(
    traj, 
    image=None,
    r=1.0, 
    subplots=None, 
    edit_plot_range=False, 
    unit="nm", 
    fontsize=14,
    is_tqdm=True,
    title=None,
    **kwargs,
    ):
    """
    Plot the PDB structure as seen from the z-axis.
    Parameters:
        pdb_file (str): PDB file path
        subplots (tuple): (fig, ax) or None (created internally if None)
    """
    # Load PDB file
    atoms = list(traj.topology.atoms)
    xyz = traj.xyz[0]

    # Move
    if image is not None:
        center = np.asarray(center_of_mass(image.reshape(image.shape[-2:])))
        z_height = (np.mean(xyz, axis=0) - np.min(xyz, axis=0))[2]
        new_com = np.concatenate([center.ravel(), z_height.ravel()], axis=0)
        traj_com = np.mean(xyz, axis=0)
        xyz = xyz - traj_com[None,:] + new_com[None,:]
        
    # Decompose coordinates
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    
    # Get atomic radii and element types
    elements = [atom.element.symbol if atom.element is not None else 'C' for atom in atoms]
    radii = r * np.array([get_atomic_radius(el) for el in elements])

    # Sort in ascending order of z (back → front)
    sort_idx = np.argsort(z)
    x = x[sort_idx]
    y = y[sort_idx]
    z = z[sort_idx]
    radii = radii[sort_idx]
    elements = [elements[i] for i in sort_idx]
    
    # Data range
    x_range = (np.min(x), np.max(x))
    y_range = (np.min(y), np.max(y))
    x_buffer = 0.1 * (x_range[1] - x_range[0])
    y_buffer = 0.1 * (y_range[1] - y_range[0])
    x_lim = (x_range[0] - x_buffer, x_range[1] + x_buffer)
    y_lim = (y_range[0] - y_buffer, y_range[1] + y_buffer)
    
    # Color based on z-coordinate (higher z → brighter color)
    norm = plt.Normalize(vmin=np.min(z), vmax=np.max(z))
    cmap = plt.cm.bone
    colors = cmap(norm(z))

    # Prepare subplot
    if subplots is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        show_plot = True
    else:
        fig, ax = subplots
        show_plot = False  # Control display externally
    
    # Plot atoms as circles
    for i, (xi, yi, ri, ci) in tqdm(enumerate(zip(x, y, radii, colors)), disable=not is_tqdm, total=len(x), desc="Plotting coordinates..."):
        circle = Circle((xi, yi), radius=ri, facecolor=ci, alpha=0.9, edgecolor='black', linewidth=0.3)
        # inner highlight (optional)
        # inner = Circle((xi - 0.2*ri, yi + 0.2*ri), radius=0.2*ri, color='white', alpha=0.4, zorder=2)
        ax.add_patch(circle)
        # ax.add_patch(inner)
    
    if unit == "A" or unit == "angstrom":
        ax.set_xlabel("X [Å]", fontsize=fontsize)
        ax.set_ylabel("Y [Å]", fontsize=fontsize)
    elif unit == "nm" or unit == "nanometer":
        ax.set_xlabel("X [nm]", fontsize=fontsize)
        ax.set_ylabel("Y [nm]", fontsize=fontsize)
    
    if title is not None:
        ax.set_title(title, fontsize=1.2*fontsize)
    ax.set_aspect("equal")
    
    if edit_plot_range:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, ax

def plot_hist(
    data,
    label=None,
    pad=0.1,
    bins=40,
    subplots=None,
    xlabel=None,
    ylabel=None,
    title=None,
    vmin=None,
    vmax=None,
    color="skyblue",
    ls="-",
    lw=1.5,
    decimals=2,   # Added: number of decimal places for x-axis labels
    fontsize=14,
    plot_normal=False,
    stepfilled=False,
    density=True,
    add_mean_vline=False,
    add_text=False,
    **kwargs
):
    if subplots is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = subplots
    
    invalid_mask = np.isnan(data) | np.isinf(data)
    valid_data = data[~invalid_mask]
    if vmin is None and vmax is None:
        data_range = (np.min(valid_data), np.max(valid_data))
        x_padding = pad * (data_range[1] - data_range[0])
    else:
        data_range = (vmin, vmax)
        x_padding = 0.0
    
    if stepfilled:
        count, bins, _ = ax.hist(
            valid_data.ravel(),
            range=(data_range[0]-x_padding, data_range[1]+x_padding),
            bins=bins, 
            histtype='stepfilled', 
            edgecolor=color, 
            facecolor='none',
            label=label,
            density=density,
            linestyle=ls,
            linewidth=lw,
            **kwargs
        )
    else:
        count, bins, _ = ax.hist(
            valid_data.ravel(),
            range=(data_range[0]-x_padding, data_range[1]+x_padding),
            bins=bins, 
            color=color,
            density=density,
            label=label,
            **kwargs
        )
    
    if density:
        ymax = np.max(count) / np.sum(count)
    else:
        ymax = np.max(count)
        
    y_padding = pad * ymax
    
    # Show mean value
    if add_mean_vline:
       ax.vlines(
            np.mean(valid_data),
            ymin=0,
            ymax=ymax+y_padding,
            colors="salmon",
            linestyles="--"
        )
    
    # Show normal distribution
    if plot_normal:
        normal = np.exp(- (bins - np.mean(valid_data))**2 / (2 * np.var(valid_data)))
        normal = normal / np.max(normal) * np.max(count)
        ax.plot(bins, normal, ls="--", c="blue")
    
    # X-axis range
    ax.set_xlim(data_range[0]-x_padding, data_range[1]+x_padding)
    
    # Axis labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    # Title
    if title is not None:
        title = title.strip() + f", Mean: {np.mean(valid_data):.{decimals}f} Std: {np.std(valid_data):.{decimals}f}"
        ax.set_title(title, fontdict={"fontsize": fontsize})

    # ★ Fix x-axis ticks at step 0.1 & show with 1 decimal place
    tick_step = np.round((data_range[1] - data_range[0]) / 5, decimals=2)
    if tick_step == 0:
        tick_step = 0.01
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    
    # ★ Display statistics in the upper-right corner of the plot
    if add_text:
        mean_val = np.mean(valid_data)
        max_val = np.max(valid_data)
        ax.text(
            0.98, 0.95,
            f"Max: {max_val:.{decimals}f}\nMean: {mean_val:.{decimals}f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=fontsize,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )
        
    return fig, ax

def plot_inter_domain_distance(
    traj, 
    domain_pairs,
    xtype=0,
    ytype=1,
    ztype=2,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    subplots=None,
    domain_pair_labels=None,
    cmap="viridis",
    basecolor="black",
    fontsize=14,
    marker="o",
    xlim=None,
    ylim=None,
    title=None,
    ):
    assert all(axistype is None or isinstance(axistype, (int, np.ndarray)) for axistype in [xtype, ytype, ztype])
    if domain_pair_labels is not None:
        assert len(domain_pair_labels) == len(domain_pairs)
    
    # Compute domain distance
    domain_distance = np.zeros((len(traj), len(domain_pairs)))
    for i, (d1, d2) in enumerate(domain_pairs):
        domain_distance[:,i] = compute_domain_distance(traj, d1, d2).ravel()
    
    # Distribute in axis
    if isinstance(xtype, int):
        x = domain_distance[:,xtype]
        if xlabel is not None:
            xlabel = xlabel
        elif domain_pair_labels is not None:
            xlabel = domain_pair_labels[xtype]
        else:
            xlabel = None
    elif isinstance(xtype, np.ndarray):
        assert len(xtype) == len(domain_distance), f"len(xtype): {len(xtype)} != len(domain_distance): {len(domain_distance)}"
        x = xtype
        if xlabel is not None:
            xlabel = xlabel
        else:
            xlabel = None
    else:
        raise NotImplementedError
    
    if isinstance(ytype, int):
        y = domain_distance[:,ytype]
        if ylabel is not None:
            ylabel = ylabel
        elif domain_pair_labels is not None:
            ylabel = domain_pair_labels[ytype]
        else:
            ylabel = None
    elif isinstance(ytype, np.ndarray):
        assert len(ytype) == len(domain_distance), f"len(ytype): {len(ytype)} != len(domain_distance): {len(domain_distance)}"
        y = ytype
        if ylabel is not None:
            ylabel = ylabel
        else:
            ylabel = None
    else:
        raise NotImplementedError
    
    if isinstance(ztype, int):
        z = domain_distance[:,ztype]
        if zlabel is not None:
            zlabel = zlabel
        elif domain_pair_labels is not None:
            zlabel = domain_pair_labels[ztype]
        else:
            zlabel = None
    elif isinstance(ztype, np.ndarray):
        assert len(ztype) == len(domain_distance), f"len(ztype): {len(ztype)} != len(domain_distance): {len(domain_distance)}"
        z = ztype
        if zlabel is not None:
            zlabel = zlabel
        else:
            zlabel = None
    elif ztype is None:
        z = None
        zlabel = None
    else:
        raise NotImplementedError
    
    # Plot
    if subplots is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig, ax = subplots
    
    if z is None:
        sc = ax.scatter(x, y, marker=marker, c=basecolor)
    else:
        sc = ax.scatter(x, y, c=z, cmap=cmap, marker=marker)
        cbar = plt.colorbar(sc, ax=ax)
        if zlabel is not None:
            cbar.set_label(zlabel, fontdict={"fontsize": fontsize})
    
    # Clean the format
    ax.set_xlabel(xlabel, fontdict={"fontsize": fontsize})
    ax.set_ylabel(ylabel, fontdict={"fontsize": fontsize})
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title is not None:
        ax.set_title(title, fontdict={"fontsize": fontsize})
    plt.tight_layout()
    return fig, ax

def plot_evaluation_results(
    traj,
    acceptable_mask,
    molprobity_results,
    bond_summeries,
    name, 
    subplots=None, 
    only_acceptable=True,
    axsize=(4, 3),
    ):
    domain_pairs = get_domain_pairs(name)
    domain_pair_names = get_domain_pair_names(name)
    domain_pair_labels = [f"{n1} - {n2}" for n1, n2 in domain_pair_names]
    
    if subplots is None:
        column = max(len(molprobity_results), len(bond_summeries))
        fig, axes = plt.subplots(3, column, figsize=(column*axsize[0], 3*axsize[1]))
    else:
        fig, axes = subplots
    
    _ = plot_inter_domain_distance(
        traj, 
        domain_pairs, 
        domain_pair_labels=domain_pair_labels,
        subplots=(fig, axes[0,0])
        )

    if only_acceptable:
        traj_acc = traj[acceptable_mask]
    else:
        traj_acc = traj
        
    _ = plot_inter_domain_distance(
        traj_acc, 
        domain_pairs, 
        domain_pair_labels=domain_pair_labels,
        subplots=(fig, axes[0,1])
        )

    for i, (name, score) in enumerate(molprobity_results.items()):
        _ = plot_inter_domain_distance(
            traj, 
            domain_pairs, 
            domain_pair_labels=domain_pair_labels,
            xtype=0,
            ytype=1,
            ztype=score,
            xlabel=None,
            ylabel=None,
            zlabel=name,
            subplots=(fig, axes[1,i])
            )
    
    for i, (name, ratios) in enumerate(bond_summeries.items()):
        _ = plot_inter_domain_distance(
            traj, 
            domain_pairs, 
            domain_pair_labels=domain_pair_labels,
            xtype=0,
            ytype=1,
            ztype=ratios,
            xlabel=None,
            ylabel=None,
            zlabel=name,
            subplots=(fig, axes[2,i])
            )

    plt.tight_layout()
    
    return fig, axes

def plot_bland_altman(
    data1,
    data2,
    data1_label=None,
    data2_label=None,
    color=None,
    color_label=None,
    cmap="viridis",
    data_unit="",
    subplots=None,
    xlim=None,
    ylim=None,
    xlabel=None,
    ylabel=None,
    title=None,
    add_zero_yline=True,
    fontsize=14,
    tick_fontsize=None,
    bbox_to_anchor=(0.5, -0.1),
    **kwargs,
):
    if subplots is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig, ax = subplots
    
    if tick_fontsize is None:
        tick_fontsize = 0.6 * fontsize
    # Data for Bland–Altman plot
    mean_values = (data1 + data2) / 2
    diff_values = data2 - data1

    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values)

    # Scatter plot
    if color is not None:
        # Determine whether color is an array or a single value
        if np.ndim(color) == 1:
            assert len(color) == len(mean_values), \
                "color must have the same length as data"
            sc = ax.scatter(
                mean_values, diff_values,
                c=color, cmap=cmap, alpha=0.7, edgecolor='k', 
            )
            # Add colorbar (works even when using subplots)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(color_label, fontsize=0.9*fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize)
        else:
            # When a single color is specified
            ax.scatter(mean_values, diff_values, color=color, alpha=0.7, edgecolor='k')
    else:
        ax.scatter(mean_values, diff_values, alpha=0.7, edgecolor='k')

    # Mean line and ±1.96 SD lines (95% limits of agreement)
    ax.axhline(mean_diff, color='salmon', linestyle='--', label=f'Mean = {mean_diff:.3f}')
    ax.axhline(mean_diff + 1.96 * std_diff, color='slateblue', linestyle=':', label='±1.96 SD')
    ax.axhline(mean_diff - 1.96 * std_diff, color='slateblue', linestyle=':')
    
    if add_zero_yline:
        ax.axhline(0.0, color='black', linestyle='--')
        
    # Axis labels and title
    if xlabel is None and (data1_label is not None and data2_label is not None):
        xlabel = f"({data2_label.strip()} + {data1_label.strip()}) / 2 {data_unit}".strip()
    ax.set_xlabel(xlabel, fontdict={"fontsize": fontsize})
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    
    if ylabel is None and (data1_label is not None and data2_label is not None):
        ylabel = f"{data2_label.strip()} - {data1_label.strip()} {data_unit}".strip()
    ax.set_ylabel(ylabel, fontdict={"fontsize": fontsize})
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    
    if title is not None:
        ax.set_title(title, fontdict={"fontsize": 1.2*fontsize})

    # Adjust axis ranges if specified
    if xlim is not None:
        ax.set_xlim(xlim)
        
    if ylim is not None:
        # ax.set_ylim(mean_diff - 3*std_diff, mean_diff + 3*std_diff)
        ax.set_ylim(ylim)

    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=bbox_to_anchor, ncol=2, fontsize=0.8*fontsize)
    
    return fig, ax



