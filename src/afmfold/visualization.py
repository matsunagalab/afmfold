import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches.Circle as Circle
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

def plot_afm(
    images, 
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
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
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
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Save or display
    if save_path is not None:
        fig.savefig(save_path, dpi=600)
        plt.close(fig)
    
    plt.tight_layout()

    return fig, axes

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

    # Set axis ticks
    ax.set_xticks(np.arange(len(X)) + 0.5)
    ax.set_xticklabels([f"{val:.{decimals}f}" for val in X])

    ax.set_yticks(np.arange(len(Y)) + 0.5)
    ax.set_yticklabels([f"{val:.{decimals}f}" for val in Y])

    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Axis limits and style
    ax.set_xlim(0, len(X))
    ax.set_ylim(0, len(Y))
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(zlabel)

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

def plot_pdb_z_projection(traj, r=1.0, subplots=None, print_progress=True, edit_plot_range=False, set_title=True, unit="A", fontsize=14):
    """
    Plot the PDB structure as seen from the z-axis.
    Parameters:
        pdb_file (str): PDB file path
        subplots (tuple): (fig, ax) or None (created internally if None)
    """
    # Load PDB file
    atoms = list(traj.topology.atoms)
    xyz = traj.xyz[0]

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
    for i, (xi, yi, ri, ci) in enumerate(zip(x, y, radii, colors)):
        if print_progress:
            total_length = 50
            percent = int((i+1) / len(x) * 1000) / 10
            bar = "#" * int((i+1) / len(x) * total_length)
            rest = "-" * (total_length - len(bar))
            end = "" if i < len(x) - 1 else "\n"
            print(f"\r{bar}{rest} [{percent}%]", end=end)
            
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
        
    if set_title:
        ax.set_title("Z-axis Projection of PDB (colored by height)", fontsize=1.2*fontsize)
    ax.set_aspect("equal")
    
    if edit_plot_range:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        
    if show_plot:
        fig.tight_layout()
        plt.show()

    return fig, ax

def apply_inverse_rotation(r1, r2):
    # r2_inv: (B1, 3, 3)
    r2_inv = np.transpose(r2, (0, 2, 1))
    # r1: (B1, B2, 3, 3), r2_inv: (B1, 3, 3)
    # Apply batch-wise using einsum
    return np.einsum('bijk,bkl->bijl', r1, r2_inv)

def distribute_rots_in_grids(rots, best_rot, correlations, lat_grid=90, lon_grid=45, **kwargs):
    # Distribute in grids
    indices = rotation_matrix_to_latlon_index(rots, lat_grid, lon_grid)
    best_index = rotation_matrix_to_latlon_index(best_rot, lat_grid, lon_grid)
    corr_values = compute_grid_averages(indices, correlations, lat_grid, lon_grid)
    return corr_values, best_index

def rotation_matrix_to_latlon_index(rotations, num_lat_bins, num_lon_bins):
    """
    Map (B, 3, 3) rotation matrices to (B, 2) integer indices representing lat-lon grid bins.

    Args:
        rotations (array-like): shape (B, 3, 3) or (3, 3); SO(3) rotation matrices.
        num_lat_bins (int)   : number of latitude bins  (-90° … +90°)
        num_lon_bins (int)   : number of longitude bins (  0° … 360°)

    Returns:
        ndarray[int]: shape (B, 2) with (lat_index, lon_index) per rotation.
    """
    # 1. Ensure input shape is (B,3,3)
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim == 2:                      # Convert single input to batch
        rotations = rotations[None, ...]
    if rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape (B,3,3) or (3,3)")
    
    # 2. Rotate the x-axis vector
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rotated_xyz = rotations @ x_axis            # (B,3)
    
    # 3. Convert coordinates → latitude/longitude (in radians)
    x, y, z = rotated_xyz.T
    lat_rad = np.arcsin(np.clip(z, -1.0, 1.0))       # [-π/2, +π/2]
    lon_rad = np.mod(np.arctan2(y, x) + np.pi, 2.0 * np.pi)  # [0, 2π)
    
    # 4. Normalize radians to [0,1)
    lat_frac = (lat_rad + np.pi / 2.0) / np.pi       # -90°→0, +90°→1
    lon_frac = lon_rad / (2.0 * np.pi)               #   0°→0, 360°→1

    # 5. Convert to integer indices in range 0 – (n-1)
    lat_idx = np.minimum((lat_frac * num_lat_bins).astype(int), num_lat_bins - 1)
    lon_idx = np.minimum((lon_frac * num_lon_bins).astype(int), num_lon_bins - 1)

    return np.stack([lat_idx, lon_idx], axis=-1)

def compute_grid_averages(indices, values, lat_grid, lon_grid):
    """
    Compute average of values per lat-lon grid cell.

    Args:
        indices: ndarray of shape (B, 2), integer indices (lat_idx, lon_idx)
        values: ndarray of shape (B,), values to average
        lat_grid: int, number of latitude bins
        lon_grid: int, number of longitude bins

    Returns:
        grid_values: ndarray of shape (lat_grid, lon_grid), mean of values in each cell (np.nan if empty)
    """
    # Initialize sum and count arrays
    sum_grid = np.zeros((lat_grid, lon_grid), dtype=np.float64)
    count_grid = np.zeros((lat_grid, lon_grid), dtype=np.int32)

    lat_idx = indices[:, 0]
    lon_idx = indices[:, 1]

    for i in range(values.shape[0]):
        lat = lat_idx[i]
        lon = lon_idx[i]
        if 0 <= lat < lat_grid and 0 <= lon < lon_grid:
            sum_grid[lat, lon] += values[i]
            count_grid[lat, lon] += 1

    # Avoid division by zero; assign np.nan where count is 0
    with np.errstate(invalid='ignore', divide='ignore'):
        grid_values = sum_grid / count_grid
        grid_values[count_grid == 0] = np.nan

    return grid_values

def get_rigid_body_fitting_axes(figsize=3.0):
    fig = plt.figure(figsize=(4*figsize, 2*figsize))

    gs = gridspec.GridSpec(2, 4, figure=fig)
    axes = np.array([[None]*4 for _ in range(2)])

    for i in range(2):
        for j in range(4):
            if (i, j) in [(0,2), (1,2)]:
                # Mollweide 投影の Axes
                ax = fig.add_subplot(gs[i, j], projection=ccrs.Mollweide(central_longitude=0))
                ax.set_global()
            else:
                # 通常の Axes
                ax = fig.add_subplot(gs[i, j])
            axes[i][j] = ax
    return fig, axes

def estimate_reasonable_range(arr, min_r=0.25, max_r=0.75, r=2.0):
    """
    Estimate and return a "reasonable range" from an arbitrary-shaped array arr.
    Procedure:
      1) Compute quantile values qmin, qmax at min_r, max_r
      2) Compute empirical probability p that data lies within [qmin, qmax]
      3) Solve for scale such that ∫_{scale*(min_r-0.5)}^{scale*(max_r-0.5)} φ(z)dz = p 
         where φ is the standard normal pdf.
         (When min_r=0.25, max_r=0.75 this interval is symmetric: ±0.25*scale)
         ⇒ Let a = 0.25*scale, then 2*Φ(a) - 1 = p,
            so a = Φ^{-1}((1+p)/2),
            thus scale = 4 * Φ^{-1}((1+p)/2)
      4) Assume I = qmax - qmin corresponds to ±aσ in normal distribution, 
         so σ̂ = I / (2a) = 2I/scale
      5) Using mean, return (mean - 2σ̂, mean + 2σ̂)

    Returns:
      (lower, upper, details)
      details is a dictionary of intermediate values (qmin, qmax, p, scale, sigma, mean)
    """
    x = np.asarray(arr).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No valid numbers in arr (requires non-NaN/finite values)")
    if not (0 < min_r < max_r < 1):
        raise ValueError("Require 0 < min_r < max_r < 1")

    # 1) Quantiles
    qmin, qmax = np.quantile(x, [min_r, max_r])

    # 2) Empirical probability (including endpoints)
    p = np.mean((x >= qmin) & (x <= qmax))
    # Numerical stabilization
    eps = 1e-12
    p = float(np.clip(p, eps, 1 - eps))

    # 3) Solve for scale
    a = _norm_ppf((1.0 + p) / 2.0)          # a = 0.25 * scale
    scale = 4.0 * a

    # 4) Estimate σ̂ (I = qmax-qmin corresponds to ±aσ)
    I = qmax - qmin
    if a <= 0 or I < 0:
        raise RuntimeError("Invalid scale or IQR (check data distribution)")
    sigma = I / (2.0 * a)   # Equivalent: sigma = 2*I/scale

    # 5) Return ±rσ range
    mean_value = float(np.mean(x))
    lower = mean_value - r * sigma
    upper = mean_value + r * sigma

    details = {
        "qmin": float(qmin),
        "qmax": float(qmax),
        "p_between": p,
        "scale": float(scale),
        "sigma": float(sigma),
        "mean": mean_value,
        "min_r": float(min_r),
        "max_r": float(max_r),
    }
    return lower, upper, details

def _norm_ppf(p: float) -> float:
    """
    Inverse CDF (ppf) of the standard normal distribution for one-sided probability p.
    Uses Peter J. Acklam’s well-known approximation (no external dependencies).
    Error is sufficiently small for practical use.
    """
    if p <= 0.0 or p >= 1.0:
        if p == 0.0:
            return -np.inf
        if p == 1.0:
            return np.inf
        raise ValueError("p must be in (0,1)")

    # Coefficients (Acklam, 2003)
    a = [ -3.969683028665376e+01,  2.209460984245205e+02,
          -2.759285104469687e+02,  1.383577518672690e+02,
          -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02,
          -1.556989798598866e+02,  6.680131188771972e+01,
          -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01,
          -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00 ]

    # Breakpoints
    plow  = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = np.sqrt(-2*np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    elif phigh < p:
        q = np.sqrt(-2*np.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    else:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def plot_mollweide_heatmap(
    data, 
    lon=None, 
    lat=None,
    label="Sample Data",
    title="Mollweide Heatmap (Grid-based)",
    subplots=None, 
    vmin=None, 
    vmax=None,
    # Show geographic grid (longitude/latitude lines)
    show_geo_grid=True,
    grid_lon_step=60,
    grid_lat_step=30,
    grid_kwargs=None,
    # Cell boundary grid (grid lines of pcolormesh)
    show_cell_grid=False,
    cell_edgecolor="k",
    cell_linewidth=0.2,
    cell_alpha=0.8,
    cmap="coolwarm",
    fontsize=14,
):
    """
    Draw a heatmap with Mollweide projection (with options for geographic lines/cell boundaries)

    Args:
        data (ndarray): shape (lat_grid, lon_grid)
        lon (1D ndarray or None): Longitude (degrees). If None, generate automatically with equal spacing [-180, 180]
        lat (1D ndarray or None): Latitude (degrees). If None, generate automatically with equal spacing [-90, 90]
    """
    lat_grid, lon_grid = data.shape

    if lon is None:
        lon = np.linspace(-180, 180, lon_grid)
    if lat is None:
        lat = np.linspace(-90, 90, lat_grid)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # If no existing subplots, create Axes with Cartopy projection
    if subplots is None:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0.0))
    else:
        fig, ax = subplots
        # Ensure existing Axes is Mollweide
        assert isinstance(getattr(ax, "projection", None), ccrs.Mollweide)

    ax.set_global()

    # pcolormesh: In Cartopy, data is transferred from PlateCarree (lat/lon) to Mollweide
    pc_kwargs = dict(shading='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                     transform=ccrs.PlateCarree())
    if show_cell_grid:
        pc_kwargs.update(edgecolors=cell_edgecolor, linewidth=cell_linewidth)
        pc_kwargs.update(alpha=cell_alpha)

    cs = ax.pcolormesh(lon2d, lat2d, data, **pc_kwargs)

    # Geographic grid lines (draw only, labels off)
    if show_geo_grid:
        gk = dict(color="k", linewidth=0.5, alpha=0.6)
        if grid_kwargs:
            gk.update(grid_kwargs)

        # Specify equivalent of dashes=[2,2] using linestyle (Cartopy/Matplotlib style)
        if 'dashes' in gk:
            dash = gk.pop('dashes')
            # Convert to Matplotlib format (offset, on_off_seq)
            gk['linestyle'] = (0, tuple(dash))

        gl = ax.gridlines(draw_labels=False, **gk)

        # Specify longitude/latitude positions with step size
        # (Note) xlocs/ylocs expect degrees. Ranges: [-180, 180], [-90, 90]
        import matplotlib.ticker as mticker
        meridians = np.arange(-180, 181, grid_lon_step)
        parallels = np.arange(-90, 91, grid_lat_step)
        gl.xlocator = mticker.FixedLocator(meridians)
        gl.ylocator = mticker.FixedLocator(parallels)

    if title is not None:
        ax.set_title(title, fontdict={"fontsize": fontsize})

    # Add colorbar at Figure level
    cbar = plt.colorbar(cs, ax=ax, orientation='horizontal', pad=0.05)
    cbar.set_label(label)

    if subplots is None:
        plt.tight_layout()
        plt.show()

    return fig, ax

def plot_mollweide_scatter(
    indices,
    lat_grid,
    lon_grid,
    lon=None,
    lat=None,
    label=None,
    subplots=None,
    marker='o',
    s=10,
    color='black',
    alpha=0.7,
    fontsize=14,
    **kwargs,
):
    """
    Draw a scatter plot with Mollweide projection (without map decorations)

    Args:
        indices (ndarray): shape (B, 2), each row is (lat_idx, lon_idx)
        lat_grid (int): Number of latitude bins
        lon_grid (int): Number of longitude bins
        lon (ndarray or None): Longitude (degrees). If None, evenly spaced [-180, 180]
        lat (ndarray or None): Latitude (degrees). If None, evenly spaced [-90, 90]
        label (str or None): Legend label
        subplots (tuple or None): (fig, ax). If not specified, create new
        marker, s, color, alpha: Scatter plot style
    Returns:
        (fig, ax)
    """
    # Generate lon/lat axes
    if lon is None:
        lon = np.linspace(-180, 180, lon_grid)
    if lat is None:
        lat = np.linspace(-90, 90, lat_grid)

    # Normalize indices
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices[None, :]
    assert indices.shape[1] == 2, "indices should have shape (B, 2)"
    lat_idx = indices[:, 0].astype(int)
    lon_idx = indices[:, 1].astype(int)

    # Index → latitude/longitude (assume center positions)
    lat_vals = lat[lat_idx]
    lon_vals = lon[lon_idx]
    
    # Prepare Axes (Mollweide projection)
    if subplots is None:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0.0))
    else:
        fig, ax = subplots
        # Ensure existing Axes is Mollweide
        assert isinstance(getattr(ax, "projection", None), ccrs.Mollweide)

    ax.set_global()

    # Scatter plot (lat/lon → Mollweide via transform specification)
    sc = ax.scatter(
        lon_vals,
        lat_vals,
        s=s,
        c=color,
        marker=marker,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
        label=label,
        **kwargs,
    )

    if label is not None:
        ax.legend(loc='upper right', fontsize=fontsize)

    return fig, ax

def plot_hist(
    data,
    pad=0.1,
    subplots=None,
    xlabel=None,
    ylabel=None,
    title=None,
    vmin=None,
    vmax=None,
    decimals=2,   # Added: number of decimal places for x-axis labels
    fontsize=14,
    plot_normal=False,
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
    else:
        data_range = (vmin, vmax)
    
    x_padding = pad * (data_range[1] - data_range[0])
    count, bins, _ = ax.hist(
        valid_data.ravel(),
        range=(data_range[0]-x_padding, data_range[1]+x_padding),
        color="skyblue",
        **kwargs
    )
    y_padding = pad * np.max(count)
    
    # Show mean value
    ax.vlines(
        np.mean(valid_data),
        ymin=0,
        ymax=np.max(count)+y_padding,
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
