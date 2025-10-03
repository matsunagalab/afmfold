import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import matplotlib.cm as cm
from matplotlib import ticker
import matplotlib.gridspec as gridspec
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
    AFM像、tip形状、原子構造散布図を 2x2 プロットで表示または保存する。

    Parameters:
        images (list or array): AFM画像. [..., H, W] 配列
        x_range (tuple): x軸
        y_range (tuple): y軸
        save_path (str): ファイル保存パス（Noneなら表示のみ）

    Returns:
        fig, axes: matplotlibのFigureとAxesオブジェクト
    """
    # 初期設定
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
    
    # サブプロットを作成（2行2列）
    if subplots is not None:
        fig, axes = subplots
    else:
        fig, axes = plt.subplots(1, B, figsize=(B*(axsize+padding), axsize))
    
    if B == 1:
        axes = [axes]
            
    for i in range(B):
        ax = axes[i]
        sub_title = title_list[i]
        
        # カラーバー範囲の自動決定
        _vmin = vmin if vmin is not None else np.min(images[i])
        _vmax = vmax if vmax is not None else np.max(images[i])
        
        # AFM イメージ
        extent = [x_range[i,0], x_range[i,-1], y_range[i,-1], y_range[i,0]]  # Y軸反転
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
    
    # 保存 or 表示
    if save_path is not None:
        fig.savefig(save_path, dpi=600)
        plt.close(fig)
    
    plt.tight_layout()

    return fig, axes

# 原子半径の辞書（単位：オングストローム）
ATOMIC_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'Na': 1.86, 'K': 2.27, 'Ca': 1.97, 'Fe': 1.26, 'Mg': 1.60,
    # その他は必要に応じて追加
}

def get_atomic_radius(element):
    """原子半径を取得。辞書にない場合はデフォルト値を返す"""
    return ATOMIC_RADII.get(element.capitalize(), 1.0)

def plot_pdb_z_projection(traj, r=1.0, subplots=None, print_progress=True, edit_plot_range=False, set_title=True, unit="A", fontsize=14):
    """
    z軸方向から見たPDB構造をプロット
    Parameters:
        pdb_file (str): PDBファイルパス
        subplots (tuple): (fig, ax) または None（その場合は内部で作成）
    """
    # PDBファイルを読み込む
    atoms = list(traj.topology.atoms)
    xyz = traj.xyz[0]

    # 座標分解
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    
    # 原子半径と種類を取得
    elements = [atom.element.symbol if atom.element is not None else 'C' for atom in atoms]
    radii = r * np.array([get_atomic_radius(el) for el in elements])

    # zの値で昇順に並べ替え（背→前）
    sort_idx = np.argsort(z)
    x = x[sort_idx]
    y = y[sort_idx]
    z = z[sort_idx]
    radii = radii[sort_idx]
    elements = [elements[i] for i in sort_idx]
    
    # データ範囲
    x_range = (np.min(x), np.max(x))
    y_range = (np.min(y), np.max(y))
    x_buffer = 0.1 * (x_range[1] - x_range[0])
    y_buffer = 0.1 * (y_range[1] - y_range[0])
    x_lim = (x_range[0] - x_buffer, x_range[1] + x_buffer)
    y_lim = (y_range[0] - y_buffer, y_range[1] + y_buffer)
    
    # 色はz座標に基づく（高いzほど明るい色）
    norm = plt.Normalize(vmin=np.min(z), vmax=np.max(z))
    cmap = plt.cm.bone
    colors = cmap(norm(z))

    # サブプロットの準備
    if subplots is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        show_plot = True
    else:
        fig, ax = subplots
        show_plot = False  # 外部で表示を制御
    
    # プロット
    for i, (xi, yi, ri, ci) in enumerate(zip(x, y, radii, colors)):
        if print_progress:
            total_length = 50
            percent = int((i+1) / len(x) * 1000) / 10
            bar = "#" * int((i+1) / len(x) * total_length)
            rest = "-" * (total_length - len(bar))
            end = "" if i < len(x) - 1 else "\n"
            print(f"\r{bar}{rest} [{percent}%]", end=end)
            
        circle = Circle((xi, yi), radius=ri, facecolor=ci, alpha=0.9, edgecolor='black', linewidth=0.3)
        #inner = Circle((xi - 0.2*ri, yi + 0.2*ri), radius=0.2*ri, color='white', alpha=0.4, zorder=2)
        ax.add_patch(circle)
        #ax.add_patch(inner)
    
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
    # einsumでバッチごとに適用
    return np.einsum('bijk,bkl->bijl', r1, r2_inv)

def distribute_rots_in_grids(rots, best_rot, correlations, lat_grid=90, lon_grid=45, **kwargs):
    # グリッドに配分
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
    # 1. 入力を (B,3,3) に揃える
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim == 2:                      # 単発をバッチ化
        rotations = rotations[None, ...]
    if rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape (B,3,3) or (3,3)")
    
    # 2. x 軸ベクトルを回転
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rotated_xyz = rotations @ x_axis            # (B,3)
    
    # 3. 座標 → 緯度・経度（ラジアン）
    x, y, z = rotated_xyz.T
    lat_rad = np.arcsin(np.clip(z, -1.0, 1.0))       # [-π/2, +π/2]
    lon_rad = np.mod(np.arctan2(y, x) + np.pi, 2.0 * np.pi)  # [0, 2π)
    
    # 4. ラジアンを [0,1) に正規化
    lat_frac = (lat_rad + np.pi / 2.0) / np.pi       # -90°→0, +90°→1
    lon_frac = lon_rad / (2.0 * np.pi)               #   0°→0, 360°→1

    # 5. 0 – (n-1) の整数インデックスへ
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

def get_default_axes(figsize=3.0):
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
    任意形状の配列 arr から「妥当な範囲」を推定して返す。
    手順:
      1) 分位点 min_r, max_r の値 qmin, qmax を求める
      2) データが [qmin, qmax] に入る経験確率 p を計算する
      3) 標準正規の ∫_{scale*(min_r-0.5)}^{scale*(max_r-0.5)} φ(z)dz = p を満たす scale を解く
         （この区間は min_r=0.25, max_r=0.75 のとき対称：±0.25*scale）
         ⇒ a = 0.25*scale,  2*Φ(a) - 1 = p から  a = Φ^{-1}((1+p)/2),
            よって scale = 4 * Φ^{-1}((1+p)/2)
      4) I = qmax - qmin が正規で ±aσ に対応すると仮定し σ̂ = I / (2a) = 2I/scale
      5) 平均値 mean を使い (mean - 2σ̂, mean + 2σ̂) を返す

    戻り値:
      (lower, upper, details)
      details は中間値を辞書で返す（qmin, qmax, p, scale, sigma, mean）
    """
    x = np.asarray(arr).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("arr に有効な数値がありません（NaN/±inf 以外が必要）")
    if not (0 < min_r < max_r < 1):
        raise ValueError("0 < min_r < max_r < 1 を満たしてください")

    # 1) 分位点
    qmin, qmax = np.quantile(x, [min_r, max_r])

    # 2) 経験確率（端点を含める）
    p = np.mean((x >= qmin) & (x <= qmax))
    # 数値安定化
    eps = 1e-12
    p = float(np.clip(p, eps, 1 - eps))

    # 3) scale を求める
    a = _norm_ppf((1.0 + p) / 2.0)          # a = 0.25 * scale
    scale = 4.0 * a

    # 4) σ̂ を求める（I = qmax-qmin が ±aσ に対応）
    I = qmax - qmin
    if a <= 0 or I < 0:
        raise RuntimeError("スケールまたは IQR が不正（データの分布を確認してください）")
    sigma = I / (2.0 * a)   # 同値: sigma = 2*I/scale

    # 5) 2σ 範囲を返す
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
    片側確率 p に対する標準正規分布の逆CDF（ppf）。
    Peter J. Acklam の有名な近似式を使用（外部依存なし）。
    誤差は実用上十分に小さい。
    """
    if p <= 0.0 or p >= 1.0:
        if p == 0.0:
            return -np.inf
        if p == 1.0:
            return np.inf
        raise ValueError("p must be in (0,1)")

    # 係数（Acklam, 2003）
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

    # 分割点
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
    # 経緯線グリッドの表示
    show_geo_grid=True,
    grid_lon_step=60,
    grid_lat_step=30,
    grid_kwargs=None,
    # セル境界グリッド（pcolormesh の格子線）
    show_cell_grid=False,
    cell_edgecolor="k",
    cell_linewidth=0.2,
    cell_alpha=0.8,
    cmap="coolwarm",
    fontsize=14,
):
    """
    Cartopy 版: Mollweide 投影でヒートマップを描画（経緯線/セル境界オプション付き）

    Args:
        data (ndarray): shape (lat_grid, lon_grid)
        lon (1D ndarray or None): 経度（度）。None の場合は等間隔で自動生成 [-180, 180]
        lat (1D ndarray or None): 緯度（度）。None の場合は等間隔で自動生成 [-90, 90]
    """
    lat_grid, lon_grid = data.shape

    if lon is None:
        lon = np.linspace(-180, 180, lon_grid)
    if lat is None:
        lat = np.linspace(-90, 90, lat_grid)
    lon2d, lat2d = np.meshgrid(lon, lat)

    # 既存の subplots が無ければ Cartopy 投影付き Axes を作成
    if subplots is None:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0.0))
    else:
        fig, ax = subplots
        # もし投影が無い/異なる場合は作り直すのが安全
        if not hasattr(ax, 'projection') or not isinstance(ax.projection, ccrs.Mollweide):
            # 既存の枠を再利用して差し替え
            pos = ax.get_position()
            fig.delaxes(ax)
            ax = fig.add_axes(pos, projection=ccrs.Mollweide(central_longitude=0.0))

    ax.set_global()

    # pcolormesh: Cartopy では PlateCarree(=経緯度) から Mollweide 転送
    pc_kwargs = dict(shading='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                     transform=ccrs.PlateCarree())
    if show_cell_grid:
        pc_kwargs.update(edgecolors=cell_edgecolor, linewidth=cell_linewidth)
        pc_kwargs.update(alpha=cell_alpha)

    cs = ax.pcolormesh(lon2d, lat2d, data, **pc_kwargs)

    # 経緯線グリッド（描画のみ、ラベルはオフ）
    if show_geo_grid:
        gk = dict(color="k", linewidth=0.5, alpha=0.6)
        if grid_kwargs:
            gk.update(grid_kwargs)

        # dashes=[2,2] 相当を linestyle で指定（Cartopy/Matplotlib の書式）
        if 'dashes' in gk:
            dash = gk.pop('dashes')
            # Matplotlib の (offset, on_off_seq) 指定に変換
            gk['linestyle'] = (0, tuple(dash))

        gl = ax.gridlines(draw_labels=False, **gk)

        # 経度/緯度の位置を刻みで指定
        # （注意）xlocs/ylocs は度を想定。範囲は [-180, 180], [-90, 90]
        import matplotlib.ticker as mticker
        meridians = np.arange(-180, 181, grid_lon_step)
        parallels = np.arange(-90, 91, grid_lat_step)
        gl.xlocator = mticker.FixedLocator(meridians)
        gl.ylocator = mticker.FixedLocator(parallels)

    if title is not None:
        ax.set_title(title, fontdict={"fontsize": fontsize})

    # カラーバーは Figure 側で付与
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
    Cartopy 版: Mollweide 投影で散布図を描画（地図装飾は描かない）

    Args:
        indices (ndarray): shape (B, 2), 各行は (lat_idx, lon_idx)
        lat_grid (int): 緯度ビン数
        lon_grid (int): 経度ビン数
        lon (ndarray or None): 経度（度）。None の場合は等間隔 [-180, 180]
        lat (ndarray or None): 緯度（度）。None の場合は等間隔 [-90, 90]
        label (str or None): 凡例ラベル
        subplots (tuple or None): (fig, ax)。指定がなければ新規作成
        marker, s, color, alpha: 散布図のスタイル
    Returns:
        (fig, ax)
    """
    # lon/lat 軸の生成
    if lon is None:
        lon = np.linspace(-180, 180, lon_grid)
    if lat is None:
        lat = np.linspace(-90, 90, lat_grid)

    # indices の正規化
    indices = np.asarray(indices)
    if indices.ndim == 1:
        indices = indices[None, :]
    assert indices.shape[1] == 2, "indices should have shape (B, 2)"
    lat_idx = indices[:, 0].astype(int)
    lon_idx = indices[:, 1].astype(int)

    # インデックス → 緯度・経度（中心位置を想定）
    lat_vals = lat[lat_idx]
    lon_vals = lon[lon_idx]
    
    # Axes 準備（Mollweide 投影）
    if subplots is None:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.Mollweide(central_longitude=0.0))
    else:
        fig, ax = subplots
        # 既存 Axes が Mollweide であることを確認
        assert isinstance(getattr(ax, "projection", None), ccrs.Mollweide)

    ax.set_global()

    # 散布図（経緯度 → Mollweide へ変換は transform 指定で行う）
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
    decimals=2,   # 追加: x軸ラベルの小数点桁数
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
    
    # 平均値の表示
    ax.vlines(
        np.mean(valid_data),
        ymin=0,
        ymax=np.max(count)+y_padding,
        colors="salmon",
        linestyles="--"
    )
    
    # 正規分布の表示
    if plot_normal:
        normal = np.exp(- (bins - np.mean(valid_data))**2 / (2 * np.var(valid_data)))
        normal = normal / np.max(normal) * np.max(count)
        ax.plot(bins, normal, ls="--", c="blue")
    
    # x軸範囲
    ax.set_xlim(data_range[0]-x_padding, data_range[1]+x_padding)
    
    # 軸ラベル
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    # タイトル
    if title is not None:
        title = title.strip() + f", Mean: {np.mean(valid_data):.{decimals}f} Std: {np.std(valid_data):.{decimals}f}"
        ax.set_title(title, fontdict={"fontsize": fontsize})

    # ★ x軸目盛りを0.1刻みに固定 & 小数点1桁表示
    tick_step = np.round((data_range[1] - data_range[0]) / 5, decimals=2)
    if tick_step == 0:
        tick_step = 0.01
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_step))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    
    # ★ 図枠内右上に統計量を表示
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