#!/usr/bin/env python3
"""Spherical weather animation on Mars using HOLYSHT.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht

Generates a ~4-5 second animation of a synthetic vorticity-like weather system
evolving on a spherical planet, with Mars-like topography baked into the flow.
The simulation uses spectral advection: forward SHT, spectral damping/forcing
in coefficient space, then inverse SHT back to the grid. The topography
steers the flow via orographic forcing (terrain gradients inject vorticity).

Visual style inspired by the SFNO blog post:
  https://developer.nvidia.com/blog/modeling-earths-atmosphere-with-spherical-fourier-neural-operators/

Output: examples/mars_weather.mp4  (also .gif fallback)
"""

import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

from holysht import RealSHT, InverseRealSHT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Grid and simulation parameters
# ============================================================================
NLAT = 256
NLON = 512
LMAX = NLAT
MMAX = NLON // 2 + 1
FPS = 30
DURATION_S = 8.0
N_FRAMES = int(FPS * DURATION_S)
DT = 0.015          # spectral timestep
DIFFUSION = 0.002   # hyperdiffusion coefficient
TOPO_COUPLING = 0.4 # how strongly terrain steers the flow

# Streamline particle system
N_PARTICLES = 500
MAX_AGE = 30        # frames before respawn (shorter lives)
TRAIL_LEN = 8       # fewer past positions per trail (shorter tails)
ADVECT_DT = 0.10    # advection step size per frame


def rgba_to_transparent_gif_frame(frame, transparency_threshold=8):
    """Convert an RGBA frame into a paletted GIF frame with real transparency.

    Pillow's implicit RGBA-to-GIF path often mattes transparent pixels against
    white. We quantise explicitly, reserve palette index 0 for transparency,
    and premultiply against black so the antialiased edge does not pick up a
    light halo.
    """
    rgba = np.asarray(frame.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]

    # Premultiply against black before quantisation so transparent edges do not
    # inherit a white matte during the GIF conversion.
    rgb = (
        (rgba[..., :3].astype(np.uint16) * alpha[..., None].astype(np.uint16) + 127) // 255
    ).astype(np.uint8)

    quantised = Image.fromarray(rgb).quantize(
        colors=255,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )

    indices = np.asarray(quantised, dtype=np.uint8) + 1
    indices[alpha <= transparency_threshold] = 0

    height, width = indices.shape
    gif_frame = Image.frombytes("P", (width, height), indices.tobytes())
    source_palette = quantised.getpalette() or []
    palette = [0, 0, 0] + source_palette[:255 * 3]
    if len(palette) < 768:
        palette.extend([0] * (768 - len(palette)))
    gif_frame.putpalette(palette[:768])
    gif_frame.info["transparency"] = 0
    gif_frame.info["disposal"] = 2
    return gif_frame


def make_mars_topography():
    """Load real Mars topography from MOLA DEM image.

    Uses the colourised MOLA elevation map (data/mola_topo.jpg) from the Mars
    Orbiter Laser Altimeter dataset. The colour-coded image is converted to a
    scalar elevation field by extracting luminance, then interpolated to the
    simulation grid (NLAT x NLON).

    The image uses a rainbow colour scale where:
      blue/purple = lowest (Hellas ~-8km), green/yellow = mid,
      red/white = highest (Olympus Mons ~21km)

    We recover approximate relative elevation by weighting RGB channels to
    invert the rainbow: R and G correlate with height, B anti-correlates.
    """
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    img_path = os.path.join(data_dir, "mola_topo.jpg")

    img = Image.open(img_path)

    # Crop off the title bar and axis labels/colorbar
    # The image is 12140x6940; the map region is roughly the interior
    w, h = img.size
    # Top ~5% is title, bottom ~3% is axis labels, sides ~2% are axis ticks
    crop_top = int(h * 0.045)
    crop_bottom = int(h * 0.97)
    crop_left = int(w * 0.025)
    crop_right = int(w * 0.975)
    img = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Resample to the simulation grid
    img = img.resize((NLON, NLAT), Image.LANCZOS)
    rgb = np.array(img, dtype=np.float32) / 255.0

    # Convert rainbow colormap to elevation estimate:
    # In the MOLA color scale, elevation roughly follows:
    #   low (blue/purple) -> mid (green/yellow) -> high (red/white)
    # A good proxy: weighted sum that increases with warm colors
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    elevation = 0.5 * r + 0.3 * g - 0.6 * b + 0.2 * (r - b)

    # Normalise to [-1, 1] range centred on the mean
    elevation = (elevation - elevation.mean()) / (elevation.std() + 1e-8)
    # Clip extremes and rescale
    elevation = np.clip(elevation, -3, 3) / 3.0

    topo = torch.from_numpy(elevation).float().to(DEVICE)
    return topo


def compute_topo_forcing(topo, sht, isht):
    """Compute orographic vorticity forcing from terrain gradients.

    grad(topo) dotted with the flow produces vorticity injection.
    We approximate d(topo)/dtheta and d(topo)/dphi spectrally.
    """
    # Use finite differences for speed (the SHT is for the weather field)
    dtheta = np.pi / NLAT
    dphi = 2 * np.pi / NLON

    dtopo_dtheta = torch.zeros_like(topo)
    dtopo_dtheta[1:-1] = (topo[2:] - topo[:-2]) / (2 * dtheta)

    dtopo_dphi = torch.zeros_like(topo)
    dtopo_dphi[:, 1:-1] = (topo[:, 2:] - topo[:, :-2]) / (2 * dphi)
    dtopo_dphi[:, 0] = (topo[:, 1] - topo[:, -1]) / (2 * dphi)
    dtopo_dphi[:, -1] = (topo[:, 0] - topo[:, -2]) / (2 * dphi)

    # Orographic forcing: curl of terrain gradient ≈ vorticity source
    forcing = dtopo_dphi - dtopo_dtheta
    # Smooth it a bit
    forcing_coeffs = sht(forcing.unsqueeze(0))
    # Zero out very high wavenumbers
    cutoff = LMAX // 3
    forcing_coeffs[:, cutoff:, :] = 0
    forcing_coeffs[:, :, cutoff:] = 0
    forcing = isht(forcing_coeffs).squeeze(0)

    return forcing * TOPO_COUPLING


def make_initial_weather():
    """Initial vorticity field: a few cyclone/anticyclone systems."""
    theta = torch.linspace(0, np.pi, NLAT, device=DEVICE)
    phi = torch.linspace(0, 2 * np.pi, NLON, device=DEVICE)
    T, P = torch.meshgrid(theta, phi, indexing="ij")

    weather = torch.zeros_like(T)

    # Large-scale jet streams
    weather += 0.3 * torch.sin(2 * T) * torch.cos(P)
    weather += 0.2 * torch.sin(3 * T) * torch.cos(2 * P - 0.5)

    # Cyclonic vortices
    vortices = [
        (1.2, 1.0, 0.5, 0.06),    # Northern hemisphere cyclone
        (1.8, 3.5, -0.4, 0.05),   # Southern hemisphere anticyclone
        (1.0, 5.2, 0.3, 0.04),    # Smaller northern vortex
        (2.0, 0.8, -0.3, 0.04),   # Southern vortex
        (1.5, 2.5, 0.2, 0.03),    # Equatorial disturbance
    ]
    for t0, p0, amp, width in vortices:
        weather += amp * torch.exp(-((T - t0) ** 2 + (P - p0) ** 2) / width)

    return weather


def spectral_timestep(coeffs, topo_forcing_coeffs, step, sht, isht):
    """One timestep of spectral weather evolution.

    - Spectral advection via phase rotation (simplified barotropic model)
    - Hyperdiffusion for stability
    - Orographic forcing
    - Nonlinear self-interaction approximated by squaring + re-projecting
    """
    lmax = coeffs.shape[-2]
    mmax = coeffs.shape[-1]

    # Degree/order indices for spectral operations
    l_idx = torch.arange(lmax, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    m_idx = torch.arange(mmax, device=DEVICE, dtype=torch.float32).unsqueeze(0)

    # Phase rotation: westward propagation scaled by zonal wavenumber
    # Higher m modes propagate faster (Rossby-wave-like dispersion)
    omega = -0.3 * m_idx / (l_idx + 1).clamp(min=1)
    # Add some l-dependent frequency for meridional structure
    omega += 0.05 * torch.sin(l_idx * 0.1)

    phase = torch.polar(
        torch.ones_like(omega),
        omega * DT
    )
    coeffs = coeffs * phase.unsqueeze(0)

    # Hyperdiffusion: damp high-wavenumber modes
    damping = torch.exp(-DIFFUSION * (l_idx * (l_idx + 1)) ** 2 * DT)
    coeffs = coeffs * damping.unsqueeze(0)

    # Orographic forcing injection
    coeffs = coeffs + topo_forcing_coeffs * DT * 0.3

    # Nonlinear cascade: go to grid space, apply weak nonlinearity, back to spectral
    if step % 3 == 0:
        field = isht(coeffs).squeeze(0)
        # Gentle nonlinear self-interaction
        nl = 0.1 * field * torch.tanh(field * 2)
        nl_coeffs = sht(nl.unsqueeze(0))
        # Only inject low/mid wavenumber energy
        cascade_cutoff = lmax // 2
        mask = torch.zeros(1, lmax, mmax, device=DEVICE)
        mask[:, :cascade_cutoff, :cascade_cutoff] = 1.0
        coeffs = coeffs + nl_coeffs * mask * DT * 0.5

    return coeffs


def project_sphere(azimuth, elevation=20, buf_size=512):
    """Compute the orthographic projection mapping (screen -> sphere).

    Returns a dict of pre-computed arrays reusable for both panels.
    """
    y_screen, x_screen = np.mgrid[-1.1:1.1:buf_size*1j, -1.1:1.1:buf_size*1j]
    r2 = x_screen**2 + y_screen**2
    on_sphere = r2 <= 1.0
    z_screen = np.sqrt(np.maximum(1.0 - r2, 0))

    az_rad = np.radians(azimuth)
    el_rad = np.radians(elevation)

    # Undo elevation rotation
    y1 = y_screen * np.cos(el_rad) + z_screen * np.sin(el_rad)
    z1 = -y_screen * np.sin(el_rad) + z_screen * np.cos(el_rad)

    # Undo azimuth rotation
    x1 = x_screen * np.cos(az_rad) - y1 * np.sin(az_rad)
    y2 = x_screen * np.sin(az_rad) + y1 * np.cos(az_rad)

    r = np.sqrt(x1**2 + y2**2 + z1**2)
    theta = np.arccos(np.clip(z1 / (r + 1e-10), -1, 1))
    phi = np.arctan2(y2, x1) % (2 * np.pi)

    # Diffuse shading (light from upper-right)
    light_dir = np.array([0.4, 0.5, 0.75])
    light_dir /= np.linalg.norm(light_dir)
    normal_dot = x_screen * light_dir[0] + y_screen * light_dir[1] + z_screen * light_dir[2]
    shade = np.clip(0.3 + 0.7 * normal_dot, 0.1, 1.0)

    # Limb atmosphere
    limb = np.sqrt(r2)
    atm_alpha = np.clip((limb - 0.85) / 0.15, 0, 1) * on_sphere

    # Edge anti-aliasing
    edge = np.clip((1.0 - limb) * 30, 0, 1)

    return dict(
        x_screen=x_screen, y_screen=y_screen, z_screen=z_screen,
        r2=r2, on_sphere=on_sphere, theta=theta, phi=phi,
        shade=shade, atm_alpha=atm_alpha, edge=edge,
    )


def smooth_2d(arr, passes=6, k=9):
    """Smooth a 2D array with repeated 1D uniform filters (shape-preserving)."""
    from scipy.ndimage import uniform_filter1d
    result = arr.astype(np.float64)
    for _ in range(passes):
        uniform_filter1d(result, k, axis=1, mode="wrap", output=result)
        uniform_filter1d(result, k, axis=0, mode="reflect", output=result)
    return result.astype(np.float32)


def bake_topo_contours(topo_np, n_levels=5, supersample=4):
    """Pre-compute anti-aliased contour mask at high resolution.

    Computes contours on a supersampled grid, then downsamples with
    averaging to produce smooth, thin, anti-aliased contour lines.
    Returns a float array (nlat, nlon) in [0, 1].
    """
    from scipy.ndimage import zoom

    # Upsample topo, smooth heavily, then detect edges at high res
    hi = zoom(topo_np, supersample, order=3)
    hi_smooth = smooth_2d(hi, passes=8, k=15)

    interval = (hi_smooth.max() - hi_smooth.min()) / n_levels
    if interval < 1e-8:
        return np.zeros_like(topo_np)

    quantised = np.floor(hi_smooth / interval)

    hi_contour = np.zeros_like(hi_smooth)
    hi_contour[:-1, :] = np.maximum(hi_contour[:-1, :], (quantised[:-1, :] != quantised[1:, :]).astype(np.float32))
    hi_contour[:, :-1] = np.maximum(hi_contour[:, :-1], (quantised[:, :-1] != quantised[:, 1:]).astype(np.float32))
    hi_contour[:, -1] = np.maximum(hi_contour[:, -1], (quantised[:, -1] != quantised[:, 0]).astype(np.float32))

    # Downsample by averaging — produces anti-aliased soft edges
    nlat, nlon = topo_np.shape
    contour = hi_contour.reshape(nlat, supersample, nlon, supersample).mean(axis=(1, 3))

    return contour.astype(np.float32)


def rasterise_terrain(topo_np, weather_np, contour_np, proj, buf_size=512):
    """Rasterise the terrain + weather shaded globe."""
    buf = np.zeros((buf_size, buf_size, 4), dtype=np.float32)

    on_sphere = proj["on_sphere"]
    theta, phi = proj["theta"], proj["phi"]
    shade, atm_alpha, edge = proj["shade"], proj["atm_alpha"], proj["edge"]

    nlat, nlon = topo_np.shape
    lat_idx = np.clip((theta / np.pi * (nlat - 1)).astype(np.int32), 0, nlat - 1)
    lon_idx = (phi / (2 * np.pi) * nlon).astype(np.int32) % nlon

    topo_sampled = topo_np[lat_idx, lon_idx]
    weather_sampled = weather_np[lat_idx, lon_idx]

    # Terrain colours
    topo_min, topo_max = topo_np.min(), topo_np.max()
    topo_norm = (topo_sampled - topo_min) / (topo_max - topo_min + 1e-8)
    terrain_rgba = plt.cm.YlOrBr(topo_norm)

    # Weather overlay
    w_max = max(abs(weather_np.min()), abs(weather_np.max()), 0.3)
    weather_norm = np.clip((weather_sampled + w_max) / (2 * w_max), 0, 1)
    weather_rgba = plt.cm.RdBu_r(weather_norm)

    # Blend terrain + weather
    weather_intensity = np.clip(np.abs(weather_sampled) / (w_max * 0.4), 0, 1)
    alpha = 0.25 + 0.65 * weather_intensity

    blended = terrain_rgba.copy()
    for c in range(3):
        blended[..., c] = (1 - alpha) * terrain_rgba[..., c] + alpha * weather_rgba[..., c]

    # Sample pre-baked contour mask (rotation-stable)
    contours = contour_np[lat_idx, lon_idx] * on_sphere
    contour_darken = 1.0 - 0.3 * contours

    # Atmosphere
    atm_color = np.array([0.6, 0.4, 0.3])

    for c in range(3):
        buf[..., c] = np.where(
            on_sphere,
            blended[..., c] * shade * contour_darken * (1 - atm_alpha * 0.5) + atm_color[c] * atm_alpha * 0.5,
            0,
        )
    buf[..., 3] = np.where(on_sphere, 1.0, 0.0) * edge

    return buf


def rasterise_flow(weather_np, proj, buf_size=512):
    """Rasterise the flow dynamics globe (vorticity only, no topography)."""
    buf = np.zeros((buf_size, buf_size, 4), dtype=np.float32)

    on_sphere = proj["on_sphere"]
    theta, phi = proj["theta"], proj["phi"]
    shade, atm_alpha, edge = proj["shade"], proj["atm_alpha"], proj["edge"]

    nlat, nlon = weather_np.shape
    lat_idx = np.clip((theta / np.pi * (nlat - 1)).astype(np.int32), 0, nlat - 1)
    lon_idx = (phi / (2 * np.pi) * nlon).astype(np.int32) % nlon

    weather_sampled = weather_np[lat_idx, lon_idx]

    # Vorticity with symmetric diverging colourmap on a dark base
    w_max = max(abs(weather_np.min()), abs(weather_np.max()), 0.15)
    vort_norm = np.clip((weather_sampled + w_max) / (2 * w_max), 0, 1)
    vort_rgba = plt.cm.coolwarm(vort_norm)

    # Desaturate slightly for the base so arrows stand out
    grey = 0.3 * vort_rgba[..., 0] + 0.59 * vort_rgba[..., 1] + 0.11 * vort_rgba[..., 2]
    desat = 0.35
    for c in range(3):
        vort_rgba[..., c] = (1 - desat) * vort_rgba[..., c] + desat * grey

    # Atmosphere glow (cool blue)
    atm_color = np.array([0.25, 0.3, 0.45])

    for c in range(3):
        buf[..., c] = np.where(
            on_sphere,
            vort_rgba[..., c] * shade * 0.8 * (1 - atm_alpha * 0.5) + atm_color[c] * atm_alpha * 0.5,
            0,
        )
    buf[..., 3] = np.where(on_sphere, 1.0, 0.0) * edge

    return buf


def forward_project(theta, phi, azimuth, elevation):
    """Project sphere (theta, phi) to screen (x, y). Returns (sx, sy, visible)."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    az = np.radians(azimuth)
    el = np.radians(elevation)

    # Azimuth rotation
    sx = x * np.cos(az) + y * np.sin(az)
    y1 = -x * np.sin(az) + y * np.cos(az)

    # Elevation rotation
    sy = y1 * np.cos(el) - z * np.sin(el)
    depth = y1 * np.sin(el) + z * np.cos(el)

    return sx, sy, depth > 0


def compute_velocity_field(weather_np):
    """Approximate wind field from vorticity via gradient rotation.

    For a barotropic flow, velocity is perpendicular to the vorticity gradient:
      v_theta ∝  (1/sin theta) dζ/dφ   (meridional)
      v_phi   ∝ -dζ/dθ                 (zonal)
    """
    dtheta = np.pi / weather_np.shape[0]
    dphi = 2 * np.pi / weather_np.shape[1]
    nlat = weather_np.shape[0]

    theta_grid = np.linspace(0, np.pi, nlat)
    sin_theta = np.sin(theta_grid)[:, None].clip(min=0.15)

    # dζ/dθ
    dz_dtheta = np.zeros_like(weather_np)
    dz_dtheta[1:-1] = (weather_np[2:] - weather_np[:-2]) / (2 * dtheta)

    # dζ/dφ (periodic)
    dz_dphi = np.zeros_like(weather_np)
    dz_dphi[:, 1:-1] = (weather_np[:, 2:] - weather_np[:, :-2]) / (2 * dphi)
    dz_dphi[:, 0] = (weather_np[:, 1] - weather_np[:, -1]) / (2 * dphi)
    dz_dphi[:, -1] = (weather_np[:, 0] - weather_np[:, -2]) / (2 * dphi)

    v_theta = dz_dphi / sin_theta
    v_phi = -dz_dtheta

    return v_theta, v_phi


class ParticleStreamlines:
    """Particle-based flow visualisation on the sphere.

    Maintains a pool of tracer particles that are advected by the velocity
    field. Each particle accumulates a trail of past positions; the trails
    are rendered as fading polylines projected onto the screen.
    """

    def __init__(self, n=N_PARTICLES, max_age=MAX_AGE, trail_len=TRAIL_LEN):
        self.n = n
        self.max_age = max_age
        self.trail_len = trail_len
        # Current positions (theta, phi)
        self.theta = np.random.uniform(0.05 * np.pi, 0.95 * np.pi, n)
        self.phi = np.random.uniform(0, 2 * np.pi, n)
        # Stagger initial ages so they don't all die at once
        self.age = np.random.randint(0, max_age, n)
        # Trail ring buffer: (n, trail_len) for theta and phi
        self.trail_th = np.tile(self.theta[:, None], (1, trail_len))
        self.trail_ph = np.tile(self.phi[:, None], (1, trail_len))
        self.trail_ptr = 0  # write pointer into ring buffer

    def _respawn(self, mask):
        """Respawn particles at random positions."""
        count = mask.sum()
        if count == 0:
            return
        self.theta[mask] = np.random.uniform(0.05 * np.pi, 0.95 * np.pi, count)
        self.phi[mask] = np.random.uniform(0, 2 * np.pi, count)
        self.age[mask] = 0
        # Reset trails for respawned particles
        self.trail_th[mask] = self.theta[mask, None]
        self.trail_ph[mask] = self.phi[mask, None]

    def advect(self, v_theta_np, v_phi_np):
        """Advance all particles one step using bilinear-interpolated velocity."""
        nlat, nlon = v_theta_np.shape

        # Continuous grid indices
        lat_f = self.theta / np.pi * (nlat - 1)
        lon_f = self.phi / (2 * np.pi) * nlon

        lat0 = np.floor(lat_f).astype(int).clip(0, nlat - 2)
        lon0 = np.floor(lon_f).astype(int) % nlon
        lat1 = lat0 + 1
        lon1 = (lon0 + 1) % nlon

        a = (lat_f - lat0).clip(0, 1)
        b = (lon_f - np.floor(lon_f)).clip(0, 1)

        # Bilinear interpolation for both velocity components
        for field, target in [(v_theta_np, "vt"), (v_phi_np, "vp")]:
            v = ((1 - a) * (1 - b) * field[lat0, lon0]
                 + a * (1 - b) * field[lat1, lon0]
                 + (1 - a) * b * field[lat0, lon1]
                 + a * b * field[lat1, lon1])
            if target == "vt":
                vt = v
            else:
                vp = v

        sin_theta = np.sin(self.theta).clip(min=0.15)
        self.theta += vt * ADVECT_DT
        self.phi += (vp / sin_theta) * ADVECT_DT

        # Wrap / clamp
        self.phi %= (2 * np.pi)
        self.theta = np.clip(self.theta, 0.05 * np.pi, 0.95 * np.pi)

        # Store in trail ring buffer
        self.trail_th[:, self.trail_ptr] = self.theta
        self.trail_ph[:, self.trail_ptr] = self.phi
        self.trail_ptr = (self.trail_ptr + 1) % self.trail_len

        # Age and respawn
        self.age += 1
        dead = self.age >= self.max_age
        self._respawn(dead)

    def get_segments(self, azimuth, elevation):
        """Project trails to screen and return (segments, colors) for LineCollection.

        Streamlines are shaded by their position on the sphere (same diffuse
        lighting as the terrain globe) and fade with particle age.
        """
        segments = []
        colors = []

        order = [(self.trail_ptr + i) % self.trail_len for i in range(self.trail_len)]

        all_th = self.trail_th[:, order]
        all_ph = self.trail_ph[:, order]

        all_sx, all_sy, all_vis = forward_project(
            all_th.ravel(), all_ph.ravel(), azimuth, elevation
        )
        all_sx = all_sx.reshape(self.n, self.trail_len)
        all_sy = all_sy.reshape(self.n, self.trail_len)
        all_vis = all_vis.reshape(self.n, self.trail_len)

        all_r = np.sqrt(all_sx**2 + all_sy**2)
        all_vis = all_vis & (all_r < 0.97)

        # Diffuse shading per trail point (same light direction as globe)
        all_z = np.sqrt(np.maximum(1.0 - all_sx**2 - all_sy**2, 0))
        light = np.array([0.4, 0.5, 0.75])
        light /= np.linalg.norm(light)
        all_shade = np.clip(0.35 + 0.65 * (all_sx * light[0] + all_sy * light[1] + all_z * light[2]), 0.15, 1.0)

        # Age-based fade
        age_frac = self.age / self.max_age
        peak_alpha = 0.7
        alpha_env = peak_alpha * np.where(
            age_frac < 0.12, age_frac / 0.12,
            np.where(age_frac > 0.8, (1 - age_frac) / 0.2, 1.0)
        )

        for p in range(self.n):
            if alpha_env[p] < 0.03:
                continue

            sx_p = all_sx[p]
            sy_p = all_sy[p]
            vis_p = all_vis[p]
            shade_p = all_shade[p]

            pts = []
            mean_shade = 0.0
            for t in range(self.trail_len):
                if vis_p[t]:
                    pts.append((sx_p[t], sy_p[t]))
                    mean_shade += shade_p[t]
                else:
                    if len(pts) >= 2:
                        s = mean_shade / len(pts)
                        segments.append(pts)
                        colors.append((s, s, s, alpha_env[p]))
                    pts = []
                    mean_shade = 0.0
            if len(pts) >= 2:
                s = mean_shade / len(pts)
                segments.append(pts)
                colors.append((s, s, s, alpha_env[p]))

        return segments, colors


def main():
    print("Mars weather animation using HOLYSHT")
    print(f"Grid: {NLAT}x{NLON}, frames: {N_FRAMES}, fps: {FPS}")

    sht = RealSHT(NLAT, NLON, grid="equiangular").to(DEVICE)
    isht = InverseRealSHT(NLAT, NLON, grid="equiangular").to(DEVICE)

    # Build terrain
    print("Generating Mars topography...")
    topo = make_mars_topography()
    topo_np = topo.cpu().numpy()

    # Pre-bake contour lines on the equirectangular grid (rotation-stable)
    print("Baking topographic contours...")
    contour_np = bake_topo_contours(topo_np, n_levels=5)

    # Orographic forcing
    print("Computing orographic forcing...")
    topo_forcing = compute_topo_forcing(topo, sht, isht)
    topo_forcing_coeffs = sht(topo_forcing.unsqueeze(0))

    # Initial weather state
    print("Setting up initial weather state...")
    weather = make_initial_weather()
    coeffs = sht(weather.unsqueeze(0))

    # Pre-compute all frames (spectral evolution on GPU, then transfer)
    print("Running spectral simulation...")
    t0 = time.perf_counter()

    frames_weather = []
    frames_velocity = []
    smooth_vt = None
    smooth_vp = None
    smooth_alpha = 0.12  # low alpha = heavy smoothing

    for i in range(N_FRAMES):
        # Multiple substeps per frame for smoother evolution
        for _ in range(2):
            coeffs = spectral_timestep(coeffs, topo_forcing_coeffs, i, sht, isht)

        field = isht(coeffs).squeeze(0)
        field_np = field.cpu().numpy()
        frames_weather.append(field_np)

        # Compute approximate velocity and time-smooth it
        vt, vp = compute_velocity_field(field_np)
        if smooth_vt is None:
            smooth_vt = vt.copy()
            smooth_vp = vp.copy()
        else:
            smooth_vt = (1 - smooth_alpha) * smooth_vt + smooth_alpha * vt
            smooth_vp = (1 - smooth_alpha) * smooth_vp + smooth_alpha * vp
        frames_velocity.append((smooth_vt.copy(), smooth_vp.copy()))

        if (i + 1) % 30 == 0:
            print(f"  Frame {i+1}/{N_FRAMES}")

    torch.cuda.synchronize()
    sim_time = time.perf_counter() - t0
    print(f"Simulation complete: {sim_time:.2f}s ({N_FRAMES / sim_time:.0f} frames/s)")

    # Render animation — two-panel layout
    print("Rendering frames...")
    t0 = time.perf_counter()

    ELEVATION = 65  # near-polar view, looking down at a slight angle

    fig = plt.figure(figsize=(16, 8), facecolor="black")

    # Left panel: terrain + weather
    ax_left = fig.add_axes([0.0, 0.0, 0.5, 1.0])
    ax_left.set_facecolor("black")
    ax_left.set_xlim(-1.15, 1.15)
    ax_left.set_ylim(-1.15, 1.15)
    ax_left.set_aspect("equal")
    ax_left.axis("off")

    # Right panel: flow dynamics
    ax_right = fig.add_axes([0.5, 0.0, 0.5, 1.0])
    ax_right.set_facecolor("black")
    ax_right.set_xlim(-1.15, 1.15)
    ax_right.set_ylim(-1.15, 1.15)
    ax_right.set_aspect("equal")
    ax_right.axis("off")

    # Panel labels (small, below each globe)
    ax_left.text(
        0.5, 0.06, "Terrain + Weather",
        transform=ax_left.transAxes, ha="center", va="bottom",
        color="#999999", fontsize=9, fontfamily="monospace",
    )
    ax_right.text(
        0.5, 0.06, "Flow Dynamics",
        transform=ax_right.transAxes, ha="center", va="bottom",
        color="#999999", fontsize=9, fontfamily="monospace",
    )

    # Title — upper-left corner (in figure coords so it doesn't overlap the globe)
    fig.text(
        0.02, 0.96, "HOLYSHT",
        ha="left", va="top",
        color="white", fontsize=18, fontweight="bold",
        fontfamily="monospace",
    )
    fig.text(
        0.02, 0.915, "Martian Weather Simulation",
        ha="left", va="top",
        color="#cccccc", fontsize=11, fontfamily="monospace",
    )
    subtitle_text = fig.text(
        0.02, 0.88, f"Spectral advection on {NLAT}\u00d7{NLON} grid",
        ha="left", va="top",
        color="#888888", fontsize=9, fontfamily="monospace",
    )

    # Attribution — lower-right corner
    fig.text(
        0.98, 0.04, "github.com/chrisvoncsefalvay/holysht",
        ha="right", va="bottom",
        color="#666666", fontsize=8, fontfamily="monospace",
    )
    fig.text(
        0.98, 0.015, "Topography: NASA/MOLA (Mars Orbiter Laser Altimeter)",
        ha="right", va="bottom",
        color="#555555", fontsize=7, fontfamily="monospace",
    )

    # Initialise image objects for both panels
    buf_size = 512
    buf_blank = np.zeros((buf_size, buf_size, 4))
    im_left = ax_left.imshow(buf_blank, extent=[-1.15, 1.15, -1.15, 1.15],
                             origin="lower", interpolation="bilinear", zorder=0)
    im_right = ax_right.imshow(buf_blank.copy(), extent=[-1.15, 1.15, -1.15, 1.15],
                               origin="lower", interpolation="bilinear", zorder=0)

    az0 = 200

    # Particle streamline system
    particles = ParticleStreamlines()
    lc = LineCollection([], linewidths=0.8, zorder=5)
    ax_right.add_collection(lc)
    lc_state = [lc]  # mutable ref

    def update(frame_idx):
        azimuth = az0 + frame_idx * (360.0 / N_FRAMES)

        # Shared projection for both views
        proj = project_sphere(azimuth, elevation=ELEVATION, buf_size=buf_size)

        # Left: terrain + weather
        buf_terrain = rasterise_terrain(topo_np, frames_weather[frame_idx], contour_np, proj, buf_size)
        im_left.set_data(buf_terrain)

        # Right: flow dynamics (vorticity, no topo)
        buf_flow = rasterise_flow(frames_weather[frame_idx], proj, buf_size)
        im_right.set_data(buf_flow)

        # Advect particles and draw streamline trails
        vt, vp = frames_velocity[frame_idx]
        particles.advect(vt, vp)
        segments, colors = particles.get_segments(azimuth, ELEVATION)

        lc_state[0].remove()
        new_lc = LineCollection(segments, colors=colors, linewidths=0.8, zorder=5)
        ax_right.add_collection(new_lc)
        lc_state[0] = new_lc

        # Update timestamp
        t_sim = frame_idx / FPS
        subtitle_text.set_text(
            f"Spectral advection on {NLAT}\u00d7{NLON} grid  |  t = {t_sim:.1f}s"
        )

    print(f"Encoding {N_FRAMES} frames...")
    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, blit=False, interval=1000//FPS)

    mp4_path = os.path.join(OUT_DIR, "mars_weather.mp4")
    gif_path = os.path.join(OUT_DIR, "mars_weather.gif")

    # Save mp4 (opaque, black background)
    try:
        writer = animation.FFMpegWriter(fps=FPS, bitrate=6000,
                                        extra_args=["-pix_fmt", "yuv420p"])
        anim.save(mp4_path, writer=writer)
        print(f"Saved: {mp4_path}")
    except Exception as e:
        print(f"mp4 failed: {e}")

    # Save loopable gif: crossfade last XFADE frames with the first XFADE
    plt.close(fig)
    print("Building loopable gif with crossfade...")
    try:
        import io

        XFADE = int(FPS * 1.0)  # 1 second crossfade

        # Re-render all frames to RGBA buffers
        fig2 = plt.figure(figsize=(16, 8), facecolor=(0, 0, 0, 0))
        fig2.patch.set_alpha(0.0)
        ax_l2 = fig2.add_axes([0.0, 0.0, 0.5, 1.0])
        ax_r2 = fig2.add_axes([0.5, 0.0, 0.5, 1.0])
        for a in [ax_l2, ax_r2]:
            a.set_facecolor((0, 0, 0, 0))
            a.patch.set_alpha(0.0)
            a.set_xlim(-1.15, 1.15); a.set_ylim(-1.15, 1.15)
            a.set_aspect("equal"); a.axis("off")

        im_l2 = ax_l2.imshow(np.zeros((buf_size, buf_size, 4)),
                              extent=[-1.15, 1.15, -1.15, 1.15],
                              origin="lower", interpolation="bilinear")
        im_r2 = ax_r2.imshow(np.zeros((buf_size, buf_size, 4)),
                              extent=[-1.15, 1.15, -1.15, 1.15],
                              origin="lower", interpolation="bilinear")
        lc2 = LineCollection([], linewidths=0.8, zorder=5)
        ax_r2.add_collection(lc2)

        # Recreate text overlays
        fig2.text(0.02, 0.96, "HOLYSHT", ha="left", va="top",
                  color="white", fontsize=18, fontweight="bold", fontfamily="monospace")
        fig2.text(0.02, 0.915, "Martian Weather Simulation", ha="left", va="top",
                  color="#cccccc", fontsize=11, fontfamily="monospace")
        sub2 = fig2.text(0.02, 0.88, "", ha="left", va="top",
                         color="#888888", fontsize=9, fontfamily="monospace")
        ax_l2.text(0.5, 0.06, "Terrain + Weather", transform=ax_l2.transAxes,
                   ha="center", va="bottom", color="#999999", fontsize=9, fontfamily="monospace")
        ax_r2.text(0.5, 0.06, "Flow Dynamics", transform=ax_r2.transAxes,
                   ha="center", va="bottom", color="#999999", fontsize=9, fontfamily="monospace")
        fig2.text(0.98, 0.04, "github.com/chrisvoncsefalvay/holysht",
                  ha="right", va="bottom", color="#666666", fontsize=8, fontfamily="monospace")
        fig2.text(0.98, 0.015, "Topography: NASA/MOLA (Mars Orbiter Laser Altimeter)",
                  ha="right", va="bottom", color="#555555", fontsize=7, fontfamily="monospace")

        particles2 = ParticleStreamlines()
        pil_frames = []

        for fi in range(N_FRAMES):
            azimuth = az0 + fi * (360.0 / N_FRAMES)
            proj = project_sphere(azimuth, elevation=ELEVATION, buf_size=buf_size)
            im_l2.set_data(rasterise_terrain(topo_np, frames_weather[fi], contour_np, proj, buf_size))
            im_r2.set_data(rasterise_flow(frames_weather[fi], proj, buf_size))

            vt, vp = frames_velocity[fi]
            particles2.advect(vt, vp)
            segs, cols = particles2.get_segments(azimuth, ELEVATION)
            lc2.remove()
            lc2 = LineCollection(segs, colors=cols, linewidths=0.8, zorder=5)
            ax_r2.add_collection(lc2)

            sub2.set_text(f"Spectral advection on {NLAT}\u00d7{NLON} grid  |  t = {fi/FPS:.1f}s")

            buf_io = io.BytesIO()
            fig2.savefig(buf_io, format="rgba", dpi=100, transparent=True)
            buf_io.seek(0)
            w_px = int(fig2.get_figwidth() * 100)
            h_px = int(fig2.get_figheight() * 100)
            frame_img = Image.frombytes("RGBA", (w_px, h_px), buf_io.read())
            pil_frames.append(frame_img)

            if (fi + 1) % 60 == 0:
                print(f"  GIF frame {fi+1}/{N_FRAMES}")

        plt.close(fig2)

        # Crossfade: blend last XFADE frames with first XFADE frames
        for i in range(XFADE):
            alpha = i / XFADE  # 0 at start of fade region, 1 at end
            tail_idx = N_FRAMES - XFADE + i
            blended = Image.blend(pil_frames[tail_idx], pil_frames[i], alpha)
            pil_frames[tail_idx] = blended

        # Trim the overlapping head frames so the loop is clean
        loop_frames = pil_frames[:N_FRAMES - XFADE] + pil_frames[N_FRAMES - XFADE:]
        # Actually keep all frames but the crossfade makes tail≈head
        loop_frames = pil_frames[:-XFADE]  # drop last XFADE (they're blended into the tail already... wait)
        # The blended frames are at tail positions. For a clean loop:
        # frames 0..N-XFADE-1 are untouched, frames N-XFADE..N-1 are blended toward frame 0..XFADE-1
        # We keep all N frames. When the gif loops, frame N-1 (≈frame XFADE-1) flows into frame 0.
        loop_frames = pil_frames
        gif_frames = [rgba_to_transparent_gif_frame(frame) for frame in loop_frames]

        gif_frames[0].save(
            gif_path, save_all=True, append_images=gif_frames[1:],
            duration=1000 // FPS, loop=0, transparency=0, disposal=2,
            optimize=False,
        )
        print(f"Saved: {gif_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"gif failed: {e}")

    render_time = time.perf_counter() - t0
    print(f"Rendering complete: {render_time:.1f}s")
    print(f"Total: simulation {sim_time:.1f}s + rendering {render_time:.1f}s")


if __name__ == "__main__":
    main()
