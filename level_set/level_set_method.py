"""
Level Set Method — a self-contained Python demonstration.
======================================================================

The level set method (Osher & Sethian, 1988) tracks a moving interface
(a curve in 2-D) *implicitly*, as the zero level set of a higher
dimensional function ``phi(x, y, t)``:

        interface(t) = { (x, y) : phi(x, y, t) = 0 }

By convention ``phi`` is a *signed distance function*: negative inside
the shape, positive outside, zero on the boundary.

Instead of moving the curve directly (which is painful when it develops
corners, splits, or merges), we evolve the whole field ``phi`` with a PDE

        d(phi)/dt + F * |grad(phi)| = 0

where ``F`` is the *speed* of the front in its normal direction. The
zero level set then moves for free, and topology changes (splitting /
merging) are handled automatically — the headline advantage of the
method over explicit "marker particle" front tracking.

This file implements the numerics from scratch (only numpy) and ships
two classic demonstrations:

  1. Motion by mean curvature  (F = kappa): a star smooths into a circle
     and then shrinks away.  Exact result for a circle: R(t) = sqrt(R0^2 - 2t).
  2. Constant normal speed  (F = const): two separate circles expand and
     *merge* into one — a topology change that explicit methods struggle
     with but the level set method takes in stride.

Run it directly to produce figures + an animation in ./figures/:

        python level_set_method.py

Author: demo for the AI_2017 repository.
"""

import os
import numpy as np


# ----------------------------------------------------------------------
# Core numerical building blocks
# ----------------------------------------------------------------------
def signed_distance_circle(X, Y, cx, cy, r):
    """Exact signed distance to a circle of radius ``r`` centred at (cx, cy)."""
    return np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r


def curvature_and_gradient(phi, dx, dy):
    """Mean curvature kappa = div(grad phi / |grad phi|) and |grad phi|.

    Central differences are used because the curvature term is parabolic
    (diffusion-like) and therefore well behaved with a centred stencil.
    Boundaries use periodic wrap (np.roll); keep the front away from the
    domain edge and this is invisible.
    """
    phi_x = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
    phi_y = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2 * dy)
    phi_xx = (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1)) / dx ** 2
    phi_yy = (np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0)) / dy ** 2
    phi_xy = (
        np.roll(np.roll(phi, -1, axis=0), -1, axis=1)
        - np.roll(np.roll(phi, -1, axis=0), 1, axis=1)
        - np.roll(np.roll(phi, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(phi, 1, axis=0), 1, axis=1)
    ) / (4 * dx * dy)

    grad_mag = np.sqrt(phi_x ** 2 + phi_y ** 2)
    num = phi_xx * phi_y ** 2 - 2 * phi_x * phi_y * phi_xy + phi_yy * phi_x ** 2
    den = (phi_x ** 2 + phi_y ** 2) ** 1.5 + 1e-12
    kappa = num / den
    return kappa, grad_mag


def godunov_gradient(phi, dx, dy, F):
    """Upwind (Godunov) approximation of |grad phi| for advective motion.

    For the hyperbolic term ``F * |grad phi|`` the derivative must be taken
    from the *upwind* direction to satisfy the entropy condition (so that
    corners stay sharp and fronts don't cross). ``F`` sets that direction:
    F > 0 (outward) uses one one-sided combination, F < 0 the other.
    """
    Dxm = (phi - np.roll(phi, 1, axis=1)) / dx   # backward difference in x
    Dxp = (np.roll(phi, -1, axis=1) - phi) / dx  # forward  difference in x
    Dym = (phi - np.roll(phi, 1, axis=0)) / dy   # backward difference in y
    Dyp = (np.roll(phi, -1, axis=0) - phi) / dy  # forward  difference in y

    grad_out = np.sqrt(
        np.maximum(Dxm, 0) ** 2 + np.minimum(Dxp, 0) ** 2 +
        np.maximum(Dym, 0) ** 2 + np.minimum(Dyp, 0) ** 2
    )
    grad_in = np.sqrt(
        np.minimum(Dxm, 0) ** 2 + np.maximum(Dxp, 0) ** 2 +
        np.minimum(Dym, 0) ** 2 + np.maximum(Dyp, 0) ** 2
    )
    return np.where(F > 0, grad_out, grad_in)


def reinitialize(phi, dx, dy, iters=8):
    """Nudge ``phi`` back toward a signed distance function (|grad phi| = 1).

    Solves the reinitialization PDE  phi_t + sign(phi0)*(|grad phi| - 1) = 0
    to a near steady state. Keeping |grad phi| ~ 1 stops the field from
    getting too flat or too steep near the interface, which keeps the
    curvature and upwind estimates accurate.
    """
    phi0 = phi.copy()
    s = phi0 / np.sqrt(phi0 ** 2 + dx ** 2)  # smoothed sign, avoids /0
    dt = 0.5 * dx
    for _ in range(iters):
        grad = godunov_gradient(phi, dx, dy, s)
        phi = phi - dt * s * (grad - 1.0)
    return phi


# ----------------------------------------------------------------------
# Demo 1 — motion by mean curvature
# ----------------------------------------------------------------------
def mean_curvature_flow(N=201, k_lobes=5, amp=0.55, base=1.1,
                        steps=2400, reinit_every=25, capture=8):
    """Evolve a star-shaped front under F = kappa (curvature flow).

    Under curvature flow every convex bump smooths out: the star relaxes
    into a circle and the circle then shrinks to a point.  Returns the
    (X, Y) grid and a list of captured phi snapshots.
    """
    L = 2.0
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, y)
    dx = dy = x[1] - x[0]

    # Star: radius modulated by angle. Not an exact signed distance, so we
    # reinitialize once up front to clean it up.
    theta = np.arctan2(Y, X)
    star_r = base + amp * np.cos(k_lobes * theta)
    phi = np.sqrt(X ** 2 + Y ** 2) - star_r
    phi = reinitialize(phi, dx, dy, iters=30)

    # Curvature flow is diffusion-like ⇒ stable time step ~ dx^2.
    dt = 0.20 * dx ** 2

    snapshots, times = [phi.copy()], [0.0]
    grab_at = set(np.linspace(1, steps, capture).astype(int))
    for n in range(1, steps + 1):
        kappa, grad_mag = curvature_and_gradient(phi, dx, dy)
        # Clamp curvature so a few spurious spikes can't blow up the step.
        kappa = np.clip(kappa, -1.0 / dx, 1.0 / dx)
        phi = phi + dt * kappa * grad_mag        # d(phi)/dt = kappa |grad phi|
        if n % reinit_every == 0:
            phi = reinitialize(phi, dx, dy, iters=5)
        if n in grab_at:
            snapshots.append(phi.copy())
            times.append(n * dt)
    return X, Y, snapshots, times


# ----------------------------------------------------------------------
# Demo 2 — constant normal speed, topology change (merging)
# ----------------------------------------------------------------------
def merging_fronts(N=201, steps=90, capture=6):
    """Two circles expand outward at constant speed and merge into one.

    F = +1 everywhere (outward normal motion). The two zero level sets
    grow, touch, and fuse — no special case needed, which is exactly the
    point of the level set method.
    """
    L = 2.0
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, y)
    dx = dy = x[1] - x[0]

    # Union of two circles = pointwise minimum of two signed distances.
    phi_a = signed_distance_circle(X, Y, -0.7, 0.0, 0.45)
    phi_b = signed_distance_circle(X, Y, 0.7, 0.0, 0.45)
    phi = np.minimum(phi_a, phi_b)

    F = 1.0                       # constant outward speed
    dt = 0.4 * dx / abs(F)        # CFL for the advective (hyperbolic) term

    snapshots, times = [phi.copy()], [0.0]
    grab_at = set(np.linspace(1, steps, capture).astype(int))
    for n in range(1, steps + 1):
        grad = godunov_gradient(phi, dx, dy, F)
        phi = phi - dt * F * grad          # d(phi)/dt = -F |grad phi|
        if n % 10 == 0:
            phi = reinitialize(phi, dx, dy, iters=4)
        if n in grab_at:
            snapshots.append(phi.copy())
            times.append(n * dt)
    return X, Y, snapshots, times


# ----------------------------------------------------------------------
# Plotting / output
# ----------------------------------------------------------------------
def _montage(X, Y, snapshots, times, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(snapshots)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, (phi, t) in enumerate(zip(snapshots, times)):
        ax = axes[i]
        ax.contourf(X, Y, phi, levels=30, cmap="RdBu")
        ax.contour(X, Y, phi, levels=[0], colors="k", linewidths=2)
        ax.set_title(f"t = {t:.3f}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(snapshots), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _animate(X, Y, snapshots, title, path, fps=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    frames = []
    for phi in snapshots:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.contourf(X, Y, phi, levels=30, cmap="RdBu")
        ax.contour(X, Y, phi, levels=[0], colors="k", linewidths=2)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(buf.copy())
        plt.close(fig)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"  wrote {path}  ({len(frames)} frames)")


def circle_shrink_check():
    """Numerical vs. analytic radius for a shrinking circle under curvature flow.

    A circle obeys dR/dt = -1/R  =>  R(t) = sqrt(R0^2 - 2t). We evolve a
    single circle and compare the measured zero-crossing radius to theory.
    """
    N = 201
    L = 2.0
    x = np.linspace(-L, L, N)
    X, Y = np.meshgrid(x, x)
    dx = x[1] - x[0]
    R0 = 1.2
    phi = signed_distance_circle(X, Y, 0, 0, R0)
    dt = 0.20 * dx ** 2
    steps = 1500
    r_mid = X[N // 2, :]                    # radius sampled along the mid row
    print("\nShrinking-circle accuracy check (R(t) = sqrt(R0^2 - 2t)):")
    print(f"  {'t':>7} {'R_numeric':>10} {'R_exact':>10} {'abs err':>9}")
    for n in range(1, steps + 1):
        kappa, grad = curvature_and_gradient(phi, dx, dx)
        kappa = np.clip(kappa, -1.0 / dx, 1.0 / dx)
        phi = phi + dt * kappa * grad
        if n % 15 == 0:
            phi = reinitialize(phi, dx, dx, iters=5)
        if n % 300 == 0:
            row = phi[N // 2, N // 2:]      # phi along +x from centre
            r_num = np.interp(0.0, row, r_mid[N // 2:])
            t = n * dt
            r_exact = np.sqrt(max(R0 ** 2 - 2 * t, 0.0))
            print(f"  {t:7.4f} {r_num:10.4f} {r_exact:10.4f} {abs(r_num - r_exact):9.4f}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(here, "figures")
    os.makedirs(figdir, exist_ok=True)

    print("Demo 1: motion by mean curvature (star -> circle -> vanish)")
    X, Y, snaps, times = mean_curvature_flow()
    _montage(X, Y, snaps, times, "Motion by mean curvature  (F = kappa)",
             os.path.join(figdir, "curvature_flow.png"))
    _animate(X, Y, snaps, "Level set: mean curvature flow",
             os.path.join(figdir, "curvature_flow.gif"), fps=6)

    print("Demo 2: constant normal speed, two fronts merging (topology change)")
    X, Y, snaps, times = merging_fronts()
    _montage(X, Y, snaps, times, "Constant normal speed  (F = 1): fronts merge",
             os.path.join(figdir, "merging_fronts.png"))

    circle_shrink_check()
    print("\nDone. See the ./figures/ directory.")


if __name__ == "__main__":
    main()
