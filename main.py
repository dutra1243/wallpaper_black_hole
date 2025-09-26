from luminet.black_hole import BlackHole
from luminet.viz import make_segments
from luminet.spatial import polar_to_cartesian
from luminet import black_hole_math as bhmath
from luminet.isoradial import Isoradial
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon



if __name__ == "__main__":
    
    # Example with black hole data
    print("\n" + "="*50)
    print("Black Hole Example:")
    
    bh = BlackHole(
        mass=1,
        incl=1.4,           # inclination in radians
        acc=1,              # accretion rate
        outer_edge=60)
    
    # Create a single plot with 90-degree rotation
    fig, ax = plt.subplots(figsize=(16, 10))
    
    rotation = np.pi/2  # 90-degree rotation
    
    # Add the accretion disk edges
    # Inner edge
    x_inner, y_inner = polar_to_cartesian(bh.disk_apparent_inner_edge.angles, 
                                         bh.disk_apparent_inner_edge.impact_parameters, 
                                         rotation=rotation)
    inner_edge_segments = make_segments(x_inner, y_inner)
    
    # Outer edge
    x_outer, y_outer = polar_to_cartesian(bh.disk_apparent_outer_edge.angles, 
                                         bh.disk_apparent_outer_edge.impact_parameters,
                                         rotation=rotation)
    outer_edge_segments = make_segments(x_outer, y_outer)
    
    # Plot disk edges in white
    lc_inner = LineCollection(inner_edge_segments, colors='white', linewidth=3, alpha=0.9, zorder=4)
    lc_outer = LineCollection(outer_edge_segments, colors='white', linewidth=3, alpha=0.9, zorder=4)
    ax.add_collection(lc_inner)
    ax.add_collection(lc_outer)

    # Overlay isoflux lines (direct image, order=0) using tricontour on Cartesian coordinates
    # 1) Gather scattered (x, y, z) points from a DENSE, HIDDEN set of isoradials independent of what's displayed
    x_pts, y_pts, z_pts = [], [], []
    # Use a denser hidden sampling to support many isoflux lines (balanced for performance)
    radii_for_flux = np.linspace(
        bh.disk_inner_edge, bh.disk_outer_edge, max(400, bh.radial_resolution * 3)
    )
    for r in radii_for_flux:
        ir = Isoradial(r, bh.incl, bh.mass, 0, bh.angular_resolution)
        # ensure solved (safe even if Isoradial auto-solves in __init__)
        try:
            ir.calculate()
        except Exception:
            pass
        flux = bhmath.calc_flux_observed(ir.radius, bh.acc, bh.mass, ir.redshift_factors)
        flux = np.asarray(flux, dtype=float) / bh.max_flux
        xi, yi = polar_to_cartesian(ir.angles, ir.impact_parameters, rotation=rotation)
        x_pts.extend(np.asarray(xi, dtype=float))
        y_pts.extend(np.asarray(yi, dtype=float))
        z_pts.extend(flux)

    if len(z_pts):
        x_pts = np.asarray(x_pts)
        y_pts = np.asarray(y_pts)
        z_pts = np.asarray(z_pts)

        # 2) Use fewer log-spaced levels so lines are more spaced (~1%..100%, 35 levels)
        # 2) Add more lines but keep spacing per decade the same by extending range downward
        #    Original was ~35 levels over 2 decades (-2..0) => ~17.5 lines/decade
        #    We'll use 18 lines/decade and extend to -2.5 => ~45 levels, same perceived spacing
        min_exp, max_exp = -2.5, 0.0
        lines_per_decade = 18
        n_levels = int(round((max_exp - min_exp) * lines_per_decade))
        levels = np.logspace(min_exp, max_exp, n_levels)
        contour = ax.tricontour(
            x_pts,
            y_pts,
            z_pts,
            levels=levels,
            colors='white',
            linewidths=0.8,
            alpha=0.35,
            zorder=2,
        )

        # 3) Mask the apparent inner edge (where direct image has no flux) to avoid triangulation artifacts
        #    Build a filled polygon from the inner edge in Cartesian coords
        x_in, y_in = polar_to_cartesian(
            bh.disk_apparent_inner_edge.angles,
            bh.disk_apparent_inner_edge.impact_parameters,
            rotation=rotation,
        )
        inner_poly = Polygon(
            np.column_stack([x_in, y_in]),
            closed=True,
            facecolor='black',
            edgecolor='none',
            zorder=3,
        )
        ax.add_patch(inner_poly)
        print(f"Isoflux: drew {len(contour.levels)} levels; normalized to max flux = {bh.max_flux:.3e}")

        # Compute bounds from isoflux points and disk edges
        xs_all = np.concatenate([x_pts, x_in, x_outer])
        ys_all = np.concatenate([y_pts, y_in, y_outer])
        min_x, max_x = float(np.min(xs_all)), float(np.max(xs_all))
        min_y, max_y = float(np.min(ys_all)), float(np.max(ys_all))
    else:
        # Fallback bounds using edges only
        min_x, max_x = float(min(x_inner.min(), x_outer.min())), float(max(x_inner.max(), x_outer.max()))
        min_y, max_y = float(min(y_inner.min(), y_outer.min())), float(max(y_inner.max(), y_outer.max()))
    
    # Set limits and styling with equal aspect ratio (no distortion)
    # Slight zoom-in towards the center
    range_x = max_x - min_x if max_x != min_x else 1
    range_y = max_y - min_y if max_y != min_y else 1
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    zoom_factor = 0.9  # <1 means zoom in a bit
    half_w = (range_x * zoom_factor) / 2.0
    half_h = (range_y * zoom_factor) / 2.0
    # Proposed zoomed-in limits
    zx0, zx1 = cx - half_w, cx + half_w
    zy0, zy1 = cy - half_h, cy + half_h
    # Guarantee all data is visible: expand if needed
    zx0 = min(zx0, min_x)
    zx1 = max(zx1, max_x)
    zy0 = min(zy0, min_y)
    zy1 = max(zy1, max_y)
    ax.set_xlim(zx0, zx1)
    ax.set_ylim(zy0, zy1)
    
    # CRITICAL: Set equal aspect ratio to prevent distortion
    ax.set_aspect('equal')

    # Full-bleed, no visible axes or labels (wallpaper-style)
    ax.set_facecolor('black')
    ax.axis('off')
    # Remove figure padding so the plot fills the canvas
    try:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    except Exception:
        pass
    
    # Print final statistics
    print(f"\nVisualization complete!")
    try:
        print(f"Plot range: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
    except NameError:
        pass
    
    # Make the figure background black for better contrast
    fig.patch.set_facecolor('black')
    
    plt.show()
