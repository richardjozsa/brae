"""Multi-view Q-criterion render: orbit + static top + static corner, one pass.

Computing Q-criterion on the mesh is the expensive step (tens of seconds on
25M cells); drawing is cheap. So this builds the isosurfaces ONCE per timestep
and then renders every camera from them -- three videos for barely more than
the cost of one.

Configured entirely by env so the same file serves both cases:
  CASE PATCH BOX CELL_L_MAX QLEV OPAC LABEL OUT WIN ONLY K0 K1 NFIT
"""
import os, gc, time, json, numpy as np
os.environ["PYVISTA_OFF_SCREEN"] = "true"
import pyvista as pv; pv.OFF_SCREEN = True

CASE   = os.environ["CASE"]
PATCH  = os.environ.get("PATCH", "f16")
BOX    = json.loads(os.environ["BOX"])
CELL_L = float(os.environ["CELL_L_MAX"])
LEVELS = [(float(v), o) for v, o in zip(os.environ["QLEV"].split(","),
          [float(x) for x in os.environ.get("OPAC", "0.14,0.38,1.0").split(",")])]
LABEL  = os.environ.get("LABEL", "")
WIN    = json.loads(os.environ.get("WIN", "[1500,860]"))
NFIT   = int(os.environ.get("NFIT", 10))
AZ0    = float(os.environ.get("AZ0", -125.0))
BODYC  = json.loads(os.environ.get("BODYC", "[0.62,0.64,0.70]"))

# (name, azimuth(frac), elevation(frac)). Static views ignore frac.
# Top view uses 72 deg rather than 90: at 90 the view direction is parallel to
# the world up vector and the camera basis degenerates.
VIEWS = [
    ("orbit",  lambda f: (AZ0 + 360.0 * f, 30.0 + (-20.0 - 30.0) * 0.5 * (1 - np.cos(2 * np.pi * f)))),
    ("top",    lambda f: (AZ0, 72.0)),
    ("corner", lambda f: (AZ0, 22.0)),
]


def view_dir(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def fit_camera(pts, az, el, win, view_angle=30.0, pad=1.06, iters=6):
    """Focal point + distance framing `pts` tightly AND centred (see notes in
    delta_render_v2.py: containment against real points, then correct for the
    fact that a 3-D bbox centre does not project to the image centre)."""
    d = view_dir(az, el); fwd = -d
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(right)
    if n < 1e-6:                       # looking straight down: pick any basis
        right = np.array([1.0, 0.0, 0.0]); n = 1.0
    right /= n
    up = np.cross(right, fwd)
    tv = np.tan(np.radians(view_angle) / 2.0)
    th = tv * (win[0] / win[1])
    c, dist = 0.5 * (pts.min(axis=0) + pts.max(axis=0)), 1.0
    for _ in range(iters):
        v = pts - c; s = v @ d
        dist = float(max((s + np.abs(v @ up) / tv).max(),
                         (s + np.abs(v @ right) / th).max())) * pad
        w = pts - (c + dist * d)
        dep = np.maximum(w @ fwd, 1e-9)
        pu, ph = (w @ up) / dep, (w @ right) / dep
        c = c + (0.5 * (pu.max() + pu.min()) * dist) * up \
              + (0.5 * (ph.max() + ph.min()) * dist) * right
    return c, dist


def build_iso(mesh):
    ip = mesh.cell_data_to_point_data()
    ip["U"] = np.nan_to_num(np.asarray(ip["U"]))
    g = ip.compute_derivative(scalars="U", qcriterion=True, faster=True)
    g["Qc"]   = np.nan_to_num(np.asarray(g["qcriterion"]))
    g["magU"] = np.linalg.norm(np.nan_to_num(np.asarray(g["U"])), axis=1)
    out = []
    for lev, op in LEVELS:
        iso = g.contour(isosurfaces=[lev], scalars="Qc")
        if iso.n_cells > 0:
            iso = iso.smooth(n_iter=12, relaxation_factor=0.1)
        out.append((iso, op))
    del ip, g
    return out


if __name__ == "__main__":
    os.chdir(CASE); open("case.foam", "w").close()
    r = pv.OpenFOAMReader("case.foam"); r.enable_all_patch_arrays()
    for a in list(r.cell_array_names):
        if a != "U":
            r.disable_cell_array(a)
    ts = sorted(r.time_values)
    r.set_active_time_value(ts[-1])
    d = r.read()
    body = d["boundary"][PATCH].extract_surface()
    im0 = d["internalMesh"]
    cc = np.asarray(im0.cell_centers().points)
    L = np.cbrt(np.abs(np.asarray(im0.compute_cell_sizes(
        length=False, area=False, volume=True).cell_data["Volume"])))
    sel = ((cc[:, 0] >= BOX[0]) & (cc[:, 0] <= BOX[1]) &
           (cc[:, 1] >= BOX[2]) & (cc[:, 1] <= BOX[3]) &
           (cc[:, 2] >= BOX[4]) & (cc[:, 2] <= BOX[5]) & (L < CELL_L))
    ids = np.where(sel)[0]
    print("cells kept %d / %d   times %d" % (ids.size, cc.shape[0], len(ts)), flush=True)
    del cc, L, sel; gc.collect()

    only = os.environ.get("ONLY")
    k0 = int(os.environ.get("K0", 0)); k1 = int(os.environ.get("K1", len(ts) - 1))
    idx = [int(x) for x in only.split(",")] if only else list(range(k0, k1 + 1))
    frac = lambda k: k / max(1, len(ts) - 1)

    for nm, _ in VIEWS:
        os.makedirs(f"frames_{nm}", exist_ok=True)

    # camera track per view, from shared isosurfaces at sampled times
    fit_idx = np.unique(np.linspace(idx[0], idx[-1], NFIT).astype(int))
    samp = {nm: {"c": [], "d": []} for nm, _ in VIEWS}
    vals = []
    for j in fit_idx:
        r.set_active_time_value(ts[j])
        pts = [np.asarray(body.points)]
        for iso, _ in build_iso(r.read()["internalMesh"].extract_cells(ids)):
            if iso.n_cells > 0:
                pts.append(np.asarray(iso.points)); vals.append(np.asarray(iso["magU"]))
        P = np.vstack(pts)
        if P.shape[0] > 400_000:
            P = P[::P.shape[0] // 400_000 + 1]
        for nm, angf in VIEWS:
            az, el = angf(frac(j))
            c, dd = fit_camera(P, az, el, WIN)
            samp[nm]["c"].append(c); samp[nm]["d"].append(dd)
        del pts, P; gc.collect()
    allk = np.arange(len(ts))
    track = {}
    for nm, _ in VIEWS:
        C = np.array(samp[nm]["c"]); D = np.array(samp[nm]["d"])
        track[nm] = (np.stack([np.interp(allk, fit_idx, C[:, i]) for i in range(3)], axis=1),
                     np.interp(allk, fit_idx, D))
        print("  %-7s camera dist %.2f-%.2f m" % (nm, D.min(), D.max()), flush=True)
    vv = np.concatenate(vals)
    CLIM = [float(np.percentile(vv, 2)), float(np.percentile(vv, 98))]
    print("auto clim = [%.1f, %.1f] m/s" % tuple(CLIM), flush=True)
    del vals, vv; gc.collect()

    t0 = time.time()
    for n, k in enumerate(idx):
        r.set_active_time_value(ts[k])
        isos = build_iso(r.read()["internalMesh"].extract_cells(ids))
        f = frac(k)
        for nm, angf in VIEWS:                    # Q computed once, drawn 3x
            az, el = angf(f)
            c, dd = track[nm][0][k], track[nm][1][k]
            p = pv.Plotter(off_screen=True, window_size=WIN)
            p.set_background("black")
            p.enable_depth_peeling(number_of_peels=12, occlusion_ratio=0.0)
            p.add_mesh(body, color=tuple(BODYC), specular=0.6, specular_power=20)
            for iso, op in isos:
                if iso.n_cells > 0:
                    p.add_mesh(iso, scalars="magU", cmap="turbo", clim=CLIM,
                               opacity=op, show_scalar_bar=False)
            p.camera_position = [tuple(c + dd * view_dir(az, el)), tuple(c), (0, 0, 1)]
            p.add_text(f"{LABEL}   t={ts[k]:.4f}s", position="lower_edge",
                       font_size=13, color="white")
            p.screenshot(f"frames_{nm}/v{k:04d}.png"); p.close()
            del p
        del isos; gc.collect()
        if n % 5 == 0:
            print("  frame %d/%d  (%.1fs/frame, all views)"
                  % (n + 1, len(idx), (time.time() - t0) / (n + 1)), flush=True)
    print("MULTIVIEW_DONE", flush=True)
