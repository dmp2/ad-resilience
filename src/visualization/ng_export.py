"""Neuroglancer assets, states, serving and verification on top of ``wsi_pipeline``.

Reused from wsi-tissue-pipeline (not reimplemented):
  * ``wsi_pipeline.omezarr.writers.write_ngff_from_mips``    (OME-Zarr v0.4; native Aperio levels are
    copied with explicit per-level scales, never resampled; see ``write_slide_zarr``)
  * ``wsi_pipeline.precomputed.write_precomputed_raw_volume`` (label volumes)
  * ``wsi_pipeline.neuroglancer.start_cors_server``          (CORS + HTTP Range server)
  * the ``open_registration_neuroglancer_view`` pattern: a state file whose sources are
    rewritten to the running server, loaded into ``neuroglancer.Viewer``.

Added here because wsi_pipeline has no equivalent: precomputed *annotation* layers
(points / polylines, via neuroglancer's own ``AnnotationWriter``), per-layer affine
source transforms that keep each stain or table in its original coordinates, and a
verifier that reads every asset back over HTTP and screenshots the scene with
headless Chrome.

Bundle layout (under one serve root, ``results/visualization_prototypes``)::

    <modality>/neuroglancer/data/...           assets (OME-Zarr, precomputed)
    <modality>/neuroglancer/<scene>.state.json  sources at http://127.0.0.1:8000/...
    <modality>/neuroglancer/<scene>.manifest.json  frames, transforms, provenance
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .scene import PROJECT, Affine2D

DEFAULT_BASE = "http://127.0.0.1:8000"
SERVE_ROOT = PROJECT / "results" / "visualization_prototypes"
WSI_REPO = PROJECT.parent / "wsi-tissue-pipeline"


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------

def wsi_provenance() -> dict[str, Any]:
    import wsi_pipeline

    repo = Path(wsi_pipeline.__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    pin_file = PROJECT / "configs" / "wsi-tissue-pipeline-upstream-commit.txt"
    return {"package": str(Path(wsi_pipeline.__file__).parent), "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": [line for line in git("status", "--porcelain").splitlines() if line.strip()][:20],
            "project_pin": pin_file.read_text().strip() if pin_file.exists() else None}


def neuroglancer_version() -> str:
    from importlib.metadata import version

    return version("neuroglancer")


# --------------------------------------------------------------------------------------
# Asset writers
# --------------------------------------------------------------------------------------

def native_levels(svs: Path, first_level: int = 1) -> dict:
    """Aperio pyramid levels ``first_level``..coarsest with their exact downsample factors.

    Aperio stores level k as floor(baseline / f_k) pixels with an integer f_k (4, 16 here),
    so the physical pixel is f_k x MPP; the baseline/level size ratio is off by up to 0.02%.
    """

    import tifffile

    from qnp_particles import _aperio_header

    with tifffile.TiffFile(svs) as tif:
        series = tif.series[0]
        header = _aperio_header(tif.pages[0].description)
        baseline = series.levels[0].shape[:2]
        mpp = float(header["MPP"])
        out = []
        for k in range(first_level, len(series.levels)):
            shape = series.levels[k].shape[:2]
            factor = round(baseline[0] / shape[0])
            if any(b // factor != s for b, s in zip(baseline, shape)):
                raise ValueError(f"{svs.name} level {k} {shape} is not floor(baseline {baseline} / {factor})")
            out.append({"aperio_level": k, "shape_yx": list(shape), "downsample": factor,
                        "pixel_um": factor * mpp})
    return {"baseline_shape_yx": list(baseline), "mpp_baseline_um": mpp, "levels": out}


def write_slide_zarr(svs: Path, out_dir: Path, level: int = 1, overwrite: bool = False) -> dict:
    """Copy Aperio pyramid levels ``level``..coarsest into an OME-Zarr multiscale, unresampled.

    The SVS already holds a pyramid (here 0.5, 2 and 8 um/px), so no level is generated:
    each native level becomes one dataset with its exact scale (factor x MPP), passed to
    wsi_pipeline's writer as explicit NGFF metadata because its default metadata assumes
    a factor of 2 per level.  Levels are corner-aligned (Aperio level pixel j covers
    baseline pixels [f j, f (j + 1))), which matches OME-Zarr/Neuroglancer's voxel
    convention without a translation.  Level 0 (~7.5 GB RGB) would need a tile-streaming
    copy like ``write_ets_pyramid_to_ngff_zarr`` and is not exported by default.
    """

    import tifffile

    from wsi_pipeline.omezarr.writers import write_ngff_from_mips

    record_path = out_dir / "visualization_source.json"
    if record_path.exists() and not overwrite:
        return json.loads(record_path.read_text())
    if out_dir.exists():
        shutil.rmtree(out_dir)
    pyramid = native_levels(svs, level)
    with tifffile.TiffFile(svs) as tif:
        arrays = [tif.series[0].levels[L["aperio_level"]].asarray()[..., :3] for L in pyramid["levels"]]
    datasets = [{"path": f"s{i}", "coordinateTransformations": [
        {"type": "scale", "scale": [1.0, L["pixel_um"], L["pixel_um"]]}]} for i, L in enumerate(pyramid["levels"])]
    root_attrs = {"multiscales": [{
        "name": svs.stem, "version": "0.4", "datasets": datasets,
        "axes": [{"name": "c^", "type": "channel"}, {"name": "y", "type": "space", "unit": "micrometer"},
                 {"name": "x", "type": "space", "unit": "micrometer"}],
        "metadata": {"description": "native Aperio pyramid levels copied without resampling",
                     "aperio_levels": [L["aperio_level"] for L in pyramid["levels"]]}}]}
    write_ngff_from_mips(arrays, out_dir, name=svs.stem, ngff_metadata=root_attrs, channel_labels=["R", "G", "B"],
                         channel_colors=["FF0000", "00FF00", "0000FF"])
    stat = svs.stat()
    record = {"source_svs": str(svs.relative_to(PROJECT)), "source_bytes": stat.st_size, **pyramid,
              "pyramid": "native Aperio levels copied; no resampling",
              "frame": f"qnp_slide_mm:{svs.stem} (in um)",
              "pixel_convention": "OME-Zarr voxel j spans [j, j+1) x pixel; qnp_slide_mm puts level pixel j at "
                                  "j x pixel and uses the size ratio for the pixel (sub-pixel offset)",
              "writer": "wsi_pipeline.omezarr.writers.write_ngff_from_mips with explicit per-level NGFF scales"}
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def write_annotations(out_dir: Path, kind: str, geometry: Sequence, properties: Mapping[str, tuple[str, np.ndarray]],
                      enums: Mapping[str, Sequence[str]] | None = None) -> dict:
    """Precomputed annotations in an (x, y) um coordinate space.

    ``geometry`` holds (N, 2) points for ``point`` or a list of (K, 2) vertex arrays for
    ``polyline``; ``properties`` maps id -> (numpy dtype name, values per annotation).
    """

    import neuroglancer as ng
    from neuroglancer.write_annotations import AnnotationWriter

    enums = enums or {}
    bad = [pid for pid in properties if not re.fullmatch(r"[a-z][a-zA-Z0-9_]*", pid)]
    if bad:  # Neuroglancer rejects the whole source otherwise (and the layer never loads)
        raise ValueError(f"Annotation property ids must match [a-z][a-zA-Z0-9_]*: {bad}")
    specs = []
    for pid, (dtype, _) in properties.items():
        spec = {"id": pid, "type": dtype}
        if pid in enums:
            spec.update(enum_values=list(range(len(enums[pid]))), enum_labels=list(enums[pid]))
        specs.append(ng.AnnotationPropertySpec(**spec))
    space = ng.CoordinateSpace(names=["x", "y"], units="um", scales=[1.0, 1.0])
    writer = AnnotationWriter(space, kind, properties=specs)
    add = writer.add_point if kind == "point" else writer.add_polyline
    for i, geom in enumerate(geometry):
        values = {pid: (int(v[i]) if dtype.startswith(("uint", "int")) else float(v[i]))
                  for pid, (dtype, v) in properties.items()}
        add(np.asarray(geom, float).tolist(), **values)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    writer.write(out_dir)
    return {"kind": kind, "count": len(geometry), "properties": {k: v[0] for k, v in properties.items()},
            "lower_bound": writer.lower_bound.tolist(), "upper_bound": writer.upper_bound.tolist()}


def write_label_volume(out_dir: Path, labels_yx: np.ndarray, pixel_um: tuple[float, float]) -> dict:
    """2-D label image (uint32, 0 = background) as a precomputed segmentation volume."""

    from wsi_pipeline.precomputed import write_precomputed_raw_volume

    if out_dir.exists():
        shutil.rmtree(out_dir)
    data = np.asarray(labels_yx, dtype=np.uint32)[None, None]
    write_precomputed_raw_volume(out_dir, data, voxel_size_um=(pixel_um[0], pixel_um[1], 1.0),
                                 chunk_size_xyz=(256, 256, 1), layer_type="segmentation")
    return {"shape_yx": list(labels_yx.shape), "pixel_um_xy": list(pixel_um),
            "writer": "wsi_pipeline.precomputed.write_precomputed_raw_volume",
            "voxel_semantics": "voxel (i, j) spans [i, i+1) x pixel in x and [j, j+1) x pixel in y"}


# --------------------------------------------------------------------------------------
# State construction
# --------------------------------------------------------------------------------------

def source_transform(affine_um: Affine2D, input_dims: Sequence[str], input_scales_um: Mapping[str, float]) -> dict:
    """Neuroglancer ``transform`` for a source whose x/y map through ``affine_um``.

    Each output dimension keeps its input dimension's scale, and the physical map is
    conjugated into those units: M = S^-1 A S, translation t / s.  With equal input and
    output scales the matrix means the same thing whether Neuroglancer reads it in
    voxel or in scaled units (an unequal output scale rendered AT8 2x too large and the
    250 um atlas volume off-screen).  Non-spatial dimensions (channels, z) pass through.
    """

    A = affine_um.matrix
    s = {"x": input_scales_um["x"], "y": input_scales_um["y"]}
    out_dims = {}
    rows = []
    for dim in input_dims:
        row = [0.0] * (len(input_dims) + 1)
        if dim in ("x", "y"):
            k = 0 if dim == "x" else 1
            row[input_dims.index("x")] = A[k, 0] * s["x"] / s[dim]
            row[input_dims.index("y")] = A[k, 1] * s["y"] / s[dim]
            row[-1] = A[k, 2] / s[dim]
            out_dims[dim] = [s[dim] * 1e-6, "m"]
        else:
            row[input_dims.index(dim)] = 1.0
            out_dims[dim] = [input_scales_um.get(dim, 1.0) * (1e-6 if dim == "z" else 1.0), "m" if dim == "z" else ""]
        rows.append(row)
    return {"outputDimensions": out_dims, "matrix": rows}


def zarr_dims(zarr_dir: Path) -> tuple[list[str], dict[str, float]]:
    attrs = json.loads((zarr_dir / ".zattrs").read_text())["multiscales"][0]
    names = [a["name"] for a in attrs["axes"]]
    scale = attrs["datasets"][0]["coordinateTransformations"][0]["scale"]
    return names, {n: float(s) for n, s in zip(names, scale)}


def property_id(name: str, prefix: str = "") -> str:
    """A Neuroglancer-valid property id (``[a-z][a-zA-Z0-9_]*``) from a free-text name."""

    return prefix + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def url_for(path: Path, base: str = DEFAULT_BASE) -> str:
    return f"{base.rstrip('/')}/{path.resolve().relative_to(SERVE_ROOT.resolve()).as_posix()}"


def image_layer(name: str, zarr_dir: Path, transform: Affine2D | None = None, *, shader: str | None = None,
                visible: bool = True, opacity: float = 1.0) -> dict:
    source: dict[str, Any] = {"url": f"zarr://{url_for(zarr_dir)}"}
    if transform is not None:
        dims, scales = zarr_dims(zarr_dir)
        source["transform"] = source_transform(transform, dims, scales)
    return {"type": "image", "name": name, "source": source, "visible": visible, "opacity": opacity,
            "shader": shader or RGB_SHADER, "blend": "default"}


def annotation_layer(name: str, ann_dir: Path, shader: str, transform: Affine2D | None = None, *,
                     visible: bool = True) -> dict:
    source: dict[str, Any] = {"url": f"precomputed://{url_for(ann_dir)}"}
    if transform is not None:
        source["transform"] = source_transform(transform, ["x", "y"], {"x": 1.0, "y": 1.0})
    return {"type": "annotation", "name": name, "source": source, "shader": shader, "visible": visible}


def segmentation_layer(name: str, vol_dir: Path, colors: Mapping[int, str], transform: Affine2D | None = None, *,
                       visible: bool = True, opacity: float = 0.45, shown: Sequence[int] | None = None) -> dict:
    source: dict[str, Any] = {"url": f"precomputed://{url_for(vol_dir)}"}
    info = json.loads((vol_dir / "info").read_text())
    res = [r / 1000.0 for r in info["scales"][0]["resolution"]]
    if transform is not None:
        source["transform"] = source_transform(transform, ["x", "y", "z"], {"x": res[0], "y": res[1], "z": res[2]})
    return {"type": "segmentation", "name": name, "source": source, "visible": visible,
            "segments": [str(k) for k in (colors if shown is None else shown)], "segmentColors": {str(k): v for k, v in colors.items()},
            "selectedAlpha": opacity, "notSelectedAlpha": 0.0, "hideSegmentZero": True}


def build_state(layers: list[dict], position_um: Sequence[float], field_um: float, viewport_px: int = 800,
                extra_dims: Mapping[str, list] | None = None) -> dict:
    """2-D cross-section state in um, centred at ``position_um`` showing ~``field_um`` across."""

    dims = {"x": [1e-6, "m"], "y": [1e-6, "m"], **(extra_dims or {})}
    position = [float(position_um[0]), float(position_um[1])] + [0.5] * len(extra_dims or {})
    return {"dimensions": dims, "position": position, "crossSectionScale": field_um / viewport_px,
            "projectionScale": field_um, "layers": layers, "layout": "xy", "showSlices": False,
            "selectedLayer": {"visible": False}, "showDefaultAnnotations": False}


def write_scene(bundle_dir: Path, scene: str, state: dict, manifest: dict) -> dict[str, Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    state_path = bundle_dir / f"{scene}.state.json"
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    manifest = {**manifest, "state": state_path.name, "default_base_url": DEFAULT_BASE,
                "serve_root": str(SERVE_ROOT.relative_to(PROJECT))}
    manifest_path = bundle_dir / f"{scene}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    url = "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(json.dumps(state, separators=(",", ":")))
    (bundle_dir / f"{scene}.demo_url.txt").write_text(url + "\n")
    return {"state": state_path, "manifest": manifest_path}


def rebase_state(state: dict, base: str) -> dict:
    text = json.dumps(state)
    return json.loads(text.replace(DEFAULT_BASE, base.rstrip("/")))


def iter_sources(state: dict):
    for layer in state["layers"]:
        source = layer["source"]
        yield layer, source["url"] if isinstance(source, dict) else source


# --------------------------------------------------------------------------------------
# Shaders
# --------------------------------------------------------------------------------------

RGB_SHADER = """void main() {
  emitRGB(vec3(toNormalized(getDataValue(0)), toNormalized(getDataValue(1)), toNormalized(getDataValue(2))));
}
"""


def _rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return f"vec3({r:.4f}, {g:.4f}, {b:.4f})"


def categorical_shader(prop: str, colors: Sequence[str], missing: str = "#8b8a85", size: float = 3.0,
                       line_width: float = 0.0, alpha: float = 1.0) -> str:
    branches = "\n".join(f"  if (k == {i}) c = {_rgb(col)};" for i, col in enumerate(colors))
    sizing = (f"  setPointMarkerSize({size:.1f});\n  setPointMarkerBorderWidth(0.0);\n"
              f"  setPointMarkerColor(vec4(c, {alpha:.2f}));\n") if size else ""
    width = f"  setLineWidth({line_width:.1f});\n  setEndpointMarkerSize(0.0);\n" if line_width else ""
    return (f"void main() {{\n  int k = int(prop_{prop}());\n  vec3 c = {_rgb(missing)};\n{branches}\n"
            f"  setColor(c);\n{sizing}{width}}}\n")


def sequential_shader(prop: str, lo: float, hi: float, ramp: Sequence[str], size: float = 3.0) -> str:
    stops = ",\n    ".join(_rgb(c) for c in ramp)
    n = len(ramp)
    return f"""#uicontrol float lo slider(min={lo:g}, max={hi:g}, default={lo:g})
#uicontrol float hi slider(min={lo:g}, max={hi * 2:g}, default={hi:g})
vec3 ramp(float t) {{
  vec3 s[{n}] = vec3[{n}](
    {stops});
  float x = clamp(t, 0.0, 1.0) * float({n - 1});
  int i = int(min(floor(x), float({n - 2})));
  return mix(s[i], s[i + 1], x - float(i));
}}
void main() {{
  vec3 c = ramp((prop_{prop}() - lo) / max(hi - lo, 1e-6));
  setColor(c);
  setPointMarkerColor(vec4(c, 1.0));
  setPointMarkerSize({size:.1f});
  setPointMarkerBorderWidth(0.0);
}}
"""


# --------------------------------------------------------------------------------------
# Serving, verification, screenshots
# --------------------------------------------------------------------------------------

def start_server(port: int = 8000, host: str = "127.0.0.1"):
    from wsi_pipeline.neuroglancer import start_cors_server

    return start_cors_server(SERVE_ROOT, host=host, port=port)


def open_viewer(state: dict, bind: str = "127.0.0.1", port: int = 0):
    import neuroglancer as ng

    ng.set_server_bind_address(bind, port)
    viewer = ng.Viewer()
    viewer.set_state(ng.ViewerState(state))
    return viewer


def screenshot(state: dict, path: Path, size: tuple[int, int] = (1200, 900), timeout_s: float = 240.0) -> dict:
    """Render ``state`` (already pointing at a running server) with a headless Chrome client.

    Uses neuroglancer's own ``Viewer.screenshot``, which waits until the visible data
    have loaded; Chrome only provides the WebGL client (SwiftShader, no GPU needed).
    """

    import io
    import threading

    from PIL import Image

    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        return {"ok": False, "reason": "no Chrome/Chromium on PATH"}
    viewer = open_viewer(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [chrome, "--headless=new", "--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
           "--hide-scrollbars", f"--window-size={size[0]},{size[1] + 200}", "--remote-debugging-port=0",
           "--enable-logging=stderr", "--v=0", viewer.get_viewer_url()]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    holder: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: holder.setdefault("reply", viewer.screenshot(size=list(size))), daemon=True)
    started = time.time()
    worker.start()
    worker.join(timeout_s)
    resolved = viewer.state.to_json()
    proc.terminate()
    try:
        stderr = proc.communicate(timeout=20)[1]
    except subprocess.TimeoutExpired:
        proc.kill()
        stderr = proc.communicate()[1]
    errors = [line.split("CONSOLE", 1)[-1][:300] for line in stderr.splitlines()
              if "CONSOLE" in line and ("rror" in line or "ailed" in line)]
    if "reply" not in holder:
        return {"ok": False, "reason": f"no screenshot within {timeout_s} s", "console_errors": errors}
    png = holder["reply"].screenshot.image
    path.write_bytes(png)
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)
    background = pixels[0, 0]
    content = float(np.mean(np.abs(pixels - background).sum(-1) > 30))
    return {"ok": content > 0.01, "path": str(path.relative_to(PROJECT) if path.is_relative_to(PROJECT) else path), "seconds": round(time.time() - started, 1),
            "non_background_fraction": round(content, 4), "size": list(size), "console_errors": errors,
            "resolved_dimensions": resolved.get("dimensions"), "resolved_position": resolved.get("position"),
            "layers": [l["name"] for l in resolved.get("layers", [])]}


def _fetch(url: str, byte_range: tuple[int, int] | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(url)
    if byte_range:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read()


def verify_source(url: str, layer: dict, manifest_layer: Mapping | None) -> dict:
    """Read a layer source back over HTTP and compare with what was written."""

    import tensorstore as ts

    scheme, http = url.split("://", 1)
    http = http if http.startswith("http") else url
    out: dict[str, Any] = {"layer": layer["name"], "url": url}
    if scheme == "zarr":
        attrs = json.loads(_fetch(http.rstrip("/") + "/.zattrs")[1])
        paths = [d["path"] for d in attrs["multiscales"][0]["datasets"]]
        store = ts.open({"driver": "zarr", "kvstore": {"driver": "http", "base_url": http.rstrip("/") + "/" + paths[-1] + "/"}},
                        read=True).result()
        remote = store[...].read().result()
        local_dir = SERVE_ROOT / urllib.parse.urlparse(http).path.lstrip("/") / paths[-1]
        local = ts.open({"driver": "zarr", "kvstore": {"driver": "file", "path": str(local_dir)}}, read=True).result()[...].read().result()
        status, partial = _fetch(http.rstrip("/") + "/" + paths[0] + "/.zarray", (0, 9))
        out.update(levels=len(paths), coarsest_shape=list(remote.shape), coarsest_equal=bool(np.array_equal(remote, local)),
                   range_request_status=status, scale_um=attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"],
                   ok=bool(np.array_equal(remote, local)) and status == 206)
    elif scheme == "precomputed":
        info = json.loads(_fetch(http.rstrip("/") + "/info")[1])
        if info.get("@type") == "neuroglancer_annotations_v1":
            from neuroglancer.read_precomputed_annotations import AnnotationReader

            reader = AnnotationReader(http.rstrip("/"))
            found = reader.get_within_spatial_bounds(lower_bound=info["lower_bound"], upper_bound=info["upper_bound"])
            found = list(found)
            expected = (manifest_layer or {}).get("count")
            out.update(kind=info["annotation_type"], count=len(found), expected=expected,
                       properties=[p["id"] for p in info["properties"]], ok=expected is None or len(found) == expected)
        else:
            store = ts.open({"driver": "neuroglancer_precomputed", "kvstore": {"driver": "http", "base_url": http.rstrip("/") + "/"}},
                            read=True).result()
            remote = store[...].read().result()
            local_dir = SERVE_ROOT / urllib.parse.urlparse(http).path.lstrip("/")
            local = ts.open({"driver": "neuroglancer_precomputed", "kvstore": {"driver": "file", "path": str(local_dir)}},
                            read=True).result()[...].read().result()
            out.update(type=info["type"], shape=list(remote.shape), equal=bool(np.array_equal(remote, local)),
                       labels_present=sorted(int(v) for v in np.unique(remote))[:20], ok=bool(np.array_equal(remote, local)))
    else:
        out.update(ok=False, reason=f"unsupported scheme {scheme}")
    return out


def verify_scene(bundle_dir: Path, scene: str, port: int = 8000, shot: bool = True) -> dict:
    state = json.loads((bundle_dir / f"{scene}.state.json").read_text())
    manifest = json.loads((bundle_dir / f"{scene}.manifest.json").read_text())
    base = f"http://127.0.0.1:{port}"
    state = rebase_state(state, base)
    server = start_server(port)
    try:
        checks = []
        for layer, url in iter_sources(state):
            try:
                checks.append(verify_source(url, layer, manifest.get("layers", {}).get(layer["name"])))
            except Exception as exc:  # report, do not hide
                checks.append({"layer": layer["name"], "url": url, "ok": False, "error": repr(exc)[:400]})
        report: dict[str, Any] = {"scene": scene, "sources": checks, "all_sources_ok": all(c["ok"] for c in checks)}
        if shot:
            report["screenshot"] = screenshot(state, bundle_dir / "screenshots" / f"{scene}.png")
    finally:
        from wsi_pipeline.neuroglancer import stop_cors_server

        stop_cors_server(server)
    report["checked_at_unix"] = int(time.time())
    (bundle_dir / f"{scene}.verify.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def serve(scenes: Sequence[Path], port: int = 8000, host: str = "127.0.0.1", ng_port: int = 0) -> None:
    """Serve every bundle and open one Python-hosted viewer per scene (blocks)."""

    import neuroglancer as ng

    server = start_server(port, host)
    base = f"http://{host}:{port}"
    viewers = []
    ng.set_server_bind_address(host, ng_port)
    for state_path in scenes:
        state = rebase_state(json.loads(Path(state_path).read_text()), base)
        viewer = ng.Viewer()
        viewer.set_state(ng.ViewerState(state))
        viewers.append(viewer)
        print(f"{Path(state_path).name}: {viewer.get_viewer_url()}", flush=True)
    print(f"Assets served from {SERVE_ROOT} at {base}. Ctrl-C to stop.", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        from wsi_pipeline.neuroglancer import stop_cors_server

        stop_cors_server(server)
