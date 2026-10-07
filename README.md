# Dirichlet Splatting: Differentiable Rendering for Wave-Based Inverse Problems

[Xingyu Chen](https://xingyuchen.me/), [Wuqiong Zhao](https://wqzhao.org/),
[Xinyu Zhang](https://xyzhang.ucsd.edu/), [Tzu-Mao Li](https://cseweb.ucsd.edu/~tzli/)

SIGGRAPH Asia 2026 · ACM Transactions on Graphics 45(6)

[Project page](https://dirichlet.xingyuchen.me/) ·
[Paper](https://dirichlet.xingyuchen.me/static/pdf/Dirichlet_Splatting.pdf) ·
[arXiv](https://arxiv.org/abs/2610.00618) ·
[ACM DL](https://doi.org/10.1145/3842559) · [中文说明](README.zh-CN.md)

![A bunny scanned by a coherent sensor, its measured spectrum, Fourier-space slices, Dirichlet kernels of three surfels, and the reconstructed bunny made of oriented surfel splats.](https://dirichlet.xingyuchen.me/static/images/teaser-1200.webp)

Coherent wave sensors such as radar, sonar and terahertz scanners record a scene
in a band-limited Fourier space, where a point reflector does not appear as a
Gaussian blob but as a complex, oscillating Dirichlet kernel. Dirichlet Splatting
uses that kernel, in closed form, as the splat: each primitive is a planar surfel
whose complex amplitude follows wave-propagation physics, and the splats add
coherently. Because the Dirichlet sidelobes make the fitting objective rugged,
the method is paired with Dirichlet Sliding Frank-Wolfe (DSFW), which places and
replaces primitives by a residual certificate and refines them with variable
projection and Gauss-Newton steps.

This repository is the official implementation. The library renders Dirichlet
splats with handwritten Slang kernels (forward, JVP, VJP and dictionaries) and
fits them with DSFW, accepting every step only if the full objective decreases.
It is a small set of functions on Torch tensors; experiments live in `examples/`.

## Install

You need an NVIDIA GPU, the CUDA Toolkit with `nvcc`, and a C++ compiler (Visual
Studio 2022 on Windows, GCC on Linux); the Slang kernels compile on first use.
Tested with Windows 11, Python 3.10, PyTorch 2.7.1+cu128, SlangTorch 1.2.1, CUDA
12.9 and an RTX 5080.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[examples,dev]"
```

On Windows outside a developer terminal, `tools/run.ps1` loads the Visual Studio
compiler environment first, e.g. `.\tools\run.ps1 -m pytest -q`.

## Usage

The mesh path is relative to the repository root.

```python
import torch
import torch.nn.functional as F
import trimesh

from dsplat import DSFWConfig, FMCWConfig, FMCWModel, ParameterConfig, SpatialSearchConfig, fit
from dsplat.groundtruth import planar_array, synthesize

# Sensor: one 128 x 128 array at 2.5 mm pitch in the plane z = 0, facing +z.
aperture_positions = planar_array(side=128, pitch=0.0025, axis=(0.0, 0.0, 1.0), device="cuda")

# Target: surfels sampled on a mesh in metres; keep the faces that point at the array.
mesh = trimesh.load("assets/bunny_60mm.ply")  # 60 mm Stanford Bunny around (0, 0, 0.35)
samples, faces = trimesh.sample.sample_surface(mesh=mesh, count=4000, seed=0)
front = mesh.face_normals[faces][:, 2] < 0
surfel_centers = torch.tensor(samples[front], dtype=torch.float32, device="cuda")
surfel_normals = torch.tensor(mesh.face_normals[faces][front], dtype=torch.float32, device="cuda")
surfel_area = mesh.area / 4000  # m^2 per sample
scan = synthesize(  # FMCW beat signal sampled by the ADC, then range FFT
    aperture_positions=aperture_positions,  # [16384, 3] metres
    surfel_centers=surfel_centers,  # [1967, 3] metres
    surfel_normals=surfel_normals,  # [1967, 3] unit
    surfel_amplitudes=torch.full((len(surfel_centers),), surfel_area**0.5, device="cuda"),
    config=FMCWConfig(),  # 120 GHz carrier, 15 GHz bandwidth, 128 ADC samples
)
print(scan.data.shape, scan.data.dtype)  # [16384, 128]: apertures x range bins

# Fit as many surfels from random positions and normals inside the mesh's bounding box.
count = len(surfel_centers)
box = torch.tensor(mesh.bounds, dtype=torch.float32, device="cuda")  # [2, 3] corners
low, high = box[0] - 0.005, box[1] + 0.005  # 5 mm margin
initial = dict(
    centers=low + (high - low) * torch.rand(count, 3, device="cuda"),
    normals=F.normalize(torch.randn(count, 3, device="cuda"), dim=1),
    areas=torch.full((count,), surfel_area, device="cuda"),
)
parameters = dict(
    centers=ParameterConfig(scale=5e-4, lower=low.tolist(), upper=high.tolist()),
    normals=ParameterConfig(scale=0.1, normalize=True),
    areas=ParameterConfig(trainable=False),
)
state = fit(
    model=FMCWModel(scan=scan),
    target=scan.data,
    params=initial,
    config=DSFWConfig(iterations=100, batch=256),
    search=SpatialSearchConfig(),
    parameters=parameters,
)
error = torch.cdist(state.params["centers"], surfel_centers).min(1).values
print(state.history[-1]["nmse"], error.median())  # NMSE, median distance to the mesh in m
```

The mesh can be any surface in metres in front of the array. `synthesize` follows
the FMCW signal chain: the beat signal of every aperture from the reference surfels and returns a `Scan`:
range profiles `data [apertures, range bins]`, the aperture positions and the
`FMCWConfig`. A measured scan goes into `Scan(data, aperture_positions, config)` the
same way. `fit` is `initialize` followed by `state = step(state, config, search)`;
every step returns a new state, whose `params` hold the fitted surfel centers,
normals and areas and whose `coefficients` are their reflectivities.
`ParameterConfig` sets the constraints: bounds, `trainable=False` to freeze a field,
`normalize=True` for unit normals. `SpatialSearchConfig` makes every step search for
new surfels inside the center bounds. Spectral data use `SpectralModel` in the same
way (see `examples/signal2d.py`).

## Examples

```bash
python -m examples.signal2d --preset paper
python -m examples.raw2d --preset paper --scene letter_A
python -m examples.bunny --preset paper
```

- `signal2d`: 80 tones along a star (or `--scene square|letter_A|random`) in a 64²
  spectrum, recovered with complex coefficients.
- `raw2d`: a planar letter reconstructed from independently synthesized ADC data
  of a 96² aperture, starting from FFT migration.
- `bunny`: 6000 surfels fitted to three orthogonal 128² views of the Stanford Bunny,
  starting from random positions and normals.

The raw2d and bunny scans are the FMCW beat signal synthesized as ADC samples in
double precision, independently of the fitted Dirichlet kernel. Each script separates
`measure`, which synthesizes the measurement and prints its dimensions, from the
reconstruction, so a measured scan can take its place. Presets are `smoke`, `demo` (default) and `paper`.
Results go to `outputs/<experiment>_<preset>_seed<seed>/`: configuration, history,
metrics, parameters, a point cloud and figures.

## Code

```text
dsplat/
  dsfw.py         DSFWConfig, initialize, step, fit
  varpro.py       streamed normal equations, coefficient solve, acceptance
  certificate.py  normalized residual correlations, peaks, continuous refinement
  replacement.py  utility (one slot) and merged (pool, fit once, prune) replacement
  refinement.py   dense reduced-VarPro LM and matrix-free joint Gauss-Newton
  spatial.py      3D candidate search
  parameters.py   ParameterConfig
  spectral.py     Dirichlet atoms on a spectral grid
  surfel.py       oriented surfels: scattering, rendering, the 13-DOF composition
  fmcw.py         FMCW sensor, scans and the surfel measurement model
  groundtruth.py  planar arrays and FMCW scans synthesized from the beat signal
  kernels/        Slang kernels and their PyTorch bindings
examples/         the three experiments, scene synthesis and plots
tests/            CUDA tests against float64 references; run python -m pytest -q
```

## Citation

```bibtex
@article{chen2026dirichlet,
  title     = {Dirichlet Splatting: Differentiable Rendering for Wave-Based Inverse Problems},
  author    = {Chen, Xingyu and Zhao, Wuqiong and Zhang, Xinyu and Li, Tzu-Mao},
  journal   = {ACM Transactions on Graphics},
  volume    = {45},
  number    = {6},
  articleno = {199},
  year      = {2026},
  month     = dec,
  doi       = {10.1145/3842559}
}
```

## License

The code is available under the [Dirichlet Splatting Research License](LICENSE)
for non-commercial research, teaching and evaluation. Modifications and free
redistribution are permitted under the same terms; commercial use requires a
separate, prior written license from the relevant copyright holders. This is a
source-available research license, not an OSI-approved open-source license.

`assets/bunny_60mm.ply` is a remeshed
Stanford Bunny (Stanford Computer Graphics Laboratory,
[3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)), scaled to
60 mm; it is not covered by the code license and remains subject to its source's terms.
