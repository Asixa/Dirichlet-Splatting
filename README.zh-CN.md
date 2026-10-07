# Dirichlet Splatting: Differentiable Rendering for Wave-Based Inverse Problems

[Xingyu Chen](https://xingyuchen.me/)、[Wuqiong Zhao](https://wqzhao.org/)、[Xinyu Zhang](https://xyzhang.ucsd.edu/)、[Tzu-Mao Li](https://cseweb.ucsd.edu/~tzli/)

SIGGRAPH Asia 2026 · ACM Transactions on Graphics 45(6)

[项目主页](https://dirichlet.xingyuchen.me/) · [论文](https://dirichlet.xingyuchen.me/static/pdf/Dirichlet_Splatting.pdf) · [arXiv](https://arxiv.org/abs/2610.00618) · [ACM DL](https://doi.org/10.1145/3842559) · [English](README.md)

![相干传感器扫描兔子、测得的频谱、傅里叶空间切片、三个 surfel 的 Dirichlet 核，以及由有向 surfel splat 组成的重建结果。](https://dirichlet.xingyuchen.me/static/images/teaser-1200.webp)

雷达、声呐和太赫兹扫描仪这类相干波传感器在带限的傅里叶空间里观测场景。点反射体在那里不是高斯斑，而是复数、振荡的 Dirichlet 核。Dirichlet Splatting 直接用这个闭式核作为 splat：每个基元是一个平面 surfel，复振幅由波传播物理决定，各 splat 相干叠加。Dirichlet 旁瓣让拟合目标很崎岖，所以方法配了 Dirichlet Sliding Frank-Wolfe（DSFW）：用残差证书放置和替换基元，再用变量投影和 Gauss–Newton 步精化。

本仓库是官方实现。库用手写的 Slang kernel 渲染 Dirichlet splat（前向、JVP、VJP 和字典），再用 DSFW 拟合，每一步只有完整目标下降才接受。

## 安装

需要 NVIDIA GPU、带 `nvcc` 的 CUDA Toolkit 和 C++ 编译器（Windows 用 Visual Studio 2022，Linux 用 GCC）；Slang kernel 第一次使用时编译。测试环境：Windows 11、Python 3.10、PyTorch 2.7.1+cu128、SlangTorch 1.2.1、CUDA 12.9、RTX 5080。

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[examples,dev]"
```

Windows 上不在开发者终端时，用 `.\tools\run.ps1` 代替 `python`，它会先加载 Visual Studio 编译环境。

## 用法

网格路径相对于仓库根目录。

```python
import torch
import torch.nn.functional as F
import trimesh

from dsplat import DSFWConfig, FMCWConfig, FMCWModel, ParameterConfig, SpatialSearchConfig, fit
from dsplat.groundtruth import planar_array, synthesize

# 传感器：z = 0 平面上一个 128 x 128、间距 2.5 mm、朝 +z 的阵列。
aperture_positions = planar_array(side=128, pitch=0.0025, axis=(0.0, 0.0, 1.0), device="cuda")

# 目标：在以米为单位的网格表面采样 surfel，只保留朝向阵列的面。
mesh = trimesh.load("assets/bunny_60mm.ply")  # 60 mm 的 Stanford Bunny，位于 (0, 0, 0.35) 附近
samples, faces = trimesh.sample.sample_surface(mesh=mesh, count=4000, seed=0)
front = mesh.face_normals[faces][:, 2] < 0
surfel_centers = torch.tensor(samples[front], dtype=torch.float32, device="cuda")
surfel_normals = torch.tensor(mesh.face_normals[faces][front], dtype=torch.float32, device="cuda")
surfel_area = mesh.area / 4000  # 每个采样点的面积，m^2
scan = synthesize(  # 合成 ADC 采样的 FMCW 混频信号，再做距离 FFT
    aperture_positions=aperture_positions,  # [16384, 3]，米
    surfel_centers=surfel_centers,  # [1967, 3]，米
    surfel_normals=surfel_normals,  # [1967, 3]，单位向量
    surfel_amplitudes=torch.full((len(surfel_centers),), surfel_area**0.5, device="cuda"),
    config=FMCWConfig(),  # 120 GHz 载频，15 GHz 带宽，128 个 ADC 采样
)
print(scan.data.shape, scan.data.dtype)  # [16384, 128]：孔径 x 距离 bin

# 在网格包围盒内用随机位置和法线初始化同样数量的 surfel，然后拟合。
count = len(surfel_centers)
box = torch.tensor(mesh.bounds, dtype=torch.float32, device="cuda")  # [2, 3] 两个角点
low, high = box[0] - 0.005, box[1] + 0.005  # 外扩 5 mm
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
print(state.history[-1]["nmse"], error.median())  # NMSE 与到网格的中位距离（米）
```

网格可以换成任何以米为单位、位于阵列前方的表面。`synthesize` 按 FMCW 信号链根据参考 surfel 合成每个孔径的混频信号，返回 `Scan`：距离剖面 `data [孔径, 距离 bin]`、孔径位置和 `FMCWConfig`。实测数据同样用 `Scan(data, aperture_positions, config)` 传入。`fit` 等于 `initialize` 加上反复调用 `state = step(state, config, search)`，每一步返回新的状态：`params` 是拟合出的 surfel 中心、法线和面积，`coefficients` 是对应的反射率。约束由 `ParameterConfig` 设置：上下界、`trainable=False` 冻结某个字段、`normalize=True` 保持单位法线。`SpatialSearchConfig` 让每一步在中心的边界内搜索新的 surfel。频谱数据用 `SpectralModel`，用法相同（见 `examples/signal2d.py`）。

## 实验

```bash
python -m examples.signal2d --preset paper
python -m examples.raw2d --preset paper --scene letter_A
python -m examples.bunny --preset paper
```

- `signal2d`：64² 频谱中沿五角星排布的 80 个音调（`--scene` 可选 square、letter_A、random），用复系数恢复。
- `raw2d`：96² 孔径、独立仿真的 ADC 数据，从 FFT 迁移出发重建平面字母。
- `bunny`：三个正交 128² 视角下的 Stanford Bunny，6000 个 surfel 从随机位置和法线出发拟合。

raw2d 和 bunny 的扫描都是用双精度合成 FMCW 混频后的 ADC 采样得到的，不依赖被拟合的 Dirichlet kernel。每个脚本把生成测量的 `measure`（会打印信号维度）和重建分开，换成实测数据时只需替换 `measure`。预设有 `smoke`、`demo`（默认）和 `paper`，结果写到 `outputs/<实验>_<预设>_seed<种子>/`。

代码结构和 BibTeX 见 [README](README.md)。代码采用自定义的
[Dirichlet Splatting Research License](LICENSE)，仅允许非商业研究、教学和评估；
允许在相同条款下修改和免费再分发，商业使用须事先取得相关版权持有人的单独书面许可。
这是源码可用的研究许可，不是 OSI 认可的开源许可。
`assets/bunny_60mm.ply` 来自 Stanford 3D Scanning Repository，不在代码许可范围内，须遵守其来源条款。
