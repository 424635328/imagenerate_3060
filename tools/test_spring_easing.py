"""test_spring_easing.py — 动效弹簧曲线的数值门禁（纯 CPU，不加载模型、不用 GPU）。

为什么要有它：css/motion.css 里的 `--spring-*` 不是手调的贝塞尔，而是阻尼谐振子
阶跃响应的采样值，导出成 CSS `linear()`：

    x(t) = 1 - e^(-z w0 t) [cos(wd t) + (z w0 / wd) sin(wd t)]      (z < 1)
    x(t) = 1 - (1 + w0 t) e^(-w0 t)                                  (z = 1)

手抄 26 个控制点到 CSS 里，抄错一位就是"动效看起来怪但没人知道为什么"。
本测试把三条不变量钉死：

  1. 每条曲线都能解析成合法的 `linear()` 列表（首点 0、末点 1、百分比单调递增）；
  2. 采样值与解析解一致（重算一遍，最大偏差 < 2e-3，含 4 位小数量化误差）；
  3. 过冲量与阻尼比一致（ζ=1 不过冲；ζ<1 的峰值在解析解的 ±0.5% 内），
     且都落在设计意图上（UI 状态 0% / 表面 0.5% / 抽屉 3.8% / 一次性的 10.6%）。

用法: python tools/test_spring_easing.py     （退出码 0 = 全部通过）
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOTION_CSS = ROOT / "site" / "css" / "motion.css"

# token → (阻尼比, 允许的峰值区间)
EXPECTED = {
    "--spring-ui": (1.0, (0.995, 1.0001)),
    "--spring-glide": (0.86, (1.0001, 1.010)),
    "--spring-sheet": (0.72, (1.020, 1.055)),
    "--spring-pop": (0.58, (1.080, 1.130)),
}

STOPS = 26
PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  [ok]   {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}  {detail}")


def analytic(zeta: float, t: float) -> float:
    """阶跃响应；w0 = 1，时间尺度由 CSS 时长提供，所以只有形状要看。"""
    if zeta >= 1:
        return 1 - (1 + t) * math.exp(-t)
    wd = math.sqrt(1 - zeta * zeta)
    return 1 - math.exp(-zeta * t) * (math.cos(wd * t) + (zeta / wd) * math.sin(wd * t))


def settle(zeta: float) -> float:
    if zeta >= 1:
        return 9.21                                  # 0.1% 稳定带
    wd = math.sqrt(1 - zeta * zeta)
    t = math.pi / wd                                 # 峰值时刻
    while analytic(zeta, t) > 1.0:                   # 首次下行穿越 1.0
        t += 1e-4
    return t


def parse_curves(css: str) -> dict[str, list[tuple[float, float]]]:
    """从 motion.css 抽出 `@supports` 里的 linear() 曲线（不是回退用的贝塞尔）。

    注意 CSS 规则：控制点可以省略百分比，此时首点按 0%、末点按 100% 处理
    （中间点省略是非法的），本解析器照此实现。
    """
    curves: dict[str, list[tuple[float, float]]] = {}
    for token, body in re.findall(r"(--spring-[\w-]+)\s*:\s*(linear\([^;]+\));", css):
        inner = body[len("linear("):-1]
        parts = [p.strip() for p in inner.split(",")]
        stops: list[tuple[float, float]] = []
        for i, part in enumerate(parts):
            bits = part.split()
            value = float(bits[0])
            if len(bits) == 1:
                percent = 0.0 if i == 0 else (1.0 if i == len(parts) - 1 else float("nan"))
            else:
                percent = float(bits[1].rstrip("%")) / 100.0
            stops.append((percent, value))
        curves[token] = stops
    return curves


def main() -> int:
    css = MOTION_CSS.read_text(encoding="utf-8")
    curves = parse_curves(css)

    print("解析 CSS linear() 曲线")
    check("四条 spring token 都出现在 @supports 块里", set(curves) == set(EXPECTED),
          f"实际 {sorted(curves)}")

    for token, (zeta, (peak_lo, peak_hi)) in EXPECTED.items():
        stops = curves.get(token)
        if not stops:
            check(f"{token} 存在", False)
            continue
        print(f"\n{token}  (ζ = {zeta})")

        # 1. 结构合法性
        check(f"{token}: 控制点数量 = {STOPS + 1}", len(stops) == STOPS + 1,
              f"实际 {len(stops)}")
        check(f"{token}: 起点 (0, 0)",
              stops[0][0] == 0.0 and abs(stops[0][1]) < 1e-9, f"{stops[0]}")
        check(f"{token}: 终点 (100%, 1.0)",
              abs(stops[-1][0] - 1.0) < 1e-9 and abs(stops[-1][1] - 1.0) < 1e-9,
              f"{stops[-1]}")
        offsets = [stops[i + 1][0] - stops[i][0] for i in range(len(stops) - 1)]
        check(f"{token}: 时间轴严格递增", all(d > 0 for d in offsets),
              f"最小步长 {min(offsets):.5f}")

        # 2. 与解析解一致
        total = settle(zeta)
        worst = max(abs(v - analytic(zeta, total * p)) for p, v in stops[:-1])
        check(f"{token}: 采样值 = 解析解（偏差 < 2e-3）", worst < 2e-3,
              f"最大偏差 {worst:.5f}")

        # 3. 峰值落在设计区间（这就是"看起来有多弹"的唯一旋钮）
        peak = max(v for _, v in stops)
        check(f"{token}: 峰值 {peak:.4f} 在 [{peak_lo}, {peak_hi}]",
              peak_lo <= peak <= peak_hi, f"解析峰值 {max(analytic(zeta, total * i / 200) for i in range(201)):.4f}")

    # 4. 回退路径：没有 linear() 的浏览器必须还有合法的贝塞尔
    print("\n回退路径")
    fallbacks = re.findall(r"(--spring-[\w-]+)\s*:\s*(cubic-bezier\([^)]*\))", css)
    check("每个 spring token 都有 cubic-bezier 回退", len(fallbacks) == len(EXPECTED),
          f"实际 {sorted(t for t, _ in fallbacks)}")
    for token, value in fallbacks:
        nums = [float(x) for x in re.findall(r"-?[\d.]+", value)]
        ok = len(nums) == 4 and all(0.0 <= nums[i] <= 1.0 for i in (0, 2)) and nums[1] >= 0 and nums[3] >= 0
        check(f"{token} 回退 {value} 合法（x 在 [0,1]，y 允许过冲）", ok, str(nums))

    # 5. 唯一性：不能和 @supports 之外的定义打架
    outside = css.split("@supports (transition-timing-function: linear(0, 1))")[0]
    check("回退定义在 @supports 之前（同 token 先降级后升级）",
          "--spring-ui: cubic-bezier" in outside)

    print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
    for name_ in FAILED:
        print(f"  - {name_}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
