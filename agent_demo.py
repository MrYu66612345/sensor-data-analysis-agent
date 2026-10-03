#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验数据自动分析与报告生成 —— 用大模型 API 完成一次端到端的分析任务。

完整链路：
    读取实验数据 -> 本地计算统计量并筛出可疑点 -> 调用大模型 API 给出判断
    -> 落盘结构化结果（JSON）与可读报告（Markdown）

在 WSL 里运行：
    python3 agent_demo.py --mock      # 不联网，先确认流程通畅
    export DEEPSEEK_API_KEY=sk-xxxx
    python3 agent_demo.py             # 真实调用

只依赖 Python 标准库，不需要 pip 安装任何东西。
"""

import argparse
import csv
import json
import os
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
SIGMA = 2.0


# --------------------------------------------------------------------------
# 1. 读取数据
# --------------------------------------------------------------------------
def load_env_file(path: Path) -> None:
    """如果同级目录有 .env，就把里面的键值对读进环境变量（不覆盖已有变量）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_rows(path: Path):
    """读 CSV，返回 (列名列表, 每一行转成 float 的列表)。"""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = [c.strip() for c in (reader.fieldnames or [])]
        rows = []
        for raw in reader:
            rows.append({c: float(raw[c]) for c in columns if raw.get(c) not in (None, "")})
    if not rows:
        raise SystemExit(f"数据文件里没有读到任何数据行：{path}")
    return columns, rows


# --------------------------------------------------------------------------
# 2. 本地先算一遍（不把原始判断权交给模型）
# --------------------------------------------------------------------------
def fit_line(xs, ys):
    """最小二乘直线拟合，返回斜率、截距与 R²。"""
    n = len(xs)
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, mean_y, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def local_analysis(rows, value_column, x_column):
    """先用最小二乘拟合出趋势，再按残差找出偏离趋势的点。

    对递增的实验数据直接对原始值做均值±σ 判定是无效的（数据本身就有趋势），
    所以这里判的是「相对拟合曲线的偏离」。
    最后把可疑点剔除再拟合一次，得到修正后的标定参数。
    """
    xs = [row[x_column] for row in rows]
    ys = [row[value_column] for row in rows]
    slope, intercept, r2 = fit_line(xs, ys)
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    residual_sd = statistics.pstdev(residuals) if len(residuals) > 1 else 0.0

    suspects = []
    if residual_sd > 0:
        for index, residual in enumerate(residuals):
            if abs(residual) > SIGMA * residual_sd:
                suspects.append((index, ys[index], residual / residual_sd))

    # 剔除可疑点后重新拟合，得到可用于标定的参数
    kept = [i for i in range(len(rows)) if i not in {s[0] for s in suspects}]
    refit = None
    if suspects and len(kept) >= 3:
        kx = [xs[i] for i in kept]
        ky = [ys[i] for i in kept]
        slope2, intercept2, r2_2 = fit_line(kx, ky)
        res2 = [y - (intercept2 + slope2 * x) for x, y in zip(kx, ky)]
        refit = {
            "slope": slope2,
            "intercept": intercept2,
            "r2": r2_2,
            "residual_sd": statistics.pstdev(res2),
            "kept": len(kept),
        }

    return {
        "mean": statistics.fmean(ys),
        "stdev": statistics.pstdev(ys) if len(ys) > 1 else 0.0,
        "min": min(ys),
        "max": max(ys),
        "count": len(ys),
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "residual_sd": residual_sd,
        "suspects": suspects,
        "refit": refit,
    }


# --------------------------------------------------------------------------
# 3. 组装提示词
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一名严谨的实验数据分析助手。你会收到一份传感器标定实验的原始数据表，"
    "以及本地程序算出的统计量。请判断数据质量、指出可疑点，并给出下一步该补做哪些检查。"
    "只输出 JSON，不要输出任何解释性文字或 Markdown 代码块，格式为："
    '{"summary": "一句话概括数据整体情况", '
    '"suspicious_points": [{"row": 行号从1开始, "reason": "判定理由"}], '
    '"conclusion": "这份数据能否用于标定，为什么", '
    '"next_checks": ["建议补充的检查1", "建议补充的检查2"]}'
)


def build_user_prompt(columns, rows, value_column, x_column, stats):
    lines = ["实验数据（" + " / ".join(columns) + "）："]
    header = "行号 | " + " | ".join(columns)
    lines.append(header)
    for index, row in enumerate(rows, start=1):
        cells = [f"{row[c]:.4g}" for c in columns]
        lines.append(f"{index} | " + " | ".join(cells))
    lines.append("")
    lines.append(
        "本地统计（对列 {col}）：样本数 {count}，均值 {mean:.4f}，"
        "总体标准差 {stdev:.4f}，最小 {min:.4f}，最大 {max:.4f}。".format(
            col=value_column, **stats
        )
    )
    lines.append(
        "以 {x} 为自变量对 {y} 做最小二乘拟合：斜率 {slope:.4f}，"
        "截距 {intercept:.4f}，R² = {r2:.5f}，残差标准差 {residual_sd:.4f}。".format(
            x=x_column, y=value_column, **stats
        )
    )
    if stats["suspects"]:
        detail = "；".join(
            f"第{i + 1}行={v:.4f}（偏离均值 {z:+.2f}σ）" for i, v, z in stats["suspects"]
        )
        lines.append(f"本地按「拟合残差超过 {SIGMA}σ」筛出的可疑点：{detail}")
    else:
        lines.append(f"本地按「拟合残差超过 {SIGMA}σ」未筛出可疑点。")
    if stats["refit"]:
        refit = stats["refit"]
        lines.append(
            "剔除上述可疑点后重新拟合（{kept} 个点）：斜率 {slope:.4f}，截距 {intercept:.4f}，"
            "R² = {r2:.5f}，残差标准差 {residual_sd:.4f}。".format(**refit)
        )
        lines.append("请判断剔除这些点是否合理，并说明这份数据最终能否用于标定。")
    lines.append("")
    lines.append("请给出你的判断。")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 4. 调用大模型 API
# --------------------------------------------------------------------------
def call_deepseek(system_prompt, user_prompt, api_key, model, timeout=90):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise SystemExit(f"接口返回错误 {error.code}：{detail}")
    except urllib.error.URLError as error:
        raise SystemExit(f"网络连接失败：{error.reason}")
    return body["choices"][0]["message"]["content"]


def mock_response():
    return json.dumps(
        {
            "summary": "这是 --mock 模式生成的示例结果，用于验证流程是否通畅，不代表真实判断。",
            "suspicious_points": [
                {"row": 7, "reason": "输出电压明显低于同位移下的线性趋势"},
                {"row": 12, "reason": "输出电压明显低于同位移下的线性趋势"},
            ],
            "conclusion": "剔除两个可疑点后数据整体呈线性，可用于初步标定。",
            "next_checks": ["对第 7、12 行重测", "补充反向行程数据以检查迟滞"],
        },
        ensure_ascii=False,
    )


# --------------------------------------------------------------------------
# 5. 解析与落盘
# --------------------------------------------------------------------------
def parse_json_block(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise SystemExit(f"模型没有返回可解析的 JSON：\n{text}")
    return json.loads(cleaned[start : end + 1])


def write_report(result, stats, value_column, columns, rows, out_dir, model):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 传感器标定数据自动分析报告",
        "",
        f"- 数据行数：{stats['count']}",
        f"- 分析列：{value_column}",
        f"- 均值 {stats['mean']:.4f}，标准差 {stats['stdev']:.4f}，"
        f"范围 [{stats['min']:.4f}, {stats['max']:.4f}]",
        f"- 最小二乘拟合：y = {stats['slope']:.4f} x "
        f"{'+' if stats['intercept'] >= 0 else '-'} {abs(stats['intercept']):.4f}，"
        f"R² = {stats['r2']:.5f}",
        f"- 调用模型：{model}",
        "",
        "## 结论",
        "",
        str(result.get("conclusion", "")).strip(),
        "",
        "## 数据整体情况",
        "",
        str(result.get("summary", "")).strip(),
        "",
        "## 可疑点",
        "",
    ]
    points = result.get("suspicious_points") or []
    if points:
        lines.append("| 行号 | 原始数据 | 模型给出的理由 |")
        lines.append("| --- | --- | --- |")
        for point in points:
            row_number = int(point.get("row", 0))
            raw = ""
            if 1 <= row_number <= len(rows):
                raw = " / ".join(f"{rows[row_number - 1][c]:.4g}" for c in columns)
            lines.append(f"| {row_number} | {raw} | {point.get('reason', '')} |")
    else:
        lines.append("未发现可疑点。")

    if stats.get("refit"):
        refit = stats["refit"]
        lines += [
            "",
            "## 剔除异常点后的重新拟合",
            "",
            f"- 参与拟合：{refit['kept']} / {stats['count']} 个点",
            f"- 标定关系：y = {refit['slope']:.4f} x "
            f"{'+' if refit['intercept'] >= 0 else '-'} {abs(refit['intercept']):.4f}",
            f"- R² = {refit['r2']:.5f}（剔除前 {stats['r2']:.5f}）",
            f"- 残差标准差 {refit['residual_sd']:.4f}（剔除前 {stats['residual_sd']:.4f}）",
        ]

    lines += ["", "## 建议补充的检查", ""]
    checks = result.get("next_checks") or []
    for item in checks:
        lines.append(f"- {item}")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="用大模型 API 自动分析实验数据并生成报告")
    parser.add_argument("--input", default="data/sample_sensor_data.csv", help="输入 CSV 路径")
    parser.add_argument("--value-column", default="输出电压_V", help="做统计分析的目标列")
    parser.add_argument("--x-column", default="时间_s", help="拟合时作为自变量的列")
    parser.add_argument("--outdir", default="results", help="结果输出目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--mock", action="store_true", help="不联网，用示例结果跑通全流程")
    args = parser.parse_args()

    load_env_file(Path(".env"))

    data_path = Path(args.input)
    if not data_path.exists():
        raise SystemExit(f"找不到数据文件：{data_path.resolve()}")

    columns, rows = load_rows(data_path)
    if args.x_column not in columns:
        raise SystemExit(f"数据里没有列 {args.x_column}，可用列：{', '.join(columns)}")
    if args.value_column not in columns:
        raise SystemExit(f"数据里没有列 {args.value_column}，可用列：{', '.join(columns)}")

    stats = local_analysis(rows, args.value_column, args.x_column)
    print(f"已读取 {len(rows)} 行数据，列：{', '.join(columns)}")
    print(
        f"本地统计：均值 {stats['mean']:.4f}，标准差 {stats['stdev']:.4f}，"
        f"线性拟合 R² = {stats['r2']:.5f}"
    )
    print(f"按拟合残差判定，筛出 {len(stats['suspects'])} 个可疑点")

    if args.mock:
        print("== mock 模式：不调用接口 ==")
        raw = mock_response()
    else:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "没有找到 DEEPSEEK_API_KEY。\n"
                "先在终端执行：export DEEPSEEK_API_KEY=sk-你的key\n"
                "或者把 .env.example 复制成 .env 并填入 key。"
            )
        print(f"正在调用 {args.model} ...")
        raw = call_deepseek(
            SYSTEM_PROMPT,
            build_user_prompt(columns, rows, args.value_column, args.x_column, stats),
            api_key,
            args.model,
        )

    result = parse_json_block(raw)
    write_report(result, stats, args.value_column, columns, rows, Path(args.outdir), args.model)

    print("\n模型结论：", result.get("conclusion", ""))
    print(f"报告已写入：{Path(args.outdir).resolve()/'report.md'}")
    print(f"结构化结果：{Path(args.outdir).resolve()/'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
