#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验数据自动分析与报告生成 —— 由智能体驱动阈值迭代的异常点剔除。

完整链路：
    读取实验数据
      -> 用当前阈值 t 判定异常点（|残差| > t·σ）
      -> 交给大模型判断这次剔除是否合理
      -> 不合理就由它给出新的阈值，回到上一步重算
      -> 收敛后落盘结构化结果（JSON）与可读报告（Markdown）

阈值迭代是这个项目的核心：模型不只是被问一次意见，它的输出会改变程序下一轮怎么算。
每一轮都从全部原始数据重新判定，不做累积剔除——阈值放宽之后，上一轮被误删的点
必须能够回到数据里，否则"放宽阈值"这个动作就没有意义。

在 WSL 里运行：
    python3 agent_demo.py --mock            # 不联网，确定性验证迭代循环
    export DEEPSEEK_API_KEY=sk-xxxx
    python3 agent_demo.py                   # 真实调用
    python3 agent_demo.py --threshold 1.5   # 换一个初始阈值

只依赖 Python 标准库，不需要 pip 安装任何东西。
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

DEFAULT_THRESHOLD = 2.0
DEFAULT_MAX_ROUNDS = 3
THRESHOLD_MIN = 1.5
THRESHOLD_MAX = 4.0
MIN_KEEP_FRACTION = 0.6
RESIDUAL_TABLE_LIMIT = 15


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
# 2. 本地计算（数值部分不交给模型）
# --------------------------------------------------------------------------
def fit_line(xs, ys):
    """最小二乘直线拟合，返回斜率、截距与 R²。"""
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


def analyze_round(rows, value_column, x_column, threshold):
    """按给定阈值跑一轮：拟合 -> 算残差 -> 判定异常点 -> 剔除后重拟合。

    判的是「相对拟合曲线的偏离」，不是「相对均值的偏离」。实验数据本身带有趋势，
    用均值±σ 判定会被趋势撑大尺度，真正的异常点反而漏判。
    """
    xs = [row[x_column] for row in rows]
    ys = [row[value_column] for row in rows]
    slope, intercept, r2 = fit_line(xs, ys)
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sigma = statistics.pstdev(residuals) if len(residuals) > 1 else 0.0

    suspects = []
    if sigma > 0:
        for index, residual in enumerate(residuals):
            if abs(residual) > threshold * sigma:
                suspects.append((index, ys[index], residual / sigma))

    kept = [i for i in range(len(rows)) if i not in {s[0] for s in suspects}]
    refit = None
    if suspects and len(kept) >= 3:
        kept_x = [xs[i] for i in kept]
        kept_y = [ys[i] for i in kept]
        slope2, intercept2, r2_2 = fit_line(kept_x, kept_y)
        residuals2 = [y - (intercept2 + slope2 * x) for x, y in zip(kept_x, kept_y)]
        refit = {
            "slope": slope2,
            "intercept": intercept2,
            "r2": r2_2,
            "residual_sd": statistics.pstdev(residuals2),
            "kept": len(kept),
        }

    return {
        "round": None,
        "threshold": threshold,
        "sigma": sigma,
        "count": len(rows),
        "residuals": residuals,
        "suspects": suspects,
        "kept_count": len(kept),
        "fit": {"slope": slope, "intercept": intercept, "r2": r2, "residual_sd": sigma},
        "refit": refit,
        "verdict": None,
        "note": None,
        "next_threshold": None,
        "adjust_notes": [],
        "accepted": False,
    }


def adjust_threshold(proposed, residuals, sigma):
    """把智能体给出的阈值限制到可执行范围，返回 (阈值, 说明列表)。

    两道限制：先夹到 [THRESHOLD_MIN, THRESHOLD_MAX]，再保证至少保留
    MIN_KEEP_FRACTION 的数据点。任何一次调整都会写进说明，最终出现在报告里。
    """
    notes = []
    value = float(proposed)
    clamped = min(max(value, THRESHOLD_MIN), THRESHOLD_MAX)
    if abs(clamped - value) > 1e-9:
        notes.append(
            f"智能体给的阈值 {value:g} 超出允许范围 "
            f"{THRESHOLD_MIN:.1f}–{THRESHOLD_MAX:.1f}，夹紧为 {clamped:.1f}"
        )
    threshold = round(clamped, 1)

    def kept_at(value_of_t):
        return sum(1 for r in residuals if abs(r) <= value_of_t * sigma + 1e-12)

    need = math.ceil(len(residuals) * MIN_KEEP_FRACTION)
    if sigma > 0 and kept_at(threshold) < need:
        before = kept_at(threshold)
        while threshold < THRESHOLD_MAX and kept_at(threshold) < need:
            threshold = round(threshold + 0.1, 1)
        notes.append(
            f"按 {clamped:.1f}σ 只能保留 {before}/{len(residuals)} 个点"
            f"（要求至少 {need} 个），阈值上调至 {threshold:.1f}"
        )
    return threshold, notes


# --------------------------------------------------------------------------
# 3. 组装提示词
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "你是一名严谨的实验数据分析助手，正在参与一个「异常点剔除阈值」的迭代流程。"
    "每一轮，程序用当前阈值 t 判定异常点：残差绝对值超过 t·σ 的点被剔除，"
    "σ 是残差的标准差。你要判断这次剔除是否合理：合理就接受；"
    "不合理就给出你认为更合适的阈值，程序会用新阈值重新算一遍。"
    "判断依据应当是残差分布本身——看排序后的残差中间是否存在天然的分界，"
    "而不是机械套用 2σ 或 3σ 的经验规则。注意数据本身可能含有物理上的非线性，"
    "偏离直线并不等于测量出错。"
    "只输出 JSON，不要输出任何解释性文字或 Markdown 代码块，格式为："
    '{"removal_reasonable": true 或 false, "reason": "一句话说明判断理由", '
    '"suggested_threshold": 数字或 null, "summary": "一句话概括数据整体情况", '
    '"conclusion": "这份数据能否用于标定，为什么", '
    '"next_checks": ["建议补充的检查1", "建议补充的检查2"]}'
)


def sorted_residual_lines(residuals, sigma, limit=RESIDUAL_TABLE_LIMIT):
    """按 |残差| 从大到小列出，让模型能看出分布里的自然分界。"""
    if sigma <= 0:
        return ["残差标准差为 0，所有点都在拟合曲线上。"]
    order = sorted(range(len(residuals)), key=lambda i: -abs(residuals[i]))
    lines = [f"第{i + 1}行: {residuals[i] / sigma:+.2f}σ" for i in order[:limit]]
    if len(order) > limit:
        tail = abs(residuals[order[limit]]) / sigma
        lines.append(f"…其余 {len(order) - limit} 个点的 |残差| 都更小（最大 {tail:.2f}σ）")
    return lines


def build_user_prompt(round_number, state, history, columns, rows, x_column, value_column):
    lines = ["实验数据（" + " / ".join(columns) + "）：", "行号 | " + " | ".join(columns)]
    for index, row in enumerate(rows, start=1):
        lines.append(f"{index} | " + " | ".join(f"{row[c]:.4g}" for c in columns))

    lines.append("")
    lines.append(f"按 |残差| 从大到小排序（单位：残差标准差 σ = {state['sigma']:.4f}）：")
    lines += ["  " + item for item in sorted_residual_lines(state["residuals"], state["sigma"])]

    lines.append("")
    rows_text = ""
    if state["suspects"]:
        rows_text = "（第 " + "、".join(str(s[0] + 1) for s in state["suspects"]) + " 行）"
    lines.append(
        f"第 {round_number} 轮：当前阈值 t = {state['threshold']:.1f}σ，"
        f"剔除 {len(state['suspects'])} 个点{rows_text}。"
    )
    fit = state["fit"]
    lines.append(
        f"以 {x_column} 为自变量、{value_column} 为因变量，"
        f"剔除前拟合（{state['count']} 个点）："
        f"y = {fit['slope']:.4f} x {'+' if fit['intercept'] >= 0 else '-'} "
        f"{abs(fit['intercept']):.4f}，R² = {fit['r2']:.5f}，"
        f"残差标准差 {fit['residual_sd']:.4f}。"
    )
    if state["refit"]:
        refit = state["refit"]
        lines.append(
            f"剔除后拟合（{refit['kept']} 个点）："
            f"y = {refit['slope']:.4f} x {'+' if refit['intercept'] >= 0 else '-'} "
            f"{abs(refit['intercept']):.4f}，R² = {refit['r2']:.5f}，"
            f"残差标准差 {refit['residual_sd']:.4f}。"
        )
    else:
        lines.append("本轮没有可用的剔除后拟合。")

    lines.append("")
    if history:
        lines.append("历史轮次：")
        lines += ["  " + item for item in history]
    else:
        lines.append("历史轮次：无，这是第一轮。")

    lines.append("")
    need = math.ceil(state["count"] * MIN_KEEP_FRACTION)
    lines.append(
        f"允许范围：新阈值必须落在 {THRESHOLD_MIN:.1f}–{THRESHOLD_MAX:.1f} 之间，"
        f"且至少要保留 {MIN_KEEP_FRACTION:.0%} 的数据点"
        f"（当前 {state['count']} 行，至少保留 {need} 个）。"
    )
    if state["suspects"]:
        lines.append("请判断这次剔除是否合理。若认为不合理，请给出你认为合适的阈值。")
    else:
        lines.append(
            "本轮没有筛出任何可疑点，无需调整阈值，"
            "请直接把 removal_reasonable 填 true、suggested_threshold 填 null，并给出数据质量结论。"
        )
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


def mock_response(round_number, has_suspects):
    """--mock 的脚本化响应：第 1 轮要求放宽阈值，第 2 轮接受。

    作用是不消耗 API 就能确定性地验证迭代循环本身，输出的内容都标注了 mock。
    """
    if not has_suspects:
        return json.dumps(
            {
                "removal_reasonable": True,
                "reason": "本轮没有筛出可疑点，无需调整（mock 响应）",
                "suggested_threshold": None,
                "summary": "这是 --mock 模式生成的示例结果，用于验证流程，不代表真实判断。",
                "conclusion": "本轮没有筛出可疑点，全部数据参与标定（mock 结论）。",
                "next_checks": ["补测反向行程以检查迟滞", "在不同温度下重复标定"],
            },
            ensure_ascii=False,
        )
    if round_number == 1:
        return json.dumps(
            {
                "removal_reasonable": False,
                "reason": "被剔除的点集中在 2σ 出头，与其余点之间存在明显空档，更像是噪声偏大的正常点，首轮剔除过度（mock 判断）。",
                "suggested_threshold": 3.0,
                "summary": "数据整体呈线性，但残差分布比高斯噪声更厚尾（mock 响应）。",
                "conclusion": "当前剔除方案需要放宽后再评价（mock 结论）。",
                "next_checks": ["对首轮被剔除的点安排重测", "确认该测量区间是否存在非线性"],
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "removal_reasonable": True,
            "reason": "放宽阈值后不再剔除任何点，标定关系由全部数据决定，接受该结果（mock 判断）。",
            "suggested_threshold": None,
            "summary": "这是 --mock 模式生成的示例结果，用于验证流程，不代表真实判断。",
            "conclusion": "全部数据参与标定，得到唯一的标定关系（mock 结论）。",
            "next_checks": ["补测反向行程以检查迟滞", "在不同温度下重复标定"],
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
        raise ValueError("没有找到 JSON 对象")
    return json.loads(cleaned[start : end + 1])


def safe_parse(text):
    """解析失败返回 None，由调用方按「无法改进」处理，而不是让程序崩掉。"""
    try:
        data = parse_json_block(text)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def history_line(state):
    verdict = state.get("verdict") or {}
    rows_text = "、".join(str(s[0] + 1) for s in state["suspects"]) or "无"
    if verdict:
        judgement = "合理" if verdict.get("removal_reasonable") else "不合理"
        suggested = verdict.get("suggested_threshold")
        suggested_text = "无" if suggested in (None, "") else f"{suggested}"
        reason = verdict.get("reason", "")
    else:
        judgement, suggested_text, reason = "无法解析", "无", ""
    return (
        f"第 {state['round']} 轮：阈值 {state['threshold']:.1f}σ，σ = {state['sigma']:.4f}，"
        f"剔除 {len(state['suspects'])} 个点（{rows_text}）；"
        f"判断={judgement}，建议阈值={suggested_text}，理由：{reason}"
    )


def signed(value):
    return f"{'+' if value >= 0 else '-'} {abs(value):.4f}"


def write_report(final_state, iterations, converged, stop_reason, columns, rows,
                 value_column, out_dir, model):
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict = final_state.get("verdict") or {}

    payload = {
        "final_threshold": final_state["threshold"],
        "rounds": len(iterations),
        "converged": converged,
        "stop_reason": stop_reason,
        "iterations": [
            {
                "round": state["round"],
                "threshold": state["threshold"],
                "sigma": round(state["sigma"], 6),
                "suspect_rows": [s[0] + 1 for s in state["suspects"]],
                "removal_reasonable": (state.get("verdict") or {}).get("removal_reasonable"),
                "reason": (state.get("verdict") or {}).get("reason"),
                "suggested_threshold": (state.get("verdict") or {}).get("suggested_threshold"),
                "adjusted_threshold": state.get("next_threshold"),
                "adjust_notes": state.get("adjust_notes") or [],
                "note": state.get("note"),
                "accepted": state.get("accepted", False),
            }
            for state in iterations
        ],
        "fit": final_state["fit"],
        "refit": final_state["refit"],
        "summary": verdict.get("summary"),
        "conclusion": verdict.get("conclusion"),
        "next_checks": verdict.get("next_checks") or [],
    }
    (out_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 传感器标定数据自动分析报告",
        "",
        f"- 数据行数：{final_state['count']}",
        f"- 分析列：{value_column}",
        f"- 调用模型：{model}",
        f"- 阈值迭代：共 {len(iterations)} 轮，最终阈值 "
        f"{final_state['threshold']:.1f}σ，{'已收敛' if converged else '未收敛'}",
        f"- 终止原因：{stop_reason}",
        "",
        "## 结论",
        "",
        str(verdict.get("conclusion") or "（模型没有返回可解析的结论）").strip(),
        "",
        "## 数据整体情况",
        "",
        str(verdict.get("summary") or "（模型没有返回可解析的整体情况）").strip(),
        "",
        "## 阈值迭代过程",
        "",
        "| 轮次 | 阈值 | 残差 σ | 剔除行号 | 智能体判断 | 建议阈值 | 理由 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for state in iterations:
        item = state.get("verdict") or {}
        rows_text = "、".join(str(s[0] + 1) for s in state["suspects"]) or "无"
        if item:
            judgement = "合理" if item.get("removal_reasonable") else "不合理"
            suggested = item.get("suggested_threshold")
            suggested_text = "—" if suggested in (None, "") else f"{suggested}"
            reason = item.get("reason", "")
        else:
            judgement, suggested_text, reason = "无法解析", "—", "模型返回内容无法解析"
        lines.append(
            f"| {state['round']} | {state['threshold']:.1f}σ | {state['sigma']:.4f} | "
            f"{rows_text} | {judgement} | {suggested_text} | {reason} |"
        )

    notes = []
    for state in iterations:
        for note in state.get("adjust_notes") or []:
            notes.append(f"- 第 {state['round']} 轮：{note}，实际采用 {state['next_threshold']:.1f}σ")
    if notes:
        lines += ["", "阈值调整说明：", ""] + notes

    lines += ["", "## 最终拟合", ""]
    fit = final_state["fit"]
    refit = final_state["refit"]
    if refit:
        lines += [
            f"- 参与拟合：{refit['kept']} / {final_state['count']} 个点",
            f"- 标定关系：y = {refit['slope']:.4f} x {signed(refit['intercept'])}",
            f"- R² = {refit['r2']:.5f}（剔除前 {fit['r2']:.5f}）",
            f"- 残差标准差 {refit['residual_sd']:.4f}（剔除前 {fit['residual_sd']:.4f}）",
        ]
    else:
        lines += [
            f"- 参与拟合：{final_state['count']} / {final_state['count']} 个点（未剔除任何点）",
            f"- 标定关系：y = {fit['slope']:.4f} x {signed(fit['intercept'])}",
            f"- R² = {fit['r2']:.5f}",
            f"- 残差标准差 {fit['residual_sd']:.4f}",
        ]

    lines += ["", "## 建议补充的检查", ""]
    checks = verdict.get("next_checks") or []
    if checks:
        lines += [f"- {item}" for item in checks]
    else:
        lines.append("- （模型没有给出建议）")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="用大模型 API 自动分析实验数据，并由智能体迭代调整异常判定阈值"
    )
    parser.add_argument("--input", default="data/sample_sensor_data.csv", help="输入 CSV 路径")
    parser.add_argument("--x-column", default="时间_s", help="拟合时作为自变量的列")
    parser.add_argument("--value-column", default="输出电压_V", help="做统计分析的目标列")
    parser.add_argument("--outdir", default="results", help="结果输出目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"初始残差阈值（单位 σ，默认 {DEFAULT_THRESHOLD:g}）",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS,
        help=f"迭代上限（默认 {DEFAULT_MAX_ROUNDS}）",
    )
    parser.add_argument("--mock", action="store_true", help="不联网，用脚本化响应跑通迭代循环")
    args = parser.parse_args()

    load_env_file(Path(".env"))

    data_path = Path(args.input)
    if not data_path.exists():
        raise SystemExit(f"找不到数据文件：{data_path.resolve()}")

    columns, rows = load_rows(data_path)
    for name, column in (("--x-column", args.x_column), ("--value-column", args.value_column)):
        if column not in columns:
            raise SystemExit(f"数据里没有列 {column}（来自 {name}），可用列：{', '.join(columns)}")

    api_key = ""
    if not args.mock:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise SystemExit(
                "没有找到 DEEPSEEK_API_KEY。\n"
                "先在终端执行：export DEEPSEEK_API_KEY=sk-你的key\n"
                "或者把 .env.example 复制成 .env 并填入 key。\n"
                "只想验证流程的话，加上 --mock。"
            )

    threshold = round(min(max(args.threshold, THRESHOLD_MIN), THRESHOLD_MAX), 1)
    if threshold != round(args.threshold, 1):
        print(f"初始阈值被夹紧到允许范围：{args.threshold:g} -> {threshold:.1f}")

    tried = {threshold}
    iterations = []
    final_state = None
    converged = False
    stop_reason = ""

    print(f"已读取 {len(rows)} 行数据，列：{', '.join(columns)}")

    for round_number in range(1, args.max_rounds + 1):
        state = analyze_round(rows, args.value_column, args.x_column, threshold)
        state["round"] = round_number
        history = [history_line(item) for item in iterations]

        rows_text = ""
        if state["suspects"]:
            rows_text = "（第 " + "、".join(str(s[0] + 1) for s in state["suspects"]) + " 行）"
        print(
            f"[第 {round_number} 轮] 阈值 {threshold:.1f}σ，σ = {state['sigma']:.4f}，"
            f"剔除 {len(state['suspects'])} 个点{rows_text}"
        )

        if args.mock:
            raw = mock_response(round_number, bool(state["suspects"]))
        else:
            print(f"        正在询问 {args.model} ...")
            raw = call_deepseek(
                SYSTEM_PROMPT,
                build_user_prompt(
                    round_number, state, history, columns, rows,
                    args.x_column, args.value_column,
                ),
                api_key,
                args.model,
            )

        verdict = safe_parse(raw)
        if verdict is None:
            state["note"] = "模型返回的内容无法解析为约定的 JSON，按未收敛处理"
            iterations.append(state)
            final_state = state
            stop_reason = state["note"]
            break

        state["verdict"] = verdict

        if not state["suspects"]:
            converged = True
            stop_reason = "本轮没有筛出可疑点，迭代结束"
            iterations.append(state)
            final_state = state
            break

        if verdict.get("removal_reasonable"):
            converged = True
            stop_reason = "智能体认可本轮剔除结果"
            iterations.append(state)
            final_state = state
            break

        proposed = verdict.get("suggested_threshold")
        if isinstance(proposed, bool) or not isinstance(proposed, (int, float)):
            state["note"] = "智能体认为剔除不合理，但没有给出可用的新阈值"
            iterations.append(state)
            final_state = state
            stop_reason = state["note"]
            break

        next_threshold, adjust_notes = adjust_threshold(
            proposed, state["residuals"], state["sigma"]
        )
        state["next_threshold"] = next_threshold
        state["adjust_notes"] = adjust_notes

        if next_threshold == threshold or next_threshold in tried:
            state["note"] = f"新阈值 {next_threshold:.1f}σ 已经尝试过，停止迭代"
            iterations.append(state)
            final_state = state
            stop_reason = state["note"]
            break

        if round_number >= args.max_rounds:
            state["note"] = f"已达到最大轮数 {args.max_rounds}，仍未得到认可"
            iterations.append(state)
            final_state = state
            stop_reason = state["note"]
            break

        iterations.append(state)
        tried.add(next_threshold)
        threshold = next_threshold

    if final_state is None:
        raise SystemExit("没有完成任何一轮分析")

    final_state["accepted"] = converged
    write_report(
        final_state, iterations, converged, stop_reason, columns, rows,
        args.value_column, Path(args.outdir), args.model,
    )

    print(
        f"\n阈值迭代：共 {len(iterations)} 轮，最终阈值 {final_state['threshold']:.1f}σ，"
        f"{'已收敛' if converged else '未收敛'}（{stop_reason}）"
    )
    print("模型结论：", (final_state.get("verdict") or {}).get("conclusion", "（无）"))
    print(f"报告已写入：{Path(args.outdir).resolve() / 'report.md'}")
    print(f"结构化结果：{Path(args.outdir).resolve() / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
