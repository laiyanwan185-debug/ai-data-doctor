from __future__ import annotations

import io
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats

# Lazy import for openai — only needed in Tab 4
try:
    from openai import OpenAI as OpenAIClient
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAIClient = None  # type: ignore[assignment]
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Doctor — 数据诊断中心",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #0f766e 0%, #06b6d4 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2.3rem; font-weight: 700; margin: 0; }
    .main-header p { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0 0; }
    .diagnosis-card {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
        border-radius: 0.75rem;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
    }
    .diagnosis-card.warn {
        background: #fffbeb;
        border-color: #fcd34d;
    }
    .diagnosis-card.bad {
        background: #fef2f2;
        border-color: #fecaca;
    }
    .stat-box {
        background: #f8fafc;
        border-radius: 0.75rem;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .stat-box .number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f766e;
    }
    .stat-box .label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    .surgery-code {
        background: #1e293b;
        color: #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 500px;
        overflow-y: auto;
    }
    .comparison-improved {
        color: #059669;
        font-weight: 600;
    }
    .comparison-worsened {
        color: #dc2626;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants — Statistical Expert System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个严谨的统计学家。你的任务是根据提供的数据诊断报告生成 Python 清洗代码。你的决策准则如下：

缺失值处理：若该列符合正态分布（p-value > 0.05），使用均值（Mean）填充；若不符合正态分布或存在显著偏度，必须使用中位数（Median）填充；对于分类变量，使用众数（Mode）或标注为"Missing"。

异常值处理：若符合正态分布，使用 3-Sigma 法则识别并截断；若不符合正态分布，使用 IQR（四分位距）法则进行盖帽处理。

数据分布优化：若偏度（Skewness）绝对值 > 1，建议执行 Log 变换或 Box-Cox 变换以平滑分布。

代码规范：仅输出 Python 代码，直接作用于名为 df 的 DataFrame。不要输出任何解释文字。"""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def detect_type_anomalies(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect columns whose dtype may not match their actual content."""
    issues: list[dict[str, Any]] = []

    for col in dataframe.select_dtypes(include=["object"]):
        col_clean = dataframe[col].dropna()
        if len(col_clean) == 0:
            continue
        numeric_like = pd.to_numeric(col_clean, errors="coerce")
        pct_numeric = numeric_like.notna().mean()
        if pct_numeric > 0.85:
            issues.append({
                "column": col,
                "current_type": "object (文本)",
                "suggested_type": "float64 (数值)",
                "evidence": f"{pct_numeric:.0%} 的行为数值，疑似因空值/特殊字符被识别为文本",
                "severity": "medium",
            })

    for col in dataframe.select_dtypes(include=[np.number]):
        col_clean = dataframe[col].dropna()
        if len(col_clean) < 2:
            continue
        unique_ratio = col_clean.nunique() / len(col_clean)
        if unique_ratio < 0.01 and len(col_clean) > 20:
            issues.append({
                "column": col,
                "current_type": str(dataframe[col].dtype),
                "suggested_type": "需人工审查",
                "evidence": f"唯一值比例仅 {unique_ratio:.2%}，可能为常量列或编码有误",
                "severity": "low",
            })

    for col in dataframe.select_dtypes(include=[np.number]):
        col_clean = dataframe[col].dropna()
        if len(col_clean) == 0:
            continue
        n_unique = col_clean.nunique()
        if n_unique <= 5 and len(col_clean) > 20:
            is_integer = (col_clean.dropna() == col_clean.dropna().astype(int)).all()
            if is_integer:
                issues.append({
                    "column": col,
                    "current_type": str(dataframe[col].dtype),
                    "suggested_type": "category (分类)",
                    "evidence": f"仅 {n_unique} 个不同值且均为整数，可能为编码分类变量",
                    "severity": "medium",
                })

    return issues


def extract_code(text: str) -> str:
    """Extract Python code from AI response, stripping markdown fences if present."""
    pattern = r"```(?:python)?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


@st.cache_data(show_spinner="正在计算诊断指标...")
def compute_diagnostics(
    _df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> dict[str, Any]:
    """Compute full diagnostic report for the dataset. Cached for performance.

    Handles 70k+ rows efficiently: Shapiro-Wilk automatically samples down to 5000.
    """
    report: dict[str, Any] = {
        "n_rows": len(_df),
        "n_cols": len(_df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing": {},
        "shapiro": {},
        "skewness": {},
        "kurtosis": {},
        "descriptive": {},
    }

    # ---- Missing values (all columns) ----
    for col in _df.columns:
        miss_count = int(_df[col].isnull().sum())
        miss_pct = round(miss_count / len(_df) * 100, 2)
        report["missing"][col] = {"count": miss_count, "pct": miss_pct}

    # ---- Numeric column diagnostics ----
    for col in numeric_cols:
        series = _df[col].dropna()
        n_valid = len(series)

        if n_valid < 3:
            report["shapiro"][col] = {
                "statistic": None, "p_value": None,
                "is_normal": None, "error": "有效数据不足 (n < 3)",
            }
            continue

        # Descriptive stats
        report["descriptive"][col] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "q25": round(float(series.quantile(0.25)), 4),
            "q50": round(float(series.median()), 4),
            "q75": round(float(series.quantile(0.75)), 4),
            "max": round(float(series.max()), 4),
        }

        # Skewness & Kurtosis
        report["skewness"][col] = round(float(series.skew()), 4)
        report["kurtosis"][col] = round(float(series.kurtosis()), 4)

        # Shapiro-Wilk (sample down to 5000 for performance)
        try:
            sample = series if n_valid <= 5000 else series.sample(5000, random_state=42)
            sw_stat, sw_p = scipy_stats.shapiro(sample)
            report["shapiro"][col] = {
                "statistic": round(float(sw_stat), 4),
                "p_value": round(float(sw_p), 6),
                "is_normal": bool(sw_p >= 0.05),
            }
        except Exception:
            report["shapiro"][col] = {
                "statistic": None, "p_value": None,
                "is_normal": None, "error": "检验执行失败",
            }

    return report


def run_diagnostics_light(_df: pd.DataFrame) -> dict[str, Any]:
    """Run diagnostics on an already-cleaned dataframe (no Streamlit caching needed)."""
    numeric_cols = _df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = _df.select_dtypes(exclude=[np.number]).columns.tolist()

    report: dict[str, Any] = {
        "n_rows": len(_df),
        "n_cols": len(_df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing": {},
        "shapiro": {},
        "skewness": {},
        "kurtosis": {},
        "descriptive": {},
    }

    for col in _df.columns:
        miss_count = int(_df[col].isnull().sum())
        miss_pct = round(miss_count / len(_df) * 100, 2) if len(_df) > 0 else 0.0
        report["missing"][col] = {"count": miss_count, "pct": miss_pct}

    for col in numeric_cols:
        series = _df[col].dropna()
        if len(series) < 3:
            continue

        report["descriptive"][col] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "q50": round(float(series.median()), 4),
            "max": round(float(series.max()), 4),
        }
        report["skewness"][col] = round(float(series.skew()), 4)
        report["kurtosis"][col] = round(float(series.kurtosis()), 4)

        try:
            sample = series if len(series) <= 5000 else series.sample(5000, random_state=42)
            sw_stat, sw_p = scipy_stats.shapiro(sample)
            report["shapiro"][col] = {
                "statistic": round(float(sw_stat), 4),
                "p_value": round(float(sw_p), 6),
                "is_normal": bool(sw_p >= 0.05),
            }
        except Exception:
            report["shapiro"][col] = {
                "statistic": None, "p_value": None, "is_normal": None,
            }

    return report


def build_diagnosis_prompt(report: dict[str, Any], type_issues: list[dict[str, Any]]) -> str:
    """Build a structured prompt from the diagnosis report for the AI statistician."""
    lines: list[str] = []

    lines.append("## 数据集基本信息")
    lines.append(f"- 总行数: {report['n_rows']:,}")
    lines.append(f"- 总列数: {report['n_cols']}")
    lines.append(f"- 数值列: {', '.join(report['numeric_cols']) if report['numeric_cols'] else '无'}")
    lines.append(f"- 分类/文本列: {', '.join(report['categorical_cols']) if report['categorical_cols'] else '无'}")
    lines.append("")

    # ---- Missing values report ----
    lines.append("## 缺失值报告")
    missing_items = [(col, info) for col, info in report["missing"].items() if info["count"] > 0]
    if missing_items:
        for col, info in missing_items:
            lines.append(f"- **{col}**: {info['count']} 个缺失 ({info['pct']}%)")
    else:
        lines.append("(无缺失值)")
    lines.append("")

    # ---- Distribution diagnostics ----
    lines.append("## 分布诊断报告（数值列）")
    for col in report["numeric_cols"]:
        lines.append(f"### 列: `{col}`")
        if col in report["descriptive"]:
            desc = report["descriptive"][col]
            lines.append(f"- 均值={desc['mean']}, 标准差={desc['std']}")
            lines.append(f"- 中位数={desc['q50']}, IQR=[{desc['q25']}, {desc['q75']}]")
            lines.append(f"- 范围=[{desc['min']}, {desc['max']}]")
        if col in report["skewness"]:
            lines.append(f"- 偏度 (Skewness) = {report['skewness'][col]}")
        if col in report["kurtosis"]:
            lines.append(f"- 峰度 (Kurtosis) = {report['kurtosis'][col]}")
        if col in report["shapiro"]:
            sh = report["shapiro"][col]
            if sh.get("is_normal") is True:
                lines.append(f"- Shapiro-Wilk: W={sh['statistic']}, p={sh['p_value']} → **符合正态分布**")
            elif sh.get("is_normal") is False:
                lines.append(f"- Shapiro-Wilk: W={sh['statistic']}, p={sh['p_value']} → **非正态分布**")
            else:
                lines.append(f"- Shapiro-Wilk: 无法判定 ({sh.get('error', '未知')})")
        lines.append("")

    # ---- Categorical columns ----
    if report["categorical_cols"]:
        lines.append("## 分类变量")
        for col in report["categorical_cols"]:
            miss = report["missing"].get(col, {})
            lines.append(f"- **{col}**: 缺失 {miss.get('count', 0)} 个 ({miss.get('pct', 0)}%)")
        lines.append("")

    # ---- Type anomalies ----
    if type_issues:
        lines.append("## 类型异常检测")
        for issue in type_issues:
            lines.append(f"- **{issue['column']}**: 当前={issue['current_type']}, 建议={issue['suggested_type']} ({issue['evidence']})")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="main-header">
    <h1>🩺 Data Doctor — 数据诊断中心</h1>
    <p>上传 · 诊断 · 分布探测 · AI 自动清洗 · 一键生成数据健康报告</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/stethoscope.png",
        width=64,
    )
    st.markdown("## 📂 数据上传")

    uploaded_file = st.file_uploader(
        "选择 CSV 或 Excel 文件",
        type=["csv", "xlsx", "xls"],
        help="支持 .csv / .xlsx / .xls 格式",
    )

    st.markdown("---")
    st.markdown("### 🔑 DeepSeek API Key")

    # API key resolution order: env var → secrets → manual input
    default_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not default_key:
        try:
            default_key = st.secrets.get("deepseek_api_key", "")
        except Exception:
            default_key = ""

    api_key = st.text_input(
        "输入 API Key",
        type="password",
        help="用于 AI 自动清洗功能。从 platform.deepseek.com 获取。",
        value=default_key,
    )

    st.markdown("---")
    st.markdown("### 📋 诊断项目清单")
    st.markdown(
        """
- 数据类型自动识别
- 描述性统计 (均值 / 标准差 / 偏度 / 峰度)
- 缺失值扫描
- 列类型异常检测
- Shapiro-Wilk 正态性检验
- 分布直方图 + 拟合曲线
- 🧪 AI 统计学家自动清洗
    """
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if uploaded_file is None:
    st.info("👈 请在左侧上传 CSV 或 Excel 文件以开始诊断。", icon="ℹ️")
    st.stop()


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame | None:
    try:
        fname = file.name.lower()
        if fname.endswith(".csv"):
            return pd.read_csv(io.StringIO(file.getvalue().decode("utf-8")))
        else:
            return pd.read_excel(file, engine="openpyxl")
    except Exception as e:
        st.error(f"文件读取失败: {e}")
        return None


# Reset session state when a new file is uploaded
file_key = uploaded_file.name + "::" + str(uploaded_file.size)
if st.session_state.get("last_file_key") != file_key:
    st.session_state.last_file_key = file_key
    for _key in ("ai_code", "df_cleaned", "surgery_done", "surgery_error"):
        st.session_state.pop(_key, None)

df = load_data(uploaded_file)
if df is None:
    st.stop()

# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------
numeric_cols: list[str] = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols: list[str] = df.select_dtypes(exclude=[np.number]).columns.tolist()

# ---------------------------------------------------------------------------
# Compute diagnostics (cached) & type anomalies
# ---------------------------------------------------------------------------
diagnosis = compute_diagnostics(df, numeric_cols, categorical_cols)
type_issues = detect_type_anomalies(df)

# ---------------------------------------------------------------------------
# Summary row
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        f"<div class='stat-box'><div class='number'>{df.shape[0]:,}</div><div class='label'>总行数</div></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<div class='stat-box'><div class='number'>{df.shape[1]}</div><div class='label'>总列数</div></div>",
        unsafe_allow_html=True,
    )
with col3:
    missing_total = sum(info["count"] for info in diagnosis["missing"].values())
    missing_pct = missing_total / (df.shape[0] * df.shape[1]) * 100 if df.shape[0] * df.shape[1] > 0 else 0
    st.markdown(
        f"<div class='stat-box'><div class='number'>{missing_pct:.1f}%</div><div class='label'>总缺失率</div></div>",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"<div class='stat-box'><div class='number'>{len(numeric_cols)}</div><div class='label'>数值列</div></div>",
        unsafe_allow_html=True,
    )
with col5:
    st.markdown(
        f"<div class='stat-box'><div class='number'>{len(categorical_cols)}</div><div class='label'>分类/文本列</div></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 数据预览",
    "🩻 统计诊断",
    "📊 分布探测器",
    "🧪 AI 自动手术室",
])

# ===========================================================================
# Tab 1: Data Preview
# ===========================================================================
with tab1:
    st.markdown("### 前 5 行预览")
    st.dataframe(df.head(5), use_container_width=True)

    st.markdown("### 列信息")
    dtypes_df = pd.DataFrame({
        "列名": df.columns,
        "数据类型": df.dtypes.astype(str).values,
        "非空数量": df.notnull().sum().values,
        "空值数": df.isnull().sum().values,
        "空值比例": (df.isnull().sum() / len(df) * 100).round(2).astype(str).values + "%",
    })
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

# ===========================================================================
# Tab 2: Statistical Diagnosis
# ===========================================================================
with tab2:
    st.markdown("### 数值列描述性统计")

    if numeric_cols:
        desc = df[numeric_cols].describe()
        skew_row = df[numeric_cols].skew().rename("偏度 (Skewness)")
        kurt_row = df[numeric_cols].kurtosis().rename("峰度 (Kurtosis)")

        stats_df = pd.concat([desc, pd.DataFrame([skew_row, kurt_row])])
        stats_df = stats_df.round(4)

        st.dataframe(stats_df, use_container_width=True)

        with st.expander("📐 指标解读"):
            st.markdown(
                """
| 指标 | 含义 | 判定参考 |
|------|------|----------|
| **均值 (mean)** | 数据中心趋势 | — |
| **标准差 (std)** | 数据离散程度 | 越大越分散 |
| **偏度 (Skewness)** | 分布不对称性 | >0 右偏（长尾在右）；<0 左偏；≈0 对称 |
| **峰度 (Kurtosis)** | 分布尾部厚度 | >0 厚尾（更多极端值）；<0 薄尾；≈0 接近正态 |
                """
            )
    else:
        st.info("未检测到数值列。")

    st.markdown("---")

    # ---- Missing value bar chart ----
    st.markdown("### 缺失值扫描")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if not missing.empty:
        fig_missing = px.bar(
            x=missing.values,
            y=missing.index,
            orientation="h",
            title="各列缺失值数量",
            labels={"x": "缺失值数量", "y": "列名"},
            color=missing.values,
            color_continuous_scale="oranges",
            text_auto=True,
        )
        fig_missing.update_layout(
            xaxis_title="缺失值数量",
            yaxis_title="",
            coloraxis_showscale=False,
            height=max(200, len(missing) * 40 + 80),
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_missing, use_container_width=True)

        st.markdown("**缺失比例明细**")
        missing_pct_df = pd.DataFrame({
            "列名": missing.index,
            "缺失数量": missing.values,
            "缺失比例": (missing.values / len(df) * 100).round(2).astype(str) + "%",
        })
        st.dataframe(missing_pct_df, use_container_width=True, hide_index=True)
    else:
        st.success("数据集完整，未发现缺失值。")

    st.markdown("---")

    # ---- Type anomaly detection ----
    st.markdown("### 列类型异常检测")

    if type_issues:
        for issue in type_issues:
            sev = issue["severity"]
            card_class = "diagnosis-card" + (" warn" if sev == "medium" else " bad" if sev == "high" else "")
            icon = "⚠️" if sev == "medium" else ("🔴" if sev == "high" else "ℹ️")
            st.markdown(
                f"""
<div class="{card_class}">
    <strong>{icon} 列「{issue["column"]}」</strong><br>
    当前类型: <code>{issue["current_type"]}</code> →
    建议类型: <code>{issue["suggested_type"]}</code><br>
    <small>证据: {issue["evidence"]}</small>
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.success("未发现列类型异常，所有列类型匹配其实际内容。")

# ===========================================================================
# Tab 3: Distribution Detector
# ===========================================================================
with tab3:
    st.markdown("### Shapiro-Wilk 正态性检验")

    if not numeric_cols:
        st.info("未检测到数值列，无法进行分布探测。")
    else:
        n_cols_per_row = 2
        for i in range(0, len(numeric_cols), n_cols_per_row):
            row_cols = st.columns(n_cols_per_row)
            for j, col_name in enumerate(numeric_cols[i : i + n_cols_per_row]):
                with row_cols[j]:
                    series = df[col_name].dropna()

                    if len(series) < 3:
                        st.warning(f"「{col_name}」有效数据不足 (< 3)，跳过检验。")
                        continue

                    # Use pre-computed Shapiro-Wilk result from diagnosis cache
                    sh_result = diagnosis["shapiro"].get(col_name, {})
                    sw_p = sh_result.get("p_value")
                    is_normal = sh_result.get("is_normal")

                    if sw_p is None:
                        st.warning(f"「{col_name}」无法执行 Shapiro-Wilk 检验。")
                        continue

                    if is_normal:
                        st.success(f"**{col_name}** 符合正态分布 (p = {sw_p:.4f})")
                    else:
                        st.warning(f"**{col_name}** 为非正态分布 (p = {sw_p:.4f})")
                        st.caption("→ 建议使用 **中位数** 填充缺失值")

                    # Histogram with normal curve overlay
                    mu, sigma = series.mean(), series.std()

                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=series,
                        histnorm="probability density",
                        name="观测数据",
                        marker_color="#0f766e",
                        marker_line_color="white",
                        marker_line_width=1,
                        nbinsx=min(50, int(np.sqrt(len(series))) + 1),
                    ))

                    x_range = np.linspace(series.min(), series.max(), 200)
                    pdf = scipy_stats.norm.pdf(x_range, mu, sigma)
                    fig.add_trace(go.Scatter(
                        x=x_range, y=pdf,
                        mode="lines",
                        name=f"正态拟合 N({mu:.2f}, {sigma:.2f})",
                        line_color="#ef4444",
                        line_width=2.5,
                    ))

                    fig.update_layout(
                        title=f"{col_name} 分布直方图",
                        xaxis_title=col_name,
                        yaxis_title="概率密度",
                        bargap=0.05,
                        margin=dict(t=40, b=20),
                        height=320,
                        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Mini stats row
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("均值", f"{mu:.2f}")
                    c2.metric("标准差", f"{sigma:.2f}")
                    c3.metric("偏度", f"{diagnosis['skewness'].get(col_name, 0):.2f}")
                    c4.metric("峰度", f"{diagnosis['kurtosis'].get(col_name, 0):.2f}")

                    st.markdown("---")

# ===========================================================================
# Tab 4: AI Auto Surgery Room
# ===========================================================================
with tab4:
    st.markdown("### 🧪 AI 自动手术室")

    st.markdown(
        """
<div style="background:#f0fdfa; border:1px solid #99f6e4; border-radius:0.75rem; padding:1rem 1.5rem; margin-bottom:1rem;">
    <strong>工作原理</strong><br>
    自动收集数据诊断报告（Shapiro-Wilk 正态性检验、缺失值比例、偏度、峰度），
    发送给 AI 统计学家（DeepSeek），由其根据统计学准则自动生成清洗代码，
    并在您的数据上执行。所有代码对您完全可见，确保透明可控。
</div>
""",
        unsafe_allow_html=True,
    )

    # ---- Diagnosis summary cards ----
    st.markdown("#### 当前诊断摘要")

    missing_cols_count = sum(1 for info in diagnosis["missing"].values() if info["count"] > 0)
    non_normal_count = sum(
        1 for col in numeric_cols
        if diagnosis["shapiro"].get(col, {}).get("is_normal") is False
    )
    high_skew_count = sum(
        1 for col in numeric_cols
        if abs(diagnosis["skewness"].get(col, 0)) > 1
    )

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("有缺失值的列", missing_cols_count)
    with sc2:
        st.metric("非正态分布列", non_normal_count)
    with sc3:
        st.metric("高偏度列 (|Skew| > 1)", high_skew_count)
    with sc4:
        st.metric("类型异常", len(type_issues))

    st.markdown("---")

    # ---- Diagnosis prompt preview ----
    diagnosis_prompt = build_diagnosis_prompt(diagnosis, type_issues)

    with st.expander("查看发送给 AI 的完整诊断报告", expanded=False):
        st.code(diagnosis_prompt, language="markdown")

    st.markdown("---")

    # ---- API key check ----
    if not _OPENAI_AVAILABLE:
        st.error("`openai` 包未安装。请运行: `pip install openai`")
    elif not api_key:
        st.warning("请在左侧输入 DeepSeek API Key 以使用 AI 自动清洗功能。")

    # ---- Launch AI Surgery button ----
    col_btn1, _ = st.columns([2, 3])
    with col_btn1:
        start_surgery = st.button(
            "启动 AI 科学清洗",
            type="primary",
            use_container_width=True,
            disabled=not (api_key and _OPENAI_AVAILABLE),
        )

    if start_surgery and api_key and _OPENAI_AVAILABLE:
        with st.spinner("AI 统计学家正在分析数据并生成清洗代码..."):
            try:
                client = OpenAIClient(
                    api_key=api_key,
                    base_url="https://api.deepseek.com",
                )

                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": diagnosis_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )

                raw_code = response.choices[0].message.content
                clean_code = extract_code(raw_code or "")

                st.session_state.ai_code = clean_code
                st.session_state.surgery_done = False
                st.session_state.surgery_error = None

            except Exception as e:
                st.session_state.ai_code = None
                st.session_state.surgery_error = f"API 调用失败: {e}"
                st.error(st.session_state.surgery_error)

    # ---- Display AI-generated code ----
    if st.session_state.get("ai_code"):
        st.markdown("#### AI 生成的清洗代码")
        st.caption("请审计以下代码后再执行。这是 Builder Mindset 的核心——您始终掌控数据。")
        st.markdown(
            f"<div class='surgery-code'>{st.session_state.ai_code}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        col_exec1, _ = st.columns([2, 3])
        with col_exec1:
            execute_code = st.button(
                "执行清洗代码",
                type="primary",
                use_container_width=True,
            )

        if execute_code:
            with st.spinner("正在执行清洗操作..."):
                try:
                    df_cleaned = df.copy()
                    namespace: dict[str, Any] = {
                        "pd": pd,
                        "np": np,
                        "df": df_cleaned,
                        "stats": scipy_stats,
                    }
                    exec(st.session_state.ai_code, namespace)
                    df_cleaned = namespace["df"]

                    if not isinstance(df_cleaned, pd.DataFrame):
                        raise ValueError(
                            "清洗代码未返回有效的 DataFrame。请确认代码是否正确操作了名为 df 的变量。"
                        )

                    st.session_state.df_cleaned = df_cleaned
                    st.session_state.surgery_done = True
                    st.session_state.surgery_error = None

                except Exception as e:
                    st.session_state.surgery_error = f"代码执行失败: {e}"
                    st.session_state.surgery_done = False
                    st.error(st.session_state.surgery_error)

    # ---- Before/After comparison ----
    if st.session_state.get("surgery_done") and st.session_state.get("df_cleaned") is not None:
        df_cleaned: pd.DataFrame = st.session_state.df_cleaned

        st.markdown("---")
        st.markdown("## 清洗前后对比")

        # Compute post-surgery diagnostics
        post_diag = run_diagnostics_light(df_cleaned)

        # ---- Overall health metrics ----
        st.markdown("#### 整体健康指标对比")
        cm1, cm2, cm3, cm4 = st.columns(4)
        with cm1:
            before_miss_total = sum(info["count"] for info in diagnosis["missing"].values())
            after_miss_total = sum(info["count"] for info in post_diag["missing"].values())
            delta_miss = before_miss_total - after_miss_total
            st.metric(
                "总缺失值",
                f"{after_miss_total:,}",
                delta=f"{-delta_miss if delta_miss >= 0 else delta_miss:+d}",
                delta_color="inverse",
            )
        with cm2:
            before_normal_cnt = sum(
                1 for col in numeric_cols
                if diagnosis["shapiro"].get(col, {}).get("is_normal") is True
            )
            after_norm_cols = post_diag["numeric_cols"]
            after_normal_cnt = sum(
                1 for col in after_norm_cols
                if post_diag["shapiro"].get(col, {}).get("is_normal") is True
            )
            st.metric(
                "符合正态分布列数",
                after_normal_cnt,
                delta=f"{after_normal_cnt - before_normal_cnt:+d}",
            )
        with cm3:
            before_skews = [abs(v) for v in diagnosis["skewness"].values() if v is not None]
            after_skews = [abs(v) for v in post_diag["skewness"].values() if v is not None]
            avg_skew_before = float(np.mean(before_skews)) if before_skews else 0.0
            avg_skew_after = float(np.mean(after_skews)) if after_skews else 0.0
            st.metric(
                "平均 |偏度|",
                f"{avg_skew_after:.3f}",
                delta=f"{avg_skew_after - avg_skew_before:+.3f}",
                delta_color="inverse",
            )
        with cm4:
            st.metric(
                "数据行数",
                f"{len(df_cleaned):,}",
                delta=f"{len(df_cleaned) - len(df):+,}",
            )

        # ---- Per-column comparison table ----
        st.markdown("#### 逐列对比明细")
        comp_rows: list[dict[str, Any]] = []

        all_columns = list(dict.fromkeys(
            list(diagnosis["missing"].keys()) + list(post_diag["missing"].keys())
        ))

        for col in all_columns:
            before_miss = diagnosis["missing"].get(col, {}).get("count", 0)
            after_miss = post_diag["missing"].get(col, {}).get("count", 0)

            # Missing values
            comp_rows.append({
                "列名": col,
                "指标": "缺失值数量",
                "清洗前": str(before_miss),
                "清洗后": str(after_miss),
                "判定": "✅ 已修复" if after_miss < before_miss else ("➖ 无变化" if after_miss == before_miss else "⚠️ 增加"),
            })

            # Normality (numeric only)
            if col in diagnosis["shapiro"] and col in post_diag["shapiro"]:
                b_norm = diagnosis["shapiro"][col].get("is_normal")
                a_norm = post_diag["shapiro"][col].get("is_normal")
                b_p = diagnosis["shapiro"][col].get("p_value")
                a_p = post_diag["shapiro"][col].get("p_value")

                if b_p is not None and a_p is not None:
                    if b_norm is False and a_norm is True:
                        verdict = "✅ 改善 (→正态)"
                    elif b_norm is True and a_norm is False:
                        verdict = "⚠️ 变为非正态"
                    elif b_norm is True and a_norm is True:
                        verdict = "➖ 保持正态"
                    else:
                        verdict = "➖ 保持非正态"

                    comp_rows.append({
                        "列名": col,
                        "指标": "正态性 (p-value)",
                        "清洗前": f"p={b_p:.4f}",
                        "清洗后": f"p={a_p:.4f}",
                        "判定": verdict,
                    })

            # Skewness (numeric only)
            if col in diagnosis["skewness"] and col in post_diag["skewness"]:
                b_skew = diagnosis["skewness"][col]
                a_skew = post_diag["skewness"][col]
                if b_skew is not None and a_skew is not None:
                    improvement = abs(a_skew) - abs(b_skew)
                    if improvement < -0.1:
                        skew_verdict = f"✅ 改善 ({b_skew:.2f} → {a_skew:.2f})"
                    elif improvement > 0.1:
                        skew_verdict = f"⚠️ 变差 ({b_skew:.2f} → {a_skew:.2f})"
                    else:
                        skew_verdict = f"➖ 持平 ({b_skew:.2f} → {a_skew:.2f})"

                    comp_rows.append({
                        "列名": col,
                        "指标": "偏度 (Skewness)",
                        "清洗前": f"{b_skew:.4f}",
                        "清洗后": f"{a_skew:.4f}",
                        "判定": skew_verdict,
                    })

        if comp_rows:
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.info("数据清洗前后无明显变化。")

        # ---- Download button ----
        st.markdown("---")
        st.markdown("#### 导出清洗后数据")

        csv_buffer = io.StringIO()
        df_cleaned.to_csv(csv_buffer, index=False)

        st.download_button(
            label="下载手术后数据 (.csv)",
            data=csv_buffer.getvalue(),
            file_name="data_doctor_cleaned.csv",
            mime="text/csv",
            type="primary",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Data Doctor v2.0 · Shapiro-Wilk 正态性检验 · 偏度/峰度分析 · 类型异常检测 · "
    "AI 统计学家自动清洗 · 支持 CSV / Excel 上传"
)
