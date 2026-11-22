"""Render the report's formulas as SVG images -> docs/report/eq/eq01.svg ... (report tooling, no results).

The report is an HTML page printed to PDF. Formulas are drawn with matplotlib's own maths renderer (no LaTeX
install needed) at 11 pt, so they print at the size of the body text.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from casetrail import config

EQUATIONS = [
    ("bm25", r"\mathrm{score}(q,d)=\sum_{t\in q} w_q(t)\;\mathrm{idf}(t)\;\frac{tf_{t,d}\,(k_1+1)}"
             r"{tf_{t,d}+k_1\left(1-b+b\,|d|\,/\,\mathrm{avgdl}\right)},\quad\quad"
             r"\mathrm{idf}(t)=\ln\left(1+\frac{N-df_t+0.5}{df_t+0.5}\right)"),
    ("pagerank", r"r_v=\frac{1-d}{N}+d\left(\sum_{u\to v} r_u\,\frac{w_{uv}}{W_u}+\sum_{u\ \mathrm{dangling}}\frac{r_u}{N}\right)"),
    ("hits", r"\mathbf{a}\leftarrow A^{\mathrm{T}}\mathbf{h},\quad\quad \mathbf{h}\leftarrow A\,\mathbf{a}"),
    ("prior", r"s(d)=\frac{\mathrm{bm25}(d)}{\max_{d'}\;\mathrm{bm25}(d')}+\lambda\, z(\log p(d))"),
    ("naive_bayes", r"\mathrm{score}_c(d)=\log\frac{P(c)}{P(\bar c)}+\sum_t tf_{t,d}\,\log\frac{\theta_{t,c}}{\theta_{t,\bar c}},"
                    r"\quad\quad \theta_{t,c}=\frac{\mathrm{count}(t,c)+\alpha}{\sum_{t'}\mathrm{count}(t',c)+\alpha V}"),
    ("knn", r"\mathrm{score}_c(d)=\sum_{n\in N_k(d)}\cos(d,n)\;\mathbf{1}[\,c\in\mathrm{labels}(n)\,]"),
    ("maxsim", r"s(q,d)=\frac{1}{|Q|}\sum_{i\in Q}\;\max_{j\in D}\;\mathbf{q}_i^{\mathrm{T}}\mathbf{d}_j"),
    ("infonce", r"\mathcal{L}_{\mathrm{NCE}}=-\log\frac{e^{s(q,p)/\tau}}{\sum_{c\in C}e^{s(q,c)/\tau}}"),
    ("margin_mse", r"\mathcal{L}_{\mathrm{MSE}}=\left(\left(s_S(q,p)-s_S(q,n)\right)-\left(s_T(q,p)-s_T(q,n)\right)\right)^2"),
    ("micro_f1", r"P=\frac{\sum_q |R_q\cap G_q|}{\sum_q |R_q|},\quad\quad R=\frac{\sum_q |R_q\cap G_q|}{\sum_q |G_q|},"
                 r"\quad\quad F_1=\frac{2PR}{P+R}"),
]


def render(tex: str, path: Path, font_size: float):
    fig = plt.figure(figsize=(0.01, 0.01))
    # Display-size fractions (mathtext draws a plain fraction in the small inline style)
    fig.text(0, 0, "$" + tex.replace(r"\frac", r"\dfrac") + "$", fontsize=font_size)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-size", type=float, default=11.0)
    args = parser.parse_args()
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["svg.fonttype"] = "path"
    out_dir = config.PROJECT_ROOT / "docs" / "report" / "eq"
    out_dir.mkdir(parents=True, exist_ok=True)
    for number, (name, tex) in enumerate(EQUATIONS, start=1):
        path = out_dir / f"eq{number:02d}_{name}.svg"
        render(tex, path, args.font_size)
        print(f"({number}) {name} -> {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
