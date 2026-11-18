"""Data figures of the coin paper in matplotlib (TrueType fonts only):
Figure 2 (entropy decomposition per top-k) and Figure 4 (predicted vs measured gain).

Both read the files that make_figs.py writes into results/tables/, so run make_figs.py first:
  python -I plots/make_figs.py
  python -I plots/fig_plots.py [preview_dir]
Outputs: figures/fig2_decomp.pdf, figures/fig4_predict.pdf (and PNG previews if preview_dir is given).
COIN_GEN / COIN_FIG override the input and output directories (the smoke fixture renders into /tmp).
Palette validated with dataviz/scripts/validate_palette.js (--mode light --pairs all).
"""
import csv, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.environ.get("COIN_GEN", os.path.join(HERE, "..", "results", "tables"))
FIG = os.environ.get("COIN_FIG", os.path.join(HERE, "..", "figures"))
PREVIEW = sys.argv[1] if len(sys.argv) > 1 else None

def _add_fonts(names, root):
    """Register the TrueType faces that exist; the venue needs TrueType (pdf.fonttype 42), not a family."""
    return [os.path.join(root, n) for n in names if os.path.exists(os.path.join(root, n))]

LIB = os.environ.get("COIN_FONT_DIR", "/usr/share/fonts/truetype/liberation")
FACES = ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf")
FILES = _add_fonts(FACES, LIB)
FONT = "Liberation Sans" if FILES else "DejaVu Sans"
if not FILES:                             # any registered TrueType sans will do
    for alt in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"):
        if os.path.exists(alt):
            FILES.append(alt)
    print(f"[warn] Liberation Sans not found in {LIB}; using {FONT}. Set COIN_FONT_DIR to the directory "
          f"holding {', '.join(FACES)} for the published figures.", file=sys.stderr)
for f in FILES:
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": FONT, "pdf.fonttype": 42, "font.size": 7.5,
    "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": FONT + ":italic",
    "mathtext.bf": FONT + ":bold", "mathtext.cal": FONT + ":italic", "mathtext.default": "it",
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "axes.edgecolor": "#555555", "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.labelcolor": "#1A1A1A", "ytick.labelcolor": "#1A1A1A",
    "legend.frameon": False, "legend.handlelength": 1.1, "legend.handleheight": 0.8,
})
MODELS = [
    ("llama", "EN", "Llama-3.1-8B", "Llama-3.1"),
    ("qwen", "ZH", "Qwen3-8B-Base", "Qwen3-8B"),
    ("llmjp", "JA", "llm-jp-3-7.2b", "llm-jp-3"),
]
C_ROOT, C_BRANCH, C_STOP = "#0072B2", "#009E73", "#D55E00"
# Validated 3-language categorical palette (ALL CHECKS PASS on light surface, CVD deutan >= 10.7, tritan >= 8.4)
C_LANG = {"EN": "#0072B2", "ZH": "#B5651D", "JA": "#9E3D76"}
MARK = {"EN": "o", "ZH": "s", "JA": "^"}
INK, MUTED = "#1A1A1A", "#5A5A5A"
H = lambda s: rf"$H_{{\mathrm{{{s}}}}}$"


def table(name):
    with open(os.path.join(GEN, name)) as fh:
        return list(csv.DictReader(fh, delimiter=" "))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------- Figure 2: decomposition of H_str per top-k ----------------
fig = plt.figure(figsize=(6.3, 1.62))
L, R, B, T, GAP = 0.058, 0.995, 0.345, 0.875, 0.018
GUT = 2.72      # gutter right of the Full bar, in bar slots, for the budget labels
w = (R - L - 2 * GAP) / 3
for i, (tag, lang, name, _) in enumerate(MODELS):
    rows = table(f"decomp_{tag}.dat")
    ax = fig.add_axes([L + i * (w + GAP), B, w, T - B])
    x = list(range(len(rows)))
    root = [float(r["root"]) for r in rows]
    branch = [float(r["branch"]) for r in rows]
    stop = [float(r["stop"]) for r in rows]
    coin = [a + b for a, b in zip(root, branch)]
    kw = dict(width=0.66, edgecolor="white", linewidth=0.55, zorder=3)
    ax.bar(x, root, color=C_ROOT, label=f"Root {H('root')} (SyncPool's budget)", **kw)
    ax.bar(x, branch, bottom=root, color=C_BRANCH, label=f"Branch {H('branch')} (added by coin projection)", **kw)
    ax.bar(x, stop, bottom=coin, color=C_STOP, label=f"Stop {H('stop')} (lost under one history)", **kw)
    # Budgets read off the Full-vocabulary bar: everything below a tick is what that design hands the coder
    rf, cf = root[-1], coin[-1]
    x0, x1 = x[-1] + 0.33, x[-1] + 0.33 + GUT - 0.22
    for y in (rf, cf):
        ax.plot([x0, x1], [y, y], color="#6E6E6E", lw=0.5, zorder=4, solid_capstyle="butt", clip_on=False)
    lab = dict(ha="left", fontsize=6.1, color=INK, zorder=5, clip_on=False)
    ax.text(x0 + 0.10, rf - 0.022, f"SyncPool {100 * rf:.0f}%", va="top", **lab)
    ax.text(x0 + 0.10, cf - 0.022, f"coin {100 * cf:.0f}%", va="top", fontweight="bold", **lab)
    ax.text(x0 + 0.10, cf + 0.012, f"stop {100 * (1 - cf):.0f}%", va="bottom", **lab)
    ax.set_xticks(x, [r["k"] for r in rows])
    ax.set_xlim(-0.54, len(rows) - 0.67 + GUT)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25", "50", "75", "100"] if i == 0 else [])
    ax.tick_params(axis="x", length=0, pad=2.2)
    ax.tick_params(axis="y", pad=1.5)
    ax.hlines([0.25, 0.5, 0.75, 1.0], -0.54, x[-1] + 0.33, color="#EAEAEA", lw=0.4, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_bounds(-0.54, x[-1] + 0.33)
    ax.set_title(f"{lang}  ·  {name}", fontsize=7.6, fontweight="bold", pad=4.0, color=INK)
    if i == 1:
        ax.set_xlabel("top-$k$ vocabulary truncation", labelpad=1.8, color=INK, x=(x[-1] + 0.87) / 2 / (len(rows) - 0.13 + GUT))
    if i == 0:
        ax.set_ylabel(f"share of {H('str')} (%)", labelpad=2.0, color=INK)
        handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.515, 0.000), ncol=3, fontsize=7.0,
           columnspacing=1.6, handletextpad=0.5, labelcolor=INK)
save(fig, "fig2_decomp")


# ---------------- Figure 4: predicted vs measured gain ----------------
fig = plt.figure(figsize=(3.03, 1.72))
ax = fig.add_axes([0.145, 0.210, 0.835, 0.765])


def finite_rows(tag):
    """Gain points with a finite estimate and interval; a dropped point is reported, never drawn silently."""
    out = []
    for r in table(f"predict_{tag}.dat"):
        vals = [float(r[a]) for a in ("pred", "meas", "eminus", "eplus")]
        if all(math.isfinite(v) for v in vals):
            out.append(r)
        else:
            print(f"[warn] fig4: {tag} k={r['k']} not finite ({vals}); point omitted")
    return out


_all = [finite_rows(t) for t, *_ in MODELS]
_v = [float(r[a]) for rows in _all for r in rows for a in ("pred", "meas")]
_v += [float(r["meas"]) + float(r["eplus"]) for rows in _all for r in rows]
_v += [float(r["meas"]) - float(r["eminus"]) for rows in _all for r in rows]
LO, HI = min(0.88, 0.9 * min(_v)), max(2.2, 1.12 * max(_v))
TICKS = [t for t in (1, 2, 4, 8, 16, 32) if LO <= t <= HI]
BAND = 0.10     # every cell lies within 10% of its prediction; checked below
ax.fill_between([LO, HI], [LO * (1 - BAND), HI * (1 - BAND)], [LO * (1 + BAND), HI * (1 + BAND)],
                color="#E4E4E4", lw=0, zorder=0.5)
ax.plot([LO, HI], [LO, HI], color="#7A7A7A", lw=0.65, ls=(0, (3.2, 2.0)), zorder=1)
_dev = max(abs(float(r["meas"]) / float(r["pred"]) - 1) for rows in _all for r in rows)
assert _dev < BAND, f"a point leaves the band: max deviation {_dev:.3f}"
print(f"fig4: max |measured/predicted - 1| = {_dev:.3f}")
# Direct callouts in neutral dark ink positioned strictly in empty regions (no line crossings)
CALLOUT = {
    "EN": dict(xytext=(5, -14), ha="center", va="top"),
    "JA": dict(xytext=(-14, 11), ha="right", va="bottom"),
    "ZH": dict(xytext=(10, -8), ha="left", va="top"),
}
ZORD = {"EN": 3, "JA": 4, "ZH": 5}
for (tag, lang, name, short), rows in zip(MODELS, _all):
    xs = [float(r["pred"]) for r in rows]
    ys = [float(r["meas"]) for r in rows]
    err = [[float(r["eminus"]) for r in rows], [float(r["eplus"]) for r in rows]]
    c = C_LANG[lang]
    z = ZORD[lang]
    ax.errorbar(xs, ys, yerr=err, color=c, ls="none", elinewidth=0.6, capsize=1.5, capthick=0.6,
                marker=MARK[lang], ms=4.0 if lang != "JA" else 4.5, mec="white", mew=0.45,
                label=f"{lang} ({short})", zorder=z)
    if "Full" not in [r["k"] for r in rows]:
        continue
    full = [r["k"] for r in rows].index("Full")
    cfg = CALLOUT[lang]
    ax.annotate(f"{lang} Full", (xs[full], ys[full]), textcoords="offset points", fontsize=6.6, color=INK,
                arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=0.45, shrinkA=1, shrinkB=3.5),
                zorder=6, **cfg)
ax.set_xscale("log")
ax.set_yscale("log")
for a in (ax.xaxis, ax.yaxis):
    a.set_major_locator(FixedLocator(TICKS))
    a.set_minor_locator(FixedLocator([m for m in (3, 5, 6, 7, 12, 24) if LO <= m <= HI]))
    a.set_minor_formatter(NullFormatter())
ax.set_xticklabels([f"{t}×" for t in TICKS])
ax.set_yticklabels([f"{t}×" for t in TICKS])
ax.set_xlim(LO, HI)
ax.set_ylim(LO, HI)
ax.grid(which="major", color="#EAEAEA", lw=0.4, zorder=0)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(pad=1.8)
ax.set_xlabel(f"predicted gain  ({H('str')} − {H('stop')}) / {H('root')}", labelpad=2.0, color=INK)
ax.set_ylabel("measured utilization ratio", labelpad=2.0, color=INK)
_yx = (LO * HI) ** 0.5 * 1.35     # label the diagonal near the geometric middle of the axes
ax.text(3.95, 3.95 * 1.19, "$y = x$ ± 10%", fontsize=6.8, color=MUTED, rotation=27.5, ha="center", va="bottom",
        rotation_mode="anchor")
ax.legend(loc="upper left", fontsize=6.8, handlelength=1.0, handletextpad=0.35,
          labelspacing=0.32, borderaxespad=0.25, labelcolor=INK)
save(fig, "fig4_predict")
