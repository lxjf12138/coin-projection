"""Figure 3 (method): coin projection at the step of Figure 1, from the sender's coins to the receiver's parse.

Top row, left to right:
  1  the prefix forest of the step with its stop coins (the hub coin at the bare space, the coin at ␣week);
  2  what the message may choose under each outcome of the ␣week coin (probability strips of total mass 1,
     averaged over all other coins), and their average over keys, which is the model's law;
  3  the receiver's parse of the text "...this␣weekend" under each key, without the key, and with several
     histories.
Bottom: the same step under the four designs (fixed cut, cut at the roots, key-drawn cut, no cut): bits chosen by
the message and by the key, exactness, and model calls per token.

All step numbers come from results/I_ll_call_you_this.json (fig3_dump_forest.py); the model calls of the multi-history
scheme come from results/tables/numbers.tex (make_figs.py).

Usage: python -I plots/fig3_story.py [out.pdf] [preview.png] [forest.json] [numbers.tex]
"""
import json, math, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figures", "fig3_story.pdf")
PNG = sys.argv[2] if len(sys.argv) > 2 else None
FOREST = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("COIN_FOREST", os.path.join(HERE, "..", "results", "I_ll_call_you_this.json"))
NUMBERS = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "..", "results", "tables", "numbers.tex")

LIB = os.environ.get("COIN_FONT_DIR", "/usr/share/fonts/truetype/liberation")
FACES = ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf")
FILES = [os.path.join(LIB, n) for n in FACES if os.path.exists(os.path.join(LIB, n))]
if not FILES:
    print(f"[warn] Liberation Sans not found in {LIB}; falling back to DejaVu Sans. Set COIN_FONT_DIR to "
          f"the directory holding {', '.join(FACES)} for the published figure.", file=sys.stderr)
for f in FILES + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "pdf.fonttype": 42, "font.size": 7.5})

# ---------------- numbers of the step ----------------
d = json.load(open(FOREST, encoding="utf-8"))
law = {bytes.fromhex(k): v for k, v in d["law"].items()}
S = set(law)
parent = {v: next((v[:j] for j in range(len(v) - 1, 0, -1) if v[:j] in S), None) for v in S}
children = {v: [] for v in S}
for v, a in parent.items():
    if a is not None:
        children[a].append(v)
M = {}
for v in sorted(S, key=len, reverse=True):
    M[v] = law[v] + sum(M[c] for c in children[v])
roots = [v for v in S if parent[v] is None]
H = lambda ps: -sum(p * math.log2(p) for p in ps if p > 0)
h2 = lambda q: H([q, 1 - q])
H_STR = H(law.values())
H_ROOT = H(M[r] for r in roots)
STOP = {v: M[v] * h2(law[v] / M[v]) for v in S if children[v]}
H_STOP = sum(STOP.values())
best, keep = {}, {}
for v in sorted(S, key=len, reverse=True):          # heaviest prefix antichain (MWIS)
    sc = sum(best[c] for c in children[v])
    best[v], keep[v] = max(law[v], sc), law[v] >= sc
A, stack = [], list(roots)
while stack:
    v = stack.pop()
    if keep[v]:
        A.append(v)
    else:
        stack.extend(children[v])
P_A = sum(law[v] for v in A)
H_MWIS = H(law[v] / P_A for v in A)
SP, WEEK, WEEKEND = b" ", b" week", b" weekend"
assert (WEEK in A) != (WEEKEND in A)
MWIS_LOST = "␣weekend" if WEEK in A else "␣week"
Q_WEEK = law[WEEK] / M[WEEK]
Q_SP = law[SP] / M[SP]
N_COINS = len(STOP)
N_SP_DESC = sum(1 for v in S if v.startswith(SP) and v != SP)
split = next(x for x in d["splits"] if x["v"] == "␣week" and x["w"] == "␣weekend")
P_END = split["p_rem_after_v"]
# Given the coin at ␣week and averaged over all other coins: a stop gives ␣week its whole subtree mass, a go
# gives every string below it p / (1 - q); every string outside the subtree keeps its probability.
BLOCK = M[WEEK]
WEEKEND_GO = law[WEEKEND] / (1 - Q_WEEK)
assert abs(Q_WEEK * BLOCK - law[WEEK]) < 1e-12 and abs((1 - Q_WEEK) * WEEKEND_GO - law[WEEKEND]) < 1e-12
TOP6 = sorted(S, key=lambda v: -law[v])[:6]
assert WEEK in TOP6 and WEEKEND in TOP6
OTHERS = [v for v in TOP6 if v not in (WEEK, WEEKEND)]        # drawn words outside the ␣week subtree
assert all(not v.startswith(WEEK) for v in OTHERS)
num = dict(re.findall(r"res@([^\\]+)\\endcsname\{([^}]*)\}", open(NUMBERS, encoding="utf-8").read()))
LAS_CALLS = float(num["llama@time@las@Full@fwdpertok"])
name = lambda b: b.decode("utf-8").replace(" ", "␣")
f2 = lambda x: f"{x:.2f}".replace("0.", ".", 1) if x < 1 else f"{x:.2f}"
print(f"H_str {H_STR:.3f} H_root {H_ROOT:.3f} H_stop {H_STOP:.3f} MWIS kept {P_A:.3f} H {H_MWIS:.3f} lost {MWIS_LOST}")
print(f"q_week {Q_WEEK:.4f} block {BLOCK:.4f} weekend|go {WEEKEND_GO:.4f} stop(week) {STOP[WEEK]:.4f}")
print(f"q_space {Q_SP:.3e} p_space {law[SP]:.3e} M_space {M[SP]:.4f} stop(space) {STOP[SP]:.4f}")
print(f"p(end|week) {P_END:.3e} coins {N_COINS} sp_desc {N_SP_DESC} LAS calls {LAS_CALLS}")
print("largest stop terms:", [(name(v) if all(32 <= c < 127 for c in v) else v.hex(), round(c, 4))
                             for c, v in sorted(((c, v) for v, c in STOP.items()), reverse=True)[:8]])
print("top6:", [(name(v), round(law[v], 4)) for v in TOP6])

# ---------------- style ----------------
C_KEY, C_MSG, C_INK, C_GREY, C_FIX = "#D55E00", "#009E73", "#1A1A1A", "#6B6B6B", "#4D4D4D"
F_MSG, F_PASS, F_KEY, F_CARD, F_OURS = "#C6E6DA", "#E2E6EC", "#FBE3D3", "#F5F6F8", "#FDF1E8"
E_EDGE, E_OURS = "#AEB6C0", "#F2C9AC"
W, HT = 6.3, 2.97
fig = plt.figure(figsize=(W, HT))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, HT)
ax.axis("off")
R = fig.canvas.get_renderer()
FS = 7.5


def tw(s, size=FS, weight="normal", style="normal"):
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight, fontstyle=style)
    w = t.get_window_extent(R).width / fig.dpi
    t.remove()
    return w


def txt(x, y, s, size=FS, color=C_INK, ha="left", weight="normal", style="normal", z=5):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center", fontweight=weight, fontstyle=style, zorder=z)


BH = 0.135                                          # node / chip height (in)


def node(x, y, s, kind, size=FS):
    w = tw(s, size) + 0.10
    fc, ec, ls, col = {"msg": (F_MSG, "none", "-", C_INK), "pass": (F_PASS, "none", "-", C_INK),
                       "key": (F_KEY, "none", "-", C_INK), "chip": ("white", "#8C939C", "-", C_INK),
                       "gone": ("#FAFAFA", "#B0B0B0", (0, (2.2, 1.4)), "#9A9A9A")}[kind]
    ax.add_patch(FancyBboxPatch((x, y - BH / 2), w, BH, boxstyle="round,pad=0,rounding_size=0.035",
                                fc=fc, ec=ec, lw=0.75, ls=ls, zorder=3))
    ax.text(x + 0.05, y + (0.012 if s == "␣" else -0.004), s, va="center", ha="left", fontsize=size, color=col, zorder=4)
    return x + w


def edge(x1, y1, x2, y2):
    xm = (x1 + x2) / 2
    ax.add_patch(PathPatch(Path([(x1, y1), (xm, y1), (xm, y2), (x2, y2)], [Path.MOVETO] + [Path.CURVE4] * 3),
                           fc="none", ec=E_EDGE, lw=0.7, zorder=1))


def coin(x, y, r=0.05):
    ax.add_patch(Circle((x, y), r, fc="#FFF6EE", ec=C_KEY, lw=0.9, zorder=6))
    ax.add_patch(Circle((x, y), r * 0.42, fc=C_KEY, ec="none", zorder=7))


def card(x0, y0, w, h, ours=False):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=F_OURS if ours else F_CARD, ec=E_OURS if ours else "none", lw=0.8, zorder=0))


def head(x0, ytop, n, title, sub):
    ax.add_patch(Circle((x0 + 0.17, ytop - 0.155), 0.075, fc=C_INK, ec="none", zorder=3))
    ax.text(x0 + 0.17, ytop - 0.158, str(n), fontsize=7.5, color="white", ha="center", va="center",
            fontweight="bold", zorder=4)
    txt(x0 + 0.30, ytop - 0.155, title, 8.5, weight="bold")
    txt(x0 + 0.11, ytop - 0.335, sub, 7.2, C_GREY)


def mark(x, y, ok):
    ax.text(x, y, "✓" if ok else "✗", fontsize=7.8, color=C_MSG if ok else C_KEY, va="center", ha="left",
            family="DejaVu Sans", zorder=5)


GAP = 0.09
Y_B, H_B = 0.0, 0.93                                # bottom band
Y_T = Y_B + H_B + GAP                               # top cards
TOP = HT - 0.01
H_T = TOP - Y_T
W1, W3 = 1.86, 1.98
W2 = W - W1 - W3 - 2 * GAP
X1, X2, X3 = 0.0, W1 + GAP, W1 + GAP + W2 + GAP

# =============== 1  sender: the forest and its stop coins ===============
card(X1, Y_T, W1, H_T, ours=True)
head(X1, TOP, 1, "The key flips stop coins", "at each string that others extend")
words = sorted(sorted(OTHERS, key=lambda v: -law[v])[:3] + [WEEK], key=lambda v: -M[v])   # rows by subtree mass
rows_y, y = [], TOP - 0.40
for i, v in enumerate(words):
    y -= 0.185 + (0.13 if v == WEEK else 0) + (0.10 if i and words[i - 1] == WEEK else 0)
    rows_y.append(y)
ry = (rows_y[0] + rows_y[-1]) / 2
rend = node(X1 + 0.10, ry, "␣", "pass")
XC = X1 + 0.48
wk_end = wk_y = None
for y, v in zip(rows_y, words):
    edge(rend, ry, XC, y)
    e = node(XC, y, name(v), "msg")
    if v == WEEK:
        wk_end, wk_y = e, y
    else:
        txt(e + 0.05, y, f2(law[v]), 6.8, C_GREY)
txt(XC + 0.05, rows_y[-1] - 0.17, f"+{N_SP_DESC - len(words) - 1:,} more under ␣", 6.8, C_GREY)
# the coin at ␣week and the string behind it
xwe = wk_end + 0.28
ax.plot([wk_end, xwe], [wk_y, wk_y], color=E_EDGE, lw=0.7, zorder=1)
coin((wk_end + xwe) / 2, wk_y)
we_end = node(xwe, wk_y, "␣weekend", "msg")
txt(wk_end - 0.03, wk_y - 0.125, f2(law[WEEK]), 6.8, C_GREY, ha="right")
txt(we_end - 0.03, wk_y - 0.125, f2(law[WEEKEND]), 6.8, C_GREY, ha="right")
txt((wk_end + xwe) / 2, wk_y + 0.15, f"stop {f2(Q_WEEK)} | go {f2(1 - Q_WEEK)}", 6.8, C_KEY, ha="center")
# the hub coin
coin(rend + 0.075, ry)
txt(X1 + 0.07, ry - 0.155, "go ≈ 1", 6.8, C_KEY)
ys = Y_T + 0.10
txt(X1 + 0.11, ys + 0.15, f"coin at ␣week: {STOP[WEEK]:.2f} bits;  at ␣: {STOP[SP]:.2f}", 6.9)
txt(X1 + 0.11, ys, f"all {N_COINS:,} coins: {H_STOP:.2f} bits (stop entropy)", 6.9)

# =============== 2  per key: a prefix-free set; over keys: the model ===============
card(X2, Y_T, W2, H_T, ours=True)
head(X2, TOP, 2, "Each key leaves a prefix-free set", "the message chooses within it")
SX, SW = X2 + 0.11, W2 - 0.22
SH = 0.125
order = [v for v in sorted(OTHERS, key=lambda v: -law[v])]
segs_before = order[:2]                                     # two heaviest words left of the ␣week block
segs_after = order[2:]


def strip(y, block, label_words):
    """block: list of (text, mass, colour) that fills the ␣week subtree; returns x-range of the block."""
    x = SX
    for v in segs_before:
        w = law[v] * SW
        ax.add_patch(Rectangle((x, y), w - 0.012, SH, fc=F_MSG, ec="none", zorder=2))
        if label_words and tw(name(v), 6.8) + 0.04 < w:
            txt(x + (w - 0.012) / 2, y + SH / 2 - 0.003, name(v), 6.8, ha="center")
        x += w
    b0 = x
    for _, m, col in block:
        w = m * SW
        ax.add_patch(Rectangle((x, y), w - 0.012, SH, fc=col, ec="none", zorder=2))
        x += w
    b1 = b0 + BLOCK * SW
    x = b1
    for v in segs_after:
        w = law[v] * SW
        ax.add_patch(Rectangle((x, y), w - 0.012, SH, fc=F_MSG, ec="none", zorder=2))
        x += w
    ax.add_patch(Rectangle((x, y), SX + SW - x, SH, fc="#DDEFE8", ec="none", zorder=2))
    return b0, b1


REST_GO = BLOCK - WEEKEND_GO                                # the other strings below ␣week, given a go
r1 = TOP - 0.74
b0, b1 = strip(r1, [("␣week", BLOCK, C_MSG)], True)
txt(SX, r1 + SH + 0.085, f"K: stop ({f2(Q_WEEK)})", 7.2, weight="bold")
txt((b0 + b1) / 2, r1 + SH + 0.085, f"␣week {f2(BLOCK)}", 7.2, ha="center")
r2 = r1 - 0.40
strip(r2, [("␣weekend", WEEKEND_GO, C_MSG), ("", REST_GO, C_MSG)], False)
txt(SX, r2 + SH + 0.085, f"K′: go ({f2(1 - Q_WEEK)})", 7.2, weight="bold")
txt((b0 + b1) / 2, r2 + SH + 0.085, f"␣weekend {f2(WEEKEND_GO)}", 7.2, ha="center")
r3 = r2 - 0.46
ax.plot([SX, SX + SW], [r3 + SH + 0.20, r3 + SH + 0.20], color="#C9CED6", lw=0.6, zorder=1)
strip(r3, [("␣week", law[WEEK], "#5DBFA3"), ("␣weekend", law[WEEKEND], C_MSG), ("", BLOCK - law[WEEK] - law[WEEKEND], C_MSG)], False)
txt(SX, r3 + SH + 0.085, "all keys", 7.2, weight="bold")
lab = f"␣week {f2(law[WEEK])}  ␣weekend {f2(law[WEEKEND])}"
txt((b0 + b1) / 2, r3 + SH + 0.085, lab, 7.2, ha="center")
for r in (r1, r2, r3):                                 # the block whose owner the coin decides
    ax.add_patch(Rectangle((b0, r - 0.03), b1 - b0 - 0.012, 0.014, fc=C_KEY, ec="none", zorder=2))
mark(SX, Y_T + 0.10, True)
txt(SX + 0.13, Y_T + 0.10, f"the model:  {f2(Q_WEEK)} × {f2(BLOCK)} = {f2(law[WEEK])},  "
    f"{f2(1 - Q_WEEK)} × {f2(WEEKEND_GO)} = {f2(law[WEEKEND])}", 6.9)

# =============== 3  receiver: one text, one parse per key ===============
card(X3, Y_T, W3, H_T, ours=True)
head(X3, TOP, 3, "One text, one parse per key", "the receiver flips the same coins")
LX = X3 + 0.11
CX = X3 + 0.36                                      # chips start


def chips(x, y, toks):
    for s_ in toks:
        x = node(x, y, s_, "chip", 7.2) + 0.025
    return x - 0.025


p0 = TOP - 0.53
txt(LX, p0, "it reads", 7.0, C_GREY)
txt(LX + tw("it reads", 7.0) + 0.06, p0, "“…this weekend”", 7.4, style="italic")
p1 = p0 - 0.21
txt(LX, p1, "K", 7.2, weight="bold")
x = chips(CX, p1, ["␣week", "end"])
mark(x + 0.05, p1, True)
txt(LX, p1 - 0.19, "K′", 7.2, weight="bold")
x2 = chips(CX, p1 - 0.19, ["␣weekend"])
mark(x2 + 0.05, p1 - 0.19, True)
txt(X3 + W3 - 0.11, p1 - 0.095, "1 call per token", 6.8, C_GREY, ha="right")
yd = p1 - 0.32
ax.plot([LX, X3 + W3 - 0.11], [yd, yd], color="#E3C7B3", lw=0.6, zorder=1)
p3 = yd - 0.12
txt(LX, p3, "without the key (retokenize)", 6.9)
x = chips(LX, p3 - 0.17, ["␣weekend"])
mark(x + 0.05, p3 - 0.17, False)
txt(x + 0.19, p3 - 0.17, "wrong under K", 6.8, C_GREY)
p4 = p3 - 0.35
txt(LX, p4, "several histories (LAS)", 6.9)
x = chips(LX, p4 - 0.17, ["␣week", "end"])
txt(x + 0.035, p4 - 0.17, "+", 7.2, C_GREY)
x = chips(x + 0.125, p4 - 0.17, ["␣weekend"])
mark(x + 0.05, p4 - 0.17, True)
txt(LX, p4 - 0.33, "keeps both: more model calls", 6.8, C_GREY)

# =============== 4  the same step under four designs ===============
card(0, Y_B, W, H_B)
XI, XN, XD, XBAR, WBAR = 0.12, 0.50, 1.78, 3.00, 1.50
XV, XE, XCALL = XBAR + WBAR + 0.08, 5.35, 6.18
yh = Y_B + H_B - 0.115
txt(XN, yh, "Same step, four designs", 7.5, weight="bold")
txt(XD, yh, "where the cut lies", 6.8, C_GREY)
ax.add_patch(Rectangle((XBAR, yh - 0.035), 0.10, 0.07, fc=C_MSG, ec="none"))
txt(XBAR + 0.13, yh, "message bits", 6.8, C_GREY)
ax.add_patch(Rectangle((XBAR + 0.80, yh - 0.035), 0.10, 0.07, fc=C_KEY, ec="none"))
txt(XBAR + 0.93, yh, "key bits", 6.8, C_GREY)
txt(XCALL, yh, "calls / token", 6.8, C_GREY, ha="right")
SC = WBAR / H_STR
RH = 0.178


def icon(x, y, kind):
    """A four-node forest (root, two children, one grandchild) with the design's cut."""
    pts = {"r": (x, y), "a": (x + 0.11, y + 0.04), "b": (x + 0.11, y - 0.04), "c": (x + 0.22, y + 0.04)}
    for u, v in (("r", "a"), ("r", "b"), ("a", "c")):
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], color="#9AA3AE", lw=0.6, zorder=2)
    for p in pts.values():
        ax.add_patch(Circle(p, 0.017, fc="#6F7884", ec="none", zorder=3))
    kw = dict(solid_capstyle="round", solid_joinstyle="round", zorder=4)
    if kind == "fixed":
        ax.plot([x + 0.275, x + 0.275, x + 0.165, x + 0.165], [y + 0.075, y, y, y - 0.075], color=C_FIX, lw=1.1, **kw)
    elif kind == "root":
        ax.plot([x + 0.055, x + 0.055], [y + 0.075, y - 0.075], color=C_FIX, lw=1.1, **kw)
    elif kind == "coin":
        ax.plot([x + 0.165, x + 0.165], [y + 0.075, y - 0.075], color=C_KEY, lw=1.1, **kw)
        ax.plot([x + 0.165, x + 0.275, x + 0.275], [y, y, y + 0.075], color=C_KEY, lw=0.8, ls=(0, (1.8, 1.3)), zorder=4)


def row(i, kind, title, where, msg, key, val, ok, verdict, calls, ours=False, soft=False):
    y = Y_B + H_B - 0.30 - RH * i
    if ours:
        ax.add_patch(FancyBboxPatch((0.05, y - RH / 2 + 0.008), W - 0.10, RH - 0.016,
                                    boxstyle="round,pad=0,rounding_size=0.04", fc=F_OURS, ec=E_OURS, lw=0.7, zorder=1))
    icon(XI, y, kind)
    txt(XN, y, title, 7.4, weight="bold" if ours else "normal")
    txt(XD, y, where, 7.0, C_GREY if not ours else C_INK)
    ax.add_patch(Rectangle((XBAR, y - 0.04), WBAR, 0.08, fc="#E7EAEE", ec="none", zorder=2))
    ax.add_patch(Rectangle((XBAR, y - 0.04), msg * SC, 0.08, fc=C_MSG, ec="none", zorder=3, alpha=0.55 if soft else 1))
    if key:
        ax.add_patch(Rectangle((XBAR + msg * SC, y - 0.04), key * SC, 0.08, fc=C_KEY, ec="none", zorder=3))
    txt(XV, y, val, 7.0)
    mark(XE, y, ok)
    txt(XE + 0.13, y, verdict, 7.0)
    txt(XCALL, y, calls, 7.2, ha="right", weight="bold" if ours else "normal")


row(0, "fixed", "Filter (MWIS)", "one fixed cut", H_MWIS, 0, f"{H_MWIS:.2f}", False, "biased", "1")
row(1, "root", "SyncPool", "at the roots", H_ROOT, H_STR - H_ROOT, f"{H_ROOT:.2f} + {H_STR - H_ROOT:.2f}", True, "exact", "1")
row(2, "coin", "Coin projection (ours)", "drawn by the key", H_STR - H_STOP, H_STOP,
    f"{H_STR - H_STOP:.2f} + {H_STOP:.2f}", True, "exact", "1", ours=True)
row(3, "none", "Multi-history (LAS)", "no cut", H_STR, 0, f"up to {H_STR:.2f}", True, "exact", f"{LAS_CALLS:.1f}", soft=True)

fig.savefig(OUT)
if PNG:
    fig.savefig(PNG, dpi=300)
print("wrote", OUT)
