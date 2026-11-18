"""Generate Figure 1 (running example) as a draw.io file with explicit geometry.

Usage: python -I fig1_drawio.py <out.drawio> <forest.json> [<v> <w>]
One step of the figure model with the full vocabulary after one context; every number comes from the forest
file of that context (fig3_dump_forest.py, stage F).  The ranked list shows the most probable next-token
strings.  v and w (default ␣week, ␣weekend) form a prefix pair: the split v + (w - v) reads as w (its
probability p(v) p(w - v | ... v) is the file's split entry), yet the stop-or-go decision at v carries
M h(q) bits that no single-history receiver can use.  The bar at the bottom splits the step's string
entropy into Hstr - Hstop and Hstop.

Built with the direct-XML route of the drawio-academic-skills overlay (the YAML DSL cannot express
HTML token labels or curly brackets); the generator applies the overlay's checks itself: print floor
(7 pt at 218 pt column width), label fit against measured text widths, and on-canvas bounds.
Fonts are TrueType so that draw.io's PDF export embeds them as CID fonts rather than Type 3.
"""
import json
import math
import sys
from xml.sax.saxutils import quoteattr

from fontTools.ttLib import TTFont

W = 292.0
PRINT_PT, FLOOR_PT = 218.0, 7.0
SANS, SERIF, SYM = "Liberation Sans", "Liberation Serif", "DejaVu Sans"
# Okabe-Ito: vermillion = stop decision and its cost, bluish green = usable entropy
C_STOP, C_USE = "#D55E00", "#009E73"
C_INK, C_GREY, C_LINE, C_SP = "#1A1A1A", "#6B6B6B", "#A6A6A6", "#ABABAB"
F_BAR, F_HL, F_STOP_LT = "#C5CBD3", "#FCEDE3", "#F2B48F"
FL, FN, FSM = 10.5, 10.0, 9.5          # label, number and small sizes in px (x0.747 = pt)
PAD, CH = 3.0, 12.5                    # chip padding and height

_FONT_FILES = {
    "sans": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "sym": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}
_FONTS = {k: TTFont(v) for k, v in _FONT_FILES.items()}


def tw(s, size, bold=False):
    """Advance width of s in px; characters missing from Liberation Sans fall back to DejaVu Sans."""
    total = 0.0
    for ch in s:
        for key in (("bold" if bold else "sans"), "sym"):
            f = _FONTS[key]
            g = f.getBestCmap().get(ord(ch))
            if g is not None:
                total += f["hmtx"][g][0] / f["head"].unitsPerEm * size
                break
    return total


cells, checks = [], []


def _id():
    _id.n += 1
    return f"c{_id.n}"


_id.n = 1


def vertex(x, y, w, h, value, style):
    cells.append(f'<mxCell id="{_id()}" value={quoteattr(value)} style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>')
    if x < -0.01 or x + w > W + 0.01:
        checks.append(f"off-canvas: x={x:.1f} w={w:.1f} {value[:30]!r}")


def line(pts, color, width=0.9, dash=False):
    (x0, y0), (x1, y1) = pts[0], pts[-1]
    mid = "".join(f'<mxPoint x="{x:.2f}" y="{y:.2f}"/>' for x, y in pts[1:-1])
    style = (f"endArrow=none;startArrow=none;html=1;rounded=0;strokeColor={color};strokeWidth={width};"
             + ("dashed=1;dashPattern=1 1.6;" if dash else ""))
    cells.append(f'<mxCell id="{_id()}" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{x0:.2f}" y="{y0:.2f}" as="sourcePoint"/><mxPoint x="{x1:.2f}" y="{y1:.2f}" as="targetPoint"/>'
                 + (f'<Array as="points">{mid}</Array>' if mid else "") + "</mxGeometry></mxCell>")


def text(x, y, w, h, html, size, color=C_INK, align="left", font=SANS, plain=None, bold=False):
    """Text cell; `plain` is the visible text used for the fit and print-floor checks."""
    vertex(x, y, w, h, html, f"text;html=1;whiteSpace=nowrap;overflow=visible;spacing=0;align={align};"
           f"verticalAlign=middle;fontFamily={font};fontSize={size};fontColor={color};")
    if size * PRINT_PT / W < FLOOR_PT - 1e-6:
        checks.append(f"print floor: {size}px = {size * PRINT_PT / W:.2f}pt < {FLOOR_PT}pt in {plain or html[:30]!r}")
    if plain is not None and font == SANS and tw(plain, size, bold) > w + 0.5:
        checks.append(f"label fit: {plain!r} needs {tw(plain, size, bold):.1f}px > box {w:.1f}px")


def rect(x, y, w, h, fill, stroke="none", sw=0.8, radius=0):
    vertex(x, y, w, h, "", f"rounded={1 if radius else 0};absoluteArcSize=1;arcSize={2 * radius};"
           f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};")


def sp(color=C_SP):
    return f'<span style="font-family:{SYM};color:{color}">␣</span>'


def label(s):
    """Token text with the leading-space mark drawn in light grey from DejaVu Sans."""
    return (sp() + s[1:]) if s.startswith("␣") else s


def chip(x, yc, s, size=FSM, stroke=C_LINE, fill="#FFFFFF", sw=0.8):
    w = tw(s, size) + 2 * PAD
    vertex(x, yc - CH / 2, w, CH, label(s),
           f"rounded=1;absoluteArcSize=1;arcSize=6;html=1;whiteSpace=nowrap;spacing=0;fillColor={fill};"
           f"strokeColor={stroke};strokeWidth={sw};fontFamily={SANS};fontSize={size};fontColor={C_INK};")
    return x + w


def italic(s, size):
    return f'<i style="font-family:{SANS};font-size:{size}px">{s}</i>'


# ---------------- data (fig3_dump_forest.py, full vocabulary) ----------------
_d = json.load(open(sys.argv[2], encoding="utf-8"))
V_S, W_S = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("␣week", "␣weekend")
_law = {bytes.fromhex(a).decode("utf-8", errors="backslashreplace").replace(" ", "␣").replace("\n", "↵"): b
        for a, b in _d["law"].items()}
ROWS = sorted(_law.items(), key=lambda kv: -kv[1])[:6]
PAIR = {V_S, W_S}
if not PAIR <= {r[0] for r in ROWS}:
    sys.exit(f"{V_S} and {W_S} are not both among the six most probable strings: {[r[0] for r in ROWS]}")
if abs([r[0] for r in ROWS].index(W_S) - [r[0] for r in ROWS].index(V_S)) != 1:
    sys.exit(f"{V_S} and {W_S} must be adjacent in the ranked list (the brace spans two rows): {[r[0] for r in ROWS]}")
_sm = _d["summary"]
HSTR, HSTOP = _sm["H_str"], _sm["H_stop"]
STOP_WEEK = next(n["stopterm"] for n in _d["nodes"] if n["s"] == V_S)
_sp = next(x for x in _d["splits"] if x["v"] == V_S and x["w"] == W_S)
REM, P_SPLIT = _sp["rem"], _sp["p_split"]
N_MORE = _sm["strings"] - len(ROWS)
CONTEXT = _sm["context"].replace("'", "’")
_e = math.floor(math.log10(P_SPLIT))
_m = round(P_SPLIT / 10 ** _e)
if _m == 10:
    _m, _e = 1, _e + 1
print(f"{_sm['model']} {_sm['context']!r}: rows {ROWS}; H_str {HSTR:.3f} H_stop {HSTOP:.3f} "
      f"stop({V_S}) {STOP_WEEK:.3f} p(split) {P_SPLIT:.2e}")

# ---------------- ranked next-token list ----------------
XL = 56.0                      # right edge of the token labels
XB = XL + 5                    # bars start
XN1 = 148.0                    # right edge of the probability column
BL = 64.0 / ROWS[0][1]         # px per unit probability: the top bar is 64 px
Y0, PITCH, BH = 30.0, 15.0, 8.0
rows_y = [Y0 + PITCH * i for i in range(len(ROWS))]

text(0, 3, 150, 14, f'<b>Context:</b> {italic(f"“{CONTEXT} …”", FL)}', FL, plain=None, font=SANS)
text(XN1 - 40, 6, 40, 11, italic("p", FL + 0.5), FN, color=C_GREY, align="right")

# top and bottom rows of the highlighted pair (either string may rank first)
iw, iwe = sorted((rows_y[[r[0] for r in ROWS].index(V_S)], rows_y[[r[0] for r in ROWS].index(W_S)]))
rect(1, iw - PITCH / 2 + 0.5, XN1 + 6, iwe - iw + PITCH - 1, F_HL, radius=3)
for (s, p), y in zip(ROWS, rows_y):
    hot = s in PAIR
    text(0, y - 6, XL, 12, label(s), FL, align="right", plain=s)
    rect(XB, y - BH / 2, p * BL, BH, C_STOP if hot else F_BAR, radius=1.5)
    text(XN1 - 24, y - 6, 24, 12, f"{p:.2f}".lstrip("0"), FN, color=C_INK if hot else C_GREY, align="right", plain=f"{p:.2f}"[1:])
ymore = rows_y[-1] + PITCH
text(0, ymore - 6, XL, 12, "…", FL, color=C_GREY, align="right", plain="…")
text(XB, ymore - 6, 120, 12, f"{N_MORE:,} more", FSM, color=C_GREY, plain=f"{N_MORE:,} more")

# ---------------- brace and callouts ----------------
xbr = XN1 + 9
ymid = (iw + iwe) / 2
vertex(xbr, iw - 6, 6, iwe - iw + 12, "",
       f"shape=curlyBracket;rounded=1;flipH=1;size=0.5;fillColor=none;strokeColor={C_STOP};strokeWidth=1;")
XC = xbr + 11                  # left edge of the callouts
WC = W - XC
# rare ambiguity: the split ␣week + end reads as ␣weekend
yr = rows_y[0] - 12
text(XC, yr - 6, WC, 12, "<b>rare ambiguity</b>", FSM, color=C_STOP, plain="rare ambiguity", bold=True)
x = chip(XC, yr + 12, V_S)
x = chip(x + 2, yr + 12, REM)
text(x + 2.5, yr + 6, 10, 12, "→", FSM, color=C_GREY, font=SYM)
chip(x + 13.5, yr + 12, W_S)
text(XC, yr + 24, WC, 12, f'{italic("p", FSM + 1)} ≈ {_m}×10<sup style="font-size:{FSM - 2.5}px">−{-_e}</sup> at this step',
     FSM, color=C_GREY, plain=f"p ≈ {_m}×10-{-_e} at this step")
# real cost: the stop-or-go decision at ␣week; the brace tip points at its heading
yc = ymid
text(XC, yc - 6, WC, 12, "<b>real cost</b>", FSM, color=C_STOP, plain="real cost", bold=True)
text(XC, yc + 8, WC, 12, f"stop at {label(V_S)} or go on?", FSM, plain=f"stop at {V_S} or go on?")
text(XC, yc + 21, WC, 12, f"unusable: <b>{STOP_WEEK:.2f} bits</b>", FSM, color=C_STOP, plain=f"unusable: {STOP_WEEK:.2f} bits")

# ---------------- the step's entropy ----------------
yb = ymore + 25
x0b, x1b = 0.5, W - 0.5
sc = (x1b - x0b) / HSTR
xs = x1b - HSTOP * sc
text(x0b, yb - 13, 200, 12, f"this step: {HSTR:.2f} bits", FSM, color=C_GREY, plain=f"this step: {HSTR:.2f} bits")
rect(x0b, yb, xs - x0b - 0.8, 12.5, C_USE, radius=2)
rect(xs + 0.8, yb, x1b - xs - 0.8, 12.5, C_STOP, radius=2)
text(x0b + 5, yb, xs - x0b - 10, 12.5, f'<span style="color:#FFFFFF">usable with one token history: <b>{HSTR - HSTOP:.2f}</b></span>',
     FSM, plain=f"usable with one token history: {HSTR - HSTOP:.2f}")
xm_stop = (xs + 0.8 + x1b) / 2
line([(xm_stop, yb + 13.0), (xm_stop, yb + 20.5), (xm_stop - 5.0, yb + 20.5)], C_STOP, width=0.85)
text(x1b - 190, yb + 14.5, 190 - (x1b - xm_stop + 6.5), 12,
     f'unusable: <b style="color:{C_STOP}">{HSTOP:.2f}</b> (stop, {HSTOP / HSTR:.0%})', FSM,
     color=C_INK, align="right", plain=f"unusable: {HSTOP:.2f} (stop, {HSTOP / HSTR:.0%})")

H_PAGE = yb + 28.5
xml = ('<mxfile host="fig1_drawio.py"><diagram id="fig1" name="Figure 1"><mxGraphModel dx="0" dy="0" grid="0" gridSize="5" '
       f'guides="1" tooltips="1" connect="0" arrows="0" fold="1" page="1" pageScale="1" pageWidth="{W:.0f}" pageHeight="{H_PAGE:.0f}" '
       'math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
open(sys.argv[1], "w", encoding="utf-8").write(xml)
print(f"wrote {sys.argv[1]}: {len(cells)} cells, page {W:.0f}x{H_PAGE:.0f}px")
print("checks:", "none failed" if not checks else "\n  " + "\n  ".join(checks))
