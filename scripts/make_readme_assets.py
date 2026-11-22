"""Generate the animated README assets (pure SVG + CSS, no JS - GitHub renders them via <img>).

python scripts/make_readme_assets.py      # writes docs/assets/{hero,cli}-{light,dark}.svg
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
THEMES = {
    "light": {
        "bg": "#ffffff",
        "ink": "#101418",
        "mute": "#5b6573",
        "line": "#d9dde3",
        "soft": "#f4f5f7",
        "yel": "#ffd84a",
        "yelD": "#c99a00",
        "red": "#e5484d",
        "term": "#101418",
        "termInk": "#e8eaed",
    },
    "dark": {
        "bg": "#0d1117",
        "ink": "#e6edf3",
        "mute": "#9aa3af",
        "line": "#30363d",
        "soft": "#161b22",
        "yel": "#ffd84a",
        "yelD": "#ffd84a",
        "red": "#ff6b6b",
        "term": "#161b22",
        "termInk": "#e6edf3",
    },
}
SANS = "Manrope, 'Segoe UI', -apple-system, BlinkMacSystemFont, Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


# ----------------------------------------------------------------------------- fingerprint geometry
def fingerprint_paths(R: float, seed: int = 9, dens: float = 9.0) -> list[str]:
    """Same ridge model as the explainer film: loops open at the bottom, tilted, clipped to a fingertip."""
    r = random.Random(seed)
    N = max(7, round(R / dens))
    tilt = -0.12 + r.random() * 0.08
    cut_y = R * 0.26

    def inside(x, y):
        return (x * x) / (R * 0.9) ** 2 + (y * y) / (R * 1.18) ** 2 <= 1

    def rot(x, y):
        return x * math.cos(tilt) - y * math.sin(tilt), x * math.sin(tilt) + y * math.cos(tilt)

    def to_d(pts):
        segs, cur = [], []
        for p in pts:
            if p is None:
                if len(cur) > 2:
                    segs.append(cur)
                cur = []
            else:
                cur.append(p)
        if len(cur) > 2:
            segs.append(cur)
        return ["M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in s) for s in segs]

    out: list[str] = []
    for i in range(N):
        f = i / (N - 1)
        rad = R * (0.08 + 1.05 * f)
        gap = 0.55 + 1.5 * f**1.2 + (r.random() - 0.5) * 0.25
        a0, span = math.pi / 2 + gap / 2, math.pi * 2 - gap
        brk = a0 + span * (0.15 + r.random() * 0.7) if r.random() < 0.6 else None
        wob, cy = r.random() * 6.28, -R * 0.12
        n = max(30, round(rad * 0.6))
        pts = []
        for k in range(n + 1):
            a = a0 + span * k / n
            if brk is not None and abs(a - brk) < 0.06:
                pts.append(None)
                continue
            sx = 0.8 * (1 + 0.04 * math.sin(3 * a + wob))
            sy = 1.18 if math.sin(a) < 0 else 0.95
            x, y = rad * math.cos(a) * sx, cy + rad * math.sin(a) * sy
            end = min(k, n - k) / n
            if end < 0.08:
                y += (0.08 - end) * rad * 0.9
            if y > cut_y + rad * 0.02 and rad > R * 0.25:
                pts.append(None)
                continue
            x, y = rot(x, y)
            pts.append((x, y) if inside(x, y) else None)
        out += to_d(pts)
    M = max(3, round(N * 0.36))
    for j in range(M):
        yb, bend, wob = R * (0.36 + 0.86 * j / M), R * (0.10 + 0.05 * j / M), r.random() * 6.28
        pts = []
        for k in range(41):
            u = k / 40 * 2 - 1
            x, y = u * R * 0.95, yb - bend * (1 - u * u) + math.sin(u * 3 + wob) * R * 0.015
            x, y = rot(x, y)
            pts.append((x, y) if inside(x, y) else None)
        if r.random() < 0.5:
            b = int(40 * (0.2 + r.random() * 0.6))
            pts[b] = pts[b + 1] = None
        out += to_d(pts)
    return out


def hexstr(seed: int, n: int) -> str:
    r = random.Random(seed)
    return "".join(r.choice("0123456789abcdef") for _ in range(n))


# ----------------------------------------------------------------------------- hero (fingerprint + name)
def hero(t: dict) -> str:
    W, H, CYC = 960, 330, 9.0  # seconds per loop
    paths = fingerprint_paths(118, seed=9)
    n = len(paths)
    ridges = []
    for i, d in enumerate(paths):
        delay = 0.15 + i / n * 2.4
        ridges.append(f'<path d="{d}" pathLength="1" class="r" style="animation-delay:{delay:.2f}s"/>')
    fp = "sha256:9d03ca7459476ddbe1e4…8a8096"
    chips = ["vision", "audio", "radar", "lidar", "IMU", "text", "time series"]

    def chip_row(items, x, y, cls, delay0):
        out, cx = [], x
        for k, s in enumerate(items):
            w = 16 + len(s) * 9.6
            out.append(
                f'<g class="chip {cls}" style="animation-delay:{delay0 + k * 0.08:.2f}s">'
                f'<rect x="{cx:.0f}" y="{y}" width="{w:.0f}" height="30" rx="15"/>'
                f'<text x="{cx + w / 2:.0f}" y="{y + 20}" text-anchor="middle">{escape(s)}</text></g>'
            )
            cx += w + 8
        return "".join(out)

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
 aria-label="mlfingerprint: a fingerprint draws itself from file hashes. A lockfile for your training data.">
<style>
  .bg{{fill:{t["bg"]}}}
  .r{{fill:none;stroke:{t["ink"]};stroke-width:2.6;stroke-linecap:round;stroke-dasharray:1 1;stroke-dashoffset:1;
      animation:draw {CYC}s cubic-bezier(.3,.6,.3,1) infinite}}
  @keyframes draw{{0%{{stroke-dashoffset:1;stroke:{t["ink"]}}}38%{{stroke-dashoffset:0;stroke:{t["ink"]}}}
      44%{{stroke:{t["yelD"]}}}52%{{stroke:{t["ink"]}}}88%{{stroke-dashoffset:0;opacity:1}}96%,100%{{stroke-dashoffset:0;opacity:0}}}}
  .name{{font:800 64px {SANS};fill:{t["ink"]}}}
  .tag{{font:600 26px {SANS};fill:{t["mute"]}}}
  .hash rect{{fill:{t["yel"]}}} .hash text{{font:600 19px {MONO};fill:#101418}}
  .hash{{clip-path:inset(0 100% 0 0);animation:type {CYC}s steps(36,end) infinite}}
  @keyframes type{{0%,30%{{clip-path:inset(0 100% 0 0);opacity:1}}45%,90%{{clip-path:inset(0 0 0 0);opacity:1}}97%,100%{{clip-path:inset(0 0 0 0);opacity:0}}}}
  .lbl{{font:600 15px {SANS};fill:{t["mute"]};opacity:0;animation:fade {CYC}s infinite;animation-delay:0s}}
  .chip{{opacity:0;animation:chip {CYC}s ease-out infinite}}
  .chip rect{{fill:{t["soft"]};stroke:{t["line"]}}} .chip text{{font:600 15px {SANS};fill:{t["ink"]}}}
  .chip.st rect{{fill:none}}
  @keyframes chip{{0%,46%{{opacity:0;transform:translateY(6px)}}52%,90%{{opacity:1;transform:none}}97%,100%{{opacity:0}}}}
  @keyframes fade{{0%,36%{{opacity:0}}42%,90%{{opacity:1}}97%,100%{{opacity:0}}}}
  @media (prefers-reduced-motion:reduce){{.r,.hash,.chip,.lbl{{animation:none;stroke-dashoffset:0;clip-path:none;opacity:1}}}}
</style>
<rect class="bg" width="{W}" height="{H}" rx="18"/>
<g transform="translate(190 178)">{"".join(ridges)}</g>
<text class="name" x="370" y="118">mlfingerprint</text>
<text class="tag" x="372" y="160">A lockfile for your training data.</text>
<text class="lbl" x="372" y="204">dataset fingerprint</text>
<g class="hash"><rect x="370" y="214" width="{len(fp) * 11.6 + 28:.0f}" height="38" rx="19"/>
<text x="384" y="239">{fp}</text></g>
{chip_row(chips, 370, 270, "md", 0)}
</svg>
"""


# ----------------------------------------------------------------------------- CLI demo (terminal)
CLI = [
    ("cmd", "pip install mlfingerprint", 0.0),
    ("out", "Successfully installed mlfingerprint-0.1.0   (zero dependencies)", 1.4),
    ("cmd", 'fingerprint commit s3://raw/drives -n drives -m "batch 05-10" --probe', 2.2),
    ("out", "drives-v1.2.0  (minor: +42 files since drives-v1.1.1)", 4.4),
    ("key", "  fingerprint : sha256:9d03ca7459476ddbe1e4b0c3…8a8096", 4.7),
    ("out", "  splits      : train=80%  val=10%  test=10%   group_by=auto:recording", 5.0),
    ("out", "  modalities  : image=240k  radar=240k  imu=24 (2.1h)  audio=24 (2.0h)", 5.3),
    ("warn", "  ! 3 augmented file(s) in val/test - evaluate on real data", 5.6),
    ("cmd", "fingerprint tag prod 1.2.0 && fingerprint verify prod --sample 0.05", 6.5),
    ("bad", "verify drives-v1.2.0: FAILED  (64,209/64,210 files intact)", 8.6),
    ("bad", "  modified : val/audio/b1_0.wav        exit 1 → training never starts", 8.9),
]


def cli(t: dict) -> str:
    W, CYC, LH, top = 880, 13.0, 30, 70
    H = top + LH * len(CLI) + 34
    colors = {"cmd": t["termInk"], "out": "#9aa3af", "key": "#ffd84a", "warn": "#ffb020", "bad": "#ff6b6b"}
    rows = []
    for i, (kind, text, at) in enumerate(CLI):
        y = top + i * LH
        start = at / CYC * 100
        if kind == "cmd":
            n = len(text) + 2
            dur = min(1.6, 0.035 * n)
            end = (at + dur) / CYC * 100
            rows.append(
                f'<text x="32" y="{y}" class="t" fill="{colors[kind]}" '
                f'style="clip-path:inset(0 100% 0 0);animation:type{i} {CYC}s steps({n},end) infinite">'
                f'<tspan fill="#ffd84a">$ </tspan>{escape(text)}</text>'
                f"<style>@keyframes type{i}{{0%,{start:.2f}%{{clip-path:inset(0 100% 0 0)}}"
                f"{end:.2f}%,94%{{clip-path:inset(0 0 0 0)}}100%{{clip-path:inset(0 0 0 0)}}}}</style>"
            )
        else:
            rows.append(
                f'<text x="32" y="{y}" class="t" fill="{colors[kind]}" style="opacity:0;animation:l{i} {CYC}s infinite">'
                f"{escape(text)}</text>"
                f"<style>@keyframes l{i}{{0%,{start:.2f}%{{opacity:0}}{start + 1.2:.2f}%,94%{{opacity:1}}100%{{opacity:0}}}}</style>"
            )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
 aria-label="Terminal demo: install mlfingerprint, commit an S3 dataset, tag it prod, verify catches a modified file.">
<style>
  .t{{font:500 17px {MONO};white-space:pre}}
  @media (prefers-reduced-motion:reduce){{text{{animation:none!important;clip-path:none!important;opacity:1!important}}}}
</style>
<rect width="{W}" height="{H}" rx="14" fill="{t["term"]}"/>
<rect width="{W}" height="40" rx="14" fill="{t["term"]}"/><rect y="26" width="{W}" height="14" fill="{t["term"]}"/>
<line x1="0" y1="40" x2="{W}" y2="40" stroke="#ffffff" stroke-opacity=".08"/>
<circle cx="26" cy="20" r="6.5" fill="#ff5f57"/><circle cx="48" cy="20" r="6.5" fill="#febc2e"/><circle cx="70" cy="20" r="6.5" fill="#28c840"/>
<text x="{W / 2}" y="25" text-anchor="middle" style="font:600 14px {SANS}" fill="#9aa3af">fingerprint · zsh</text>
{"".join(rows)}
</svg>
"""


# ----------------------------------------------------------------------------- concept explainer
def concept(t: dict) -> str:
    """How it works, in five beats: hash -> split -> seal -> lock -> verify. Loops every CYC seconds."""
    W, H, CYC = 1000, 500, 17.0
    css: list[str] = []
    n_anim = [0]

    def show(at: float, extra_from: str = "", extra_to: str = "", dur: float = 0.45) -> str:
        """Hidden until `at` seconds, then fades (and optionally moves) in and stays until the loop fades out."""
        n_anim[0] += 1
        name = f"a{n_anim[0]}"
        a, b = at / CYC * 100, (at + dur) / CYC * 100
        css.append(f"@keyframes {name}{{0%,{a:.2f}%{{opacity:0;{extra_from}}}{b:.2f}%,100%{{opacity:1;{extra_to}}}}}")
        return f'style="opacity:0;animation:{name} {CYC}s infinite"'

    def fly(at: float, dx: float, dy: float, dur: float = 0.55) -> str:
        """A chip that appears at its origin and travels by (dx, dy), then disappears."""
        n_anim[0] += 1
        name = f"a{n_anim[0]}"
        a, m, b, e = (at / CYC * 100, (at + 0.08) / CYC * 100, (at + dur) / CYC * 100, (at + dur + 0.1) / CYC * 100)
        css.append(
            f"@keyframes {name}{{0%,{a:.2f}%{{opacity:0;transform:translate(0,0)}}{m:.2f}%{{opacity:1;transform:translate(0,0)}}"
            f"{b:.2f}%{{opacity:1;transform:translate({dx:.0f}px,{dy:.0f}px)}}{e:.2f}%,100%{{opacity:0;transform:translate({dx:.0f}px,{dy:.0f}px)}}}}"
        )
        return f'style="opacity:0;animation:{name} {CYC}s infinite"'

    def sweep(at: float, x0: float, x1: float, dur: float) -> str:
        n_anim[0] += 1
        name = f"a{n_anim[0]}"
        a, b, e = at / CYC * 100, (at + dur) / CYC * 100, (at + dur + 0.15) / CYC * 100
        css.append(
            f"@keyframes {name}{{0%,{a:.2f}%{{opacity:0;transform:translate({x0:.0f}px,0)}}{a + 0.3:.2f}%{{opacity:1;transform:translate({x0:.0f}px,0)}}"
            f"{b:.2f}%{{opacity:1;transform:translate({x1:.0f}px,0)}}{e:.2f}%,100%{{opacity:0;transform:translate({x1:.0f}px,0)}}}}"
        )
        return f'style="opacity:0;animation:{name} {CYC}s infinite"'

    def step(at: float, until: float) -> str:
        n_anim[0] += 1
        name = f"a{n_anim[0]}"
        a, b, c, d = at / CYC * 100, (at + 0.3) / CYC * 100, until / CYC * 100, (until + 0.3) / CYC * 100
        css.append(f"@keyframes {name}{{0%,{a:.2f}%{{opacity:.32}}{b:.2f}%,{c:.2f}%{{opacity:1}}{d:.2f}%,100%{{opacity:.32}}}}")
        return f'style="opacity:.32;animation:{name} {CYC}s infinite"'

    ink, mute, line, soft, yel, red = t["ink"], t["mute"], t["line"], t["soft"], t["yel"], t["red"]
    split_col = {"train": ink, "val": "#8b97aa", "test": yel}
    runs = [("run_01", "train"), ("run_02", "val"), ("run_03", "test")]
    kinds = [("cam", ".jpg"), ("radar", ".npy"), ("imu", ".csv")]
    out: list[str] = []

    # panel frames
    def panel(x, y, w, h, title, sub):
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{t["bg"]}" stroke="{line}"/>'
            f'<text x="{x + 16}" y="{y + 28}" class="h">{title}</text><text x="{x + 16}" y="{y + 48}" class="s">{sub}</text>'
        )

    out.append(panel(20, 20, 290, 370, "Your data", "stays where it is: S3 · Azure · GCS · disk"))
    out.append(panel(340, 20, 300, 370, "index.jsonl.gz", "one row per file: path · sha256 · split"))
    out.append(panel(670, 20, 310, 205, "manifest.json", "a few KB · written once"))
    out.append(panel(670, 245, 310, 145, "Your model", "stores the lock it trained on"))

    # storage tiles
    tiles = {}
    for r, (run, sp) in enumerate(runs):
        y = 104 + r * 92
        out.append(f'<text x="36" y="{y - 12}" class="s">{run}</text>')
        for k, (kind, ext) in enumerate(kinds):
            x = 36 + k * 88
            tiles[(r, k)] = (x, y)
            out.append(f'<rect x="{x}" y="{y}" width="76" height="58" rx="9" fill="{soft}" stroke="{line}"/>')
            cx, cy = x + 38, y + 24
            if kind == "cam":
                out.append(
                    f'<rect x="{cx - 15}" y="{cy - 11}" width="30" height="22" rx="4" fill="none" stroke="{ink}" stroke-width="2"/>'
                    f'<circle cx="{cx}" cy="{cy}" r="5.5" fill="none" stroke="{ink}" stroke-width="2"/>'
                )
            elif kind == "radar":
                out.append(
                    "".join(
                        f'<rect x="{cx - 14 + i * 10}" y="{cy - 12 + j * 9}" width="8" height="7" rx="1.5" fill="{yel if (i + j) % 3 == 0 else line}"/>'
                        for i in range(3)
                        for j in range(3)
                    )
                )
            else:
                out.append(f'<path d="M{cx - 16},{cy} q4,-10 8,0 t8,0 t8,0 t8,0" fill="none" stroke="{ink}" stroke-width="2"/>')
            out.append(f'<text x="{cx}" y="{y + 50}" class="xs" text-anchor="middle">{kind}{ext}</text>')

    # beat 1: scan + hash chips fly into index rows
    out.append(f'<rect x="0" y="62" width="4" height="318" rx="2" fill="{yel}" {sweep(0.4, 30, 300, 2.9)}/>')
    order = [(r, k) for r in range(3) for k in range(3)]
    for i, (r, k) in enumerate(order):
        x, y = tiles[(r, k)]
        ry = 86 + i * 31
        at = 0.6 + i * 0.3
        h = hexstr(100 + i, 6)
        out.append(
            f'<g {fly(at, 362 - x, ry - 10 - y)}><rect x="{x}" y="{y + 10}" width="76" height="20" rx="10" fill="{yel}"/>'
            f'<text x="{x + 38}" y="{y + 24}" class="xs" text-anchor="middle" fill="#101418">{h}…</text></g>'
        )
        run, sp = runs[r]
        kind, ext = kinds[k]
        out.append(
            f'<g {show(at + 0.55)}><text x="358" y="{ry + 4}" class="m">{run}/{kind}{ext}</text>'
            f'<text x="512" y="{ry + 4}" class="m" fill="{mute}">{h}…</text></g>'
        )
        # beat 2: split chip per row
        c = split_col[sp]
        out.append(
            f'<g {show(4.4 + r * 0.25)}><rect x="572" y="{ry - 9}" width="54" height="18" rx="9" fill="{c}"/>'
            f'<text x="599" y="{ry + 4}" class="xs" text-anchor="middle" fill="{"#101418" if sp == "test" else t["bg"] if sp == "train" else "#ffffff"}">{sp}</text></g>'
        )

    # beat 2: whole recordings outlined in their split colour
    for r, (run, sp) in enumerate(runs):
        y = 104 + r * 92
        out.append(
            f'<rect x="28" y="{y - 30}" width="274" height="94" rx="12" fill="none" stroke="{split_col[sp]}" stroke-width="3" {show(4.4 + r * 0.25)}/>'
        )
    out.append(f'<text x="165" y="378" class="s" text-anchor="middle" {show(5.2)}>a recording never straddles splits</text>')

    # beat 3: rows converge into the manifest, fingerprint appears, sealed
    for i in range(0, 9, 2):
        ry = 86 + i * 31
        out.append(f'<path d="M632,{ry} C660,{ry} 650,150 690,150" fill="none" stroke="{line}" stroke-width="1.6" {show(6.6 + i * 0.05)}/>')
    mlines = [("version", "drives-v1.2.0", 6.9), ("splits", "train 80 · val 10 · test 10", 7.2), ("lineage", "from v1.1.1, +42 files", 7.5)]
    for j, (k, v, at) in enumerate(mlines):
        out.append(
            f'<g {show(at)}><text x="690" y="{92 + j * 24}" class="m" fill="{mute}">{k}</text><text x="756" y="{92 + j * 24}" class="m">{v}</text></g>'
        )
    out.append(
        f'<g {show(7.9, "transform:translateY(6px)", "transform:none")}><rect x="686" y="160" width="278" height="30" rx="15" fill="{yel}"/>'
        f'<text x="825" y="180" class="m" text-anchor="middle" fill="#101418">fingerprint sha256:9d03ca74…</text></g>'
    )
    out.append(
        f'<g class="seal" {show(8.4, "transform:scale(1.25)", "transform:none")}>'
        f'<circle cx="930" cy="62" r="24" fill="none" stroke="{ink}" stroke-width="2.5"/>'
        f'<text x="930" y="66" class="xs" text-anchor="middle" font-weight="800">SEALED</text></g>'
    )

    # beat 4: model stores the lock
    out.append(f'<path d="M825,192 L825,246" fill="none" stroke="{ink}" stroke-width="2" stroke-dasharray="5 5" {show(9.1)}/>')
    out.append(
        f'<g {show(9.4)}><text x="690" y="324" class="b">model-v4 · mAP 0.61</text>'
        f'<text x="690" y="350" class="m" fill="{mute}">trained on drives-v1.2.0</text>'
        f'<text x="690" y="374" class="m">sha256:9d03ca74…  ✓ reproducible</text></g>'
    )

    # beat 5: someone edits a file -> verify re-hashes -> exact file named
    ex, ey = tiles[(1, 1)]
    out.append(f'<rect x="{ex}" y="{ey}" width="76" height="58" rx="9" fill="none" stroke="{red}" stroke-width="3.5" {show(11.0)}/>')
    out.append(f'<text x="{ex + 38}" y="{ey - 8}" class="xs" text-anchor="middle" fill="{red}" font-weight="700" {show(11.0)}>edited later</text>')
    out.append(f'<rect x="0" y="66" width="4" height="300" rx="2" fill="{red}" {sweep(11.6, 348, 636, 1.4)}/>')
    vy = 86 + 4 * 31
    out.append(f'<rect x="348" y="{vy - 13}" width="284" height="24" rx="6" fill="none" stroke="{red}" stroke-width="2.5" {show(12.4)}/>')
    out.append(
        f'<g {show(13.0, "transform:translateY(6px)", "transform:none")}><rect x="250" y="402" width="500" height="34" rx="17" fill="{red}"/>'
        f'<text x="500" y="424" class="m" text-anchor="middle" fill="#ffffff">verify: FAILED · run_02/radar.npy changed → exit 1</text></g>'
    )

    # step bar
    steps = [
        ("Hash every file", 0.3, 4.2),
        ("Split whole recordings", 4.3, 6.4),
        ("Seal a manifest", 6.5, 8.9),
        ("Model keeps the lock", 9.0, 10.8),
        ("Verify catches drift", 10.9, 16.2),
    ]
    sx = 20
    for label, a, b in steps:
        w = 22 + len(label) * 8.6
        out.append(f'<g {step(a, b)}><circle cx="{sx + 8}" cy="470" r="5" fill="{yel}"/><text x="{sx + 20}" y="475" class="st">{label}</text></g>')
        sx += w + 20

    a_out, b_out = (CYC - 0.7) / CYC * 100, (CYC - 0.1) / CYC * 100
    css.append(f"@keyframes all{{0%{{opacity:0}}1.5%,{a_out:.2f}%{{opacity:1}}{b_out:.2f}%,100%{{opacity:0}}}}")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
 aria-label="How mlfingerprint works: every file is hashed into an index, whole recordings are assigned to splits, a small manifest with one fingerprint is sealed, the model stores that fingerprint, and verify names the exact file that changed later.">
<style>
  .h{{font:800 18px {SANS}}} .s{{font:500 13px {SANS}}} .xs{{font:600 11px {MONO}}}
  .m{{font:500 13px {MONO}}} .b{{font:800 17px {SANS}}} .st{{font:700 14px {SANS}}}
  .h:not([fill]),.m:not([fill]),.b:not([fill]),.st:not([fill]){{fill:{ink}}} .s:not([fill]),.xs:not([fill]){{fill:{mute}}}
  .all{{animation:all {CYC}s infinite}} .seal{{transform-origin:930px 62px}}
  {"".join(css)}
  @media (prefers-reduced-motion:reduce){{*{{animation:none!important;opacity:1!important}}}}
</style>
<rect width="{W}" height="{H}" rx="18" fill="{t["bg"]}"/>
<g class="all">{"".join(out)}</g>
</svg>
"""


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        (OUT / f"hero-{name}.svg").write_text(hero(t))
        (OUT / f"cli-{name}.svg").write_text(cli(t))
        (OUT / f"concept-{name}.svg").write_text(concept(t))
    for p in sorted(OUT.glob("*.svg")):
        print(f"{p.relative_to(OUT.parent.parent)}  {p.stat().st_size / 1024:.1f} KB")
