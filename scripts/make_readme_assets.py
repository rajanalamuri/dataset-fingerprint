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


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        (OUT / f"hero-{name}.svg").write_text(hero(t))
        (OUT / f"cli-{name}.svg").write_text(cli(t))
    for p in sorted(OUT.glob("*.svg")):
        print(f"{p.relative_to(OUT.parent.parent)}  {p.stat().st_size / 1024:.1f} KB")
