"""Generate a self-contained HTML report from a FeatureScope screen JSON.

Usage: python report_html.py screen_sentiment.json featurescope_report.html
Palette (validated): driver-like #2a78d6 / thermometer #1baf7a / indeterminate #eda100,
shapes as secondary encoding, direct labels + table (contrast-WARN relief).
"""
import json, sys, html

VC = {"not_ruled_out": ("#2a78d6", "driver-like", "circle"),
      "ruled_out": ("#1baf7a", "thermometer", "square"),
      "indeterminate": ("#eda100", "indeterminate", "triangle")}

def mark(x, y, shape, color, label):
    ring = f'stroke="#ffffff" stroke-width="2"'
    if shape == "circle":  m = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" {ring}/>'
    elif shape == "square": m = f'<rect x="{x-5.5:.1f}" y="{y-5.5:.1f}" width="11" height="11" fill="{color}" {ring}/>'
    else: m = f'<polygon points="{x:.1f},{y-7:.1f} {x-6:.1f},{y+5:.1f} {x+6:.1f},{y+5:.1f}" fill="{color}" {ring}/>'
    return m + f'<text x="{x+9:.1f}" y="{y+4:.1f}" font-size="11" fill="#555">{label}</text>'

def main(src, dst):
    d = json.load(open(src))
    rows = d.get("results") or d.get("features") or []
    if not rows and "drivers" in d:  # tolerate schema variants
        rows = d.get("rows", [])
    concept = d.get("concept", "?")
    n = len(rows); nd = sum(1 for r in rows if r.get("verdict") == "not_ruled_out")
    nt = sum(1 for r in rows if r.get("verdict") == "ruled_out")

    # scatter: x=read, y=peak dose effect
    W,H,P = 660, 420, 52
    xs = [r["read"] for r in rows]; ys = [max(r["dose"]) for r in rows]
    x0,x1 = min(xs+[0])-0.1, max(xs+[0])+0.1; y0,y1 = min(ys+[0])-0.1, max(ys+[0])+0.1
    fx = lambda v:(v-x0)/(x1-x0)*(W-2*P)+P; fy = lambda v:H-P-(v-y0)/(y1-y0)*(H-2*P)
    pts = "".join(f'<g>{mark(fx(r["read"]), fy(max(r["dose"])), VC[r["verdict"]][2], VC[r["verdict"]][0], r["feature"])}'
                  f'<title>feat {r["feature"]}  read {r["read"]:.2f}  peak {max(r["dose"]):.2f}  z {r.get("z",0):.1f}  {VC[r["verdict"]][1]}</title></g>'
                  for r in rows)
    grid = (f'<line x1="{fx(0)}" y1="{P}" x2="{fx(0)}" y2="{H-P}" stroke="#ddd"/>' if x0<0<x1 else "") + \
           (f'<line x1="{P}" y1="{fy(0)}" x2="{W-P}" y2="{fy(0)}" stroke="#ddd"/>' if y0<0<y1 else "")
    scatter = (f'<svg viewBox="0 0 {W} {H}" style="max-width:100%">{grid}'
               f'<text x="{W/2}" y="{H-10}" text-anchor="middle" font-size="12" fill="#666">read score (correlation with concept)</text>'
               f'<text x="14" y="{H/2}" font-size="12" fill="#666" transform="rotate(-90 14 {H/2})">peak steering effect (cause)</text>{pts}</svg>')

    # dose-response small multiples (shared y)
    dy1 = max(max(r["dose"]) for r in rows)*1.15 or 1; dy0 = min(min(r["dose"]) for r in rows+[{"dose":[0]}])
    mw,mh,mp = 150,110,26
    def mini(r):
        c = VC[r["verdict"]][0]
        gx = lambda i: mp+i*(mw-2*mp)/2; gy = lambda v: mh-mp-(v-dy0)/(dy1-dy0)*(mh-2*mp)
        pl = " ".join(f"{gx(i):.1f},{gy(v):.1f}" for i,v in enumerate(r["dose"]))
        return (f'<svg viewBox="0 0 {mw} {mh}" width="{mw}"><title>feat {r["feature"]} dose {r["dose"]}</title>'
                f'<text x="{mw/2}" y="13" text-anchor="middle" font-size="11" fill="#333">{r["feature"]}</text>'
                f'<line x1="{mp}" y1="{gy(0)}" x2="{mw-mp}" y2="{gy(0)}" stroke="#e5e5e5"/>'
                f'<polyline points="{pl}" fill="none" stroke="{c}" stroke-width="2"/>'
                + "".join(f'<circle cx="{gx(i)}" cy="{gy(v)}" r="3" fill="{c}"/>' for i,v in enumerate(r["dose"]))
                + f'<text x="{mw/2}" y="{mh-4}" text-anchor="middle" font-size="9" fill="#999">dose 1x 2x 4x</text></svg>')
    minis = "".join(mini(r) for r in rows)

    trs = "".join(
        f'<tr><td>{r["feature"]}</td><td class="fire">{html.escape(str(r.get("fires_on") or ""))[:60]}</td>'
        f'<td>{r["read"]:.2f}</td><td>{" / ".join(f"{v:.2f}" for v in r["dose"])}</td><td>{r.get("z",float("nan")):.1f}</td>'
        f'<td><span class="chip" style="background:{VC[r["verdict"]][0]}"></span>{VC[r["verdict"]][1]}</td></tr>'
        for r in sorted(rows, key=lambda r:-max(r["dose"])))

    legend = "".join(f'<span style="margin-right:18px"><span class="chip" style="background:{c}"></span>{name}</span>'
                     for c,name,_ in VC.values())
    page = f"""<meta charset="utf-8"><title>Lens4SAE report — {concept}</title>
<style>body{{font:14px/1.5 -apple-system,system-ui,sans-serif;color:#1a1a19;max-width:900px;margin:32px auto;padding:0 16px}}
h1{{font-size:20px}} .tiles{{display:flex;gap:14px;margin:18px 0}} .tile{{border:1px solid #e5e5e5;border-radius:8px;padding:12px 18px}}
.tile b{{font-size:26px;display:block}} .tile span{{color:#666;font-size:12px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{padding:6px 8px;border-bottom:1px solid #eee;text-align:left}}
.chip{{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}} .fire{{color:#666;font-style:italic}}
.minis{{display:flex;flex-wrap:wrap;gap:6px}} .note{{color:#666;font-size:12px;margin-top:22px}}</style>
<h1>Lens4SAE screening report — concept: {concept}</h1>
<div class="tiles"><div class="tile"><b>{n}</b><span>features screened</span></div>
<div class="tile"><b style="color:#2a78d6">{nd}</b><span>driver-like (not ruled out)</span></div>
<div class="tile"><b style="color:#1baf7a">{nt}</b><span>thermometers (ruled out)</span></div></div>
<div>{legend}</div>
<h2 style="font-size:16px">Read vs cause — the founding picture</h2>
<p style="color:#666;font-size:13px">Every point is one SAE feature. Right = correlates with the concept; up = steering it moves behavior. Readable features that stay low are thermometers.</p>
{scatter}
<h2 style="font-size:16px">Dose-response per feature</h2><div class="minis">{minis}</div>
<h2 style="font-size:16px">Verdict table</h2>
<table><tr><th>feature</th><th>fires on</th><th>read</th><th>dose 1x/2x/4x</th><th>z</th><th>verdict</th></tr>{trs}</table>
<p class="note">One-sided negative screen: "driver-like" is NOT a safety certificate. Generated from {html.escape(src)}.</p>"""
    open(dst, "w").write(page)
    print(f"wrote {dst} ({n} features)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "featurescope_report.html")
