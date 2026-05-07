"""
report_generator.py
Generates a styled HTML report for the Ridge Regression federated computation,
matching the visual style of the CSV Analyzer report.
"""
from typing import List, Dict, Any


# ── Site colour palette (matches CSV Analyzer) ──────────────────────────────
_SITE_COLORS = [
    "rgba(99,102,241,0.65)",  "rgba(20,184,166,0.65)",
    "rgba(245,158,11,0.65)",  "rgba(239,68,68,0.65)",
    "rgba(168,85,247,0.65)",  "rgba(34,197,94,0.65)",
]
_SITE_COLORS_SOLID = [c.replace("0.65", "1.0") for c in _SITE_COLORS]
_SITE_PILL_BG    = [c.replace("0.65", "0.12") for c in _SITE_COLORS]
_SITE_PILL_TEXT  = ["#3730a3","#0f766e","#92400e","#991b1b","#6b21a8","#166534"]


def _site_maps(all_sites):
    solid = {s: _SITE_COLORS_SOLID[i % len(_SITE_COLORS_SOLID)] for i, s in enumerate(all_sites)}
    pill_bg = {s: _SITE_PILL_BG[i % len(_SITE_PILL_BG)]         for i, s in enumerate(all_sites)}
    pill_txt = {s: _SITE_PILL_TEXT[i % len(_SITE_PILL_TEXT)]    for i, s in enumerate(all_sites)}
    return solid, pill_bg, pill_txt


def _pill(site, solid_map, bg_map, txt_map):
    return (f'<span style="display:inline-flex;align-items:center;gap:.35rem;'
            f'background:{bg_map[site]};color:{txt_map[site]};border:1px solid {solid_map[site]};'
            f'border-radius:999px;padding:.18rem .6rem;font-size:.75rem;font-weight:600">'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{solid_map[site]};'
            f'flex-shrink:0"></span>{site}</span>')


def _p_style(p):
    """Continuous green intensity for significant p-values; muted grey otherwise."""
    if p is None or p >= 0.05:
        return "color:var(--text3)"
    import math
    t = math.log(max(p, 1e-10) / 0.05) / math.log(0.001 / 0.05)  # 0 at p=0.05, 1 at p=0.001
    t = max(0.0, min(1.0, t))
    alpha  = round(0.08 + t * 0.47, 3)
    weight = "700" if t > 0.55 else "600" if t > 0.2 else "400"
    text   = "#065f46" if alpha > 0.3 else "#047857"
    return f"background:rgba(16,185,129,{alpha});color:{text};font-weight:{weight}"


def generate_report_html(
    results: List[Dict[str, Any]],
    computation_parameters: Dict[str, Any],
    user_name: str = None,
    user_id: str = None,
) -> str:
    """
    Main entry point. Takes the agg_results list (same format as global_regression_result.json)
    and returns a complete, styled HTML string.
    """
    rois = [r["ROI"] for r in results]
    all_sites = sorted(results[0]["local_stats"].keys()) if results else []
    covariates = results[0]["global_stats"]["covariate_labels"] if results else []
    solid_map, bg_map, txt_map = _site_maps(all_sites)

    body = (
        _build_header(rois, all_sites, covariates, computation_parameters, solid_map, bg_map, txt_map, user_name, user_id)
        + _build_summary_section(results, all_sites, solid_map, bg_map, txt_map)
        + _build_coefficients_section(results, all_sites, solid_map, bg_map, txt_map)
        + _build_pvalue_section(results, all_sites, solid_map, bg_map, txt_map)
    )
    nav_titles = ["Study Overview", "Global Summary", "Coefficients", "P-values & Significance"]
    return _wrap_page(body, nav_titles=nav_titles)


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────

def _build_header(rois, all_sites, covariates, params, solid_map, bg_map, txt_map,
                  user_name=None, user_id=None):
    n_sites = len(all_sites)
    n_rois  = len(rois)
    n_cov   = len(covariates)
    lam     = params.get("Lambda", "—")

    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:.45rem;font-size:.83rem;color:var(--legend-color)">'
        f'<div style="width:11px;height:11px;border-radius:3px;background:{solid_map[s]}"></div>{s}</div>'
        for s in all_sites
    )

    roi_pills = "".join(
        f'<span style="background:var(--chip-bg);border:1px solid var(--border);border-radius:999px;'
        f'padding:.2rem .65rem;font-size:.76rem;color:var(--chip-color)">{r}</span>'
        for r in rois
    )

    return f'''<div class="page-header">
  <h1>Single Round Ridge Regression</h1>
  <p>Federated ridge regression across {n_sites} site{"s" if n_sites!=1 else ""}</p>
  <div class="chips">
    <div class="chip">Sites <b>{n_sites}</b></div>
    <div class="chip">Outcomes (ROIs) <b>{n_rois}</b></div>
    <div class="chip">Covariates <b>{n_cov}</b></div>
    <div class="chip">λ (Ridge) <b>{lam}</b></div>
  </div>
  <div style="margin-top:.9rem;display:flex;flex-wrap:wrap;gap:.4rem">{roi_pills}</div>
  <div class="site-legend" style="margin-top:.75rem">{legend}</div>
</div>'''


def _build_summary_section(results, all_sites, solid_map, bg_map, txt_map):
    solid, bg, txt = solid_map, bg_map, txt_map
    cards = ""
    for result in results:
        roi = result["ROI"]
        gs  = result["global_stats"]
        r2g = gs.get("R Squared", "—")
        dof = gs.get("Degrees of Freedom", "—")
        sse = gs.get("Sum Square of Errors", "—")

        # per-site R² rows
        site_rows = ""
        for site in all_sites:
            ls  = result["local_stats"].get(site, {})
            r2s = ls.get("R Squared")
            site_rows += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:.3rem 0;border-bottom:1px solid var(--border);font-size:.8rem">'
                f'<span>{_pill(site, solid, bg, txt)}</span>'
                f'<span style="font-family:monospace;color:var(--td-mono)">'
                + (f'{r2s:.4f}' if isinstance(r2s, float) else '—') + '</span></div>'
            )

        r2_bar = ""
        if isinstance(r2g, float):
            pct = min(max(r2g * 100, 0), 100)
            clr = "#059669" if r2g >= 0.5 else "#d97706" if r2g >= 0.2 else "#dc2626"
            r2_bar = (
                f'<div style="margin-top:.5rem;background:var(--bg3);border-radius:999px;height:6px">'
                f'<div style="width:{pct:.1f}%;background:{clr};height:6px;border-radius:999px"></div></div>'
                f'<div style="font-size:.7rem;color:var(--text3);margin-top:.25rem">{pct:.1f}% variance explained (global)</div>'
            )

        cards += f'''<div class="stat-card">
  <div class="stat-card-header">
    <div class="stat-card-title">{roi}</div>
    <span class="badge {'badge-green' if isinstance(r2g,float) and r2g>=0.5 else 'badge-yellow' if isinstance(r2g,float) and r2g>=0.2 else 'badge-red'}">
      R²&nbsp;{'%.4f'%r2g if isinstance(r2g,float) else '—'}
    </span>
  </div>
  <div style="padding:.85rem 1rem">
    {r2_bar}
    <div style="margin-top:.75rem;font-size:.72rem;font-weight:700;text-transform:uppercase;
                letter-spacing:.07em;color:var(--text3);margin-bottom:.35rem">R² by site</div>
    {site_rows}
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-top:.75rem">
      <div style="background:var(--bg3);border-radius:8px;padding:.6rem .8rem">
        <div style="font-size:.7rem;color:var(--text3)">Degrees of Freedom</div>
        <div style="font-family:monospace;font-size:.9rem;color:var(--td-mono);font-weight:600">{dof}</div>
      </div>
      <div style="background:var(--bg3);border-radius:8px;padding:.6rem .8rem">
        <div style="font-size:.7rem;color:var(--text3)">Global SSE</div>
        <div style="font-family:monospace;font-size:.9rem;color:var(--td-mono);font-weight:600">{'%.2f'%sse if isinstance(sse,float) else sse}</div>
      </div>
    </div>
  </div>
</div>'''

    return _section("Global Summary", f'<div class="stats-grid">{cards}</div>', "global-summary")


def _build_coefficients_section(results, all_sites, solid_map, bg_map, txt_map):
    solid, bg, txt = solid_map, bg_map, txt_map
    html = ""
    for result in results:
        roi  = result["ROI"]
        gs   = result["global_stats"]
        cov_labels = gs["covariate_labels"]

        # header row
        site_hdrs = "".join(f'<th colspan="2">{_pill(s, solid, bg, txt)}</th>' for s in all_sites)
        sub_hdrs  = "".join("<th>Coef</th><th>t</th>" for _ in all_sites)

        rows = ""
        for i, cov in enumerate(cov_labels):
            g_coef = gs["Coefficient"][i] if i < len(gs["Coefficient"]) else None
            g_t    = gs["t Stat"][i]      if i < len(gs["t Stat"])      else None
            g_p    = gs["P-value"][i]     if i < len(gs["P-value"])     else None
            g_p_str = ('%.4f' % g_p if isinstance(g_p, float) and g_p >= 0.0001 else ('<0.0001' if isinstance(g_p, float) else '—'))
            p_cell = f'<td style="font-family:monospace;font-size:.79rem;{_p_style(g_p)}">{g_p_str}</td>'

            site_cells = ""
            for site in all_sites:
                ls     = result["local_stats"].get(site, {})
                l_coef = ls.get("Coefficient", [])[i] if i < len(ls.get("Coefficient",[])) else None
                l_t    = ls.get("t Stat", [])[i]      if i < len(ls.get("t Stat",[]))      else None
                site_cells += (
                    f'<td style="font-family:monospace;font-size:.79rem;color:var(--td-mono)">'
                    f'{"%.4f"%l_coef if isinstance(l_coef,float) else "—"}</td>'
                    f'<td style="font-family:monospace;font-size:.79rem;color:var(--text3)">'
                    f'{"%.3f"%l_t if isinstance(l_t,float) else "—"}</td>'
                )

            rows += f'''<tr>
  <td style="font-weight:600;color:var(--text2);white-space:nowrap">{cov}</td>
  <td style="font-family:monospace;font-size:.82rem;color:var(--td-mono);font-weight:700">
    {'%.4f'%g_coef if isinstance(g_coef,float) else '—'}
  </td>
  <td style="font-family:monospace;font-size:.79rem;color:var(--text3)">
    {'%.3f'%g_t if isinstance(g_t,float) else '—'}
  </td>
  {p_cell}
  {site_cells}
</tr>'''

        html += f'''<div class="hist-section" style="margin-bottom:1.5rem">
  <div class="hist-header">
    <div class="hist-title">{roi}</div>
    <span style="font-size:.8rem;color:var(--text3)">Global coef · t-stat · p-value, then per-site coef · t-stat</span>
  </div>
  <div class="stat-card-scroll">
    <table class="stat-table" style="min-width:600px">
      <thead>
        <tr>
          <th style="text-align:left">Covariate</th>
          <th>Global Coef</th><th>t</th><th>p-value</th>
          {site_hdrs}
        </tr>
        <tr>
          <th></th><th></th><th></th><th></th>
          {sub_hdrs}
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <div style="padding:.5rem 1rem .75rem;font-size:.72rem;color:var(--text3);border-top:1px solid var(--border)">
    *** p&lt;0.001 &nbsp;** p&lt;0.01 &nbsp;* p&lt;0.05
  </div>
</div>'''

    return _section("Coefficients", html, "coefficients")


def _build_pvalue_section(results, all_sites, solid_map, bg_map, txt_map):
    """Heat-map style p-value table across all ROIs and covariates."""
    solid, bg, txt = solid_map, bg_map, txt_map

    def _p_cell(p):
        if p is None: return '<td style="color:var(--text3)">—</td>'
        val = f'{p:.4f}' if p >= 0.0001 else '<0.0001'
        return f'<td style="font-family:monospace;{_p_style(p)}">{val}</td>'

    html = '<div class="stat-card-scroll"><table class="stat-table" style="min-width:500px"><thead><tr>'
    html += '<th style="text-align:left">Covariate</th>'
    for result in results:
        html += f'<th>{result["ROI"]}</th>'
    html += '</tr></thead><tbody>'

    cov_labels = results[0]["global_stats"]["covariate_labels"] if results else []
    for i, cov in enumerate(cov_labels):
        html += f'<tr><td style="font-weight:600;color:var(--text2)">{cov}</td>'
        for result in results:
            p = result["global_stats"]["P-value"][i] if i < len(result["global_stats"]["P-value"]) else None
            html += _p_cell(p)
        html += '</tr>'
    html += '</tbody></table></div>'

    legend = ('<div style="margin-top:.75rem;font-size:.77rem;color:var(--text3);display:flex;gap:1.2rem;flex-wrap:wrap">'
              '<span style="background:rgba(16,185,129,.55);color:#065f46;font-weight:700;padding:.1rem .5rem;border-radius:4px">p&lt;0.001</span>'
              '<span style="background:rgba(16,185,129,.3);color:#065f46;font-weight:600;padding:.1rem .5rem;border-radius:4px">p&lt;0.01</span>'
              '<span style="background:rgba(16,185,129,.12);color:#047857;padding:.1rem .5rem;border-radius:4px">p&lt;0.05</span>'
              '<span style="color:var(--text3)">— not significant</span></div>')

    # per-site p-value tables
    site_tabs = ""
    for site in all_sites:
        site_tabs += f'<div style="margin-top:1.2rem"><div style="font-size:.8rem;font-weight:700;margin-bottom:.5rem">{_pill(site, solid, bg, txt)}</div>'
        site_tabs += '<div class="stat-card-scroll"><table class="stat-table" style="min-width:400px"><thead><tr>'
        site_tabs += '<th style="text-align:left">Covariate</th>'
        for result in results:
            site_tabs += f'<th>{result["ROI"]}</th>'
        site_tabs += '</tr></thead><tbody>'
        for i, cov in enumerate(cov_labels):
            site_tabs += f'<tr><td style="font-weight:600;color:var(--text2)">{cov}</td>'
            for result in results:
                p = result["local_stats"].get(site, {}).get("P-value", [])[i] \
                    if i < len(result["local_stats"].get(site, {}).get("P-value", [])) else None
                site_tabs += _p_cell(p)
            site_tabs += '</tr>'
        site_tabs += '</tbody></table></div></div>'

    return _section("P-values & Significance",
                    '<div style="margin-bottom:.4rem;font-size:.85rem;color:var(--text2);font-weight:600">Global p-values</div>'
                    + html + legend
                    + '<div style="margin-top:1.5rem;font-size:.85rem;color:var(--text2);font-weight:600">Per-site p-values</div>'
                    + site_tabs,
                    "p-values-significance")


# ─────────────────────────────────────────────────────────────────────────────
# Shell helpers
# ─────────────────────────────────────────────────────────────────────────────

def _section(title, content, slug=None):
    if slug is None:
        slug = title.lower().replace(" ", "-").replace("(","").replace(")","").replace("&","").replace("/","")
    return (f'<div class="container"><div class="section" id="sec-{slug}">'
            f'<div class="section-title">{title}</div>{content}</div></div>')


def _wrap_page(body: str, nav_titles: list = None) -> str:
    nav_titles = nav_titles or []
    def _slug(t):
        import re
        s = t.lower().replace("(","").replace(")","").replace("&","").replace("/","")
        return re.sub(r'-+', '-', s.replace(" ","-")).strip("-")
    nav_items_html = "\n".join(
        '<button class="nav-item" data-sec="sec-' + _slug(t) + '" '
        'onclick="document.getElementById(\'sec-' + _slug(t) + '\').scrollIntoView({behavior:\'smooth\',block:\'start\'})">' + t + '</button>'
        for t in nav_titles
    )
    sidebar_and_body = (
        '<div class="layout">'
        '<aside class="sidebar" id="sidebar">'
        '<div class="sidebar-inner">'
        '<div class="sidebar-header">'
        '<span class="sidebar-label">Sections</span>'
        '<button class="sidebar-toggle" onclick="toggleSidebar()">&#x2715;</button>'
        '</div>'
        + nav_items_html +
        '</div></aside>'
        '<button class="sidebar-peek" id="sidebarPeek" onclick="toggleSidebar()">Sections</button>'
        '<div class="main-content">'
        + body +
        '</div></div>'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Federated Ridge Regression Report</title>
<style>
:root {{
  --bg:#ffffff;--bg2:#ffffff;--bg3:#f1f5f9;--bg4:#e2e8f0;
  --border:#e2e8f0;--border2:#cbd5e1;
  --text:#0f172a;--text2:#334155;--text3:#64748b;--text4:#94a3b8;
  --header-bg:linear-gradient(135deg,#e0e9ff 0%,#f8fafc 100%);
  --header-border:#e2e8f0;
  --chip-bg:#f1f5f9;--chip-color:#475569;--chip-b:#6366f1;
  --legend-color:#334155;
  --card-bg:#ffffff;--card-hover:#f8fafc;
  --th-bg:#f8fafc;--td-mono:#1e293b;
  --global-row:rgba(99,102,241,.06);--global-color:#4338ca;
}}
[data-theme="dark"] {{
  --bg:#0f172a;--bg2:#1e293b;--bg3:#0f172a;--bg4:#1a2640;
  --border:#334155;--border2:#334155;
  --text:#e2e8f0;--text2:#cbd5e1;--text3:#64748b;--text4:#94a3b8;
  --header-bg:linear-gradient(135deg,#1e1b4b 0%,#0f172a 100%);
  --header-border:#334155;
  --chip-bg:#1e293b;--chip-color:#94a3b8;--chip-b:#a5b4fc;
  --legend-color:#cbd5e1;
  --card-bg:#1e293b;--card-hover:#1a2640;
  --th-bg:#161f30;--td-mono:#cbd5e1;
  --global-row:rgba(99,102,241,.06);--global-color:#a5b4fc;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;transition:background .2s,color .2s;margin:1rem 0}}
button{{font-family:inherit}}
.theme-toggle{{position:fixed;top:2rem;right:1.25rem;z-index:999;background:var(--card-bg);border:1px solid var(--border);border-radius:999px;padding:.35rem .85rem;font-size:.8rem;font-weight:600;color:var(--text2);cursor:pointer;display:flex;align-items:center;gap:.4rem;box-shadow:0 1px 4px rgba(0,0,0,.08);transition:background .2s,color .2s,border-color .2s}}
.theme-toggle:hover{{background:var(--bg3)}}
.page-header{{background:var(--header-bg);border-bottom:1px solid var(--header-border);padding:2rem 2.5rem;padding-right:8rem}}
.page-header h1{{font-size:1.7rem;font-weight:700;color:var(--text);letter-spacing:-.02em}}
.page-header p{{color:var(--text3);margin-top:.35rem;font-size:.93rem}}
.chips{{display:flex;gap:.6rem;margin-top:1rem;flex-wrap:wrap}}
.chip{{background:var(--chip-bg);border:1px solid var(--border);border-radius:999px;padding:.25rem .75rem;font-size:.78rem;color:var(--chip-color)}}
.chip b{{color:var(--chip-b)}}
.site-legend{{display:flex;gap:.9rem;flex-wrap:wrap;margin-top:1rem}}
.container{{max-width:1400px;margin:0 auto;padding:2rem 0}}
.section{{margin-bottom:3rem}}
.section-title{{font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:1.1rem;padding-bottom:.5rem;border-bottom:1px solid var(--border)}}
.stats-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1rem}}
.stat-card{{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;overflow:hidden}}
.stat-card-scroll{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
.stat-card-header{{padding:.75rem 1rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:.6rem}}
.stat-card-title{{font-weight:700;color:var(--text);font-size:.9rem;flex:1}}
.hist-section{{background:var(--card-bg);border:1px solid var(--border);border-radius:14px;overflow:hidden}}
.hist-header{{padding:1.1rem 1.4rem;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:.8rem;flex-wrap:wrap}}
.hist-title{{font-size:1rem;font-weight:700;color:var(--text)}}
table.stat-table{{width:100%;border-collapse:collapse;font-size:.8rem}}
table.stat-table th{{color:var(--text3);font-weight:600;padding:.45rem .85rem;text-align:right;background:var(--th-bg)}}
table.stat-table th:first-child{{text-align:left}}
table.stat-table td{{padding:.42rem .85rem;border-top:1px solid var(--border);text-align:right}}
table.stat-table td:first-child{{text-align:left}}
table.stat-table tr.global-row td{{color:var(--global-color);font-weight:600;background:var(--global-row)}}

.badge{{display:inline-block;padding:.14rem .55rem;border-radius:999px;font-size:.73rem;font-weight:600}}
.badge-green{{background:rgba(16,185,129,.12);color:#15803d;border:1px solid rgba(16,185,129,.3)}}
.badge-red{{background:rgba(239,68,68,.12);color:#dc2626;border:1px solid rgba(239,68,68,.3)}}
.badge-yellow{{background:rgba(245,158,11,.12);color:#b45309;border:1px solid rgba(245,158,11,.3)}}
[data-theme="dark"] .badge-green{{color:#34d399}}
[data-theme="dark"] .badge-red{{color:#f87171}}
[data-theme="dark"] .badge-yellow{{color:#fbbf24}}
/* sidebar */
.layout{{display:flex;align-items:flex-start;padding:1.5rem 2rem 0}}
.sidebar{{width:190px;flex-shrink:0;position:sticky;top:1.5rem;max-height:calc(100vh - 3rem);overflow-y:auto;margin-right:1.5rem;transition:width .2s,opacity .2s,margin .2s}}
.sidebar.hidden{{width:0;opacity:0;overflow:hidden;margin-right:0;pointer-events:none}}
.sidebar-inner{{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:.6rem .5rem}}
.sidebar-header{{display:flex;align-items:center;justify-content:space-between;padding:.2rem .3rem .45rem;border-bottom:1px solid var(--border);margin-bottom:.4rem}}
.sidebar-label{{font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text3)}}
.sidebar-toggle{{background:none;border:none;cursor:pointer;font-size:.75rem;color:var(--text3);padding:.15rem .35rem;border-radius:5px;line-height:1}}
.sidebar-toggle:hover{{background:var(--bg3);color:var(--text)}}
.nav-item{{display:block;width:100%;padding:.4rem .65rem;border-radius:7px;font-size:.78rem;color:var(--text3);line-height:1.35;cursor:pointer;border:none;background:none;text-align:left;transition:background .12s,color .12s}}
.nav-item:hover{{background:var(--bg3);color:var(--text)}}
.nav-item.active{{background:rgba(99,102,241,.13);color:#6366f1;font-weight:600}}
[data-theme="dark"] .nav-item.active{{background:rgba(165,180,252,.1);color:#a5b4fc}}
.main-content{{flex:1;min-width:0}}
.sidebar-peek{{position:fixed;left:0;top:50%;transform:translateY(-50%);background:var(--card-bg);border:1px solid var(--border);border-left:none;border-radius:0 8px 8px 0;padding:.55rem .35rem;cursor:pointer;font-size:.72rem;color:var(--text3);display:none;z-index:200;writing-mode:vertical-rl;letter-spacing:.06em;line-height:1}}
.sidebar-peek:hover{{color:var(--text)}}
.sidebar-peek.visible{{display:block}}
</style>
</head>
<body>
<button class="theme-toggle" onclick="toggleTheme()" id="themeBtn">🌙 Dark mode</button>
{sidebar_and_body}
<script>
function getTheme(){{try{{return localStorage.getItem('theme')||'light'}}catch(e){{return'light'}}}}
function applyTheme(t){{
  document.documentElement.setAttribute('data-theme',t==='dark'?'dark':'');
  document.getElementById('themeBtn').textContent=t==='dark'?'☀️ Light mode':'🌙 Dark mode';
}}
function toggleTheme(){{var n=getTheme()==='dark'?'light':'dark';try{{localStorage.setItem('theme',n)}}catch(e){{}}applyTheme(n);}}
function toggleSidebar(){{
  var sb=document.getElementById('sidebar'),pk=document.getElementById('sidebarPeek');
  var h=sb.classList.toggle('hidden');
  try{{localStorage.setItem('sidebarHidden',h?'1':'0')}}catch(e){{}}
  if(pk)pk.classList.toggle('visible',h);
}}
(function(){{try{{
  if(localStorage.getItem('sidebarHidden')==='1'){{
    var sb=document.getElementById('sidebar'),pk=document.getElementById('sidebarPeek');
    if(sb)sb.classList.add('hidden');if(pk)pk.classList.add('visible');
  }}
}}catch(e){{}}}}());
document.addEventListener('DOMContentLoaded',function(){{
  var items=document.querySelectorAll('.nav-item[data-sec]');
  if(!items.length)return;
  var secs=Array.from(items).map(function(el){{return document.getElementById(el.getAttribute('data-sec'));}}).filter(Boolean);
  function onScroll(){{
    var y=window.scrollY+140,active=secs[0];
    secs.forEach(function(s){{if(s.offsetTop<=y)active=s;}});
    items.forEach(function(el){{el.classList.toggle('active',active&&el.getAttribute('data-sec')===active.id);}});
  }}
  window.addEventListener('scroll',onScroll,{{passive:true}});
  onScroll();
  applyTheme(getTheme());
}});
</script>
</body>
</html>"""
