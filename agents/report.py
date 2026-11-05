from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from truco.features import FEATURE_NAMES

from .base import limit_threads
from .plots import plot_curves, plot_grouped_bars, plot_infoset_curve, plot_intervals, plot_matrix, series_color
from .registry import spec_label
from .stats import wilson_interval

ABLATION_LABELS = {
    "normal": "normal",
    "compañero_mezclado": "señas del compañero mezcladas",
    "rivales_mezclados": "señas espiadas mezcladas",
    "ambos_mezclados": "compañero y rivales mezclados",
    "rivales_ocultos": "sin espiar a los rivales",
    "seña_constante": "seña constante",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[int(row["iteration"])] = row
    return [rows[k] for k in sorted(rows)]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _skip(path: Path, out_dir: Path) -> bool:
    try:
        path.resolve().relative_to(out_dir.resolve())
        return True
    except ValueError:
        return False


def discover(runs_dir: Path, out_dir: Path) -> dict:
    def find(pattern: str) -> list[Path]:
        return sorted(p for p in runs_dir.rglob(pattern) if not _skip(p, out_dir))

    return {
        "runs": sorted({p.parent for p in find("eval.jsonl")}),
        "ligas": find("liga.json"),
        "crossplay": find("crossplay.json"),
        "ablaciones": find("ablaciones.json"),
        "cfr_curvas": find("curva_infosets.csv"),
        "diccionarios": find("diccionario.json"),
    }


def run_name(run: Path, runs_dir: Path) -> str:
    try:
        return run.resolve().relative_to(runs_dir.resolve()).as_posix()
    except ValueError:
        return run.name


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "run"


def _run_config(run: Path) -> dict:
    path = run / "config.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _hands_by_iteration(run: Path) -> dict[int, float]:
    path = run / "metrics.csv"
    if not path.exists():
        return {}
    out = {}
    for row in read_csv(path):
        it, hands = _float(row.get("iteration")), _float(row.get("hands"))
        if it is not None and hands is not None:
            out[int(it)] = hands
    return out


def learning_series(run: Path, label: str, key: str, use_hands: bool) -> dict | None:
    rows = [r for r in read_jsonl(run / "eval.jsonl") if key in r]
    if not rows:
        return None
    hands = _hands_by_iteration(run) if use_hands else {}
    x, y, lo, hi = [], [], [], []
    for r in rows:
        entry = r[key]
        n = int(entry["matches"])
        k = round(float(entry["win_rate"]) * n)
        a, b = wilson_interval(k, n)
        it = int(r["iteration"])
        x.append(hands[it] / 1e6 if use_hands and it in hands else it)
        y.append(k / n if n else 0.0)
        lo.append(a)
        hi.append(b)
    return {"label": label, "x": x, "y": y, "lo": lo, "hi": hi}


def groups_of(runs: list[Path], runs_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for run in runs:
        name = run_name(run, runs_dir)
        group = name.rsplit("/", 1)[0] if "/" in name else "principales"
        groups.setdefault(group, []).append(run)
    return groups


def plot_learning(runs: list[Path], runs_dir: Path, out_dir: Path, group: str, figures: list[dict]) -> list[dict]:
    use_hands = all(_hands_by_iteration(r) for r in runs)
    xlabel = "manos jugadas (millones)" if use_hands else "iteración"
    finals = []
    for key, title, stem, ref in (
        ("vs_heuristic", "Victorias contra el heurístico", "curvas_heuristico", 0.5),
        ("vs_random", "Victorias contra random", "curvas_random", 0.5),
        ("vs_self_scrambled", "Valor de las señas: contra una copia con las señas entre compañeros mezcladas", "curvas_senas", 0.5),
    ):
        series = []
        for i, run in enumerate(runs):
            s = learning_series(run, run_name(run, runs_dir), key, use_hands)
            if s is not None:
                s["color"] = series_color(i)
                series.append(s)
                finals.append({"run": s["label"], "medida": key, "tasa": round(s["y"][-1], 4), "ci95": [round(s["lo"][-1], 4), round(s["hi"][-1], 4)]})
        if not series:
            continue
        path = out_dir / f"{stem}_{_safe(group)}.png"
        plot_curves(series, path, f"{title} · {group} (IC 95% de Wilson)", xlabel, "tasa de victoria", reference=ref, ylim=(0, 1))
        figures.append({"archivo": path.name, "titulo": f"{title} ({group})", "seccion": "Curvas de aprendizaje"})
    return finals


def plot_training(runs: list[Path], runs_dir: Path, out_dir: Path, group: str, figures: list[dict]) -> None:
    from .plots import pyplot

    metrics = {}
    for run in runs:
        path = run / "metrics.csv"
        if path.exists():
            rows = read_csv(path)
            if rows:
                metrics[run] = rows
    if not metrics:
        return
    plt = pyplot()
    panels = (
        ("signal_info", "información de las señas en el lote (nats)"),
        ("entropy", "entropía de la política de juego"),
        ("mean_reward", "recompensa media"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (column, title) in zip(axes, panels):
        for i, (run, rows) in enumerate(metrics.items()):
            x = np.array([_float(r.get("iteration")) or 0 for r in rows])
            y = np.array([_float(r.get(column)) if _float(r.get(column)) is not None else np.nan for r in rows])
            if np.all(np.isnan(y)):
                continue
            window = max(1, len(y) // 40)
            smooth = np.convolve(np.nan_to_num(y), np.ones(window) / window, mode="valid")
            ax.plot(x[window - 1 :], smooth, color=series_color(i), label=run_name(run, runs_dir))
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel("iteración")
    if len(metrics) > 1:
        axes[0].legend(loc="best")
    fig.tight_layout()
    path = out_dir / f"entrenamiento_{_safe(group)}.png"
    from .plots import _save

    _save(fig, path)
    figures.append({"archivo": path.name, "titulo": f"Métricas de entrenamiento ({group})", "seccion": "Curvas de aprendizaje"})


def signal_summary(run: Path, hands: int, seed: int, bootstrap: int) -> dict | None:
    checkpoint = run / "last.pt"
    if checkpoint.exists() and hands > 0:
        from .ppo.agent import PPOAgent
        from .signal_analysis import collect_signals, summarize

        agent = PPOAgent.from_checkpoint(str(checkpoint), seed=seed)
        limit_threads()
        if agent.rules is None or not agent.rules.signals:
            return None
        data = collect_signals(agent.rules, [agent] * agent.rules.num_players, hands, seed=seed)
        return summarize(data, agent.rules.signal_vocab, bootstrap=bootstrap, seed=seed)
    cached = run / "senas" / "diccionario.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    return None


def plot_signals(runs: list[Path], runs_dir: Path, out_dir: Path, figures: list[dict], hands: int, seed: int, bootstrap: int) -> dict:
    from .signal_analysis import plot_dictionary

    summaries = {}
    for run in runs:
        config = _run_config(run)
        if config and not config.get("rules", {}).get("signals", True):
            continue
        summary = signal_summary(run, hands, seed, bootstrap)
        if summary is None:
            continue
        name = run_name(run, runs_dir)
        summaries[name] = summary
        (out_dir / f"diccionario_{_safe(name)}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        path = out_dir / f"diccionario_{_safe(name)}.png"
        plot_dictionary(summary, str(path), title=f"{name}: qué anuncia cada seña · MI con la carta más alta {summary['mi_carta_mas_alta_bits']:.2f} bits")
        figures.append({"archivo": path.name, "titulo": f"Diccionario de señas de {name}", "seccion": "Señas"})
    if summaries:
        chosen = list(summaries.items())[:4]
        groups = []
        for i, (name, summary) in enumerate(chosen):
            values = [summary["mi_por_rasgo_bits"][f] for f in FEATURE_NAMES]
            group = {"label": name, "values": values, "color": series_color(i)}
            if "mi_por_rasgo_ci95" in summary:
                group["lo"] = [summary["mi_por_rasgo_ci95"][f][0] for f in FEATURE_NAMES]
                group["hi"] = [summary["mi_por_rasgo_ci95"][f][1] for f in FEATURE_NAMES]
            groups.append(group)
        bound = None
        first = chosen[0][1]
        if "entropia_por_rasgo_bits" in first:
            bound = [first["entropia_por_rasgo_bits"][f] for f in FEATURE_NAMES]
        path = out_dir / "mi_por_rasgo.png"
        plot_grouped_bars(list(FEATURE_NAMES), groups, path, "Información mutua entre la seña y cada rasgo de la mano (IC 95% bootstrap)", "bits", bound=bound)
        figures.append({"archivo": path.name, "titulo": "Información mutua por rasgo", "seccion": "Señas"})
    return summaries


def plot_leagues(paths: list[Path], runs_dir: Path, out_dir: Path, figures: list[dict]) -> list[dict]:
    out = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        name = _safe(run_name(path.parent, runs_dir))
        rows = summary["ratings"]
        anchor = summary.get("ancla")
        players = summary.get("rules", {}).get("num_players", "?")
        suffix = "" if name in ("liga", "liga_1v1") else f" ({name})"
        figure = out_dir / f"liga_{name}.png"
        plot_intervals(
            [r["nombre"] for r in rows],
            [r["elo"] for r in rows],
            [r["elo_lo"] for r in rows],
            [r["elo_hi"] for r in rows],
            figure,
            f"Liga de {players} jugadores{suffix}: Elo Bradley-Terry, IC 95% bootstrap",
            f"Elo relativo a {anchor}" if anchor else "Elo relativo a la media",
            reference=0.0,
            highlight=[r["nombre"] == anchor for r in rows],
            fmt="{:+.0f}",
        )
        figures.append({"archivo": figure.name, "titulo": f"Liga {name}", "seccion": "Liga"})
        names = summary["nombres"]
        order = [names.index(r["nombre"]) for r in rows]
        matrix = [[np.nan if summary["tasa_observada"][i][j] is None else summary["tasa_observada"][i][j] for j in order] for i in order]
        figure = out_dir / f"liga_matriz_{name}.png"
        plot_matrix(
            [names[i] for i in order],
            [names[i] for i in order],
            matrix,
            figure,
            f"Liga de {players} jugadores{suffix}: victorias de la fila contra la columna",
            colorbar="tasa de victoria",
        )
        figures.append({"archivo": figure.name, "titulo": f"Cruces de la liga {name}", "seccion": "Liga"})
        out.append({"liga": name, "jugadores": players, "ancla": anchor, "ratings": rows})
    return out


def plot_crossplays(paths: list[Path], runs_dir: Path, out_dir: Path, figures: list[dict]) -> list[dict]:
    out = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        name = _safe(run_name(path.parent, runs_dir))
        names = summary["nombres"]
        matrix = [[np.nan if v is None else v for v in row] for row in summary["matriz"]]
        figure = out_dir / f"crossplay_matriz_{name}.png"
        rival = spec_label(summary["rival"])
        plot_matrix(names, names, matrix, figure, f"Parejas de semillas contra {rival}", colorbar="tasa de victoria")
        figures.append({"archivo": figure.name, "titulo": f"Matriz de cross-play ({name})", "seccion": "Cross-play"})
        labels, values, lows, highs = [], [], [], []
        for key, label in (("self_play", "self-play"), ("cross_play", "cross-play"), ("self_play_mezclado", "self-play con señas mezcladas")):
            entry = summary["agregado"].get(key)
            if entry:
                labels.append(label)
                values.append(entry["p"])
                lows.append(entry["ci95"][0])
                highs.append(entry["ci95"][1])
        if labels:
            figure = out_dir / f"crossplay_{name}.png"
            plot_intervals(
                labels,
                values,
                lows,
                highs,
                figure,
                f"Cross-play contra self-play (rival: {rival})",
                "tasa de victoria, IC 95% de Wilson",
                reference=0.5,
            )
            figures.append({"archivo": figure.name, "titulo": f"Cross-play contra self-play ({name})", "seccion": "Cross-play"})
        out.append(
            {
                "crossplay": name,
                "agregado": summary["agregado"],
                "diferencias": summary.get("diferencias", {}),
                "fraccion_perdida": summary.get("fraccion_del_valor_de_las_señas_perdida"),
                "diccionarios": summary.get("diccionarios_media"),
            }
        )
    return out


def plot_ablations(paths: list[Path], runs_dir: Path, out_dir: Path, figures: list[dict]) -> list[dict]:
    from .plots import BLUE, INK_SECONDARY, MUTED, SURFACE, _save, pyplot

    loaded = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        name = summary.get("run") or path.parent.parent.name
        loaded.append((name, summary))
    if not loaded:
        return []
    plt = pyplot()
    conditions = [c for c in ABLATION_LABELS if any(c in s["condiciones"] for _, s in loaded)]
    fig, axes = plt.subplots(1, len(loaded), figsize=(4.6 * len(loaded) + 1.6, 0.9 + 0.42 * len(conditions)), squeeze=False, sharey=True)
    for ax, (name, summary) in zip(axes[0], loaded):
        y = np.arange(len(conditions))[::-1]
        normal = 0.5 if summary.get("referencia_valores") == "simetria_0.5" else summary["condiciones"].get("normal", {}).get("win_rate")
        if normal is not None:
            ax.axvline(normal, color=MUTED, linewidth=0.8)
        for k, condition in enumerate(conditions):
            entry = summary["condiciones"].get(condition)
            if entry is None:
                continue
            color = INK_SECONDARY if condition == "normal" else BLUE
            ax.plot(entry["ci95"], [y[k], y[k]], color=color, linewidth=1.6)
            ax.plot([entry["win_rate"]], [y[k]], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_yticks(y, [ABLATION_LABELS[c] for c in conditions])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel("tasa de victoria, IC 95% de Wilson")
    fig.suptitle("Ablaciones de señas: el equipo ablacionado contra el agente sin tocar", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    figure = out_dir / "ablaciones.png"
    _save(fig, figure)
    figures.append({"archivo": figure.name, "titulo": "Ablaciones de señas", "seccion": "Ablaciones"})
    return [{"run": name, "condiciones": s["condiciones"], "valores": s.get("valores", {})} for name, s in loaded]


def plot_cfr(paths: list[Path], runs_dir: Path, out_dir: Path, figures: list[dict]) -> list[dict]:
    out = []
    for path in paths:
        name = _safe(run_name(path.parent, runs_dir))
        curve = read_csv(path)
        if not curve:
            continue
        figure = out_dir / f"cfr_infosets_{name}.png"
        plot_infoset_curve(curve, figure)
        figures.append({"archivo": figure.name, "titulo": f"MCCFR: crecimiento de la tabla ({name})", "seccion": "MCCFR"})
        last = {}
        for row in curve:
            last[row["worker"]] = row
        entry = {
            "cfr": name,
            "iteraciones_por_worker": max(int(r["iteration"]) for r in curve),
            "infosets_por_worker": [int(r["infosets"]) for r in last.values()],
            "bytes_por_infoset": [float(r["bytes_per_infoset"]) for r in last.values()],
            "rss_mb_pico": round(max(float(r["rss_bytes"]) for r in curve) / 2**20, 1),
        }
        summaries = sorted(path.parent.glob("*.json"))
        for s in summaries:
            data = json.loads(s.read_text(encoding="utf-8"))
            if "vs_heuristic" in data and "infosets" in data:
                entry["politica"] = {
                    k: data[k] for k in ("infosets", "policy_bytes_per_infoset", "file_bytes", "vs_heuristic", "vs_random", "coverage") if k in data
                }
        out.append(entry)
    return out


def _fmt_ci(entry: dict) -> str:
    return f"{entry['win_rate'] if 'win_rate' in entry else entry['p']:.3f} [{entry['ci95'][0]:.3f}, {entry['ci95'][1]:.3f}]"


def write_latex(out_dir: Path, report: dict) -> None:
    lines = []
    for league in report.get("ligas", []):
        lines += [
            f"% liga {league['liga']}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Agente & Elo & IC 95\\% & $P(\\text{mejor})$ & Partidos \\\\",
            "\\midrule",
        ]
        for r in league["ratings"]:
            nombre = r["nombre"].replace("_", "\\_")
            lines.append(f"{nombre} & {r['elo']:+.0f} & [{r['elo_lo']:+.0f}, {r['elo_hi']:+.0f}] & {r['p_mejor']:.2f} & {r['partidos']} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
    for ab in report.get("ablaciones", []):
        lines += [f"% ablaciones {ab['run']}", "\\begin{tabular}{lrr}", "\\toprule", "Condición & Victorias & IC 95\\% \\\\", "\\midrule"]
        for condition, entry in ab["condiciones"].items():
            lines.append(f"{ABLATION_LABELS.get(condition, condition)} & {entry['win_rate']:.3f} & [{entry['ci95'][0]:.3f}, {entry['ci95'][1]:.3f}] \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
    for cp in report.get("crossplay", []):
        lines += [f"% cross-play {cp['crossplay']}", "\\begin{tabular}{lrr}", "\\toprule", "Pareja & Victorias & IC 95\\% \\\\", "\\midrule"]
        for key, entry in cp["agregado"].items():
            lines.append(f"{key.replace('_', '-')} & {entry['p']:.3f} & [{entry['ci95'][0]:.3f}, {entry['ci95'][1]:.3f}] \\\\")
        lines += ["\\bottomrule", "\\end{tabular}", ""]
    (out_dir / "tablas.tex").write_text("\n".join(lines), encoding="utf-8")


MEASURE_LABELS = {
    "vs_heuristic": "contra el heurístico",
    "vs_random": "contra random",
    "vs_self_scrambled": "contra una copia con señas mezcladas",
}

VALUE_LABELS = {
    "valor_escuchar_compañero": "escuchar al compañero",
    "valor_espiar_rivales": "espiar a los rivales",
    "costo_rivales_mezclados": "señas espiadas falsas",
    "valor_de_todas_las_señas": "todas las señas",
    "valor_neto_de_señalar": "señalar (comunicar y filtrar)",
}


def _table(headers: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    out = ["<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr>"]
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            attribute = " class='n'" if i >= numeric_from else ""
            cells.append(f"<td{attribute}>{html.escape(str(cell))}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</table>")
    return "".join(out)


def _interval(entry: dict, key: str = "p") -> str:
    return f"{entry[key]:.3f} [{entry['ci95'][0]:.3f}, {entry['ci95'][1]:.3f}]"


def _summary_tables(report: dict) -> list[str]:
    parts = []
    finals = report.get("curvas_finales", [])
    if finals:
        parts.append("<h2>Última evaluación de cada corrida</h2>")
        rows = [[r["run"], MEASURE_LABELS.get(r["medida"], r["medida"]), f"{r['tasa']:.3f}", f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"] for r in finals]
        parts.append(_table(["corrida", "medida", "tasa", "IC 95%"], rows, numeric_from=2))
    for cp in report.get("crossplay", []):
        parts.append(f"<h2>Cross-play ({html.escape(cp['crossplay'])})</h2>")
        rows = [[key.replace("_", " "), _interval(entry)] for key, entry in cp["agregado"].items()]
        rows += [
            [key.replace("_", " "), f"{entry['diff']:+.3f} [{entry['ci95'][0]:+.3f}, {entry['ci95'][1]:+.3f}]"] for key, entry in cp["diferencias"].items()
        ]
        lost = cp.get("fraccion_perdida") or {}
        if lost.get("valor") is not None:
            interval = f" [{lost['ci95'][0]:.2f}, {lost['ci95'][1]:.2f}]" if "ci95" in lost else ""
            rows.append(["fracción del valor de las señas perdida", f"{lost['valor']:.2f}{interval}"])
        for key, value in (cp.get("diccionarios") or {}).items():
            rows.append([f"diccionarios: {key.replace('_', ' ')}", f"{value:.3f}"])
        parts.append(_table(["medida", "valor (IC 95%)"], rows))
    ablations = report.get("ablaciones", [])
    if ablations:
        parts.append("<h2>Valor de cada componente de las señas</h2>")
        rows = []
        for ab in ablations:
            for key, entry in ab["valores"].items():
                rows.append([ab["run"], VALUE_LABELS.get(key, key), f"{entry['diff']:+.3f} [{entry['ci95'][0]:+.3f}, {entry['ci95'][1]:+.3f}]"])
        parts.append(_table(["corrida", "componente", "diferencia de victorias (IC 95%)"], rows, numeric_from=2))
    for cfr in report.get("cfr", []):
        parts.append(f"<h2>MCCFR ({html.escape(cfr['cfr'])})</h2>")
        policy = cfr.get("politica", {})
        rows = [
            ["infosets en la tabla", f"{max(cfr['infosets_por_worker']):,}"],
            ["bytes por infoset al entrenar", f"{max(cfr['bytes_por_infoset']):.1f}"],
            ["pico de RSS por worker (MB)", f"{cfr['rss_mb_pico']:.0f}"],
        ]
        if policy:
            rows += [
                ["infosets en la política", f"{policy['infosets']:,}"],
                ["bytes por infoset en la política", f"{policy['policy_bytes_per_infoset']:.1f}"],
                ["archivo (MB)", f"{policy['file_bytes'] / 2**20:.1f}"],
                ["contra el heurístico", _interval(policy["vs_heuristic"], "win_rate")],
                ["contra random", _interval(policy["vs_random"], "win_rate")],
            ]
        parts.append(_table(["medida", "valor"], rows))
    return parts


def write_html(out_dir: Path, report: dict, figures: list[dict]) -> None:
    sections: dict[str, list[dict]] = {}
    for f in figures:
        sections.setdefault(f["seccion"], []).append(f)
    parts = [
        "<!doctype html>",
        '<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Reporte truco-senas</title>",
        "<style>body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;background:#f9f9f7;color:#0b0b0b;"
        "margin:0 auto;max-width:1100px;padding:24px 16px}"
        "h1{font-size:22px}h2{font-size:17px;margin-top:32px}figure{background:#fcfcfb;margin:16px 0;padding:12px;border-radius:8px}"
        "img{max-width:100%;height:auto}figcaption{color:#52514e;font-size:13px;margin-top:6px}"
        "table{border-collapse:collapse;font-size:13px;margin:8px 0}td,th{padding:4px 10px;border-bottom:1px solid #e1e0d9;text-align:left}"
        "th{color:#52514e;font-weight:600}td.n{text-align:right;font-variant-numeric:tabular-nums}</style></head><body>",
        "<h1>Reporte de investigación · truco con señas</h1>",
        *_summary_tables(report),
    ]
    for section, items in sections.items():
        parts.append(f"<h2>{html.escape(section)}</h2>")
        for f in items:
            parts.append(
                f"<figure><img src='{html.escape(f['archivo'])}' alt='{html.escape(f['titulo'])}'><figcaption>{html.escape(f['titulo'])}</figcaption></figure>"
            )
    parts.append("</body></html>")
    (out_dir / "reporte.html").write_text("\n".join(parts), encoding="utf-8")


def build_report(runs_dir: str | Path, out_dir: str | Path, signal_hands: int = 4000, bootstrap: int = 200, seed: int = 0) -> dict:
    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    found = discover(runs_dir, out_dir)
    figures: list[dict] = []
    report: dict = {"runs": [run_name(r, runs_dir) for r in found["runs"]]}
    finals = []
    for group, runs in groups_of(found["runs"], runs_dir).items():
        finals += plot_learning(runs, runs_dir, out_dir, group, figures)
        plot_training(runs, runs_dir, out_dir, group, figures)
    report["curvas_finales"] = finals
    signals = plot_signals(found["runs"], runs_dir, out_dir, figures, signal_hands, seed, bootstrap)
    report["señas"] = {
        name: {
            "mi_carta_mas_alta_bits": s["mi_carta_mas_alta_bits"],
            "mi_carta_mas_alta_ci95": s.get("mi_carta_mas_alta_ci95"),
            "entropia_señas_bits": s["entropia_señas_bits"],
            "mi_por_rasgo_bits": s["mi_por_rasgo_bits"],
        }
        for name, s in signals.items()
    }
    report["ligas"] = plot_leagues(found["ligas"], runs_dir, out_dir, figures)
    report["crossplay"] = plot_crossplays(found["crossplay"], runs_dir, out_dir, figures)
    report["ablaciones"] = plot_ablations(found["ablaciones"], runs_dir, out_dir, figures)
    report["cfr"] = plot_cfr(found["cfr_curvas"], runs_dir, out_dir, figures)
    report["figuras"] = figures
    (out_dir / "resumen.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_latex(out_dir, report)
    write_html(out_dir, report, figures)
    return report
