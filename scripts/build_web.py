import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents.ppo.numpy_net import export_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("truco", "agents", "demo")
SKIP_DIRS = {"__pycache__", "web"}


def bundle_sources(zf: zipfile.ZipFile) -> int:
    count = 0
    for package in PACKAGES:
        for path in sorted((ROOT / package).rglob("*.py")):
            rel = path.relative_to(ROOT)
            if SKIP_DIRS & set(rel.parts[:-1]):
                continue
            zf.write(path, rel.as_posix())
            count += 1
    return count


def last_eval_line(path: Path) -> str | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[-1] if lines else None


def bundle_run(zf: zipfile.ZipFile, run_dir: Path, work: Path, dictionary: Path | None) -> dict:
    name = run_dir.name
    npz = export_checkpoint(run_dir / "last.pt", work / name / "last.npz", dtype=np.float16)
    zf.write(npz, f"runs/{name}/last.npz")
    info = {"run": name, "npz_bytes": npz.stat().st_size}
    line = last_eval_line(run_dir / "eval.jsonl")
    if line:
        zf.writestr(f"runs/{name}/eval.jsonl", line + "\n")
    config = run_dir / "config.json"
    if config.exists():
        zf.write(config, f"runs/{name}/config.json")
    source = dictionary if dictionary is not None and dictionary.exists() else run_dir / "senas" / "diccionario.json"
    if source.exists():
        zf.write(source, f"runs/{name}/senas/diccionario.json")
        info["dictionary"] = True
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="arma la version estatica de la demo, que corre entera en el navegador con Pyodide")
    parser.add_argument("--runs", default="runs", help="carpeta con los runs entrenados")
    parser.add_argument("--names", nargs="+", default=["parejas_senas", "parejas_sin_senas", "parejas_espias"])
    parser.add_argument("--dictionaries", default=None, help="carpeta con diccionario_<run>.json, como runs/reporte, para runs sin senas/diccionario.json")
    parser.add_argument("--out", default="docs")
    args = parser.parse_args()
    runs = Path(args.runs)
    out = Path(args.out)
    if out.exists():
        for child in ("index.html", "favicon.svg", "css", "js", "py"):
            target = out / child
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    out.mkdir(parents=True, exist_ok=True)
    web = ROOT / "truco" / "web"
    for item in web.iterdir():
        if item.is_dir():
            shutil.copytree(item, out / item.name)
        else:
            shutil.copy2(item, out / item.name)
    index = out / "index.html"
    html = index.read_text(encoding="utf-8")
    if 'data-backend="local"' not in html:
        html = html.replace('<html lang="es">', '<html lang="es" data-backend="local">', 1)
    index.write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    (out / "py").mkdir(exist_ok=True)
    work = out / "py" / "_tmp"
    report = {"runs": []}
    with zipfile.ZipFile(out / "py" / "truco.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        report["sources"] = bundle_sources(zf)
        for name in args.names:
            run_dir = runs / name
            if not (run_dir / "last.pt").exists():
                print(f"aviso: {run_dir} no tiene last.pt, lo salteo")
                continue
            dictionary = Path(args.dictionaries) / f"diccionario_{name}.json" if args.dictionaries else None
            report["runs"].append(bundle_run(zf, run_dir, work, dictionary))
    shutil.rmtree(work, ignore_errors=True)
    size = (out / "py" / "truco.zip").stat().st_size
    report["zip_mb"] = round(size / 1e6, 2)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
