@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
call scripts\buscar_python.bat
if not defined PY (pause & exit /b 1)
"%PY%" -m pytest -q || exit /b 1
"%PY%" scripts\train_seeds.py --config configs\ppo_semillas.yaml --seeds 0 1 2 3 4 || exit /b 1
"%PY%" scripts\train_cfr.py --iterations 1000000 --workers 5 || exit /b 1
"%PY%" scripts\analyze_signals.py --checkpoint runs\parejas_senas\last.pt || exit /b 1
"%PY%" scripts\analyze_signals.py --checkpoint runs\parejas_espias\last.pt || exit /b 1
"%PY%" scripts\ablations.py --checkpoint runs\parejas_senas\last.pt --matches 1000 || exit /b 1
"%PY%" scripts\ablations.py --checkpoint runs\parejas_espias\last.pt --matches 1000 || exit /b 1
"%PY%" scripts\crossplay.py --runs runs\semillas\seed_0 runs\semillas\seed_1 runs\semillas\seed_2 runs\semillas\seed_3 runs\semillas\seed_4 --short-names --matches 400 || exit /b 1
"%PY%" scripts\league.py --runs runs\parejas_senas runs\parejas_sin_senas runs\parejas_espias --per-run 4 --matches 200 --out runs\liga || exit /b 1
"%PY%" scripts\league.py --players 2 --specs cfr:runs\cfr\heads_up.npz --matches 400 --out runs\liga_1v1 || exit /b 1
"%PY%" scripts\report.py || exit /b 1
