@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
call scripts\buscar_python.bat
if not defined PY (pause & exit /b 1)
"%PY%" -m pytest -q || exit /b 1
"%PY%" scripts\train_ppo.py --config configs\ppo_principal.yaml || exit /b 1
"%PY%" scripts\analyze_signals.py --checkpoint runs\parejas_senas\last.pt
"%PY%" scripts\train_ppo.py --config configs\ppo_sin_senas.yaml
"%PY%" scripts\train_ppo.py --config configs\ppo_espias.yaml
"%PY%" scripts\train_cfr.py --iterations 300000 --workers 5
