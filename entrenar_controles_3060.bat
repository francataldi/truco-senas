@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
call scripts\buscar_python.bat
if not defined PY (pause & exit /b 1)
set DESPIERTA=1
start "Manteniendo la PC despierta" /min cmd /c mantener_despierta.bat
"%PY%" -m pytest -q || (pause & exit /b 1)
title Controles 1/3: sin bonus ni cabeza auxiliar
"%PY%" scripts\train_ppo.py --config configs\ppo_sin_sesgos.yaml
title Controles 2/3: sin bonus de informacion
"%PY%" scripts\train_ppo.py --config configs\ppo_sin_bonus.yaml
title Controles 3/3: sin cabeza auxiliar
"%PY%" scripts\train_ppo.py --config configs\ppo_sin_aux.yaml
call analizar_experimentos_3060.bat
