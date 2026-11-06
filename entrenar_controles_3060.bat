@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
python -m pytest -q || exit /b 1
title Controles 1/3: sin bonus ni cabeza auxiliar
python scripts\train_ppo.py --config configs\ppo_sin_sesgos.yaml
title Controles 2/3: sin bonus de informacion
python scripts\train_ppo.py --config configs\ppo_sin_bonus.yaml
title Controles 3/3: sin cabeza auxiliar
python scripts\train_ppo.py --config configs\ppo_sin_aux.yaml
call analizar_experimentos_3060.bat
