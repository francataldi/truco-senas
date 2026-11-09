@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
call scripts\buscar_python.bat
if not defined PY (pause & exit /b 1)
set E=runs_experimentos
if not defined DESPIERTA (set DESPIERTA=1& start "Manteniendo la PC despierta" /min cmd /c mantener_despierta.bat)
title Analisis 1/6: diccionarios de los controles
if not exist %E%\parejas_sin_sesgos\senas\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %E%\parejas_sin_sesgos\last.pt
if not exist %E%\parejas_sin_bonus\senas\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %E%\parejas_sin_bonus\last.pt
if not exist %E%\parejas_sin_aux\senas\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %E%\parejas_sin_aux\last.pt
title Analisis 2/6: contra el heuristico, partidos duplicados
if not exist %E%\vs_heuristico\parejas_senas.json "%PY%" scripts\evaluate.py --a ppo:runs\parejas_senas\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_senas.json
if not exist %E%\vs_heuristico\parejas_espias.json "%PY%" scripts\evaluate.py --a ppo:runs\parejas_espias\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_espias.json
if not exist %E%\vs_heuristico\parejas_sin_senas.json "%PY%" scripts\evaluate.py --a ppo:runs\parejas_sin_senas\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_sin_senas.json
if not exist %E%\vs_heuristico\parejas_sin_senas_mesa_sin_senas.json "%PY%" scripts\evaluate.py --a ppo:runs\parejas_sin_senas\last.pt --no-signals --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_sin_senas_mesa_sin_senas.json
if not exist %E%\vs_heuristico\parejas_sin_sesgos.json "%PY%" scripts\evaluate.py --a ppo:%E%\parejas_sin_sesgos\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_sin_sesgos.json
if not exist %E%\vs_heuristico\parejas_sin_bonus.json "%PY%" scripts\evaluate.py --a ppo:%E%\parejas_sin_bonus\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_sin_bonus.json
if not exist %E%\vs_heuristico\parejas_sin_aux.json "%PY%" scripts\evaluate.py --a ppo:%E%\parejas_sin_aux\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_sin_aux.json
title Analisis 3/6: ablaciones con partidos duplicados
if not exist %E%\ablaciones_duplicado\parejas_senas\ablaciones.json "%PY%" scripts\ablations.py --checkpoint runs\parejas_senas\last.pt --matches 1000 --duplicado --out %E%\ablaciones_duplicado\parejas_senas
if not exist %E%\ablaciones_duplicado\parejas_espias\ablaciones.json "%PY%" scripts\ablations.py --checkpoint runs\parejas_espias\last.pt --matches 1000 --duplicado --out %E%\ablaciones_duplicado\parejas_espias
if not exist %E%\ablaciones_duplicado\parejas_sin_sesgos\ablaciones.json "%PY%" scripts\ablations.py --checkpoint %E%\parejas_sin_sesgos\last.pt --matches 1000 --duplicado --out %E%\ablaciones_duplicado\parejas_sin_sesgos
if not exist %E%\ablaciones_duplicado\parejas_sin_bonus\ablaciones.json "%PY%" scripts\ablations.py --checkpoint %E%\parejas_sin_bonus\last.pt --matches 1000 --duplicado --out %E%\ablaciones_duplicado\parejas_sin_bonus
if not exist %E%\ablaciones_duplicado\parejas_sin_aux\ablaciones.json "%PY%" scripts\ablations.py --checkpoint %E%\parejas_sin_aux\last.pt --matches 1000 --duplicado --out %E%\ablaciones_duplicado\parejas_sin_aux
title Analisis 4/6: diccionarios con marcadores sorteados
if not exist %E%\marcador_sorteado\parejas_senas\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint runs\parejas_senas\last.pt --zero-score-prob 0.3 --out %E%\marcador_sorteado\parejas_senas
if not exist %E%\marcador_sorteado\parejas_espias\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint runs\parejas_espias\last.pt --zero-score-prob 0.3 --out %E%\marcador_sorteado\parejas_espias
if not exist %E%\marcador_sorteado\parejas_sin_sesgos\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %E%\parejas_sin_sesgos\last.pt --zero-score-prob 0.3 --out %E%\marcador_sorteado\parejas_sin_sesgos
if not exist %E%\marcador_sorteado\parejas_sin_bonus\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %E%\parejas_sin_bonus\last.pt --zero-score-prob 0.3 --out %E%\marcador_sorteado\parejas_sin_bonus
if not exist %E%\marcador_sorteado\parejas_sin_aux\diccionario.json "%PY%" scripts\analyze_signals.py --checkpoint %E%\parejas_sin_aux\last.pt --zero-score-prob 0.3 --out %E%\marcador_sorteado\parejas_sin_aux
title Analisis 5/6: liga
if not exist %E%\liga\liga.json "%PY%" scripts\league.py --runs runs\parejas_senas runs\parejas_espias runs\parejas_sin_senas %E%\parejas_sin_sesgos %E%\parejas_sin_bonus %E%\parejas_sin_aux --per-run 1 --matches 400 --out %E%\liga
title Analisis 6/6: reporte
if not exist %E%\reporte\resumen.json "%PY%" scripts\report.py --runs %E% --out %E%\reporte
title Experimentos terminados
pause
