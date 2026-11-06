@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set E=runs_experimentos
title Analisis 1/6: diccionarios de los controles
python scripts\analyze_signals.py --checkpoint %E%\parejas_sin_sesgos\last.pt
python scripts\analyze_signals.py --checkpoint %E%\parejas_sin_bonus\last.pt
python scripts\analyze_signals.py --checkpoint %E%\parejas_sin_aux\last.pt
title Analisis 2/6: contra el heuristico, partidos duplicados
python scripts\evaluate.py --a ppo:runs\parejas_senas\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_senas.json
python scripts\evaluate.py --a ppo:runs\parejas_espias\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_espias.json
python scripts\evaluate.py --a ppo:runs\parejas_sin_senas\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_sin_senas.json
python scripts\evaluate.py --a ppo:runs\parejas_sin_senas\last.pt --no-signals --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_sin_senas_mesa_sin_senas.json
python scripts\evaluate.py --a ppo:%E%\parejas_sin_sesgos\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_sin_sesgos.json
python scripts\evaluate.py --a ppo:%E%\parejas_sin_bonus\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_sin_bonus.json
python scripts\evaluate.py --a ppo:%E%\parejas_sin_aux\last.pt --matches 2000 --duplicado --json %E%\vs_heuristico\parejas_sin_aux.json
title Analisis 3/6: ablaciones con partidos duplicados
python scripts\ablations.py --checkpoint runs\parejas_senas\last.pt --matches 1000 --duplicado --out %E%\ablaciones_duplicado\parejas_senas
python scripts\ablations.py --checkpoint runs\parejas_espias\last.pt --matches 1000 --duplicado --out %E%\ablaciones_duplicado\parejas_espias
python scripts\ablations.py --checkpoint %E%\parejas_sin_sesgos\last.pt --matches 1000 --duplicado --out %E%\ablaciones_duplicado\parejas_sin_sesgos
python scripts\ablations.py --checkpoint %E%\parejas_sin_bonus\last.pt --matches 1000 --duplicado --out %E%\ablaciones_duplicado\parejas_sin_bonus
python scripts\ablations.py --checkpoint %E%\parejas_sin_aux\last.pt --matches 1000 --duplicado --out %E%\ablaciones_duplicado\parejas_sin_aux
title Analisis 4/6: diccionarios con marcadores sorteados
python scripts\analyze_signals.py --checkpoint runs\parejas_senas\last.pt --zero-score-prob 0.3 --out %E%\marcador_sorteado\parejas_senas
python scripts\analyze_signals.py --checkpoint runs\parejas_espias\last.pt --zero-score-prob 0.3 --out %E%\marcador_sorteado\parejas_espias
python scripts\analyze_signals.py --checkpoint %E%\parejas_sin_sesgos\last.pt --zero-score-prob 0.3 --out %E%\marcador_sorteado\parejas_sin_sesgos
python scripts\analyze_signals.py --checkpoint %E%\parejas_sin_bonus\last.pt --zero-score-prob 0.3 --out %E%\marcador_sorteado\parejas_sin_bonus
python scripts\analyze_signals.py --checkpoint %E%\parejas_sin_aux\last.pt --zero-score-prob 0.3 --out %E%\marcador_sorteado\parejas_sin_aux
title Analisis 5/6: liga
python scripts\league.py --runs runs\parejas_senas runs\parejas_espias runs\parejas_sin_senas %E%\parejas_sin_sesgos %E%\parejas_sin_bonus %E%\parejas_sin_aux --per-run 1 --matches 400 --out %E%\liga
title Analisis 6/6: reporte
python scripts\report.py --runs %E% --out %E%\reporte
title Experimentos terminados
pause
