partidos_ganados = int(input("Cuantos partidos se ganaron?"))
partidos_perdidos = int(input("Cuantos partidos se ganaron?"))
partidos_empatados = int(input("Cuantos partidos se ganaron?"))
partidos = partidos_empatados+partidos_ganados+partidos_perdidos
if partidos>20:
    print("numero de partidos incorrectos")
else:
    puntos_ganados=partidos_ganados*3
    puntos_empatados=partidos_empatados
promedio=(puntos_empatados+puntos_ganados)/partidos

print(f"El promedio total es {promedio}")