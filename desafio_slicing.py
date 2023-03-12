cadena = "acitametaM ,5.8 ,otipeP ordeP"
print(f"su cadena se daño y sale asi {cadena}, ya se corrige...")

cadena_formateada = cadena[::-1]
nombre_alumno = cadena_formateada[:12]
nota = cadena_formateada[14:17]
materia = cadena_formateada[19:]

print(f"{nombre_alumno} ha sacado un {nota} en {materia}")

