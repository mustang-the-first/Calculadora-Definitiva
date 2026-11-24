# Calculadora matemática

Página estática (`public/index.html`) más una función Python en Vercel (`api/calc.py`) que resuelve con SymPy. El resultado se muestra con KaTeX.

Operaciones: derivadas, integrales (indefinida y definida), límites, series de Taylor, sumatorias, simplificar, expandir, factorizar, fracciones parciales, ecuaciones y desigualdades, y evaluar una cuenta.

```
calculadora-calculo/
├── api/calc.py        # POST /api/calc: todas las operaciones
├── public/index.html  # interfaz
├── pyproject.toml     # dependencias y punto de entrada de la función (lo pide Vercel)
├── requirements.txt   # sympy (copia de las dependencias; mpmath viene con él)
└── vercel.json        # tiempo máximo de la función
```

## Desplegar

Opción A, desde GitHub:

1. Subí esta carpeta a un repositorio.
2. En vercel.com: Add New, Project, importá el repositorio y apretá Deploy. No hace falta cambiar ningún ajuste.

Opción B, desde la terminal:

```
npm i -g vercel
vercel --prod
```

Para probarla en tu máquina antes de publicar: `vercel dev` dentro de esta carpeta.

## Qué hace cada modo

| Modo | Entrada | Qué devuelve |
| --- | --- | --- |
| Derivada | `f(x)` y orden (1 a 5) | La derivada, con paso a paso |
| Integral indefinida | `f(x)` | La primitiva más `C`, con paso a paso |
| Integral definida | `f(x)`, límites (pueden ser `-oo` y `oo`) | El valor exacto y el aproximado; si no hay primitiva elemental, se aproxima y se avisa |
| Límite | `f(x)`, punto (puede ser `oo`), lado (ambos, `+` o `-`) | El valor, `oo`, o «no existe» si los lados difieren o oscila; con paso a paso |
| Taylor | `f(x)`, centro y grado (1 a 10) | El polinomio más `O(...)`, con paso a paso |
| Sumatoria | término en `n`, límites enteros o `oo` | La suma en forma cerrada, su valor aproximado, o «diverge» |
| Simplificar, expandir, factorizar, fracciones parciales | una expresión | El resultado (avisa si ya estaba así) |
| Resolver | ecuación (`=`) o desigualdad (`<`, `>`, `<=`, `>=`) en `x` | Soluciones reales; paso a paso para polinomios y cocientes |
| Evaluar | una cuenta sin variables | Valor exacto y decimal |

## Cómo funciona

- `POST /api/calc` recibe `{"op": ..., "expr": "x^2*sin(x)", "a": "0", "b": "pi", "order": 1, "side": "both"}` y devuelve `{ok, input_latex, result_latex, result_text, ...}`. `op` es una de: `diff`, `int`, `defint`, `limit`, `taylor`, `sum`, `simplify`, `expand`, `factor`, `apart`, `solve`, `eval`. Según la operación se usan `a` y `b` (límites, punto o centro), `order` (derivada o grado) y `side` (solo en límites).
- La respuesta puede traer también `lead` (el signo que va antes del resultado, por ejemplo `S =` en las ecuaciones), `note` (un aviso), `approx` y `numeric` (valor aproximado) y `steps` (el paso a paso).
- `pyproject.toml` declara el punto de entrada (`api.calc:handler`). Sin ese archivo, el build de Vercel falla con `PYTHON_ENTRYPOINT_NOT_FOUND`.
- Además de `POST /api/calc`, la función responde `GET /` con la página, por si Vercel no sirve `public/` por su cuenta.
- El paso a paso de las derivadas usa reglas propias (suma, producto, cociente, cadena, potencia, tablas). El de las integrales usa `manualintegrate` de SymPy y se verifica derivando la primitiva. El de Taylor, los límites y las ecuaciones también se verifica. Si no se puede armar un paso a paso confiable, la respuesta va igual sin él.
- El resultado se envía primero y el paso a paso después, con su propio tope de 4 segundos, así que un paso a paso lento no pierde el resultado.
- Los campos de la página son un editor matemático, [MathLive](https://mathlive.io), que se carga desde un CDN: la fracción se ve apilada, el exponente arriba y la raíz con su símbolo. La página convierte lo que se ve (LaTeX) a texto (`texToAscii`, dentro de `index.html`) antes de mandarlo al servidor. Si el editor no carga, los campos quedan como texto simple (`x^2`, `sqrt(x)`).
- El teclado de la página inserta fracciones, potencias, raíces, logaritmos, etc. en el campo activo. Cada modo agrega las teclas que necesita: `=`, `<`, `>`, `≤`, `≥` al resolver; `n`, `n!` y `(−1)ⁿ` en sumatorias; `n!` al evaluar. El editor no abre el teclado del sistema, así que en celulares el panel queda siempre abierto.
- `log(x)` es el logaritmo en base 10 en el editor (se manda como `log10(x)`), `ln(x)` el natural y `logₐ` el de cualquier base (`log(x, a)`). En el servidor, `log(x)` a secas sigue siendo el natural.
- La entrada solo acepta una lista de nombres (`x`, `n`, `pi`, `e`, `oo`, `sin`, `cos`, `exp`, `ln`, `log10`, `sqrt`, `factorial`, `sen`, `tg`, etc.). Todo lo demás se rechaza antes de llegar a SymPy, porque su parser usa `eval` por dentro. También se rechazan números de 7 o más cifras y expresiones de más de 200 caracteres.
- Cada cálculo corre en un proceso aparte con tope de 8 segundos y 1 GB de memoria. Si SymPy se cuelga con una integral difícil, el proceso se corta y el usuario recibe un mensaje.
- `maxDuration` en `vercel.json` está en 15 segundos. Si tu plan permite menos, bajalo.

## Limitaciones conocidas

- La variable es `x` (y `n` en sumatorias). Ecuaciones y desigualdades se resuelven solo en `x` y solo con soluciones reales.
- En el editor, después de escribir un exponente, una fracción o una raíz hay que salir de ahí con la flecha derecha (la del teclado de la página o la de la compu); si no, lo que sigue queda adentro.
- El editor y las fórmulas dependen de CDN (jsDelivr). Sin conexión a ellos, la página sigue andando con campos de texto simple.
- Desigualdades con funciones trigonométricas no se resuelven: SymPy devuelve un solo período y eso sería engañoso, así que se avisa.
- Ecuaciones sin solución con fórmulas se aproximan numéricamente buscando raíces entre −50 y 50, y se avisa que son valores aproximados.
- En sumatorias, el veredicto de convergencia usa criterios propios (comparación, razón, raíz, integral, series alternadas). Si ninguno alcanza, la respuesta dice que no se pudo determinar en lugar de adivinar.
- En integrales con funciones especiales (erf, Si, etc.) el paso a paso muestra solo la fórmula de tabla.
- Algunas sustituciones que elige SymPy son válidas pero distintas de las que usaría un profesor.
- `∫ 1/x dx` devuelve `ln(x) + C`, no `ln|x| + C`.
- Las integrales sin primitiva elemental dan error en la indefinida y se aproximan numéricamente en la definida.
- Los ángulos se interpretan en radianes.
- La primera llamada después de un rato sin uso puede tardar unos segundos (arranque en frío de SymPy).
