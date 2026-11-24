"""Calculadora de cálculo y álgebra: función serverless para Vercel, con SymPy.

POST /api/calc
  {"op": "diff", "expr": "x^2*sin(x)", "order": 1}
  {"op": "int", "expr": "x*exp(x)"}
  {"op": "defint", "expr": "x^2", "a": "0", "b": "1"}
  {"op": "limit", "expr": "sin(x)/x", "a": "0", "side": "both" | "+" | "-"}
  {"op": "taylor", "expr": "exp(x)", "a": "0", "order": 5}
  {"op": "sum", "expr": "1/n^2", "a": "1", "b": "oo"}
  {"op": "simplify" | "expand" | "factor" | "apart", "expr": "(x^2-1)/(x-1)"}
  {"op": "solve", "expr": "x^2-4=0"}   (también <, >, <=, >=)
  {"op": "eval", "expr": "sqrt(2)+1/3"}
"""
import json
import math
import multiprocessing as mp
import os
import re
from http.server import BaseHTTPRequestHandler

import mpmath
from sympy import (
    AccumBounds, Abs, Add, CRootOf, Complement, ConditionSet, Dummy, E, Expr, FiniteSet, Float, I,
    ImageSet, Integer, Integral, Interval, Le, Limit, Lt, Ge, Gt, Mul, N, Order, Poly,
    Pow, Rational, S, Sum, Symbol, Union, acos, apart, asin, atan, cancel, cos, cosh, cot,
    csc, diff, exp, expand, factor, factorial, floor, fraction, gammasimp, integrate, lambdify, limit,
    log, nan, nsimplify, oo, pi, sec, series, simplify, sin, sinh, solveset, sqrt, sympify,
    summation, tan, tanh, together, zoo,
)
from sympy import latex as _latex
from sympy.integrals import manualintegrate as mi
from sympy.logic.boolalg import BooleanAtom
from sympy.polys.polyerrors import PolynomialError
from sympy.parsing.sympy_parser import (
    convert_xor, implicit_multiplication_application, parse_expr, rationalize,
    standard_transformations,
)

TIMEOUT_S = 8            # tiempo máximo de cálculo antes de cortar
STEPS_TIMEOUT_S = 4      # tiempo extra para armar el paso a paso (el resultado ya se envió)
MAX_STEPS = 80           # más pasos que esto no se muestran
MEM_LIMIT = 1024 ** 3    # memoria virtual máxima del proceso que calcula
MAX_LEN = 200
MAX_ORDER = 5            # derivadas
MAX_TAYLOR = 10          # grado máximo del polinomio de Taylor
MAX_OUT = 4000           # largo máximo del resultado en LaTeX
BIG_BITS = 12000         # enteros más grandes que esto se muestran en notación científica
STEP_OPS = ("diff", "int", "defint", "limit", "taylor", "solve")
OPS = ("diff", "int", "defint", "limit", "taylor", "sum",
       "simplify", "expand", "factor", "apart", "solve", "eval")
SIDES = ("both", "+", "-")
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "index.html")


class UserError(ValueError):
    """Error con un mensaje pensado para mostrarse tal cual al usuario."""


# Real: así abs(x) deriva a sign(x) y sqrt(x^2) da |x|, como se espera en un curso de análisis.
x = Symbol("x", real=True)
# Índice de las sumatorias.
n = Symbol("n", integer=True)
VARS = {"x": x, "n": n}

# Vocabulario permitido. parse_expr usa eval por dentro, así que todo identificador
# que no esté acá se rechaza ANTES de parsear.
BASE_NAMES = {
    "pi": pi, "e": E, "oo": oo,
    "sin": sin, "cos": cos, "tan": tan, "cot": cot, "sec": sec, "csc": csc,
    "asin": asin, "acos": acos, "atan": atan,
    "arcsin": asin, "arccos": acos, "arctan": atan,
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "exp": exp, "log": log, "ln": log, "sqrt": sqrt, "abs": Abs, "factorial": factorial,
    # log(x) y ln(x) son el logaritmo natural; log10(x) es en base 10 y log(x, b) en base b.
    "log10": lambda u: log(u, 10),
    # Nombres en castellano.
    "sen": sin, "tg": tan, "ctg": cot, "cosec": csc, "senh": sinh, "tgh": tanh,
    "arcsen": asin, "arctg": atan,
}
TRIG = (sin, cos, tan, cot, sec, csc)
TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)
TRANSFORMS_EXACT = TRANSFORMS + (rationalize,)  # 0.1 pasa a ser 1/10: cuentas exactas

# Se rechaza todo salvo estos caracteres. "!" es el factorial; = < > solo llegan a _parse
# ya separados (ver _split_relation).
_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/^().,!\s]+$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_BIGNUM = re.compile(r"\d{7,}")
_ATTR = re.compile(r"\.\s*[A-Za-z_]")  # x.algo: acceso a atributos, nunca hace falta


def _names(var):
    names = dict(BASE_NAMES)
    if var:
        names[var] = VARS[var]
    return names


def _parse(text, what="la expresión", var="x", exact=False, evaluate=True):
    """Convierte texto en una expresión de SymPy. var es la única variable permitida (o None)."""
    s = text.strip() if isinstance(text, str) else ""
    if not s:
        raise UserError(f"Falta {what}.")
    if len(s) > MAX_LEN:
        raise UserError(f"{what.capitalize()} es demasiado larga (máximo {MAX_LEN} caracteres).")
    if not _CHARS.match(s) or "__" in s or _ATTR.search(s):
        raise UserError(f"Hay caracteres no permitidos en {what}.")
    if _BIGNUM.search(s):
        raise UserError("Hay un número demasiado grande.")
    names = _names(var)
    for name in _IDENT.findall(s):
        if name in names:
            continue
        if name in VARS:
            if var:
                raise UserError(f"En este modo la variable es {var}, no {name}.")
            raise UserError(
                f"{what.capitalize()} tiene que ser un número (por ejemplo 0, 1, pi, oo): no puede llevar «{name}»."
            )
        usable = f"{var}, pi, e" if var else "pi, e"
        raise UserError(f"No reconozco «{name}». Usá {usable} y funciones como sin, cos, exp, ln, sqrt.")
    try:
        out = parse_expr(s, local_dict=names, evaluate=evaluate,
                         transformations=TRANSFORMS_EXACT if exact else TRANSFORMS)
    except Exception:
        raise UserError(f"No pude interpretar {what}. Revisá paréntesis y operadores.") from None
    if not isinstance(out, Expr):
        raise UserError(f"No pude interpretar {what}.")
    return out


def _num(text, what):
    """Número exacto sin variables (0, 1/2, pi, oo, -oo...). Los decimales pasan a fracciones."""
    v = _parse(text, what, var=None)
    if v.has(Float):
        v = nsimplify(v, rational=True)
    if v.has(nan, zoo) or not v.is_number:
        raise UserError(f"{what.capitalize()} tiene que ser un número (por ejemplo 0, 1, pi, oo).")
    return v


def latex(expr):
    return _latex(expr, ln_notation=True)


def _pretty(expr):
    """Texto plano que se puede volver a pegar en la calculadora (^ y ln)."""
    txt = re.sub(r"\blog\(", "ln(", str(expr).replace("**", "^"))
    return re.sub(r"\bE\b", "e", txt).replace("Abs(", "abs(")


def _plain(v):
    """Texto corto para notas: ∞ en vez de oo."""
    if v == oo:
        return "+∞"
    if v == -oo:
        return "−∞"
    return _pretty(v)


def _fin(v):
    """True si v es un número finito y bien definido."""
    return v.is_finite is True and not v.has(nan, zoo, AccumBounds)


def _same(u, v):
    """True si u y v son el mismo valor (exacto o, si no, numéricamente)."""
    try:
        if u == v:
            return True
        if not (_fin(u) and _fin(v)):
            return False
        d = u - v
        return d == 0 or simplify(d) == 0 or abs(complex(N(d, 20))) < 1e-9
    except Exception:
        return False


def _approx_str(v, digits=12):
    return str(N(v, digits))


def _dec_str(v, digits=12):
    """Decimal sin ceros de relleno (0.300000000000 -> 0.3)."""
    t = str(N(v, digits))
    if "." in t and "e" not in t:
        t = t.rstrip("0").rstrip(".")
    return t


def _irrational(v):
    return _fin(v) and v.is_number and not v.is_Rational


def _huge_fraction(v):
    """Fracción cuyo numerador o denominador tiene más de 40 dígitos: no se muestra exacta."""
    return v.is_Rational and not v.is_Integer and max(len(str(abs(v.p))), len(str(v.q))) > 40


def _bigint(v):
    return v.is_Rational and max(abs(v.p).bit_length(), v.q.bit_length()) > BIG_BITS


def _finish(out):
    """Último control antes de responder: el resultado tiene que poder mostrarse."""
    if len(out.get("result_latex", "")) > MAX_OUT or len(out.get("result_text", "")) > MAX_OUT:
        raise UserError("El resultado es demasiado largo para mostrarlo.")
    return out


# ------------------------------------------------------------------ conjuntos (ecuaciones)

_REL_TEX = {"=": "=", "<": "<", ">": ">", "<=": r"\le", ">=": r"\ge"}
_REL_FN = {"<": Lt, ">": Gt, "<=": Le, ">=": Ge}
_REL_RE = re.compile(r"(<=|>=|=|<|>)")


def _split_relation(text):
    """'x^2-4=0' -> ('x^2-4', '=', '0'). Sin signo de relación se toma '= 0'."""
    s = text.strip() if isinstance(text, str) else ""
    if not s:
        raise UserError("Falta la ecuación.")
    s = s.replace("≤", "<=").replace("≥", ">=").replace("==", "=")
    parts = _REL_RE.split(s)
    if len(parts) == 1:
        return s, "=", "0"
    if len(parts) != 3:
        raise UserError("Escribí una sola igualdad o desigualdad (por ejemplo x^2 - 4 = 0 o x + 1 < 3).")
    left, rel, right = (p.strip() for p in parts)
    if not left or not right:
        raise UserError("Falta uno de los lados de la ecuación.")
    return left, rel, right


def _sorted_elems(s):
    try:
        return sorted(s, key=lambda e: float(N(e)))
    except Exception:
        return list(s)


def _set_text(s):
    """Texto plano de un conjunto solución."""
    if s == S.EmptySet:
        return "{}"
    if s == S.Reals:
        return "R"
    if isinstance(s, FiniteSet):
        return "{" + ", ".join(_pretty(e) for e in _sorted_elems(s)) + "}"
    if isinstance(s, Interval):
        lb = "(" if s.left_open else "["
        rb = ")" if s.right_open else "]"
        return f"{lb}{_pretty(s.start)}, {_pretty(s.end)}{rb}"
    if isinstance(s, Union):
        return " U ".join(_set_text(a) for a in s.args)
    if isinstance(s, Complement):
        return f"{_set_text(s.args[0])} \\ {_set_text(s.args[1])}"
    if isinstance(s, ImageSet):
        lam = s.lamda
        return "{" + _pretty(lam.expr) + " | " + _pretty(lam.variables[0]) + " en Z}"
    return _pretty(s)


def _numeric_roots(g, lo=-50.0, hi=50.0, steps=4000):
    """Raíces reales aproximadas de g(x)=0 por cambio de signo y bisección (para ecuaciones sin fórmula)."""
    fn = lambdify(x, g, "math")

    def ev(t):
        try:
            v = float(fn(t))
            return v if math.isfinite(v) else None
        except Exception:
            return None

    roots = []
    h = (hi - lo) / steps
    pt, pv = lo, ev(lo)
    for i in range(1, steps + 1):
        t = lo + i * h
        v = ev(t)
        if pv is not None and v is not None:
            if pv == 0:
                roots.append(pt)
            elif pv * v < 0:
                a_, b_, fa = pt, t, pv
                for _ in range(100):
                    m = (a_ + b_) / 2
                    fm = ev(m)
                    if fm is None:
                        break
                    if fm == 0:
                        a_ = b_ = m
                        break
                    if fa * fm < 0:
                        b_ = m
                    else:
                        a_, fa = m, fm
                r = (a_ + b_) / 2
                fr = ev(r)
                # en un polo la función se va a infinito: ahí no hay raíz
                if fr is not None and abs(fr) <= 1e-6 * max(1.0, abs(pv), abs(v)):
                    roots.append(r)
        pt, pv = t, v
    if pv == 0:
        roots.append(pt)
    roots.sort()
    uniq = []
    for r in roots:
        if not uniq or abs(r - uniq[-1]) > 1e-7:
            uniq.append(r)
    return uniq[:12]


# ------------------------------------------------------------------ cálculo

_CACHE = {}  # datos que _compute le deja a _steps (corren en el mismo proceso)


def _lim_tex(f, a, side):
    arrow = latex(a)
    if not a.is_infinite and side in ("+", "-"):
        arrow += "^{" + side + "}"
    inner = r"\left(" + latex(f) + r"\right)" if f.is_Add else latex(f)
    return rf"\lim_{{x \to {arrow}}} {inner}"


def _op_limit(f, a_s, side):
    a = _num(a_s, "el punto al que tiende x") if (a_s or "").strip() else None
    if a is None:
        raise UserError("Falta el punto al que tiende x.")
    if a.is_infinite and a not in (oo, -oo):
        raise UserError("El punto tiene que ser un número, oo o -oo.")
    head = _lim_tex(f, a, side)
    sides_note = None
    try:
        if a.is_infinite:
            val = limit(f, x, a)
        elif side == "both":
            lp, lm = limit(f, x, a, "+"), limit(f, x, a, "-")
            if lp.has(Limit) or lm.has(Limit):
                raise UserError("No pude calcular este límite.")
            if lp.has(AccumBounds) or lm.has(AccumBounds):
                val = AccumBounds(-oo, oo)
            elif _same(lp, lm) or lp == lm:
                val = lp
            else:
                sides_note = (f"Por la izquierda tiende a {_plain(lm)} y por la derecha a {_plain(lp)}: "
                              "los límites laterales son distintos.")
                val = None
        else:
            val = limit(f, x, a, side)
    except UserError:
        raise
    except Exception:
        raise UserError("No pude calcular este límite.") from None

    if val is None:
        return {"ok": True, "input_latex": head, "result_latex": r"\text{no existe}",
                "result_text": "no existe", "note": sides_note}
    if val.has(Limit) or val is nan or val.has(nan):
        raise UserError("No pude calcular este límite.")
    if val.has(AccumBounds):
        return {"ok": True, "input_latex": head, "result_latex": r"\text{no existe}",
                "result_text": "no existe",
                "note": "La función oscila y no se acerca a ningún valor."}
    if val == zoo:
        return {"ok": True, "input_latex": head, "result_latex": r"\text{no existe}",
                "result_text": "no existe", "note": "La función crece sin acotarse y sin signo definido."}
    _CACHE["limit"] = val
    out = {"ok": True, "input_latex": head, "result_latex": latex(val), "result_text": _pretty(val)}
    if _irrational(val):
        out["numeric"] = _approx_str(val)
    return out


def _asc_terms(coeffs, base):
    """[(negativo, término)] de c·base^k en orden creciente de k.

    El signo se separa del coeficiente y el término se arma sin evaluar: si no, SymPy reparte
    -1/4·(x - 2) como -x/4 + 1/2 y al volver a unir los signos la cuenta queda mal.
    """
    out = []
    for k in sorted(coeffs):
        c = sympify(coeffs[k])
        if c == 0:
            continue
        neg = bool(c.could_extract_minus_sign())
        m = -c if neg else c
        if k == 0:
            t = m
        else:
            pw = base if k == 1 else Pow(base, k)
            t = pw if m == 1 else Mul(*(m.args if m.is_Mul else (m,)), pw, evaluate=False)
        out.append((neg, t))
    return out


def _join_terms(terms, fmt):
    if not terms:
        return "0"
    parts = []
    for i, (neg, t) in enumerate(terms):
        body = fmt(t)
        parts.append(("- " if neg else "") + body if i == 0 else ("- " if neg else "+ ") + body)
    return " ".join(parts)


def _taylor_poly(f, a, order):
    """({k: coef}, hay_resto) del desarrollo de f en potencias de (x - a) hasta grado order; None si no es polinómico."""
    ser = series(f, x, a, order + 1)
    has_rest = ser.has(Order)
    poly = ser.removeO()
    t = Dummy("t")
    q = expand(poly.subs(x, a + t))
    try:
        P = Poly(q, t)
    except PolynomialError:
        return None, ser, has_rest
    coeffs = {m[0]: simplify(c) for m, c in P.terms() if m[0] <= order}
    return coeffs, ser, has_rest


def _op_taylor(f, a_s, order):
    a = _num(a_s, "el centro") if (a_s or "").strip() else Integer(0)
    if a.is_infinite:
        raise UserError("El centro tiene que ser un número finito.")
    try:
        coeffs, ser, has_rest = _taylor_poly(f, a, order)
    except Exception:
        raise UserError("No pude desarrollar esta función alrededor de ese punto.") from None
    base = x if a == 0 else x - a
    where = "x = 0 (Maclaurin)" if a == 0 else f"x = {_plain(a)}"
    if coeffs is None:
        raise UserError("Esta función no tiene un desarrollo de Taylor en ese punto "
                        "(no está definida ahí o tiene una singularidad).")
    terms = _asc_terms(coeffs, base)
    rest = ""
    if has_rest:
        inner = latex(base) if a == 0 else r"\left(" + latex(base) + r"\right)"
        rest = r" + O\left(" + inner + rf"^{{{order + 1}}}\right)"
    note = f"Polinomio de Taylor hasta grado {order} centrado en {where}."
    if any(coeffs.get(k, 0) == 0 for k in range(order + 1)):
        note += " Los términos que no aparecen tienen coeficiente 0."
    _CACHE["taylor"] = (a, coeffs)
    return {"ok": True, "input_latex": r"f(x) = " + latex(f), "result_latex": _join_terms(terms, latex) + rest,
            "result_text": _join_terms(terms, _pretty), "note": note}


def _lim_n(e):
    try:
        v = limit(e, n, oo)
        return None if v.has(Limit) else v
    except Exception:
        return None


def _series_verdict(f):
    """('conv' | 'div' | None, motivo). Criterios propios: el de SymPy se equivoca (p. ej. con 1/(n·ln n))."""
    t = _lim_n(f)
    if t is not None and t != 0 and not (isinstance(t, AccumBounds) and t.min <= 0 <= t.max):
        return "div", "los términos no tienden a 0"
    try:
        ratio = _lim_n(Abs(gammasimp(f.subs(n, n + 1) / f)))
        if ratio is not None and ratio.is_number and not ratio.has(AccumBounds):
            if bool(ratio < 1):
                return "conv", "criterio del cociente"
            if bool(ratio > 1):
                return "div", "criterio del cociente"
    except Exception:
        pass
    for p in (Integer(2), Rational(3, 2), Rational(5, 4)):
        c = _lim_n(n ** p * Abs(f))
        if c is not None and (c.is_finite is True or (isinstance(c, AccumBounds) and c.is_finite is True)):
            return "conv", f"comparación con la serie 1/n^{_pretty(p)}"
    c = _lim_n(n * f)
    if c is not None and not c.has(AccumBounds) and c != 0:
        return "div", "comparación con la serie armónica 1/n"
    return None, None


def _partial_sum(fn, a, terms):
    return sum((fn(mpmath.mpf(k)) for k in range(int(a), int(a) + terms)), mpmath.mpf(0))


def _numeric_series(f, a):
    """Valor (texto) de una serie que ya se sabe convergente, o None si no se puede dar con confianza.

    El evalf de SymPy falla en la 4.ª cifra con series oscilantes (sin(n)/n²). Se usa mpmath con dos
    métodos que fallan en casos opuestos: Richardson+Shanks se desvía en series positivas de decaimiento
    lento y Euler–Maclaurin tiene error ~1e-6 en las oscilantes. Se exige coherencia entre ambos y con
    las sumas parciales, y se muestran solo las cifras que esa coherencia justifica.
    """
    fn = lambdify(n, f, "mpmath")
    with mpmath.workdps(30):
        v_rs = mpmath.nsum(fn, [int(a), mpmath.inf])                    # Richardson + Shanks + E-M
        v_em = mpmath.nsum(fn, [int(a), mpmath.inf], method="e")        # Euler-Maclaurin
        if mpmath.im(v_rs) != 0 or mpmath.im(v_em) != 0 or not mpmath.isfinite(v_em):
            return None
        v_rs, v_em = mpmath.re(v_rs), mpmath.re(v_em)
        s1, s2 = _partial_sum(fn, a, 1000), _partial_sum(fn, a, 2000)
        if abs(v_em - s2) > 0.05 * max(1, abs(v_em)) + 20 * abs(s2 - s1):
            return None
        d = abs(v_rs - v_em) / max(1, abs(v_em))
        head = [fn(mpmath.mpf(k)) for k in range(int(a), int(a) + 200)]
        one_signed = all(t >= 0 for t in head) or all(t <= 0 for t in head)
        if d < 1e-9:
            return mpmath.nstr(v_em, 8)
        if one_signed and d < 0.1:
            return mpmath.nstr(v_em, 6)       # Euler-Maclaurin es el confiable con términos suaves de un signo
        if not one_signed and d < 1e-4:
            return mpmath.nstr(v_rs, 5)       # con signos alternados el confiable es Richardson+Shanks
        return None


def _op_sum(f, a_s, b_s):
    a = _num(a_s, "el límite inferior")
    b = _num(b_s, "el límite superior")
    if a.is_infinite or a.is_integer is not True:
        raise UserError("El límite inferior tiene que ser un entero (por ejemplo 0 o 1).")
    if not (b == oo or b.is_integer is True):
        raise UserError("El límite superior tiene que ser un entero o oo.")
    if b != oo and b < a:
        raise UserError("El límite superior tiene que ser mayor o igual que el inferior.")
    head = rf"\sum_{{n={latex(a)}}}^{{{latex(b)}}} " + (r"\left(" + latex(f) + r"\right)" if f.is_Add else latex(f))
    out = {"ok": True, "input_latex": head}

    try:
        res = summation(f, (n, a, b))
    except Exception:
        res = None
    if res is not None and res.has(nan, zoo):
        raise UserError("La expresión no está definida para algún valor de n dentro del rango.")

    if res is None or res.has(Sum):
        if b != oo:
            if b - a > 2000:
                raise UserError("No encontré una fórmula y son demasiados términos para sumarlos uno por uno.")
            terms = [f.subs(n, k) for k in range(int(a), int(b) + 1)]
            res = Add(*terms)
            if res.has(nan, zoo):
                raise UserError("La expresión no está definida para algún valor de n dentro del rango.")
            res = simplify(res) if not res.is_Rational else res
        else:
            verdict, why = _series_verdict(f)
            if verdict is None:
                raise UserError("No pude determinar si esta serie converge ni encontrar su suma exacta.")
            if verdict == "div":
                out.update(result_latex=r"\text{diverge}", result_text="diverge",
                           note=f"La serie diverge ({why}).")
                return out
            try:
                txt = _numeric_series(f, a)
            except Exception:
                txt = None
            if txt is None:
                raise UserError("La serie converge, pero no pude calcular su valor de forma confiable.")
            out.update(result_latex=latex(Float(txt)), result_text=txt, approx=True,
                       note=f"La serie converge ({why}); no encontré una suma exacta, así que el valor es aproximado.")
            return out

    if res in (oo, -oo):
        out.update(result_latex=latex(res), result_text=_pretty(res),
                   note="La serie diverge." if res == oo else "La serie diverge a −∞.")
        return out
    if res.is_Add and len(res.args) > 12 and b != oo:
        # sin fórmula cerrada, SymPy devuelve los términos uno por uno: se da el valor numérico
        val = N(res, 12)
        out.update(result_latex=latex(val), result_text=_dec_str(val), approx=True,
                   note=f"No hay una fórmula cerrada: es la suma de {int(b - a + 1)} términos, calculada numéricamente.")
        return out
    if res.has(AccumBounds):
        out.update(result_latex=r"\text{diverge}", result_text="diverge",
                   note="La suma oscila y no converge.")
        return out
    if _huge_fraction(res):
        out.update(result_latex=latex(N(res, 12)), result_text=_dec_str(res), approx=True,
                   note="La suma exacta es una fracción con muchísimos dígitos; se muestra aproximada.")
        return out
    out.update(result_latex=latex(res), result_text=_pretty(res))
    if _irrational(res):
        out["numeric"] = _approx_str(res)
    return out


def _op_algebra(op, f):
    try:
        if op == "apart":
            res = apart(f, x)
        elif op == "simplify":
            res = simplify(f)
        elif op == "expand":
            res = expand(f)
        else:
            res = factor(f)
    except Exception:
        if op == "apart":
            raise UserError("Las fracciones parciales necesitan un cociente de polinomios, "
                            "por ejemplo (x+1)/(x^2-1).") from None
        raise UserError("No pude procesar esta expresión.") from None
    out = {"ok": True, "input_latex": latex(f), "result_latex": latex(res), "result_text": _pretty(res)}
    if res == f:
        out["note"] = {
            "simplify": "Ya estaba en su forma más simple.",
            "expand": "Ya estaba expandida.",
            "factor": "No se puede factorizar más con coeficientes racionales.",
            "apart": "Ya estaba descompuesta, o no se puede descomponer más.",
        }[op]
    return out


def _op_eval(expr_s):
    has_dec = isinstance(expr_s, str) and "." in expr_s
    f = _parse(expr_s, "la cuenta", var=None, exact=True)
    res = f if f.is_Rational else simplify(f)
    if res.has(nan):
        raise UserError("La cuenta no está definida (por ejemplo 0/0 o ∞ − ∞).")
    if res.has(zoo):
        raise UserError("La cuenta no está definida: hay una división por cero o un logaritmo de 0.")
    if res.has(I) or res.is_real is False:
        raise UserError("El resultado no es un número real.")
    if not res.is_number:
        raise UserError("No pude calcular esta cuenta.")
    out = {"ok": True, "input_latex": None}
    if _bigint(res):
        val = N(res, 12)
        out.update(result_latex=latex(val), result_text=str(val), approx=True,
                   note="Es un número enorme: se muestra en notación científica.")
        return out
    if _huge_fraction(res) and not has_dec:
        out.update(result_latex=latex(N(res, 12)), result_text=_dec_str(res), approx=True,
                   note="El resultado exacto es una fracción con muchísimos dígitos; se muestra aproximado.")
        return out
    if has_dec and res.is_Rational:
        # Cuenta con decimales: se responde con decimales (0.1 + 0.2 = 0.3).
        val = N(res, 12)
        exact_dec = all(p in (2, 5) for p in _prime_factors(res.q))
        out.update(result_latex=_dec_str(res), result_text=_dec_str(res))
        if not exact_dec:
            out["approx"] = True
        return out
    out.update(result_latex=latex(res), result_text=_pretty(res))
    if res in (oo, -oo):
        return out
    if _irrational(res) or (res.is_Rational and not res.is_Integer and res.q > 1):
        out["numeric"] = _approx_str(res)
    return out


def _prime_factors(q):
    out, d = [], 2
    q = int(q)
    while d * d <= q:
        while q % d == 0:
            out.append(d)
            q //= d
        d += 1
    if q > 1:
        out.append(q)
    return out


def _op_solve(expr_s):
    ls, rel, rs = _split_relation(expr_s)
    lhs = _parse(ls, "el primer miembro")
    rhs = _parse(rs, "el segundo miembro")
    g = lhs - rhs
    if not (lhs.has(x) or rhs.has(x)):
        raise UserError("Falta la incógnita: la ecuación tiene que llevar x.")
    if rel == "=":
        head = latex(lhs) + " = " + latex(rhs)
    else:
        head = latex(lhs) + " " + _REL_TEX[rel] + " " + latex(rhs)
    if rel != "=" and g.has(*TRIG):
        raise UserError("Las desigualdades con funciones trigonométricas se repiten en cada período "
                        "y no las resuelvo. Las ecuaciones sí.")
    try:
        if rel == "=":
            sol = solveset(g, x, S.Reals)
        else:
            r = _REL_FN[rel](lhs, rhs)
            if isinstance(r, BooleanAtom):
                sol = S.Reals if bool(r) else S.EmptySet
            else:
                sol = solveset(r, x, S.Reals)
    except Exception:
        raise UserError("No pude resolver esta ecuación." if rel == "=" else "No pude resolver esta desigualdad.") from None

    out = {"ok": True, "input_latex": head, "lead": "S ="}
    if sol.has(ConditionSet):
        if rel != "=":
            raise UserError("No pude resolver esta desigualdad en forma exacta.")
        roots = _numeric_roots(g)
        if not roots:
            raise UserError("No encontré soluciones: ni con fórmula ni buscando entre −50 y 50.")
        vals = FiniteSet(*[Float(r, 10) for r in roots])
        out.update(lead=r"S \approx", result_latex=latex(vals), result_text=_set_text(vals), approx=True,
                   note="Soluciones aproximadas, buscadas entre −50 y 50. Pueden faltar raíces dobles o fuera de ese rango.")
        return out
    if sol.has(CRootOf) and isinstance(sol, FiniteSet):
        vals = FiniteSet(*[N(e, 10) for e in sol])
        out.update(lead=r"S \approx", result_latex=latex(vals), result_text=_set_text(vals), approx=True,
                   note="El polinomio no se puede resolver con fórmulas: son las raíces reales aproximadas.")
        return out
    out.update(result_latex=latex(sol), result_text=_set_text(sol))
    _CACHE["solve"] = (lhs, rhs, rel, sol)
    if sol == S.EmptySet:
        out["result_latex"] = r"\varnothing"
        out["note"] = "No hay soluciones reales."
    elif sol == S.Reals:
        out["result_latex"] = r"\mathbb{R}"
        out["note"] = "Se cumple para todo x real."
    elif isinstance(sol, FiniteSet) and any(_irrational(e) for e in sol):
        out["note"] = "Valores aproximados: " + ", ".join("x ≈ " + str(N(e, 8)) for e in _sorted_elems(sol))
    elif isinstance(sol, (Interval, Union)):
        try:
            ends = [e for e in _sorted_elems(sol.boundary) if e.is_finite and _irrational(e)]
        except Exception:
            ends = []  # uniones de conjuntos periódicos (ImageSet) no tienen extremos que listar
        if ends:
            out["note"] = "Extremos aproximados: " + ", ".join(str(N(e, 8)) for e in ends)
    return out


def _compute(op, expr_s, a_s, b_s, order, side="both"):
    _CACHE.clear()
    if op == "solve":
        return _finish(_op_solve(expr_s))
    if op == "eval":
        return _finish(_op_eval(expr_s))
    if op == "sum":
        return _finish(_op_sum(_parse(expr_s, var="n"), a_s, b_s))

    f = _parse(expr_s)

    if op in ("simplify", "expand", "factor", "apart"):
        return _finish(_op_algebra(op, f))
    if op == "limit":
        return _finish(_op_limit(f, a_s, side))
    if op == "taylor":
        return _finish(_op_taylor(f, a_s, order))

    if op == "diff":
        res = simplify(diff(f, x, order))
        d = r"\frac{d}{dx}" if order == 1 else rf"\frac{{d^{{{order}}}}}{{dx^{{{order}}}}}"
        return _finish({
            "ok": True,
            "input_latex": rf"{d}\left({latex(f)}\right)",
            "result_latex": latex(res),
            "result_text": _pretty(res),
        })

    if op == "int":
        res = integrate(f, x)
        if res.has(Integral):
            raise UserError(
                "SymPy no encontró una primitiva en forma cerrada (puede que no exista con "
                "funciones elementales). Probá la integral definida: esa se puede aproximar numéricamente."
            )
        return _finish({
            "ok": True,
            "input_latex": rf"\int {latex(f)}\,dx",
            "result_latex": latex(res) + " + C",
            "result_text": _pretty(res) + " + C",
        })

    # integral definida
    a = _parse(a_s, "el límite inferior", var=None)
    b = _parse(b_s, "el límite superior", var=None)
    head = rf"\int_{{{latex(a)}}}^{{{latex(b)}}} {latex(f)}\,dx"

    res = integrate(f, (x, a, b))
    if res.has(zoo, nan) or res in (oo, -oo):
        raise UserError("La integral diverge o no está definida en ese intervalo.")

    if res.has(Integral):
        # Sin primitiva simbólica: aproximación numérica.
        try:
            val = N(Integral(f, (x, a, b)), 12)
        except Exception:
            raise UserError("No pude calcular esta integral ni exacta ni numéricamente.") from None
        if not (val.is_number and val.is_finite):
            raise UserError("La integral no converge o no se pudo aproximar.")
        return _finish({
            "ok": True,
            "input_latex": head,
            "result_latex": latex(val),
            "result_text": str(val),
            "approx": True,
        })

    out = {
        "ok": True,
        "input_latex": head,
        "result_latex": latex(res),
        "result_text": _pretty(res),
        "approx": False,
    }
    if res.is_number:
        out["numeric"] = str(N(res, 12))
    return _finish(out)


# ------------------------------------------------------------------ paso a paso
# Cada paso es {"depth": sangría, "text": explicación, "latex": fórmula o None}.

Uf = Symbol("u", real=True)  # variable genérica para las derivadas "de tabla"

_FN_ES = {
    sin: "seno", cos: "coseno", tan: "tangente", cot: "cotangente", sec: "secante",
    csc: "cosecante", asin: "arcoseno", acos: "arcocoseno", atan: "arcotangente",
    sinh: "seno hiperbólico", cosh: "coseno hiperbólico", tanh: "tangente hiperbólica",
    log: "logaritmo natural", Abs: "valor absoluto",
}


def _tex(expr):
    """LaTeX de expr; los símbolos auxiliares de SymPy (Dummy) se muestran con su nombre."""
    expr = sympify(expr)
    dummies = expr.atoms(Dummy)
    if dummies:
        expr = expr.xreplace({d: Symbol(d.name) for d in dummies})
    return latex(expr)


def _st(depth, text, tex=None):
    return {"depth": depth, "text": text, "latex": tex}


def _vname(sym):
    """Nombre de una variable para usar en el texto (sin el guion bajo de los Dummy)."""
    name = getattr(sym, "name", None) or str(sym)
    return "θ" if name == "theta" else name


def _dx(expr):
    return r"\frac{d}{dx}\left(" + _tex(expr) + r"\right)"


def _dsteps(f, d, out):
    """Agrega a out los pasos para derivar f respecto de x, con sangría d."""
    r = diff(f, x)
    lhs = _dx(f)

    if not f.has(x):
        out.append(_st(d, "La derivada de una constante es 0.", lhs + " = 0"))
        return
    if f == x:
        out.append(_st(d, "La derivada de x es 1.", lhs + " = 1"))
        return

    if f.is_Add:
        out.append(_st(d, "Regla de la suma: se deriva cada término por separado.",
                       lhs + " = " + " + ".join(_dx(t) for t in f.args)))
        for t in f.args:
            _dsteps(t, d + 1, out)
        return

    if f.is_Mul:
        c, g = f.as_independent(x, as_Add=False)
        if c != 1:
            out.append(_st(d, f"Regla del factor constante: la constante {_pretty(c)} sale de la derivada.",
                           lhs + " = " + _tex(c) + r" \cdot " + _dx(g)))
            _dsteps(g, d + 1, out)
            return
        neg = [a for a in f.args if a.is_Pow and a.exp.is_negative and a.has(x)]
        if neg:
            den = Mul(*[Pow(a.base, -a.exp) for a in neg])
            num = Mul(*[a for a in f.args if a not in neg])
            if num.has(x):
                out.append(_st(d, "Regla del cociente: (u/v)′ = (u′·v − u·v′) / v².",
                               "u = " + _tex(num) + r", \quad v = " + _tex(den)))
                _dsteps(num, d + 1, out)
                _dsteps(den, d + 1, out)
                out.append(_st(
                    d, "Se reemplaza en la fórmula:",
                    lhs + r" = \frac{\left(" + _tex(diff(num, x)) + r"\right)\left(" + _tex(den)
                    + r"\right) - \left(" + _tex(num) + r"\right)\left(" + _tex(diff(den, x))
                    + r"\right)}{\left(" + _tex(den) + r"\right)^{2}}"))
                return
        a, b = f.args[0], Mul(*f.args[1:])
        out.append(_st(d, "Regla del producto: (u·v)′ = u′·v + u·v′.",
                       "u = " + _tex(a) + r", \quad v = " + _tex(b)))
        _dsteps(a, d + 1, out)
        _dsteps(b, d + 1, out)
        out.append(_st(
            d, "Se reemplaza en la fórmula:",
            lhs + r" = \left(" + _tex(diff(a, x)) + r"\right)\left(" + _tex(b)
            + r"\right) + \left(" + _tex(a) + r"\right)\left(" + _tex(diff(b, x)) + r"\right)"))
        return

    if f.is_Pow:
        b, e = f.args
        if not b.has(x):
            if e == x:
                out.append(_st(d, "Derivada de una exponencial: (aˣ)′ = aˣ·ln(a).", lhs + " = " + _tex(r)))
            else:
                out.append(_st(d, f"Regla de la cadena con la exponencial: (aᵘ)′ = aᵘ·ln(a)·u′, con u = {_pretty(e)}.",
                               "u = " + _tex(e)))
                _dsteps(e, d + 1, out)
                out.append(_st(d, "Se reemplaza en la fórmula:",
                               lhs + " = " + _tex(f) + r" \cdot " + _tex(log(b))
                               + r" \cdot \left(" + _tex(diff(e, x)) + r"\right)"))
            return
        if not e.has(x):
            if b == x:
                out.append(_st(d, "Regla de la potencia: (xⁿ)′ = n·xⁿ⁻¹.", lhs + " = " + _tex(r)))
            else:
                out.append(_st(d, f"Regla de la cadena con la potencia: (uⁿ)′ = n·uⁿ⁻¹·u′, con u = {_pretty(b)} y n = {_pretty(e)}.",
                               "u = " + _tex(b)))
                _dsteps(b, d + 1, out)
                out.append(_st(d, "Se reemplaza en la fórmula:",
                               lhs + " = " + _tex(e * b ** (e - 1))
                               + r" \cdot \left(" + _tex(diff(b, x)) + r"\right)"))
            return
        g = exp(e * log(b))
        if not g.is_Pow:
            out.append(_st(d, "La base y el exponente dependen de x: se reescribe uᵛ = e^(v·ln u).",
                           _tex(f) + " = " + _tex(g)))
            _dsteps(g, d + 1, out)
            return

    if isinstance(f, exp):
        u = f.args[0]
        if u == x:
            out.append(_st(d, "La derivada de eˣ es eˣ.", lhs + " = " + _tex(f)))
        else:
            out.append(_st(d, f"Regla de la cadena: (eᵘ)′ = eᵘ·u′, con u = {_pretty(u)}.", "u = " + _tex(u)))
            _dsteps(u, d + 1, out)
            out.append(_st(d, "Se reemplaza en la fórmula:",
                           lhs + " = " + _tex(f) + r" \cdot \left(" + _tex(diff(u, x)) + r"\right)"))
        return

    if len(f.args) == 1 and f.func in _FN_ES:
        u = f.args[0]
        nombre = _FN_ES[f.func]
        outer = diff(f.func(Uf), Uf).subs(Uf, u)
        if u == x:
            out.append(_st(d, f"Derivada de tabla ({nombre}).", lhs + " = " + _tex(outer)))
        else:
            out.append(_st(d, f"Regla de la cadena ({nombre}): se deriva la función exterior y se multiplica por u′, con u = {_pretty(u)}.",
                           "u = " + _tex(u)))
            _dsteps(u, d + 1, out)
            out.append(_st(d, "Se reemplaza en la fórmula:",
                           lhs + " = " + _tex(outer) + r" \cdot \left(" + _tex(diff(u, x)) + r"\right)"))
        return

    out.append(_st(d, "Se deriva con las reglas del cálculo.", lhs + " = " + _tex(r)))


def _diff_steps(f, order):
    out = []
    cur = f
    base = 1 if order > 1 else 0
    for k in range(1, order + 1):
        if order > 1:
            out.append(_st(0, "Primera derivada." if k == 1 else f"Derivada de orden {k}: se deriva el resultado anterior."))
        _dsteps(cur, base, out)
        cur = simplify(diff(cur, x))
        if len(out) > MAX_STEPS:
            return None
    name = "f'(x)" if order == 1 else "f^{(" + str(order) + ")}(x)"
    out.append(_st(0, "Resultado", name + " = " + _tex(simplify(diff(f, x, order)))))
    return out


def _int_tex(expr, var):
    return r"\int " + _tex(expr) + r"\,d" + _tex(var)


def _signed_ints(terms, var):
    parts = []
    for i, t in enumerate(terms):
        t = sympify(t)
        neg = i > 0 and t.could_extract_minus_sign()
        parts.append((" - " if neg else (" + " if i else "")) + _int_tex(-t if neg else t, var))
    return "".join(parts)


def _minus_int(rest, var):
    rest = sympify(rest)
    if rest.could_extract_minus_sign():
        return " + " + _int_tex(-rest, var)
    return " - " + _int_tex(rest, var)


def _isteps(rule, d, out):
    """Agrega los pasos de una regla de manualintegrate. Devuelve False si algún tramo no se puede explicar."""
    t = type(rule).__name__
    v = rule.variable
    if t == "DontKnowRule":
        return False
    if t == "AlternativeRule":
        for alt in rule.alternatives:
            tmp = []
            if _isteps(alt, d, tmp):
                out.extend(tmp)
                return True
        return False

    big_i = _int_tex(rule.integrand, v) if rule.integrand is not None else ""

    if t == "ConstantRule":
        out.append(_st(d, "La integral de una constante es la constante por la variable.",
                       big_i + " = " + _tex(rule.eval())))
        return True
    if t == "ConstantTimesRule":
        out.append(_st(d, f"Se saca la constante {_pretty(rule.constant)} fuera de la integral.",
                       big_i + " = " + _tex(rule.constant) + " " + _int_tex(rule.other, v)))
        return _isteps(rule.substep, d + 1, out)
    if t == "PowerRule":
        out.append(_st(d, "Regla de la potencia: ∫ xⁿ dx = xⁿ⁺¹ / (n + 1).", big_i + " = " + _tex(rule.eval())))
        return True
    if t == "AddRule":
        out.append(_st(d, "Se integra cada término por separado.",
                       big_i + " = " + _signed_ints([s.integrand for s in rule.substeps], v)))
        return all(_isteps(s, d + 1, out) for s in rule.substeps)
    if t == "URule":
        uf, uv = rule.u_func, rule.u_var
        out.append(_st(d, "Sustitución: se elige u y se calcula du.",
                       "u = " + _tex(uf) + r", \quad du = " + _tex(diff(uf, v)) + r"\,d" + _tex(v)))
        out.append(_st(d, "La integral queda en términos de u:",
                       big_i + " = " + _int_tex(rule.substep.integrand, uv)))
        if not _isteps(rule.substep, d + 1, out):
            return False
        out.append(_st(d, f"Se vuelve a la variable {_vname(v)} reemplazando u.", _tex(rule.eval())))
        return True
    if t == "PartsRule":
        if rule.second_step is None:
            return False
        u, dv = rule.u, rule.dv
        vv = rule.v_step.eval()
        out.append(_st(d, "Integración por partes: ∫ u dv = u·v − ∫ v du.",
                       "u = " + _tex(u) + r", \quad dv = " + _tex(dv) + r"\,d" + _tex(v)))
        out.append(_st(d, "Se deriva u y se integra dv:",
                       "du = " + _tex(diff(u, v)) + r"\,d" + _tex(v) + r", \quad v = " + _tex(vv)))
        out.append(_st(d, "Se reemplaza en la fórmula:",
                       big_i + " = " + _tex(u * vv) + _minus_int(rule.second_step.integrand, v)))
        return _isteps(rule.second_step, d + 1, out)
    if t == "CyclicPartsRule":
        out.append(_st(d, "Integración por partes repetida: la integral original reaparece y se despeja.", big_i))
        for pr in rule.parts_rules:
            out.append(_st(d + 1, "Por partes:", "u = " + _tex(pr.u) + r", \quad dv = " + _tex(pr.dv) + r"\,d" + _tex(v)))
        out.append(_st(d, "Despejando la integral original:", big_i + " = " + _tex(rule.eval())))
        return True
    if t in ("RewriteRule", "CompleteSquareRule"):
        txt = "Se reescribe el integrando para poder integrarlo." if t == "RewriteRule" else "Se completa el cuadrado."
        out.append(_st(d, txt, big_i + " = " + _int_tex(rule.rewritten, v)))
        return _isteps(rule.substep, d + 1, out)
    if t == "TrigSubstitutionRule":
        out.append(_st(d, "Sustitución trigonométrica.", _tex(v) + " = " + _tex(rule.func)))
        out.append(_st(d, f"La integral queda en términos de {_vname(rule.theta)}:",
                       big_i + " = " + _int_tex(rule.rewritten, rule.theta)))
        if not _isteps(rule.substep, d + 1, out):
            return False
        out.append(_st(d, f"Se vuelve a la variable {_vname(v)} deshaciendo la sustitución.", _tex(rule.eval())))
        return True

    try:
        res = rule.eval()
    except Exception:
        return False
    out.append(_st(d, "Integral de tabla.", big_i + " = " + _tex(res)))
    return True


def _antiderivative_steps(f):
    """(pasos, F) con F una primitiva de f verificada por derivación; (None, None) si no se puede."""
    rule = mi.integral_steps(f, x)
    if rule.contains_dont_know():
        return None, None
    out = []
    if not _isteps(rule, 0, out) or len(out) > MAX_STEPS:
        return None, None
    F = rule.eval()
    chk = diff(F, x) - f
    if simplify(chk) != 0 and chk.equals(0) is not True:
        return None, None
    return out, F


def _int_steps(f):
    out, F = _antiderivative_steps(f)
    if out is None:
        return None
    out.append(_st(0, "Resultado: se agrega la constante de integración C.",
                   _int_tex(f, x) + " = " + _tex(F) + " + C"))
    return out


def _eval_at(F, p, side):
    """F evaluada en p (con límite si hace falta); None si no es un valor finito."""
    try:
        if p.is_infinite:
            val = limit(F, x, p)
        else:
            val = F.subs(x, p)
            if val.has(nan, zoo, oo, -oo):
                val = limit(F, x, p, side)
        if val.has(nan, zoo) or val in (oo, -oo):
            return None
        return val
    except Exception:
        return None


def _defint_steps(f, a, b, ref):
    out, F = _antiderivative_steps(f)
    if out is None:
        return None
    head = r"\int_{" + _tex(a) + r"}^{" + _tex(b) + "} " + _tex(f) + r"\,dx"
    out.append(_st(0, "Regla de Barrow: ∫ₐᵇ f(x) dx = F(b) − F(a), con F una primitiva de f.",
                   head + r" = \left[" + _tex(F) + r"\right]_{" + _tex(a) + "}^{" + _tex(b) + "}"))
    fb, fa = _eval_at(F, b, "-"), _eval_at(F, a, "+")
    if fb is None or fa is None:
        return None
    val = simplify(fb - fa)
    if ref is not None:  # el paso a paso tiene que coincidir con el resultado ya calculado
        try:
            want = complex(ref)
            if abs(complex(N(val, 15)) - want) > 1e-8 * max(1.0, abs(want)):
                return None
        except Exception:
            return None
    out.append(_st(0, "Se evalúa en los límites (con límites si hace falta).",
                   r"F\left(" + _tex(b) + r"\right) - F\left(" + _tex(a) + r"\right) = \left("
                   + _tex(fb) + r"\right) - \left(" + _tex(fa) + r"\right)"))
    out.append(_st(0, "Resultado", head + " = " + _tex(val)))
    return out


# ------------------------------------------------------------------ pasos: límites

def _lim_wrap(a, side, inner):
    arrow = _tex(a)
    if not a.is_infinite and side in ("+", "-"):
        arrow += "^{" + side + "}"
    return rf"\lim_{{x \to {arrow}}} {inner}"


def _lim_of(a, side, g):
    return _lim_wrap(a, side, r"\left(" + _tex(g) + r"\right)" if g.is_Add else _tex(g))


def _lim_frac(a, side, num, den):
    return _lim_wrap(a, side, r"\frac{" + _tex(num) + "}{" + _tex(den) + "}")


def _subs_at(f, a):
    try:
        return f.subs(x, a)
    except Exception:
        return None


def _lim1(g, a, dirn):
    try:
        v = limit(g, x, a, dirn)
        return None if v.has(Limit) else v
    except Exception:
        return None


def _sign_near(g, a, s):
    """Signo (+1/-1) de g justo a la derecha (s='+') o a la izquierda (s='-') de a; 0 si no se sabe."""
    h = Rational(1, 10 ** 6) if s == "+" else -Rational(1, 10 ** 6)
    try:
        v = float(N(g.subs(x, a + h), 15))
    except Exception:
        return 0
    return 1 if v > 0 else -1 if v < 0 else 0


def _ok_value(v, inf_a):
    return v is not None and (_fin(v) or (inf_a and v in (oo, -oo)))


def _limit_steps(f, a, side, val):
    if not (_fin(val) or val in (oo, -oo)):
        return None
    inf_a = bool(a.is_infinite)
    head = _lim_of(a, side, f)
    dirn = "+" if (inf_a or side == "both") else side
    out = []

    def done():
        out.append(_st(0, "Resultado", head + " = " + _tex(val)))
        return out

    # 1) sustitución directa
    v = _subs_at(f, a)
    if _ok_value(v, inf_a) and _same(v, val):
        if inf_a:
            txt = "Cuando x crece sin límite la expresión se puede evaluar directamente (1/∞ vale 0, ∞² vale ∞, etc.)."
        else:
            txt = f"Se reemplaza x por {_plain(a)}: la expresión queda bien definida, así que ese valor es el límite."
        out.append(_st(0, txt, head + r" = \left. " + _tex(f) + r" \right|_{x=" + _tex(a) + "} = " + _tex(v)))
        return done()

    num, den = fraction(together(f))
    if not den.has(x):
        return None
    nl, dl = _lim1(num, a, dirn), _lim1(den, a, dirn)
    if nl is None or dl is None:
        return None

    # 2) numerador → c ≠ 0 y denominador → 0
    if not inf_a and _fin(nl) and nl != 0 and dl == 0:
        sides = ["+", "-"] if side == "both" else [side]
        signs = {_sign_near(den, a, s_) for s_ in sides}
        if len(signs) == 1 and 0 not in signs:
            ds = signs.pop()
            sg = ds * (1 if nl.is_positive else -1)
            if (oo if sg > 0 else -oo) == val:
                out.append(_st(
                    0, f"El numerador tiende a {_plain(nl)} y el denominador tiende a 0 con valores "
                       f"{'positivos' if ds > 0 else 'negativos'}.",
                    r"\frac{" + _tex(nl) + "}{0^{" + ("+" if ds > 0 else "-") + "}}"))
                out.append(_st(0, "Un número distinto de 0 dividido por algo cada vez más chico crece sin límite, "
                                  f"con signo {'positivo' if sg > 0 else 'negativo'}.", None))
                return done()
        return None

    # 3) indeterminaciones 0/0 e ∞/∞
    zero_form = nl == 0 and dl == 0
    inf_form = nl.is_infinite and dl.is_infinite
    if not (zero_form or inf_form):
        return None
    out.append(_st(0, "Numerador y denominador tienden a " + ("0" if zero_form else "∞")
                   + ": es una indeterminación " + ("0/0." if zero_form else "∞/∞."), None))

    # 3a) cociente de polinomios en el infinito: se comparan los grados
    if inf_a:
        try:
            P, Q = Poly(num, x), Poly(den, x)
        except PolynomialError:
            P = Q = None
        if P is not None:
            p, q, lc = P.degree(), Q.degree(), P.LC() / Q.LC()
            if p < q:
                guess, why = Integer(0), "el denominador tiene mayor grado, crece más rápido y el cociente tiende a 0."
            elif p == q:
                guess, why = lc, "los grados son iguales y el límite es el cociente de los coeficientes principales."
            else:
                s_ = (1 if lc.is_positive else -1) * (1 if a == oo else (-1) ** (p - q))
                guess, why = (oo if s_ > 0 else -oo), (
                    "el numerador tiene mayor grado y el cociente crece sin límite; el signo sale del "
                    "coeficiente principal y de si x tiende a +∞ o −∞.")
            if _same(guess, val):
                out.append(_st(0, "Se comparan los grados del numerador (p) y del denominador (q).",
                               rf"p = {p}, \quad q = {q}"))
                out.append(_st(0, "Para |x| muy grande la expresión se comporta como el cociente de los términos de "
                                  "mayor grado: " + why,
                               r"\frac{" + _tex(num) + "}{" + _tex(den) + r"} \sim \frac{" + _tex(P.LC() * x ** p)
                               + "}{" + _tex(Q.LC() * x ** q) + "} = " + _tex(lc * x ** (p - q))))
                return done()

    # 3b) factor común (0/0 con polinomios)
    if not inf_a and zero_form:
        g = cancel(together(f))
        v2 = _subs_at(g, a)
        if g != f and _ok_value(v2, False) and _same(v2, val):
            out.append(_st(0, "Se factorizan numerador y denominador.",
                           _tex(f) + " = " + r"\frac{" + _tex(factor(num)) + "}{" + _tex(factor(den)) + "}"))
            out.append(_st(0, "Se simplifica el factor que se repite (vale 0 en x = " + _plain(a)
                           + ", pero en un límite x nunca llega a ese valor).", _tex(f) + " = " + _tex(g)))
            out.append(_st(0, "Ahora sí se puede reemplazar.",
                           _lim_of(a, side, g) + r" = \left. " + _tex(g) + r" \right|_{x=" + _tex(a) + "} = " + _tex(v2)))
            return done()

    # 3c) regla de L'Hôpital
    cn, cd = num, den
    for _ in range(3):
        n1, d1 = simplify(diff(cn, x)), simplify(diff(cd, x))
        out.append(_st(0, "Se aplica la regla de L'Hôpital: se derivan numerador y denominador por separado.",
                       _lim_frac(a, side, cn, cd) + " = " + _lim_frac(a, side, n1, d1)))
        g = n1 / d1
        v2 = _subs_at(g, a)
        if _ok_value(v2, inf_a) and _same(v2, val):
            out.append(_st(0, "Ahora se puede reemplazar.", _lim_frac(a, side, n1, d1) + " = " + _tex(v2)))
            return done()
        nl2, dl2 = _lim1(n1, a, dirn), _lim1(d1, a, dirn)
        if nl2 is None or dl2 is None:
            return None
        if not ((nl2 == 0 and dl2 == 0) or (nl2.is_infinite and dl2.is_infinite)):
            return None
        out.append(_st(0, "Sigue siendo una indeterminación: se aplica otra vez.", None))
        cn, cd = n1, d1
    return None


# ------------------------------------------------------------------ pasos: Taylor

def _taylor_steps(f, a, order, coeffs):
    out = [_st(0, "Fórmula del polinomio de Taylor de grado n centrado en a.",
               r"P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}\,(x-a)^{k}"),
           _st(0, "En este caso:", rf"n = {order}, \quad a = {_tex(a)}"),
           _st(0, "Se calcula cada derivada y se evalúa en x = " + _plain(a) + ":", None)]
    cur, vals = f, {}
    for k in range(order + 1):
        if k:
            cur = simplify(diff(cur, x))
        v = _subs_at(cur, a)
        if not (v is not None and _fin(v)):
            v = _lim1(cur, a, "+")
            if v is None or not _fin(v):
                return None
        v = simplify(v)
        vals[k] = v
        nm = "f" + ("'" * k if k <= 3 else f"^{{({k})}}")
        out.append(_st(1, "La función misma." if k == 0 else f"Derivada de orden {k}.",
                       nm + "(x) = " + _tex(cur) + r" \;\Rightarrow\; " + nm + "(" + _tex(a) + ") = " + _tex(v)))
        if len(out) > MAX_STEPS:
            return None
    cs = {k: simplify(vals[k] / factorial(k)) for k in vals}
    for k in range(order + 1):
        if not _same(cs[k], coeffs.get(k, Integer(0))):
            return None
    out.append(_st(0, "Cada coeficiente es la derivada evaluada dividida por k!.",
                   r",\ ".join(rf"c_{{{k}}} = \frac{{{_tex(vals[k])}}}{{{k}!}} = {_tex(cs[k])}" for k in range(order + 1))))
    base = x if a == 0 else x - a
    out.append(_st(0, "Resultado", "P_{" + str(order) + "}(x) = " + _join_terms(_asc_terms(cs, base), _tex)))
    return out


# ------------------------------------------------------------------ pasos: ecuaciones y desigualdades

def _par(v):
    t = _tex(v)
    return r"\left(" + t + r"\right)" if (v.could_extract_minus_sign() or not (v.is_Atom or v.is_Pow)) else t


def _set_tex(s):
    if s == S.EmptySet:
        return r"\varnothing"
    if s == S.Reals:
        return r"\mathbb{R}"
    return latex(s)


def _same_set(A, B):
    try:
        if A == B:
            return True
        if isinstance(A, FiniteSet) and isinstance(B, FiniteSet) and len(A) == len(B):
            a_ = sorted(A, key=lambda e: float(N(e)))
            b_ = sorted(B, key=lambda e: float(N(e)))
            return all(abs(float(N(u - v, 20))) < 1e-9 for u, v in zip(a_, b_))
        return A.symmetric_difference(B) == S.EmptySet
    except Exception:
        return False


def _roots_poly(p, d, out, roots, tag=None):
    """Resuelve p(x) = 0 (polinomio no constante) explicando cada paso; agrega las raíces reales a roots."""
    P = Poly(p, x)
    deg = P.degree()
    if deg == 1:
        a_, b_ = P.all_coeffs()
        r = simplify(-b_ / a_)
        mid = "" if a_ == 1 else _tex(a_ * x) + " = " + _tex(-b_) + r" \;\Rightarrow\; "
        what = (tag + ": se iguala a 0 y se despeja x.") if tag else "Ecuación de primer grado: se despeja x."
        if b_ == 0:
            out.append(_st(d, (tag + ": ") + "x = 0." if tag else "Ya está despejada.", "x = 0"))
        else:
            out.append(_st(d, what, _tex(p) + r" = 0 \;\Rightarrow\; " + mid + "x = " + _tex(r)))
        roots.append(r)
        return True
    if deg == 2:
        a_, b_, c_ = P.all_coeffs()
        D = b_ ** 2 - 4 * a_ * c_
        out.append(_st(d, "Ecuación de segundo grado: se identifican los coeficientes.",
                       rf"a = {_tex(a_)}, \quad b = {_tex(b_)}, \quad c = {_tex(c_)}"))
        out.append(_st(d, "Se calcula el discriminante Δ = b² − 4ac.",
                       r"\Delta = " + _par(b_) + r"^{2} - 4 \cdot " + _par(a_) + r" \cdot " + _par(c_) + " = " + _tex(D)))
        if D < 0:
            out.append(_st(d, "Δ < 0: no hay soluciones reales.", None))
            return True
        if D == 0:
            r = simplify(-b_ / (2 * a_))
            out.append(_st(d, "Δ = 0: hay una única solución (doble).", r"x = \frac{-b}{2a} = " + _tex(r)))
            roots.append(r)
            return True
        sq = sqrt(D)
        r1, r2 = simplify((-b_ - sq) / (2 * a_)), simplify((-b_ + sq) / (2 * a_))
        out.append(_st(d, "Δ > 0: hay dos soluciones reales (fórmula resolvente).",
                       r"x = \frac{-b \pm \sqrt{\Delta}}{2a} = \frac{" + _tex(-b_) + r" \pm " + _tex(sq) + "}{" + _tex(2 * a_) + "}"))
        out.append(_st(d, "Las dos soluciones:", "x_1 = " + _tex(r1) + r", \quad x_2 = " + _tex(r2)))
        roots.extend([r1, r2])
        return True
    # grado ≥ 3: se factoriza y se resuelve cada factor
    from sympy import factor_list
    _, fl = factor_list(p, x)
    if len(fl) == 1 and fl[0][1] == 1:
        return False  # irreducible de grado ≥ 3: no hay fórmula sencilla para explicar
    out.append(_st(d, "Se factoriza el polinomio.", _tex(p) + " = " + _tex(factor(p))))
    out.append(_st(d, "Un producto vale 0 cuando algún factor vale 0: se iguala cada factor a 0.", None))
    def order_key(item):
        pf = Poly(item[0], x)
        return (pf.degree(), float(N(-pf.all_coeffs()[1] / pf.all_coeffs()[0])) if pf.degree() == 1 else 0.0)

    for fac, mult in sorted(fl, key=order_key):
        tag = "Factor " + _pretty(fac) + (f" (aparece {mult} veces)" if mult > 1 else "")
        if Poly(fac, x).degree() == 1:
            if not _roots_poly(expand(fac), d + 1, out, roots, tag):
                return False
        else:
            out.append(_st(d + 1, tag + ":", _tex(fac) + " = 0"))
            if not _roots_poly(expand(fac), d + 2, out, roots):
                return False
        if len(out) > MAX_STEPS:
            return False
    return True


def _eq_steps(lhs, rhs, sol):
    p = expand(lhs - rhs)
    try:
        P = Poly(p, x)
    except PolynomialError:
        return None
    if P.degree() < 1:
        return None
    out = []
    if not (rhs == 0 and lhs == p):
        out.append(_st(0, "Se pasa todo a un miembro y se ordena: queda un polinomio igual a 0.", _tex(p) + " = 0"))
    roots = []
    if not _roots_poly(p, 0, out, roots):
        return None
    if not _same_set(FiniteSet(*roots), sol):
        return None
    out.append(_st(0, "Resultado", "S = " + _set_tex(sol)))
    return out


def _real_zeros(P):
    if P.degree() < 1:
        return []
    s = solveset(P.as_expr(), x, S.Reals)
    from sympy import CRootOf
    if not isinstance(s, FiniteSet) or s.has(CRootOf):
        return None
    return list(s)


def _pick(lo, hi):
    """Un número cómodo estrictamente entre lo y hi."""
    if lo == -oo:
        return Integer(int(floor(N(hi))) - 1)
    if hi == oo:
        return Integer(int(floor(N(lo))) + 1)
    cand = int(floor(N(lo))) + 1
    if cand < N(hi):
        return Integer(cand)
    mid = Rational((N(lo) + N(hi)) / 2).limit_denominator(10 ** 4)
    return mid if lo < mid < hi else Rational((N(lo) + N(hi)) / 2)


def _ineq_steps(lhs, rhs, rel, sol):
    g = together(lhs - rhs)
    num, den = fraction(g)
    try:
        Pn, Pd = Poly(num, x), Poly(den, x)
    except PolynomialError:
        return None
    zn, zd = _real_zeros(Pn), _real_zeros(Pd)
    if zn is None or zd is None:
        return None
    crit = sorted(set(zn) | set(zd), key=lambda e: float(N(e)))
    if not crit or len(crit) > 8:
        return None
    shown = r"\frac{" + _tex(factor(num)) + "}{" + _tex(factor(den)) + "}" if den != 1 else _tex(factor(num))
    want_pos = rel in (">", ">=")
    strict = rel in ("<", ">")
    out = [_st(0, "Se pasa todo a un miembro y se factoriza para estudiar el signo de E(x).",
               "E(x) = " + shown + r", \quad E(x) " + _REL_TEX[rel] + " 0")]
    out.append(_st(0, "Puntos críticos: donde se anula el numerador (la expresión vale 0) o el denominador (no existe).",
                   r"\text{numerador} = 0 \Rightarrow x \in " + latex(FiniteSet(*zn))
                   + r", \quad \text{denominador} = 0 \Rightarrow x \in " + latex(FiniteSet(*zd))))
    out.append(_st(0, "Los puntos críticos parten la recta en intervalos; en cada uno el signo no cambia. "
                      "Se prueba un valor de cada intervalo:", None))
    bounds = [-oo] + crit + [oo]
    good = []
    for lo, hi in zip(bounds, bounds[1:]):
        t = _pick(lo, hi)
        val = simplify(g.subs(x, t))
        fv = float(N(val))
        ok = fv > 0 if want_pos else fv < 0
        out.append(_st(1, f"Con x = {_pretty(t)} la expresión es {'positiva' if fv > 0 else 'negativa'}: "
                          + ("cumple." if ok else "no cumple."),
                       r"x \in " + latex(Interval.open(lo, hi)) + r": \quad E(" + _tex(t) + ") = " + _tex(val)))
        if ok:
            good.append(Interval.open(lo, hi))
    pts = [z for z in zn if z not in zd] if not strict else []
    out.append(_st(0, "La desigualdad es " + ("estricta: los ceros del numerador no entran. " if strict else
                       "no estricta: los ceros del numerador sí entran. ")
                   + "Donde se anula el denominador la expresión no existe, así que nunca entra.", None))
    try:
        assembled = Union(*good, *[FiniteSet(z) for z in pts])
    except Exception:
        return None
    if not _same_set(assembled, sol):
        return None
    out.append(_st(0, "Resultado: se unen los intervalos que cumplen.", "S = " + _set_tex(sol)))
    return out


def _solve_steps():
    data = _CACHE.get("solve")
    if not data:
        return None
    lhs, rhs, rel, sol = data
    return _eq_steps(lhs, rhs, sol) if rel == "=" else _ineq_steps(lhs, rhs, rel, sol)


def _steps(op, expr_s, a_s, b_s, order, side, result):
    if op == "solve":
        steps = _solve_steps()
    elif op == "limit":
        val = _CACHE.get("limit")
        steps = None if val is None else _limit_steps(_parse(expr_s), _num(a_s, "el punto"), side, val)
    elif op == "taylor":
        data = _CACHE.get("taylor")
        steps = None if data is None else _taylor_steps(_parse(expr_s), data[0], order, data[1])
    elif op == "diff":
        steps = _diff_steps(_parse(expr_s), order)
    elif op == "int":
        steps = _int_steps(_parse(expr_s))
    elif op == "defint":
        steps = _defint_steps(_parse(expr_s), _parse(a_s, "el límite inferior", var=None),
                              _parse(b_s, "el límite superior", var=None), result.get("numeric"))
    else:
        steps = None  # simplify, expand, factor, apart, sum, eval: sin paso a paso
    if steps is not None and len(steps) > MAX_STEPS:
        return None
    return steps


def _worker(conn, args):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT, MEM_LIMIT))
    except Exception:
        pass
    try:
        out = _compute(*args)
    except UserError as exc:
        out = {"ok": False, "error": str(exc)}
    except Exception:
        out = {"ok": False, "error": "No pude resolver esta expresión."}
    try:
        conn.send(out)  # primero el resultado, así no se pierde si el paso a paso se cuelga
        if out.get("ok") and not out.get("approx") and args[0] in STEP_OPS:
            try:
                steps = _steps(*args, out)
            except Exception:
                steps = None
            conn.send({"steps": steps})
    finally:
        conn.close()


def run(op, expr, a, b, order, side="both"):
    """Calcula en un proceso aparte para poder cortarlo si SymPy se cuelga."""
    ctx = mp.get_context("fork")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(send, (op, expr, a, b, order, side)), daemon=True)
    proc.start()
    send.close()
    try:
        if recv.poll(TIMEOUT_S):
            try:
                result = recv.recv()
            except EOFError:
                return {"ok": False, "error": "El cálculo se cortó (¿demasiada memoria?). Probá con algo más simple."}
            if result.get("ok") and not result.get("approx") and op in STEP_OPS:
                try:
                    if recv.poll(STEPS_TIMEOUT_S):
                        result["steps"] = recv.recv().get("steps")
                except (EOFError, OSError):
                    pass
            return result
        return {"ok": False, "error": f"Tardó más de {TIMEOUT_S} s y lo corté. Probá con una expresión más simple."}
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(1)


class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_page(self):
        # Respaldo: si Vercel no sirve public/ por su cuenta, la función entrega la página.
        try:
            with open(PAGE, "rb") as fh:
                body = fh.read()
        except OSError:
            self._send(404, {"ok": False, "error": "No se encontró la página."})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send_page()
            return
        self._send(405, {"ok": False, "error": "Usá POST con un JSON."})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= 4096:
                raise ValueError("Pedido vacío o demasiado grande.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Formato inválido.")
            op = data.get("op")
            if op not in OPS:
                raise ValueError("Operación inválida.")
            order = 1
            if op in ("diff", "taylor"):
                top = MAX_ORDER if op == "diff" else MAX_TAYLOR
                what = "de derivación" if op == "diff" else "del polinomio"
                try:
                    order = int(data.get("order", 1 if op == "diff" else 5))
                except (TypeError, ValueError):
                    raise ValueError(f"El orden {what} tiene que ser un entero.") from None
                if not 1 <= order <= top:
                    raise ValueError(f"El orden {what} va de 1 a {top}.")
            side = data.get("side", "both")
            if side not in SIDES:
                raise ValueError("El lado del límite tiene que ser both, + o -.")
        except (ValueError, json.JSONDecodeError) as exc:
            msg = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else "JSON inválido."
            self._send(400, {"ok": False, "error": msg})
            return

        result = run(op, data.get("expr"), data.get("a"), data.get("b"), order, side)
        self._send(200 if result.get("ok") else 422, result)
