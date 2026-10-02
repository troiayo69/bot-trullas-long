#!/usr/bin/env python3
"""
Bot_long_trullas_server.py — v3 · MODELO EMA 70 (servidor / GitHub Actions)

Escanea la lista de activos (assets_trullas.txt) una vez al día y avisa por Telegram
de las ENTRADAS LONG según esta regla (la misma que el backtest):

  1. El precio estaba bajo la EMA 70 y la cruza al alza.
  2. Confirmación: la vela SIGUIENTE al cruce cierra al menos `confirmacion_pct` %
     por encima de la EMA 70. La señal es esa vela de confirmación (última vela cerrada).
  3. Filtros: volumen, precio mínimo, RSI máximo, mercado (SPY sobre su SMA200),
     resultados próximos. (ADX y tendencia EMA70/200 desactivados por defecto.)

  Stop sugerido: bajo el mínimo de las últimas 10 velas (máx. 3 ATR; mín. 0,5 ATR).
  Salida (la haces tú): cuando la EMA 6 vuelva a caer bajo la EMA 70 tras haber subido
  sobre ella y con un mínimo de `dias_min_salida` velas desde la entrada.

Variables de entorno: bot_token, chat_id  (Telegram).
Parámetros: DEFAULTS de abajo o un archivo settings_ema70.json (opcional) con las
claves que quieras cambiar.

Uso:
    python Bot_long_trullas_server.py                  # escaneo normal
    python Bot_long_trullas_server.py --sin-telegram   # prueba: imprime en vez de enviar
    python Bot_long_trullas_server.py --solo intc aapl --sin-telegram
"""
import argparse
import html
import json
import logging
import os
import re
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ===========================
# RUTAS Y CONSTANTES
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_FILE = os.path.join(BASE_DIR, "assets_trullas.txt")
HISTORIAL_FILE = os.path.join(BASE_DIR, "historial_senales.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings_ema70.json")
LOG_FILE = os.path.join(BASE_DIR, "senal_long_trullas.log")

TAMANO_LOTE = 40            # tickers por petición a Yahoo
MIN_VELAS = 210             # historial mínimo para calentar la EMA 200
DIAS_LIMPIEZA_HISTORIAL = 90
MODELO = "EMA70"

TELEGRAM_TOKEN = None
TELEGRAM_CHAT_ID = None
ENVIAR_TELEGRAM = True      # --sin-telegram lo pone a False

# ===========================
# LOGGING
# ===========================
logger = logging.getLogger("trullas_ema70")
logger.setLevel(logging.INFO)
if not logger.handlers:
    try:
        _fh = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s"))
        logger.addHandler(_fh)
    except Exception:
        pass
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def log_consola(texto):
    print(texto)
    logger.info(str(texto).strip())


# ===========================
# CONFIGURACIÓN
# ===========================
DEFAULTS = {
    # medias y señal
    "ema_fast": 6, "ema_mid": 70, "ema_slow": 200,
    "confirmacion_pct": 1.0,      # % mínimo sobre la EMA 70 de la vela siguiente al cruce (None = sin confirmación)
    "exigir_tendencia": False,    # True = exige EMA 70 y 200 subiendo y precio > EMA 200
    "pendiente_barras": 5,
    "dias_min_salida": 5,         # solo informativo (se muestra en el aviso)
    # MACD (solo para la puntuación)
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    # datos
    "periodo_datos": "2y", "dias_no_repetir": 2,
    # riesgo
    "atr_periodo": 14, "stop_barras": 10, "stop_max_atr": 3.0, "stop_atr_defecto": 1.5,
    "capital": 10000.0, "riesgo_pct": 1.0,
    # filtros
    "filtro_volumen": True, "volumen_min": 500000,
    "filtro_precio": True, "precio_min": 5.0,
    "filtro_adx": False, "adx_min": 20.0,
    "filtro_rsi": True, "rsi_max": 70.0,
    "filtro_mercado": True, "indice_mercado": "SPY",
    "filtro_resultados": True, "dias_resultados": 5,
    "score_min": 0,
    # avisos
    "enviar_individuales": True, "enviar_resumen": True,
}


def cargar_settings():
    p = dict(DEFAULTS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    if k in DEFAULTS:
                        p[k] = v
            log_consola(f"Parámetros leídos de {os.path.basename(SETTINGS_FILE)}")
        except Exception as e:
            logger.warning(f"No se pudo leer {SETTINGS_FILE}: {e}")
    return p


def guardar_json_seguro(ruta, datos):
    carpeta = os.path.dirname(ruta) or "."
    fd, tmp = tempfile.mkstemp(dir=carpeta, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=4, ensure_ascii=False)
        os.replace(tmp, ruta)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def nombre_modelo(p):
    conf = p.get("confirmacion_pct")
    txt_conf = "sin confirmación" if conf is None or float(conf) < 0 else f"confirmación {float(conf):g} %"
    return f"EMA {p['ema_fast']}/{p['ema_mid']}/{p['ema_slow']} · cruce EMA {p['ema_mid']} + {txt_conf}"


# ===========================
# TELEGRAM
# ===========================
def cargar_config_telegram():
    global TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_TOKEN = os.environ.get("bot_token")
    TELEGRAM_CHAT_ID = os.environ.get("chat_id")
    if ENVIAR_TELEGRAM and (not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID):
        log_consola("⚠️ No se encontraron 'bot_token' o 'chat_id' en las variables de entorno.")


def esc(texto):
    return html.escape(str(texto), quote=False)


def _trocear(mensaje, limite=4000):
    partes, actual = [], ""
    for linea in mensaje.split("\n"):
        while len(linea) > limite:
            partes.append(linea[:limite])
            linea = linea[limite:]
        if len(actual) + len(linea) + 1 > limite:
            partes.append(actual)
            actual = ""
        actual += linea + "\n"
    if actual.strip():
        partes.append(actual)
    return partes


def enviar_telegram(mensaje, log=log_consola):
    if not ENVIAR_TELEGRAM:
        print("\n--- (modo prueba, no se envía a Telegram) ---\n" + re.sub(r"<[^>]+>", "", mensaje) + "\n---")
        return True
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("⚠️ Faltan bot_token o chat_id: no se puede enviar a Telegram.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    ok_total = True
    for parte in _trocear(mensaje):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": parte, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            resp = requests.post(url, data=payload, timeout=15)
            data = resp.json()
            if not data.get("ok"):
                ok_total = False
                log(f"⚠️ Telegram rechazó el mensaje: {data.get('description', 'sin descripción')}")
        except Exception as e:
            ok_total = False
            log(f"⚠️ Error enviando Telegram: {e}")
    if ok_total:
        log("📨 Mensaje enviado a Telegram.")
    return ok_total


def enlace_tradingview(ticker):
    mapa = {".MC": "BME", ".L": "LSE", ".PA": "EURONEXT", ".AS": "EURONEXT",
            ".DE": "XETR", ".MI": "MIL", ".TO": "TSX", ".SW": "SIX", ".LS": "EURONEXT"}
    for suf, bolsa in mapa.items():
        if ticker.endswith(suf):
            return f"https://www.tradingview.com/chart/?symbol={bolsa}:{ticker[:-len(suf)]}"
    return f"https://www.tradingview.com/chart/?symbol={ticker.replace('-', '.')}"


def mensaje_senal(r, p):
    t = esc(r["ticker"])
    lineas = [
        "🟢 <b>[ENTRADA LONG · EMA 70]</b>", "",
        f"📊 <b>Ticker:</b> <a href=\"{enlace_tradingview(r['ticker'])}\">{t}</a>",
        f"🧭 <b>Modelo:</b> {esc(nombre_modelo(p))}",
        f"📅 <b>Vela de señal:</b> {r['fecha']} (cruce el {r['fecha_cruce']})",
        f"💰 <b>Cierre:</b> {r['precio']:.2f} · EMA {p['ema_mid']}: {r['ema_m']:.2f} ({r['sobre_ema_pct']:+.1f}% sobre ella)",
        f"🛑 <b>Stop:</b> {r['sl']:.2f} ({r['sl_pct']:+.1f}%)",
        f"📦 <b>Tamaño:</b> {r['acciones']} acc. (riesgo {r['riesgo_eur']:.0f} €)",
        f"🚪 <b>Salida:</b> cuando la EMA {p['ema_fast']} vuelva a caer bajo la EMA {p['ema_mid']} "
        f"(tras subir sobre ella y mín. {p['dias_min_salida']} velas)",
        f"⭐ <b>Puntuación:</b> {r['score']}/100",
        f"ADX {r['adx']:.0f} · RSI {r['rsi']:.0f} · Vol x{r['vol_ratio']:.1f}",
        "ℹ️ Entrada orientativa: apertura de la siguiente sesión.",
    ]
    if r.get("resultados_dias") is not None:
        lineas.append(f"📆 Resultados en {r['resultados_dias']} días")
    return "\n".join(lineas)


def enviar_resumen_telegram(senales, info, p, log=log_consola):
    cab = ["📋 <b>[RESUMEN ESCANEO]</b>", "", f"🧭 {esc(nombre_modelo(p))}"]
    if info.get("mercado") is not None:
        estado = "✅ alcista" if info["mercado"] else "⛔ bajista"
        cab.append(f"🌍 Mercado ({esc(p['indice_mercado'])} vs SMA200): {estado}")
    cab.append(f"🔎 Analizados: {info['analizados']} · Sin datos: {info['sin_datos']} · "
               f"Cruces confirmados hoy: {info['gatillos']} · Descartados por filtros: {info['descartadas']}")
    if info["motivos"]:
        cab.append("Motivos: " + ", ".join(f"{k} {v}" for k, v in sorted(info["motivos"].items(), key=lambda x: -x[1])))
    if info["total"] and info["sin_datos"] / info["total"] > 0.15:
        cab.append("⚠️ Muchos activos sin datos: posible fallo de Yahoo, revisa el log.")
    cab.append("")
    if not senales:
        enviar_telegram("\n".join(cab + ["No se han encontrado señales de entrada LONG en este escaneo."]), log=log)
        return
    lineas = cab + [f"Se han detectado <b>{len(senales)}</b> señal(es), de mejor a peor:", ""]
    for n, s in enumerate(sorted(senales, key=lambda x: -x["score"]), start=1):
        lineas.append(f"{n}. 🟢 <b>{esc(s['ticker'])}</b> ⭐{s['score']} — {s['precio']:.2f} (SL {s['sl']:.2f})")
    enviar_telegram("\n".join(lineas), log=log)


# ===========================
# LISTA DE ACTIVOS E HISTORIAL
# ===========================
def normalizar_ticker(t):
    t = (t or "").strip().upper()
    m = re.fullmatch(r"([A-Z]+)\.([ABC])", t)
    if m:
        t = f"{m.group(1)}-{m.group(2)}"
    return t


def cargar_activos():
    if not os.path.exists(ASSETS_FILE):
        log_consola(f"⚠️ No existe {os.path.basename(ASSETS_FILE)}")
        return []
    vistos, activos = set(), []
    with open(ASSETS_FILE, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.split("#")[0].strip()
            if not linea:
                continue
            t = normalizar_ticker(linea)
            if t and t not in vistos:
                vistos.add(t)
                activos.append(t)
    return activos


def cargar_historial():
    if not os.path.exists(HISTORIAL_FILE):
        return {}
    try:
        with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def guardar_historial(historial):
    try:
        guardar_json_seguro(HISTORIAL_FILE, historial)
    except Exception as e:
        logger.error(f"No se pudo guardar el historial: {e}")


def es_repetida(historial, ticker, vela, dias):
    reg = historial.get(f"{ticker}|{MODELO}")
    if not isinstance(reg, dict):
        return False
    if reg.get("vela") == vela:
        return True
    try:
        enviado = datetime.strptime(reg["enviado"], "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - enviado).total_seconds() / 86400 < dias
    except Exception:
        return False


def registrar_senal(historial, ticker, vela):
    historial[f"{ticker}|{MODELO}"] = {"vela": vela, "enviado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def limpiar_historial(historial):
    limite = datetime.now() - timedelta(days=DIAS_LIMPIEZA_HISTORIAL)
    limpio = {}
    for k, v in historial.items():
        try:
            if isinstance(v, dict) and datetime.strptime(v["enviado"], "%Y-%m-%d %H:%M:%S") >= limite:
                limpio[k] = v
        except Exception:
            pass
    return limpio


# ===========================
# INDICADORES Y SEÑAL
# ===========================
def smma(series, period):
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calcular_indicadores(df, p):
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]
    df["ema_f"] = ema(c, int(p["ema_fast"]))
    df["ema_m"] = ema(c, int(p["ema_mid"]))
    df["ema_s"] = ema(c, int(p["ema_slow"]))
    df["macd"] = ema(c, int(p["macd_fast"])) - ema(c, int(p["macd_slow"]))
    df["macd_signal"] = ema(df["macd"], int(p["macd_signal"]))
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = c.diff()
    subida = smma(delta.clip(lower=0), 14)
    bajada = smma(-delta.clip(upper=0), 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rsi"] = 100 - 100 / (1 + subida / bajada)

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = smma(tr, int(p["atr_periodo"]))
    up, down = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr14 = smma(tr, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100 * smma(plus_dm, 14) / atr14
        minus_di = 100 * smma(minus_dm, 14) / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx"] = smma(dx.fillna(0), 14)

    vol = df["Volume"] if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    df["Volume"] = vol.fillna(0)
    df["vol_media"] = df["Volume"].rolling(20).mean()
    return df


def serie_mercado(df_indice):
    if df_indice is None or df_indice.empty:
        return None
    c = df_indice["Close"]
    sma = c.rolling(200).mean()
    return (c > sma) | sma.isna()


def _alinear_mercado(mercado, index):
    if mercado is None:
        return pd.Series(True, index=index)
    m = mercado.astype(float)
    m = m[~m.index.duplicated()].sort_index()
    alineado = m.reindex(m.index.union(index)).ffill().reindex(index)
    return alineado.fillna(1.0) > 0.5


def calcular_senal(df, p, mercado=None):
    """Devuelve, por vela: gatillo (señal EMA 70), filtros_ok y una columna f_* por filtro."""
    m, s = df["ema_m"], df["ema_s"]
    close = df["Close"]
    cruce = (close > m) & (close.shift(1) <= m.shift(1))
    conf = p.get("confirmacion_pct")
    if conf is None or float(conf) < 0:
        gatillo = cruce
    else:
        gatillo = cruce.shift(1, fill_value=False).astype(bool) & (close >= m * (1 + float(conf) / 100))
    if p["exigir_tendencia"]:
        k = max(1, int(p["pendiente_barras"]))
        gatillo = gatillo & (m > m.shift(k)) & (s > s.shift(k)) & (close > s)
    calentado = pd.Series(np.arange(len(df)) >= int(p["ema_slow"]), index=df.index)
    gatillo = (gatillo & calentado).astype(bool)

    filtros = pd.DataFrame(index=df.index)
    if p["filtro_volumen"]:
        filtros["volumen"] = df["vol_media"] >= float(p["volumen_min"])
    if p["filtro_precio"]:
        filtros["precio"] = close >= float(p["precio_min"])
    if p["filtro_adx"]:
        filtros["adx"] = df["adx"] >= float(p["adx_min"])
    if p["filtro_rsi"]:
        filtros["rsi"] = df["rsi"] <= float(p["rsi_max"])
    if p["filtro_mercado"]:
        filtros["mercado"] = _alinear_mercado(mercado, df.index)
    filtros_ok = filtros.all(axis=1) if len(filtros.columns) else pd.Series(True, index=df.index)
    sal = pd.DataFrame({"gatillo": gatillo, "filtros_ok": filtros_ok.astype(bool)}, index=df.index)
    return sal.join(filtros.add_prefix("f_"))


def calcular_score(df, i):
    r, ant = df.iloc[i], df.iloc[i - 1]
    pts = 0.0
    if pd.notna(r["adx"]):
        pts += 25 * min(max((r["adx"] - 10) / 30, 0), 1)
    if pd.notna(r["vol_media"]) and r["vol_media"] > 0:
        pts += 15 * min(max((r["Volume"] / r["vol_media"] - 0.5) / 1.5, 0), 1)
    rsi = r["rsi"]
    if pd.notna(rsi):
        if 50 <= rsi <= 65:
            pts += 20
        elif 40 <= rsi < 50 or 65 < rsi <= 70:
            pts += 12
        elif rsi < 40:
            pts += 5
    if pd.notna(r["atr"]) and r["atr"] > 0:
        ext = (r["Close"] - r["ema_m"]) / r["atr"]
        if ext < 0:
            pts += 10
        elif ext <= 1.5:
            pts += 20
        elif ext < 5:
            pts += 20 * (5 - ext) / 3.5
    if r["macd_hist"] > ant["macd_hist"]:
        pts += 10
    pts += 10  # señal de cruce (no es una simple continuación)
    return int(round(pts))


def calcular_stop_y_riesgo(df, i, precio, atr, p):
    """Stop bajo el mínimo de las últimas N velas (máx. stop_max_atr ATR; mín. 0,5 ATR)."""
    if not (atr and atr > 0):
        sl = precio * 0.95
    else:
        sl = precio - float(p["stop_atr_defecto"]) * atr
        n = int(p["stop_barras"])
        minimo = float(df["Low"].iloc[max(0, i - n + 1): i + 1].min())
        sl_sw = max(minimo - 0.1 * atr, precio - float(p["stop_max_atr"]) * atr)
        if precio - sl_sw >= 0.5 * atr:
            sl = sl_sw
    riesgo = precio - sl
    riesgo_max = float(p["capital"]) * float(p["riesgo_pct"]) / 100
    acciones = int(riesgo_max // riesgo) if riesgo > 0 else 0
    acciones = max(0, min(acciones, int(float(p["capital"]) // precio) if precio > 0 else 0))
    return {"sl": sl, "sl_pct": (sl / precio - 1) * 100, "acciones": acciones, "riesgo_eur": acciones * riesgo}


def ultima_vela_cerrada_idx(df, ticker):
    n = len(df)
    if ticker.endswith("-USD") or ticker.endswith("=X") or ticker.endswith("=F"):
        tz, hh, mm = "UTC", 24, 0
    else:
        tz, hh, mm = "America/New_York", 16, 0
    HORARIOS = {".MC": ("Europe/Madrid", 17, 35), ".PA": ("Europe/Paris", 17, 35), ".DE": ("Europe/Berlin", 17, 35),
                ".AS": ("Europe/Amsterdam", 17, 35), ".MI": ("Europe/Rome", 17, 35), ".L": ("Europe/London", 16, 35)}
    for suf, datos in HORARIOS.items():
        if ticker.endswith(suf):
            tz, hh, mm = datos
            break
    try:
        ahora = pd.Timestamp.now(tz=tz)
        ultima = pd.Timestamp(df.index[-1]).date()
        if ultima < ahora.date():
            return n - 1
        cierre = ahora.normalize() + pd.Timedelta(hours=hh, minutes=mm + 20)
        return n - 1 if ahora >= cierre else n - 2
    except Exception:
        return n - 2


# ===========================
# DESCARGA
# ===========================
def _extraer(data, ticker):
    if data is None or data.empty:
        return None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.get_level_values(0):
                df = data[ticker].copy()
            elif ticker in data.columns.get_level_values(1):
                df = data.xs(ticker, axis=1, level=1).copy()
            else:
                return None
        else:
            df = data.copy()
        if "Close" not in df.columns:
            return None
        df = df.dropna(subset=["Close"])
        for col in ("Open", "High", "Low"):
            if col not in df.columns:
                df[col] = df["Close"]
            df[col] = df[col].fillna(df["Close"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df if len(df) else None
    except Exception:
        return None


def descargar_lote(tickers, periodo="2y", log=log_consola):
    resultado = {}
    tickers = list(dict.fromkeys(tickers))
    bloques = [tickers[i:i + TAMANO_LOTE] for i in range(0, len(tickers), TAMANO_LOTE)]
    for nb, bloque in enumerate(bloques, start=1):
        data = None
        for intento in range(3):
            try:
                data = yf.download(bloque, period=periodo, interval="1d", group_by="ticker",
                                   threads=True, progress=False, auto_adjust=True)
                if data is not None and not data.empty:
                    break
            except Exception as e:
                log(f"⚠️ Error descargando bloque {nb}/{len(bloques)} (intento {intento + 1}): {e}")
            time.sleep(2 * (intento + 1))
        n_ok = 0
        for t in bloque:
            df = _extraer(data, t)
            if df is not None:
                resultado[t] = df
                n_ok += 1
        if n_ok < len(bloque):
            log(f"ℹ️ Bloque {nb}/{len(bloques)}: {n_ok}/{len(bloque)} tickers con datos.")
    return resultado


def dias_hasta_resultados(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        fechas = cal.get("Earnings Date") if isinstance(cal, dict) else (
            list(cal.loc["Earnings Date"].values) if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index else None)
        if not fechas:
            return None
        if not isinstance(fechas, (list, tuple, np.ndarray)):
            fechas = [fechas]
        hoy = date.today()
        futuras = sorted(d for d in (pd.Timestamp(x).date() for x in fechas) if d >= hoy)
        return (futuras[0] - hoy).days if futuras else None
    except Exception:
        return None


# ===========================
# ANÁLISIS DE UN TICKER
# ===========================
def analizar_ticker(ticker, dfi, p, mercado=None):
    """Devuelve None si no hay señal en la última vela cerrada; si la hay, un dict con los datos."""
    i = ultima_vela_cerrada_idx(dfi, ticker)
    if i < 2:
        return None
    sen = calcular_senal(dfi, p, mercado)
    s = sen.iloc[i]
    if not bool(s["gatillo"]):
        return None
    fila = dfi.iloc[i]
    precio = float(fila["Close"])
    atr = float(fila["atr"]) if pd.notna(fila["atr"]) else float("nan")
    conf = p.get("confirmacion_pct")
    i_cruce = i if (conf is None or float(conf) < 0) else i - 1
    vol_ratio = float(fila["Volume"] / fila["vol_media"]) if pd.notna(fila["vol_media"]) and fila["vol_media"] else 0.0
    res = {
        "ticker": ticker,
        "fecha": dfi.index[i].strftime("%d/%m/%Y"),
        "vela": dfi.index[i].strftime("%Y-%m-%d"),
        "fecha_cruce": dfi.index[i_cruce].strftime("%d/%m/%Y"),
        "precio": precio, "atr": atr,
        "ema_m": float(fila["ema_m"]), "sobre_ema_pct": (precio / float(fila["ema_m"]) - 1) * 100,
        "adx": float(fila["adx"]) if pd.notna(fila["adx"]) else 0.0,
        "rsi": float(fila["rsi"]) if pd.notna(fila["rsi"]) else 0.0,
        "vol_ratio": vol_ratio,
        "filtros": {c[2:]: bool(s[c]) for c in sen.columns if c.startswith("f_")},
        "filtros_ok": bool(s["filtros_ok"]),
        "score": calcular_score(dfi, i), "resultados_dias": None,
    }
    res.update(calcular_stop_y_riesgo(dfi, i, precio, atr, p))
    return res


# ===========================
# ESCANEO PRINCIPAL
# ===========================
def ejecutar_escaneo(p, solo=None, log=log_consola):
    log("=" * 60)
    log(f"🚀 INICIANDO ESCANEO — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    log(f"🧭 Modelo: {nombre_modelo(p)}")
    log("=" * 60)

    activos = [normalizar_ticker(t) for t in solo] if solo else cargar_activos()
    if not activos:
        log("⚠️ La lista de activos está vacía.")
        return []
    indice = normalizar_ticker(p["indice_mercado"])
    lista = list(activos) + ([indice] if p["filtro_mercado"] and indice not in activos else [])

    t0 = time.time()
    datos = descargar_lote(lista, periodo=p["periodo_datos"], log=log)
    log(f"⬇️ Descargados {len(datos)}/{len(lista)} tickers en {time.time() - t0:.1f} s")

    mercado, estado_mercado = None, None
    if p["filtro_mercado"]:
        mercado = serie_mercado(datos.get(indice))
        if mercado is None:
            log(f"⚠️ No hay datos de {indice}: el filtro de mercado se ignora.")
        else:
            estado_mercado = bool(mercado.iloc[-1])
            log(f"🌍 Mercado ({indice} vs SMA200): {'ALCISTA ✅' if estado_mercado else 'BAJISTA ⛔'}")

    historial = cargar_historial()
    senales, motivos = [], {}
    sin_datos = gatillos = descartadas = 0

    for ticker in activos:
        try:
            df = datos.get(ticker)
            if df is None or len(df) < MIN_VELAS:
                sin_datos += 1
                continue
            dfi = calcular_indicadores(df, p)
            r = analizar_ticker(ticker, dfi, p, mercado)
            if r is None:
                continue
            gatillos += 1
            if not r["filtros_ok"]:
                descartadas += 1
                falla = [k for k, ok in r["filtros"].items() if not ok]
                for k in falla:
                    motivos[k] = motivos.get(k, 0) + 1
                log(f"⏭️ {ticker:<10} cruce confirmado pero descartado por filtro: {', '.join(falla)}")
                continue
            if r["score"] < int(p["score_min"]):
                descartadas += 1
                motivos["score"] = motivos.get("score", 0) + 1
                continue
            dias = dias_hasta_resultados(ticker)
            r["resultados_dias"] = dias
            if p["filtro_resultados"] and dias is not None and dias <= int(p["dias_resultados"]):
                descartadas += 1
                motivos["resultados"] = motivos.get("resultados", 0) + 1
                log(f"⏭️ {ticker:<10} descartado: resultados en {dias} días")
                continue
            senales.append(r)
            log(f"🟢 {ticker:<10} ENTRADA LONG @ {r['precio']:.2f} · SL {r['sl']:.2f} · ⭐{r['score']}")
        except Exception as e:
            log(f"⚠️ {ticker}: error inesperado: {e}")
            logger.exception(f"Error analizando {ticker}")

    senales.sort(key=lambda s: -s["score"])

    if p["enviar_individuales"]:
        for r in senales:
            if es_repetida(historial, r["ticker"], r["vela"], int(p["dias_no_repetir"])):
                log(f"⏭️ {r['ticker']} ya avisada (anti-spam).")
                continue
            if enviar_telegram(mensaje_senal(r, p), log=log):
                registrar_senal(historial, r["ticker"], r["vela"])
    guardar_historial(limpiar_historial(historial))

    if p["enviar_resumen"]:
        enviar_resumen_telegram(senales, {
            "mercado": estado_mercado, "analizados": len(activos) - sin_datos, "sin_datos": sin_datos,
            "gatillos": gatillos, "descartadas": descartadas, "motivos": motivos, "total": len(activos),
        }, p, log=log)

    log("\n✔ Escaneo finalizado.")
    log(f"📋 {len(senales)} señal(es) · {descartadas} descartada(s) · {sin_datos} sin datos.\n")
    return senales


# ===========================
# MAIN
# ===========================
def main():
    global ENVIAR_TELEGRAM
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Escáner de entradas LONG — modelo EMA 70")
    ap.add_argument("--sin-telegram", action="store_true", help="Prueba: imprime los avisos en vez de enviarlos")
    ap.add_argument("--solo", nargs="+", metavar="TICKER", help="Escanear solo estos tickers (en vez de assets_trullas.txt)")
    args = ap.parse_args()
    if args.sin_telegram:
        ENVIAR_TELEGRAM = False

    try:
        cargar_config_telegram()
        p = cargar_settings()
        log_consola("Iniciando bot en modo servidor (headless)...")
        ejecutar_escaneo(p, solo=args.solo)
    except Exception as e:
        error_msg = f"❌ <b>Error crítico en el escáner</b>\n{html.escape(str(e))}"
        log_consola(error_msg)
        logger.exception("Error crítico en modo servidor")
        try:
            enviar_telegram(error_msg)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
