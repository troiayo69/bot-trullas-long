#!/usr/bin/env python3
"""
Bot_Long_Trullas_Server.py — v2 PRO (Versión Servidor / PythonAnywhere)
Escáner de señales de ENTRADA LONG optimizado para ejecución headless (sin interfaz gráfica).
Mantiene: Estrategias EMA/SMMA, Filtros, Gestión de Riesgo, Backtest y Notificaciones Telegram.
SIN GRÁFICOS: Solo texto + enlaces a TradingView (más rápido y eficiente para servidores).
"""
import argparse
import html
import json
import logging
import math
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
# RUTAS Y ARCHIVOS (Adaptado para PythonAnywhere)
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_FILE = os.path.join(BASE_DIR, "assets_trullas.txt")
HISTORIAL_FILE = os.path.join(BASE_DIR, "historial_senales.json")
CONFIG_TELEGRAM_FILE = os.path.join(BASE_DIR, "telegram_config.json")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings_trullas.json")
LOG_FILE = os.path.join(BASE_DIR, "senal_long_trullas.log")

TELEGRAM_TOKEN = None
TELEGRAM_CHAT_ID = None
TAMANO_LOTE = 40          # tickers por petición a Yahoo
DIAS_LIMPIEZA_HISTORIAL = 90

# ===========================
# LOGGING (Rotativo, ideal para servidores)
# ===========================
logger = logging.getLogger("trullas_server")
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
    """Log dual: pantalla (consola del servidor) + archivo."""
    print(texto)
    logger.info(str(texto).strip())

# ===========================
# CONFIGURACIÓN
# ===========================
DEFAULTS = {
    "estrategia": "EMA",
    "ema_fast": 6, "ema_mid": 70, "ema_slow": 200,
    "smma_fast": 5, "smma_mid": 20, "smma_slow": 50,
    "pendiente_barras": 5, "exigir_rapida_sobre_media": True,
    "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
    "solo_nuevas": True, "periodo_datos": "2y", "dias_no_repetir": 2,
    "atr_periodo": 14, "atr_mult_sl": 1.5, "ratio_tp": 2.0,
    "capital": 10000.0, "riesgo_pct": 1.0,
    "filtro_volumen": True, "volumen_min": 500000,
    "filtro_precio": True, "precio_min": 5.0,
    "filtro_adx": True, "adx_min": 20.0,
    "filtro_rsi": True, "rsi_max": 70.0,
    "filtro_mercado": True, "indice_mercado": "SPY",
    "filtro_resultados": True, "dias_resultados": 5,
    "score_min": 0,
    "enviar_individuales": True, "enviar_resumen": True,
    "bt_anos": 5, "bt_max_barras": 20, "bt_comision_pct": 0.1,
}

def cargar_settings():
    p = dict(DEFAULTS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                guardado = json.load(f)
                for k, v in guardado.items():
                    if k in DEFAULTS:
                        p[k] = v
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

def guardar_settings(p):
    guardar_json_seguro(SETTINGS_FILE, p)

def nombre_estrategia(est, p):
    if est == "EMA":
        return f"EMA {p['ema_fast']}/{p['ema_mid']}/{p['ema_slow']}"
    return f"SMMA {p['smma_fast']}/{p['smma_mid']}/{p['smma_slow']}"

def lista_estrategias(sel):
    sel = (sel or "EMA").upper()
    if sel == "AMBAS":
        return ["EMA", "SMMA"]
    return [sel if sel in ("EMA", "SMMA") else "EMA"]

# ===========================
# TELEGRAM
# ===========================

def cargar_config_telegram():
    global TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
    TELEGRAM_TOKEN = os.environ.get("bot_token")
    TELEGRAM_CHAT_ID = os.environ.get("chat_id")
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log_consola("⚠️ No se encontraron 'bot_token' o 'chat_id' en las variables de entorno.")

def _telegram_listo(log):
    if not TELEGRAM_TOKEN:
        log("️ Falta 'bot_token' en telegram_config.json.")
        return False
    if not TELEGRAM_CHAT_ID:
        log("⚠️ Falta 'chat_id' en telegram_config.json.")
        return False
    return True

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
    if not _telegram_listo(log):
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
        "🟢 <b>[ENTRADA EN COMPRA]</b>", "",
        f"📊 <b>Ticker:</b> <a href=\"{enlace_tradingview(r['ticker'])}\">{t}</a>",
        f"🧭 <b>Estrategia:</b> {esc(nombre_estrategia(r['estrategia'], p))}",
        f"️ <b>Tipo:</b> {esc(r['tipo'])}", f"📅 <b>Vela:</b> {r['fecha']}",
        f"💰 <b>Precio:</b> {r['precio']:.2f}", f" <b>Stop:</b> {r['sl']:.2f} ({r['sl_pct']:+.1f}%)",
        f" <b>Objetivo:</b> {r['tp']:.2f} ({r['tp_pct']:+.1f}%)", f"⚖️ <b>R:B</b> 1:{p['ratio_tp']:g}",
        f" <b>Tamaño:</b> {r['acciones']} acc. (riesgo {r['riesgo_eur']:.0f} €)",
        f"⭐ <b>Puntuación:</b> {r['score']}/100", f" ADX {r['adx']:.0f} · RSI {r['rsi']:.0f} · Vol x{r['vol_ratio']:.1f}",
    ]
    if r.get("resultados_dias") is not None:
        lineas.append(f"📆 Resultados en {r['resultados_dias']} días")
    return "\n".join(lineas)

def enviar_resumen_telegram(senales, log=log_consola, p=None, info=None):
    p = p or DEFAULTS
    info = info or {}
    cabecera = [" <b>[RESUMEN ESCANEO]</b>", ""]
    if info.get("estrategias"):
        cabecera.append(f"🧭 {esc(info['estrategias'])}")
    if info.get("mercado") is not None:
        estado = "✅ alcista" if info["mercado"] else "⛔ bajista"
        cabecera.append(f"🌍 Mercado ({esc(p['indice_mercado'])} vs SMA200): {estado}")
    if info.get("analizados") is not None:
        cabecera.append(f" Analizados: {info['analizados']} · Sin datos: {info.get('sin_datos', 0)} · Descartadas: {info.get('descartadas', 0)}")
    cabecera.append("")
    if not senales:
        enviar_telegram("\n".join(cabecera + ["No se han encontrado señales de entrada LONG en este escaneo."]), log=log)
        return
    lineas = cabecera + [f"Se han detectado <b>{len(senales)}</b> señal(es), de mejor a peor:", ""]
    for n, s in enumerate(sorted(senales, key=lambda x: -x["score"]), start=1):
        lineas.append(f"{n}. 🟢 <b>{esc(s['ticker'])}</b> ⭐{s['score']} — {s['precio']:.2f} (SL {s['sl']:.2f} / TP {s['tp']:.2f}) — {esc(s['estrategia'])}")
    enviar_telegram("\n".join(lineas), log=log)

# ===========================
# LISTA DE ACTIVOS Y HISTORIAL
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

def es_repetida(historial, ticker, estrategia, vela, dias):
    reg = historial.get(f"{ticker}|{estrategia}")
    if not isinstance(reg, dict):
        return False
    if reg.get("vela") == vela:
        return True
    try:
        enviado = datetime.strptime(reg["enviado"], "%Y-%m-%d %H:%M:%S")
        return (datetime.now() - enviado).total_seconds() / 86400 < dias
    except Exception:
        return False

def registrar_senal(historial, ticker, estrategia, vela):
    historial[f"{ticker}|{estrategia}"] = {"vela": vela, "enviado": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

def limpiar_historial(historial):
    limite = datetime.now() - timedelta(days=DIAS_LIMPIEZA_HISTORIAL)
    limpio = {}
    for k, v in historial.items():
        if not isinstance(v, dict):
            continue
        try:
            if datetime.strptime(v["enviado"], "%Y-%m-%d %H:%M:%S") >= limite:
                limpio[k] = v
        except Exception:
            pass
    return limpio

# ===========================
# INDICADORES Y SEÑALES
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
    df["smma_f"] = smma(c, int(p["smma_fast"]))
    df["smma_m"] = smma(c, int(p["smma_mid"]))
    df["smma_s"] = smma(c, int(p["smma_slow"]))
    df["macd"] = ema(c, int(p["macd_fast"])) - ema(c, int(p["macd_slow"]))
    df["macd_signal"] = ema(df["macd"], int(p["macd_signal"]))
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    delta = c.diff()
    subida = smma(delta.clip(lower=0), 14)
    bajada = smma(-delta.clip(upper=0), 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["rsi"] = 100 - 100 / (1 + subida / bajada)
    
    n = int(p["atr_periodo"])
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = smma(tr, n)
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

def calcular_senales(df, estrategia, p, mercado=None):
    pref = "ema" if estrategia == "EMA" else "smma"
    f, m, s = df[f"{pref}_f"], df[f"{pref}_m"], df[f"{pref}_s"]
    close = df["Close"]
    k = max(1, int(p["pendiente_barras"]))
    lento = int(p[f"{pref}_slow"])
    tendencia = (m > m.shift(k)) & (s > s.shift(k))
    cond = tendencia & (close > f) & (df["macd"] > df["macd_signal"])
    if p["exigir_rapida_sobre_media"]:
        cond &= f > m
    calentado = pd.Series(np.arange(len(df)) >= lento, index=df.index)
    cond = (cond & calentado).astype(bool)
    previa = cond.shift(1, fill_value=False).astype(bool)
    nueva = cond & ~previa
    cruce_ma = (f > m) & (f.shift(1) <= m.shift(1))
    cruce_macd = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    etiqueta_ma = f"CRUCE_{pref.upper()}{p[pref + '_fast']}_{pref.upper()}{p[pref + '_mid']}"
    tipo = np.select([cruce_ma.values, cruce_macd.values], [etiqueta_ma, "CRUCE_MACD_ALCISTA"], default="CONFIRMACION_LONG")
    
    filtros = pd.DataFrame(index=df.index)
    if p["filtro_volumen"]: filtros["volumen"] = df["vol_media"] >= float(p["volumen_min"])
    if p["filtro_precio"]: filtros["precio"] = close >= float(p["precio_min"])
    if p["filtro_adx"]: filtros["adx"] = df["adx"] >= float(p["adx_min"])
    if p["filtro_rsi"]: filtros["rsi"] = df["rsi"] <= float(p["rsi_max"])
    if p["filtro_mercado"]: filtros["mercado"] = _alinear_mercado(mercado, df.index)
    filtros_ok = filtros.all(axis=1) if len(filtros.columns) else pd.Series(True, index=df.index)
    
    return pd.DataFrame({"cond": cond, "nueva": nueva, "tipo": tipo, "filtros_ok": filtros_ok.astype(bool)}, index=df.index).join(filtros.add_prefix("f_"))

def calcular_score(df, i, estrategia, tipo):
    pref = "ema" if estrategia == "EMA" else "smma"
    r, ant = df.iloc[i], df.iloc[i - 1]
    pts = 0.0
    if pd.notna(r["adx"]): pts += 25 * min(max((r["adx"] - 10) / 30, 0), 1)
    if pd.notna(r["vol_media"]) and r["vol_media"] > 0:
        pts += 15 * min(max((r["Volume"] / r["vol_media"] - 0.5) / 1.5, 0), 1)
    rsi = r["rsi"]
    if pd.notna(rsi):
        if 50 <= rsi <= 65: pts += 20
        elif 40 <= rsi < 50 or 65 < rsi <= 70: pts += 12
        elif rsi < 40: pts += 5
    if pd.notna(r["atr"]) and r["atr"] > 0:
        ext = (r["Close"] - r[f"{pref}_m"]) / r["atr"]
        if ext < 0: pts += 10
        elif ext <= 1.5: pts += 20
        elif ext < 5: pts += 20 * (5 - ext) / 3.5
    if r["macd_hist"] > ant["macd_hist"]: pts += 10
    if tipo != "CONFIRMACION_LONG": pts += 10
    return int(round(pts))

def calcular_riesgo(precio, atr, p):
    riesgo = float(p["atr_mult_sl"]) * atr if atr > 0 else precio * 0.05
    sl = precio - riesgo
    tp = precio + float(p["ratio_tp"]) * riesgo
    riesgo_max = float(p["capital"]) * float(p["riesgo_pct"]) / 100
    acciones = int(riesgo_max // riesgo) if riesgo > 0 else 0
    acciones = max(0, min(acciones, int(float(p["capital"]) // precio) if precio > 0 else 0))
    return {"sl": sl, "tp": tp, "sl_pct": (sl / precio - 1) * 100, "tp_pct": (tp / precio - 1) * 100, "acciones": acciones, "riesgo_eur": acciones * riesgo}

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
# DESCARGA Y ANÁLISIS
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

def descargar_lote(tickers, periodo=None, inicio=None, log=log_consola, stop_event=None):
    resultado = {}
    tickers = list(dict.fromkeys(tickers))
    bloques = [tickers[i:i + TAMANO_LOTE] for i in range(0, len(tickers), TAMANO_LOTE)]
    kwargs = {"interval": "1d", "group_by": "ticker", "threads": True, "progress": False, "auto_adjust": True}
    if inicio is not None:
        kwargs["start"] = pd.Timestamp(inicio).strftime("%Y-%m-%d")
    else:
        kwargs["period"] = periodo or "2y"
    
    for nb, bloque in enumerate(bloques, start=1):
        if stop_event is not None and stop_event.is_set():
            break
        data = None
        for intento in range(3):
            try:
                data = yf.download(bloque, **kwargs)
                if data is not None and not data.empty:
                    break
            except Exception as e:
                log(f"⚠️ Error descargando bloque {nb} (intento {intento + 1}): {e}")
                time.sleep(2 * (intento + 1))
        for t in bloque:
            df = _extraer(data, t)
            if df is not None:
                resultado[t] = df
    return resultado

def dias_hasta_resultados(ticker):
    try:
        cal = yf.Ticker(ticker).calendar
        fechas = cal.get("Earnings Date") if isinstance(cal, dict) else (list(cal.loc["Earnings Date"].values) if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index else None)
        if not fechas:
            return None, None
        if not isinstance(fechas, (list, tuple, np.ndarray)):
            fechas = [fechas]
        hoy = date.today()
        futuras = sorted(d for d in (pd.Timestamp(x).date() for x in fechas) if d >= hoy)
        if not futuras:
            return None, None
        return (futuras[0] - hoy).days, futuras[0]
    except Exception:
        return None, None

def analizar_ticker(ticker, dfi, estrategia, p, mercado=None, senales=None):
    i = ultima_vela_cerrada_idx(dfi, ticker)
    if i < 2:
        return None
    sen = senales if senales is not None else calcular_senales(dfi, estrategia, p, mercado)
    fila, s = dfi.iloc[i], sen.iloc[i]
    activa, nueva = bool(s["cond"]), bool(s["nueva"])
    j = i
    if activa:
        cond = sen["cond"].values
        while j > 0 and cond[j - 1]:
            j -= 1
    tipo = str(sen["tipo"].iloc[j]) if activa else "SIN_SEÑAL"
    precio, atr = float(fila["Close"]), float(fila["atr"]) if pd.notna(fila["atr"]) else float("nan")
    riesgo = calcular_riesgo(precio, atr, p)
    vol_ratio = float(fila["Volume"] / fila["vol_media"]) if fila["vol_media"] and pd.notna(fila["vol_media"]) else 0.0
    filtros = {c[2:]: bool(s[c]) for c in sen.columns if c.startswith("f_")}
    res = {
        "ticker": ticker, "estrategia": estrategia, "activa": activa, "nueva": nueva, "tipo": tipo,
        "desde": dfi.index[j].strftime("%d/%m/%Y"), "velas_activa": i - j + 1, "fecha": dfi.index[i].strftime("%d/%m/%Y"),
        "vela": dfi.index[i].strftime("%Y-%m-%d"), "idx": i, "precio": precio, "atr": atr,
        "adx": float(fila["adx"]) if pd.notna(fila["adx"]) else 0.0, "rsi": float(fila["rsi"]) if pd.notna(fila["rsi"]) else 0.0,
        "vol_ratio": vol_ratio, "filtros": filtros, "filtros_ok": bool(s["filtros_ok"]),
        "score": calcular_score(dfi, i, estrategia, tipo) if activa else 0, "resultados_dias": None,
    }
    res.update(riesgo)
    if activa and not nueva:
        res["tipo"] = f"{tipo} (activa desde {res['desde']})"
    return res

# ===========================
# ESCANEO PRINCIPAL (Headless)
# ===========================
def ejecutar_escaneo(log=log_consola, p=None, estrategias=None, stop_event=None):
    p = p or cargar_settings()
    estrategias = estrategias or lista_estrategias(p["estrategia"])
    nombres = " + ".join(nombre_estrategia(e, p) for e in estrategias)
    log("=" * 60)
    log(f"🚀 INICIANDO ESCANEO ENTRADAS LONG — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    log(f"🧭 Estrategia: {nombres}")
    log("=" * 60)
    
    activos = cargar_activos()
    if not activos:
        log("️ La lista de activos está vacía.")
        return [], {}, None
        
    indice = normalizar_ticker(p["indice_mercado"])
    lista = list(activos) + ([indice] if p["filtro_mercado"] and indice not in activos else [])
    
    t0 = time.time()
    datos = descargar_lote(lista, periodo=p["periodo_datos"], log=log, stop_event=stop_event)
    log(f"⬇️ Descargados {len(datos)}/{len(lista)} tickers en {time.time() - t0:.1f} s")
    
    mercado, estado_mercado = None, None
    if p["filtro_mercado"]:
        mercado = serie_mercado(datos.get(indice))
        if mercado is None:
            log(f"️ No hay datos de {indice}: el filtro de mercado se ignora.")
        else:
            estado_mercado = bool(mercado.iloc[-1])
            log(f"🌍 Mercado ({indice} vs SMA200): {'ALCISTA ✅' if estado_mercado else 'BAJISTA '}")
            
    historial = cargar_historial()
    senales, cache = [], {}
    sin_datos = descartadas = 0
    total = len(activos)
    
    for idx, ticker in enumerate(activos, start=1):
        if stop_event is not None and stop_event.is_set():
            log("\n⏹️ Escaneo detenido.")
            break
        try:
            df = datos.get(ticker)
            if df is None or len(df) < 60:
                sin_datos += 1
                continue
            dfi = calcular_indicadores(df, p)
            cache[ticker] = dfi
            for est in estrategias:
                r = analizar_ticker(ticker, dfi, est, p, mercado)
                if r is None:
                    continue
                visible = r["nueva"] if p["solo_nuevas"] else r["activa"]
                if not visible:
                    continue
                if not r["filtros_ok"]:
                    descartadas += 1
                    continue
                if r["score"] < int(p["score_min"]):
                    descartadas += 1
                    continue
                    
                dias, _ = dias_hasta_resultados(ticker)
                r["resultados_dias"] = dias
                if p["filtro_resultados"] and dias is not None and dias <= int(p["dias_resultados"]):
                    descartadas += 1
                    continue
                    
                senales.append(r)
                log(f"🟢 {ticker:<10} [{est}] ENTRADA LONG @ {r['precio']:.2f} · SL {r['sl']:.2f} · TP {r['tp']:.2f} · ⭐{r['score']}")
        except Exception as e:
            log(f"⚠️ {ticker}: error inesperado: {e}")
            logger.exception(f"Error analizando {ticker}")
            
    senales.sort(key=lambda s: -s["score"])
    
    # --- Avisos individuales ---
    if p["enviar_individuales"]:
        for r in senales:
            if es_repetida(historial, r["ticker"], r["estrategia"], r["vela"], int(p["dias_no_repetir"])):
                log(f"⏭️ {r['ticker']} [{r['estrategia']}] ya avisada (anti-spam).")
                continue
            texto = mensaje_senal(r, p)
            enviado = enviar_telegram(texto, log=log)
            if enviado:
                registrar_senal(historial, r["ticker"], r["estrategia"], r["vela"])
                
    guardar_historial(limpiar_historial(historial))
    
    if p["enviar_resumen"]:
        enviar_resumen_telegram(senales, log=log, p=p, info={
            "estrategias": nombres, "mercado": estado_mercado,
            "analizados": total - sin_datos, "sin_datos": sin_datos, "descartadas": descartadas
        })
        
    log("\n✔ Escaneo finalizado.")
    log(f"📋 {len(senales)} señal(es) · {descartadas} descartada(s) · {sin_datos} sin datos.\n")
    return senales, cache, mercado

# ===========================
# MAIN (Entry Point para Servidor)
# ===========================
def main():
    parser = argparse.ArgumentParser(description="Escáner de entradas LONG Trullas PRO (Versión Servidor)")
    parser.add_argument("--estrategia", choices=["EMA", "SMMA", "AMBAS"], type=str.upper, help="Estrategia a usar (por defecto la guardada en settings)")
    args = parser.parse_args()
    
    try:
        cargar_config_telegram()
        p = cargar_settings()
        est = lista_estrategias(args.estrategia or p["estrategia"])
        
        log_consola("Iniciando bot en modo servidor (headless)...")
        ejecutar_escaneo(p=p, estrategias=est)
        
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
