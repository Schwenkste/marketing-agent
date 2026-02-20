"""
Tools für den Keyword Agent.

Enthält:
- Google Trends Abfrage (via pytrends) als Tool für Google ADK
- Fallback, wenn Trends nicht verfügbar ist
- Kleine Hilfsfunktionen (Normalisierung, Sicherheit)

WICHTIG:
pytrends liefert typischerweise KEIN echtes "Suchvolumen" (MSV).
Wir geben daher:
- trend_index (0-100 Proxy/Index)
- suchvolumen = None
- verwandte_suchanfragen (related queries)
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
import re


# -----------------------------
# Kleine Utilities
# -----------------------------
def _normalisiere_keyword(keyword: str) -> str:
    keyword = (keyword or "").strip()
    keyword = re.sub(r"\s+", " ", keyword)
    return keyword


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# Optional: sehr grober Brand-Safety Filter (nur als zusätzliche Sicherung)
# (Der eigentliche Brand-Safety-Check sollte in den Agenten-Instructions passieren.)
_VERBOTENE_MUSTER = [
    r"\b(kokain|heroin|meth|crystal|drogen kaufen)\b",
    r"\b(waffe kaufen|schusswaffe|bombenbau)\b",
    r"\b(kindesmissbrauch)\b",
]


def brand_safety_ok(keyword: str) -> bool:
    k = keyword.lower()
    for pat in _VERBOTENE_MUSTER:
        if re.search(pat, k):
            return False
    return True


# -----------------------------
# Google Trends Tool (pytrends)
# -----------------------------
def get_trend_daten_fuer_keywords(
    keywords: List[str],
    land: str = "DE",
    zeitraum: str = "today 12-m",
    sprache: str = "de-DE",
    max_verwandte: int = 5,
) -> Dict[str, Any]:
    """
    ADK Tool: Holt Trenddaten (Proxy-Index) und verwandte Suchanfragen.

    Args:
        keywords: Liste von Keywords (Strings)
        land: Geo (DE)
        zeitraum: z.B. "today 12-m", "today 3-m", "today 5-y"
        sprache: z.B. "de-DE"
        max_verwandte: wie viele related queries max.

    Returns:
        {
          "trends_verfuegbar": bool,
          "ergebnisse": [
             {
               "keyword": str,
               "trend_index": int|None,  # 0-100 Proxy
               "suchvolumen": None,      # pytrends liefert i.d.R. kein MSV
               "verwandte_suchanfragen": [str, ...]
             }, ...
          ],
          "hinweis": str
        }
    """
    # Normalisieren + Brand safety + Dedup
    cleaned = [_normalisiere_keyword(k) for k in (keywords or [])]
    cleaned = [k for k in cleaned if k and brand_safety_ok(k)]
    cleaned = _dedupe_preserve_order(cleaned)

    if not cleaned:
        return {
            "trends_verfuegbar": False,
            "ergebnisse": [],
            "hinweis": "Keine gültigen Keywords für Trends-Abfrage vorhanden.",
        }

    # pytrends kann pro Request nur eine begrenzte Anzahl Keywords sinnvoll handeln.
    # Wir machen es robust: batchweise und aggregieren.
    try:
        from pytrends.request import TrendReq  # type: ignore
        import pandas as pd  # noqa: F401

        pytrends = TrendReq(hl=sprache, tz=360)

        ergebnisse: List[Dict[str, Any]] = []

        # Batch-Größe: 5 ist erfahrungsgemäß stabil
        BATCH_SIZE = 5
        for i in range(0, len(cleaned), BATCH_SIZE):
            batch = cleaned[i : i + BATCH_SIZE]
            pytrends.build_payload(batch, timeframe=zeitraum, geo=land)

            # 1) Interest over time -> wir reduzieren auf einen Proxy-Index:
            #    z.B. Maximum über Zeitraum oder Mittelwert (hier: Mittelwert).
            iot = pytrends.interest_over_time()
            # iot enthält Spalten je Keyword + evtl. isPartial

            # 2) Related queries
            related = pytrends.related_queries()  # dict: {keyword: {"top": df, "rising": df}}

            for kw in batch:
                trend_index: Optional[int] = None
                try:
                    if kw in iot.columns:
                        series = iot[kw].dropna()
                        if len(series) > 0:
                            # Proxy: Durchschnitt (0-100)
                            trend_index = int(round(float(series.mean())))
                except Exception:
                    trend_index = None

                verwandte: List[str] = []
                try:
                    rq = related.get(kw, {})
                    top_df = rq.get("top")
                    rising_df = rq.get("rising")

                    # Wir nehmen zuerst rising, dann top (oder andersrum).
                    if rising_df is not None and not rising_df.empty:
                        verwandte += rising_df["query"].head(max_verwandte).tolist()
                    if top_df is not None and not top_df.empty and len(verwandte) < max_verwandte:
                        needed = max_verwandte - len(verwandte)
                        verwandte += top_df["query"].head(needed).tolist()

                    verwandte = [_normalisiere_keyword(x) for x in verwandte if isinstance(x, str)]
                    verwandte = [x for x in verwandte if x and brand_safety_ok(x)]
                    verwandte = _dedupe_preserve_order(verwandte)
                except Exception:
                    verwandte = []

                ergebnisse.append(
                    {
                        "keyword": kw,
                        "trend_index": trend_index,
                        "suchvolumen": None,
                        "verwandte_suchanfragen": verwandte,
                    }
                )

        return {
            "trends_verfuegbar": True,
            "ergebnisse": ergebnisse,
            "hinweis": "trend_index ist ein Proxy (0-100) aus Google Trends; echtes Suchvolumen ist hier nicht enthalten.",
        }

    except Exception as e:
        # Fallback ohne Trends
        return {
            "trends_verfuegbar": False,
            "ergebnisse": [
                {
                    "keyword": k,
                    "trend_index": None,
                    "suchvolumen": None,
                    "verwandte_suchanfragen": [],
                }
                for k in cleaned
            ],
            "hinweis": f"Trends-Abfrage nicht verfügbar (pytrends Fehler): {str(e)}",
        }