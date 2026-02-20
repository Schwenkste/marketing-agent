"""
Keyword Agent (Google ADK) – Agenten-Pipeline.

Pipeline:
1) Eingabeprüfung
2) Keyword-Kandidaten (30)
3) Trend-Anreicherung (Tool: get_trend_daten_fuer_keywords)
4) Auswahl + Bewertung (Top 10) + SEO-Texte
5) Visualisierung (Altair Code)
6) Zusammenfassung
"""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import InMemoryRunner

from keyword_agent.tools import get_trend_daten_fuer_keywords

GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================================
# 1) Eingabeprüfung
# ============================================================================
eingabe_pruefung_agent = LlmAgent(
    model=GEMINI_MODEL,
    name="eingabe_pruefung_agent",
    description="Prüft Thema, Text und Altersbereich und normalisiert die Eingaben.",
    instruction="""
<system_prompt>

Du validierst Nutzereingaben für ein deutsches SEO-Keyword-Tool.

Du erhältst diese Werte aus dem State:
- {thema}
- {artikeltext}
- {alter_min}
- {alter_max}

Regeln:
- Thema darf nach trim() nicht leer sein
- alter_min und alter_max sind ganze Zahlen
- alter_min >= 0, alter_max >= 0
- alter_min <= alter_max
- artikeltext darf leer sein, dann aber Warnung ausgeben

Gib NUR JSON aus, exakt in diesem Format:
{
  "thema": "...",
  "artikeltext": "...",
  "alter_min": 0,
  "alter_max": 0,
  "gueltig": true,
  "fehlermeldung": "",
  "warnungen": []
}

Wenn ungültig:
- gueltig=false
- fehlermeldung mit klarer Handlungsanweisung

Keine Markdown-Fences. Kein zusätzlicher Text.

</system_prompt>
""",
    output_key="validierte_eingabe",
)


# ============================================================================
# 2) Keyword-Kandidaten-Generierung (30)
# ============================================================================
keyword_generierung_agent = LlmAgent(
    model=GEMINI_MODEL,
    name="keyword_generierung_agent",
    description="Erstellt 30 deutsche Keyword-Kandidaten basierend auf Thema, Text und Zielgruppenalter.",
    instruction="""
<system_prompt>

Du bist SEO-Experte für deutschsprachige Inhalte.

Input (als JSON im State):
{validierte_eingabe}

Aufgabe:
Erstelle exakt 30 Keyword-Kandidaten auf Deutsch.
Mischung aus:
- Hauptkeywords (Head)
- Longtail-Keywords
- Frage-Keywords

Alters-Anpassung (ohne Slang, neutral):
- Jüngere Zielgruppen: häufiger Einsteiger, Basics, Preis/Preis-Leistung, kurz erklärt
- Mittlere Zielgruppen: Vergleich, Test, Erfahrungen, beste Optionen
- Ältere Zielgruppen: Schritt-für-Schritt, Sicherheit, Komfort, Zuverlässigkeit

Brand-Safety:
- Keine sensiblen/illegalen/anstößigen Keywords.

Output (NUR JSON):
{
  "keyword_kandidaten": [
    {
      "keyword": "...",
      "suchintention": "informativ|kommerziell|transaktional|navigational",
      "begruendung": "..."
    }
  ]
}

Constraints:
- Genau 30 Einträge.
- begruendung: 1 kurzer Satz.

</system_prompt>
""",
    output_key="keyword_kandidaten",
)


# ============================================================================
# 3) Trend-Anreicherung (Tool)
# ============================================================================
trend_anreicherung_agent = LlmAgent(
    model=GEMINI_MODEL,
    name="trend_anreicherung_agent",
    description="Reichert Keywords mit Trenddaten (Proxy) und verwandten Suchanfragen an.",
    instruction="""
<system_prompt>

Du reicherst Keyword-Kandidaten mit Trenddaten für Deutschland an.

Input:
{keyword_kandidaten}

HARD CONSTRAINT:
- Du MUSST das Tool get_trend_daten_fuer_keywords verwenden, um Trenddaten zu holen.
- Übergib dem Tool die Liste der Keywords (Strings).

Vorgehen:
1) Extrahiere alle 30 Keywords aus keyword_kandidaten.
2) Rufe das Tool get_trend_daten_fuer_keywords(keywords=[...]) auf.
3) Merge die Tool-Ergebnisse zurück zu jedem Keyword.
4) Gib NUR JSON aus im Format:

{
  "angereicherte_keywords": [
    {
      "keyword": "...",
      "suchintention": "...",
      "begruendung": "...",
      "trend_index": 0-100|null,
      "suchvolumen": int|null,
      "verwandte_suchanfragen": ["...", "..."]
    }
  ],
  "trends_verfuegbar": true|false
}

Hinweise:
- Falls Tool trends_verfuegbar=false liefert, setze trend_index=null, suchvolumen=null, verwandte_suchanfragen=[].
- Wenn ein Keyword im Tool-Output fehlt: ebenfalls null/[] setzen.

Keine Markdown-Fences. Kein zusätzlicher Text.

</system_prompt>
""",
    tools=[get_trend_daten_fuer_keywords],
    output_key="trend_daten",
)


# ============================================================================
# 4) Auswahl + Bewertung + SEO-Texte (Top 10)
# ============================================================================
seo_bewertung_agent = LlmAgent(
    model=GEMINI_MODEL,
    name="seo_bewertung_agent",
    description="Wählt Top 10 Keywords aus, bewertet sie und erzeugt SEO-Texte.",
    instruction="""
<system_prompt>

Du bist SEO-Redakteur und Keyword-Analyst.

Inputs:
{validierte_eingabe}
{trend_daten}

Aufgaben:
1) Wähle exakt 10 Keywords aus den 30 angereicherten Keywords.
2) Vergib pro Keyword einen gesamt_score (0-100):
   - Themenrelevanz (0-50)
   - Alters-Passung (0-20)
   - Trendstärke (0-30, NUR wenn trends_verfuegbar=true, sonst 0)
3) auswahl_begruendung: 1 Satz je Keyword.
4) Erstelle folgende Textausgaben (deutsch):
   - seo_titel (ca. 55-60 Zeichen)
   - meta_beschreibung (ca. 150-160 Zeichen)
   - hook_ueberschrift (kurz, prägnant, nicht reißerisch)
   - vorspann (2-4 Sätze, neutral-professionell)

Brand-Safety:
- Keine sensiblen/illegalen Inhalte, keine anstößigen Formulierungen.

Output (NUR JSON):
{
  "top_keywords": [
    {
      "keyword": "...",
      "gesamt_score": 0-100,
      "trend_index": int|null,
      "suchvolumen": int|null,
      "suchintention": "informativ|kommerziell|transaktional|navigational",
      "auswahl_begruendung": "..."
    }
  ],
  "seo_titel": "...",
  "meta_beschreibung": "...",
  "hook_ueberschrift": "...",
  "vorspann": "...",
  "trends_verfuegbar": true|false
}

Constraints:
- Genau 10 Einträge in top_keywords.

</system_prompt>
""",
    output_key="seo_ergebnis",
)


# ============================================================================
# 5) Visualisierung (Altair Code)
# ============================================================================
visualisierungs_agent = LlmAgent(
    model=GEMINI_MODEL,
    name="visualisierungs_agent",
    description="Erstellt ein Altair-Diagramm (Python-Code) für Keyword-Scores.",
    instruction="""
<system_prompt>

Input:
{seo_ergebnis}

Erzeuge exakt EIN Altair-Diagramm als ausführbaren Python-Code.

Ziel:
- Horizontales Balkendiagramm: Keyword vs. gesamt_score
- Sortiert nach gesamt_score absteigend

HARD CONSTRAINTS:
- Output NUR Python-Code
- Keine Markdown-Fences
- Muss enthalten:
  - import altair as alt
  - import pandas as pd
- Das finale Chart muss in einer Variable 'chart' gespeichert sein
- width 400-600, height 300-400

Tooltips:
- keyword, gesamt_score
- optional trend_index, suchvolumen (wenn vorhanden)

</system_prompt>
""",
    output_key="diagramm_code",
)


# ============================================================================
# 6) Zusammenfassung (2-4 Sätze)
# ============================================================================
zusammenfassung_agent = LlmAgent(
    model=GEMINI_MODEL,
    name="zusammenfassung_agent",
    description="Erstellt eine kurze Zusammenfassung der Ergebnisse (2-4 Sätze).",
    instruction="""
<system_prompt>

Input:
{seo_ergebnis}

Schreibe exakt 2 bis 4 Sätze auf Deutsch:
- Was zeigen die Top-Keywords inhaltlich?
- Wie hat der Altersbereich die Auswahl beeinflusst?
- Nenne mindestens eine konkrete Zahl (z. B. höchster gesamt_score).
- Wenn trends_verfuegbar=false, erwähne das in 1 Satz (insgesamt 2-4 Sätze).

Keine technischen Begriffe, kein SQL, kein Tool-Jargon.

Output NUR Text.

</system_prompt>
""",
    output_key="zusammenfassung",
)


# ============================================================================
# Insight Pipeline: Visualisierung → Zusammenfassung
# ============================================================================
insight_pipeline = SequentialAgent(
    name="insight_pipeline",
    sub_agents=[visualisierungs_agent, zusammenfassung_agent],
    description="Erzeugt Diagramm-Code und Kurz-Zusammenfassung.",
)


# ============================================================================
# Root Agent
# ============================================================================
root_agent = SequentialAgent(
    name="keyword_pipeline",
    description="Prüfung → Keywords → Trends → Bewertung+SEO → Diagramm → Zusammenfassung",
    sub_agents=[
        eingabe_pruefung_agent,
        keyword_generierung_agent,
        trend_anreicherung_agent,
        seo_bewertung_agent,
        insight_pipeline,
    ],
)


# ============================================================================
# Runner
# ============================================================================
root_runner = InMemoryRunner(agent=root_agent, app_name="keyword_agent")