"""
Gradio UI für den Keyword Agent (Google ADK).
"""

import gradio as gr
import asyncio
import pandas as pd
import altair as alt
from dotenv import load_dotenv
from google.genai import types

from keyword_agent import root_runner

load_dotenv(dotenv_path=".env")


# ============================================================================
# Helpers
# ============================================================================

def _safe_json_load(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        import json
        try:
            return json.loads(value)
        except Exception:
            return None
    return None


def _clean_code(code_str: str) -> str:
    if not isinstance(code_str, str):
        return ""
    cleaned = code_str.strip()
    if cleaned.startswith("```python"):
        cleaned = cleaned.replace("```python", "").replace("```", "").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "").strip()
    return cleaned


def _build_df_top(seo_ergebnis: dict) -> pd.DataFrame:
    top = seo_ergebnis.get("top_keywords", []) if isinstance(seo_ergebnis, dict) else []
    if not isinstance(top, list) or not top:
        return pd.DataFrame()

    df = pd.DataFrame(top)
    preferred = ["keyword", "gesamt_score", "trend_index", "suchvolumen", "suchintention", "auswahl_begruendung"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def _make_error_outputs(msg: str):
    return (
        f"*{msg}*",
        pd.DataFrame(),
        None,
        "",
        "",
        "",
        "",
        f"*{msg}*",
    )


# ============================================================================
# Async Runner
# ============================================================================

async def run_keyword_agent_async(thema: str, artikeltext: str, alter_min: int, alter_max: int):
    session = await root_runner.session_service.create_session(
        user_id="user",
        app_name="keyword_agent"
    )

    nutzer_prompt = f"""
thema: {thema}
artikeltext: {artikeltext}
alter_min: {alter_min}
alter_max: {alter_max}
""".strip()

    content = types.Content(role="user", parts=[types.Part(text=nutzer_prompt)])

    events_async = root_runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=content
    )

    results = {}
    async for event in events_async:
        if event.actions and event.actions.state_delta:
            results.update(event.actions.state_delta)

    return results


async def process_request_async(thema: str, artikeltext: str, alter_min: int, alter_max: int):
    if not str(thema).strip():
        return _make_error_outputs("Bitte ein Thema eingeben.")

    try:
        alter_min = int(alter_min)
        alter_max = int(alter_max)
    except Exception:
        return _make_error_outputs("Alterswerte müssen ganze Zahlen sein.")

    if alter_min < 0 or alter_max < 0 or alter_min > alter_max:
        return _make_error_outputs("Bitte einen gültigen Altersbereich angeben (min <= max, beide >= 0).")

    results = await run_keyword_agent_async(thema.strip(), artikeltext or "", alter_min, alter_max)

    validierte_eingabe = _safe_json_load(results.get("validierte_eingabe"))
    if isinstance(validierte_eingabe, dict) and validierte_eingabe.get("gueltig") is False:
        return _make_error_outputs(validierte_eingabe.get("fehlermeldung") or "Eingabe ungültig.")

    seo_ergebnis = _safe_json_load(results.get("seo_ergebnis")) or {}
    df_top = _build_df_top(seo_ergebnis)

    seo_titel = seo_ergebnis.get("seo_titel", "")
    meta = seo_ergebnis.get("meta_beschreibung", "")
    hook = seo_ergebnis.get("hook_ueberschrift", "")
    vorspann = seo_ergebnis.get("vorspann", "")

    zusammenfassung = results.get("zusammenfassung", "")
    zusammenfassung_md = zusammenfassung if zusammenfassung else "*Keine Zusammenfassung verfügbar.*"

    chart = None
    diagramm_code = _clean_code(results.get("diagramm_code", ""))

    if diagramm_code and not df_top.empty:
        try:
            namespace = {"alt": alt, "pd": pd, "df": df_top, "daten": df_top.to_dict(orient="records")}
            exec(diagramm_code, namespace)
            chart = namespace.get("chart")
        except Exception:
            chart = None

    trends_flag = seo_ergebnis.get("trends_verfuegbar", None)
    status = "*Fertig.*" if trends_flag is not False else "*Fertig (Trenddaten nicht verfügbar).*"

    return status, df_top, chart, seo_titel, meta, hook, vorspann, zusammenfassung_md


def process_request(thema: str, artikeltext: str, alter_min: int, alter_max: int):
    return asyncio.run(process_request_async(thema, artikeltext, alter_min, alter_max))


# ============================================================================
# UI
# ============================================================================

with gr.Blocks(title="Keyword Agent (Google ADK)") as demo:
    gr.Markdown(
        """
# Keyword Agent (Google ADK)

Gib Thema, optional Artikeltext und Altersbereich ein.  
Du bekommst Top-Keywords + SEO-Texte + Diagramm + Kurz-Zusammenfassung.
"""
    )

    thema_input = gr.Textbox(label="Thema", lines=1)
    artikeltext_input = gr.Textbox(label="Artikeltext (optional)", lines=8)

    with gr.Row():
        alter_min_input = gr.Number(label="Alter (von)", value=18, precision=0)
        alter_max_input = gr.Number(label="Alter (bis)", value=24, precision=0)

    with gr.Row():
        generieren_btn = gr.Button("Generieren", variant="primary")
        reset_btn = gr.Button("Zurücksetzen")

    status_output = gr.Markdown(value="*Warte auf Eingabe…*")

    with gr.Row():
        keywords_df_output = gr.DataFrame(label="Top 10 Keywords", wrap=True)
        chart_output = gr.Plot(label="Diagramm")

    seo_titel_output = gr.Textbox(label="SEO Titel", lines=2)
    meta_output = gr.Textbox(label="Meta-Beschreibung", lines=3)
    hook_output = gr.Textbox(label="Hook-Überschrift", lines=2)
    vorspann_output = gr.Textbox(label="Vorspann", lines=5)

    zusammenfassung_output = gr.Markdown(value="*Warte auf Eingabe…*")

    generieren_btn.click(
        fn=process_request,
        inputs=[thema_input, artikeltext_input, alter_min_input, alter_max_input],
        outputs=[status_output, keywords_df_output, chart_output, seo_titel_output, meta_output, hook_output, vorspann_output, zusammenfassung_output],
    )

    reset_btn.click(
        fn=lambda: ("*Warte auf Eingabe…*", pd.DataFrame(), None, "", "", "", "", "*Warte auf Eingabe…*"),
        inputs=None,
        outputs=[status_output, keywords_df_output, chart_output, seo_titel_output, meta_output, hook_output, vorspann_output, zusammenfassung_output],
    )

if __name__ == "__main__":
    demo.launch()