"""Streamlit-Frontend für den Insurance Claim Classifier & Smart Router."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import KATEGORIEN, analysiere_schreiben


st.set_page_config(
    page_title="Insurance Claim Classifier & Smart Router",
    page_icon=None,
    layout="wide",
)


BEISPIEL_SCHREIBEN = {
    "Schadensmeldung mit hoher Dringlichkeit": """Sehr geehrte Damen und Herren,

am 12.05.2026 kam es in meiner Wohnung zu einem erheblichen Wasserschaden. Die Küche ist aktuell nicht nutzbar, der Kostenvoranschlag beträgt 4.850,00 EUR. Meine Versicherungsnummer lautet VS-7845129. Bitte bearbeiten Sie den Schaden dringend, da weitere Folgekosten entstehen.

Mit freundlichen Grüßen
Anna Keller""",
    "Beschwerde wegen Bearbeitungsdauer": """Sehr geehrte Damen und Herren,

ich bin sehr unzufrieden mit der Bearbeitung meines Anliegens. Seit drei Wochen erhalte ich keine Antwort, obwohl ich bereits mehrfach nachgefragt habe. Das ist aus meiner Sicht nicht akzeptabel. Wenn ich bis morgen keine Rückmeldung bekomme, wende ich mich an meinen Anwalt.

Freundliche Grüße
Thomas Berger""",
    "Vertragsänderung": """Guten Tag,

bitte ändern Sie zum 01.06.2026 meine Bankverbindung für den bestehenden Vertrag VNR 99347120. Die neue IBAN habe ich im Kundenportal hinterlegt. Außerdem möchte ich prüfen lassen, ob ein Tarifwechsel sinnvoll ist.

Mit freundlichen Grüßen
M. Schulte""",
    "Kündigung": """Sehr geehrte Damen und Herren,

hiermit kündige ich meinen Vertrag VS-671245 zum nächstmöglichen Zeitpunkt. Bitte senden Sie mir eine schriftliche Bestätigung der Kündigung und nennen Sie mir das genaue Vertragsende.

Mit freundlichen Grüßen
Laura Neumann""",
}


def _css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f9fb;
            color: #17202a;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dde5ea;
            border-radius: 8px;
            padding: 14px 16px;
        }
        .status-box {
            background: #ffffff;
            border: 1px solid #dde5ea;
            border-radius: 8px;
            padding: 16px;
            min-height: 116px;
        }
        .status-hoch {
            border-left: 6px solid #c53030;
        }
        .status-mittel {
            border-left: 6px solid #b7791f;
        }
        .status-niedrig {
            border-left: 6px solid #2f855a;
        }
        .small-label {
            color: #52616b;
            font-size: 0.86rem;
            margin-bottom: 0.25rem;
        }
        .big-value {
            color: #17202a;
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _initialisiere_state() -> None:
    if "beispiel_name" not in st.session_state:
        st.session_state.beispiel_name = next(iter(BEISPIEL_SCHREIBEN))
    if "eingangstext" not in st.session_state:
        st.session_state.eingangstext = BEISPIEL_SCHREIBEN[st.session_state.beispiel_name]
    if "analyse" not in st.session_state:
        st.session_state.analyse = None


def _lade_beispiel() -> None:
    st.session_state.eingangstext = BEISPIEL_SCHREIBEN[st.session_state.beispiel_name]
    st.session_state.analyse = None


def _format_prozent(wert: float) -> str:
    return f"{wert * 100:.1f} %"


def _zeige_statusbox(titel: str, wert: str, klasse: str = "") -> None:
    st.markdown(
        f"""
        <div class="status-box {klasse}">
            <div class="small-label">{titel}</div>
            <div class="big-value">{wert}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _zeige_analyse(ergebnis: dict) -> None:
    prioritaet_klasse = {
        "Hoch": "status-hoch",
        "Mittel": "status-mittel",
        "Niedrig": "status-niedrig",
    }.get(ergebnis["prioritaet"], "")

    spalte_1, spalte_2, spalte_3, spalte_4 = st.columns(4)
    with spalte_1:
        _zeige_statusbox(
            "Kategorie",
            f"{ergebnis['kategorie']}<br><span class='small-label'>{_format_prozent(ergebnis['klassifikation_score'])}</span>",
        )
    with spalte_2:
        _zeige_statusbox(
            "Stimmung",
            f"{ergebnis['stimmung']}<br><span class='small-label'>{_format_prozent(ergebnis['stimmung_score'])}</span>",
        )
    with spalte_3:
        _zeige_statusbox(
            "Priorität",
            f"{ergebnis['prioritaet']}<br><span class='small-label'>Score {ergebnis['prioritaet_score']}/100</span>",
            prioritaet_klasse,
        )
    with spalte_4:
        _zeige_statusbox("SLA", ergebnis["sla"])

    st.divider()

    linke_spalte, rechte_spalte = st.columns([1.1, 1])
    with linke_spalte:
        st.subheader("Routing")
        st.write(ergebnis["route"])

        st.subheader("Begründung")
        for grund in ergebnis["gruende"]:
            st.write(f"- {grund}")

        st.subheader("Extrahierte Merkmale")
        metadaten = ergebnis["metadaten"]
        merkmale_df = pd.DataFrame(
            [
                {"Merkmal": "Vertragsnummern", "Wert": ", ".join(metadaten["vertragsnummern"]) or "-"},
                {"Merkmal": "Schadennummern", "Wert": ", ".join(metadaten["schadennummern"]) or "-"},
                {"Merkmal": "E-Mail-Adressen", "Wert": ", ".join(metadaten["emails"]) or "-"},
                {"Merkmal": "Beträge", "Wert": ", ".join(metadaten["geldbetraege"]) or "-"},
                {"Merkmal": "Datumsangaben", "Wert": ", ".join(metadaten["daten"]) or "-"},
                {"Merkmal": "Textumfang", "Wert": f"{metadaten['woerter']} Wörter"},
            ]
        )
        st.dataframe(merkmale_df, use_container_width=True, hide_index=True)

    with rechte_spalte:
        st.subheader("Klassifikationsscores")
        score_df = (
            pd.DataFrame(
                {
                    "Kategorie": list(ergebnis["kategorien_scores"].keys()),
                    "Score": list(ergebnis["kategorien_scores"].values()),
                }
            )
            .sort_values("Score", ascending=True)
            .reset_index(drop=True)
        )
        fig = go.Figure(
            go.Bar(
                x=score_df["Score"],
                y=score_df["Kategorie"],
                orientation="h",
                marker_color=["#2f855a" if k == ergebnis["kategorie"] else "#8aa0ae" for k in score_df["Kategorie"]],
                hovertemplate="%{y}: %{x:.1%}<extra></extra>",
            )
        )
        fig.update_layout(
            height=285,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(tickformat=".0%", range=[0, 1]),
            yaxis_title=None,
            xaxis_title=None,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Antwortentwurf")
        st.text_area(
            "Antwort",
            value=ergebnis["antwort"],
            height=260,
            label_visibility="collapsed",
        )

    if ergebnis["modellhinweise"]:
        with st.expander("Modellstatus"):
            for hinweis in ergebnis["modellhinweise"]:
                st.write(f"- {hinweis}")


def _brief_simulator() -> None:
    st.header("Brief-Simulator")
    eingabe_spalte, optionen_spalte = st.columns([2.2, 1])

    with optionen_spalte:
        st.selectbox(
            "Szenario",
            options=list(BEISPIEL_SCHREIBEN.keys()),
            key="beispiel_name",
            on_change=_lade_beispiel,
        )
        nutze_sprachmodell = st.toggle(
            "German-GPT2 für Antwortentwurf",
            value=False,
        )
        analysieren = st.button("Analyse starten", type="primary", use_container_width=True)

    with eingabe_spalte:
        st.text_area(
            "Eingangsschreiben",
            key="eingangstext",
            height=310,
        )

    if analysieren:
        try:
            with st.spinner("Modelle analysieren das Schreiben..."):
                st.session_state.analyse = analysiere_schreiben(
                    st.session_state.eingangstext,
                    nutze_sprachmodell=nutze_sprachmodell,
                )
        except Exception as exc:
            st.error(f"Analyse fehlgeschlagen: {exc}")

    if st.session_state.analyse:
        _zeige_analyse(st.session_state.analyse)


def _roi_daten(
    volumen: int,
    minuten_manuell: float,
    minuten_ki: float,
    automatisierungsquote: float,
    stundensatz: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    manuelle_stunden = volumen * minuten_manuell / 60
    ki_stunden = volumen * (
        automatisierungsquote * minuten_ki
        + (1 - automatisierungsquote) * minuten_manuell
    ) / 60
    gesparte_stunden = max(0.0, manuelle_stunden - ki_stunden)
    monatliche_einsparung = gesparte_stunden * stundensatz

    monate = list(range(1, 13))
    daten = pd.DataFrame(
        {
            "Monat": monate,
            "Manuell": [manuelle_stunden * monat for monat in monate],
            "KI-gestützt": [ki_stunden * monat for monat in monate],
            "Ersparte Stunden": [gesparte_stunden * monat for monat in monate],
            "Kumulierte Einsparung EUR": [monatliche_einsparung * monat for monat in monate],
        }
    )
    kennzahlen = {
        "manuelle_stunden": manuelle_stunden,
        "ki_stunden": ki_stunden,
        "gesparte_stunden": gesparte_stunden,
        "monatliche_einsparung": monatliche_einsparung,
    }
    return daten, kennzahlen


def _business_metrics() -> None:
    st.header("Business Metrics")
    regler_spalte, chart_spalte = st.columns([1, 2.1])

    with regler_spalte:
        volumen = st.number_input(
            "Briefe pro Monat",
            min_value=10_000,
            max_value=500_000,
            value=100_000,
            step=10_000,
        )
        minuten_manuell = st.slider(
            "Manuelle Minuten pro Brief",
            min_value=2.0,
            max_value=15.0,
            value=7.5,
            step=0.5,
        )
        minuten_ki = st.slider(
            "KI-gestützte Minuten pro Brief",
            min_value=0.5,
            max_value=5.0,
            value=1.4,
            step=0.1,
        )
        automatisierungsquote_prozent = st.slider(
            "Automatisierungsquote",
            min_value=40,
            max_value=95,
            value=78,
            step=1,
            format="%d %%",
        )
        stundensatz = st.slider(
            "Vollkosten pro Arbeitsstunde",
            min_value=25.0,
            max_value=80.0,
            value=42.0,
            step=1.0,
        )

    daten, kennzahlen = _roi_daten(
        volumen=int(volumen),
        minuten_manuell=minuten_manuell,
        minuten_ki=minuten_ki,
        automatisierungsquote=automatisierungsquote_prozent / 100,
        stundensatz=stundensatz,
    )

    with chart_spalte:
        metrik_1, metrik_2, metrik_3, metrik_4 = st.columns(4)
        metrik_1.metric("Manuell pro Monat", f"{kennzahlen['manuelle_stunden']:,.0f} h")
        metrik_2.metric("KI-gestützt pro Monat", f"{kennzahlen['ki_stunden']:,.0f} h")
        metrik_3.metric("Ersparnis pro Monat", f"{kennzahlen['gesparte_stunden']:,.0f} h")
        metrik_4.metric("Monatlicher ROI", f"{kennzahlen['monatliche_einsparung']:,.0f} €")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=daten["Monat"],
                y=daten["Manuell"],
                name="Manuell",
                marker_color="#8aa0ae",
                hovertemplate="Monat %{x}<br>%{y:,.0f} Stunden<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=daten["Monat"],
                y=daten["KI-gestützt"],
                name="KI-gestützt",
                marker_color="#2f855a",
                hovertemplate="Monat %{x}<br>%{y:,.0f} Stunden<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daten["Monat"],
                y=daten["Ersparte Stunden"],
                name="Ersparte Stunden",
                mode="lines+markers",
                line=dict(color="#b7791f", width=3),
                marker=dict(size=7),
                hovertemplate="Monat %{x}<br>%{y:,.0f} Stunden gespart<extra></extra>",
            )
        )
        fig.update_layout(
            barmode="group",
            height=500,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Monat",
            yaxis_title="Kumulierte Arbeitsstunden",
            legend_title=None,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            daten.assign(
                **{
                    "Kumulierte Einsparung EUR": daten["Kumulierte Einsparung EUR"].map(
                        lambda wert: f"{wert:,.0f} €"
                    )
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    _css()
    _initialisiere_state()

    st.title("Insurance Claim Classifier & Smart Router")
    tab_simulator, tab_metrics = st.tabs(["Brief-Simulator", "Business Metrics"])

    with tab_simulator:
        _brief_simulator()

    with tab_metrics:
        _business_metrics()


if __name__ == "__main__":
    main()
