"""NLP-Backend für den Insurance Claim Classifier & Smart Router."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import torch
from transformers import pipeline


LOGGER = logging.getLogger(__name__)

SENTIMENT_MODELL = os.getenv("SENTIMENT_MODELL", "oliverguhr/german-sentiment-bert")
ZERO_SHOT_MODELL = os.getenv("ZERO_SHOT_MODELL", "Sahajpreet/german-zero-shot")
ZERO_SHOT_FALLBACK_MODELL = os.getenv(
    "ZERO_SHOT_FALLBACK_MODELL", "Sahajtomar/German_Zeroshot"
)
GENERIERUNGS_MODELL = os.getenv("GENERIERUNGS_MODELL", "dbmdz/german-gpt2")

KATEGORIEN = ["Schadensmeldung", "Beschwerde", "Vertragsänderung", "Kündigung"]
HYPOTHESE_TEMPLATE = "In diesem Schreiben geht es um {}."


@dataclass(frozen=True)
class AnalyseErgebnis:
    """Strukturierter Output für UI, Tests und spätere API-Anbindung."""

    kategorie: str
    klassifikation_score: float
    kategorien_scores: dict[str, float]
    stimmung: str
    stimmung_score: float
    prioritaet: str
    prioritaet_score: int
    route: str
    sla: str
    antwort: str
    gruende: list[str]
    metadaten: dict[str, Any]
    modellhinweise: list[str]


SCHLUESSELWOERTER = {
    "Schadensmeldung": [
        "schaden",
        "unfall",
        "wasserschaden",
        "brand",
        "diebstahl",
        "einbruch",
        "sturm",
        "hagel",
        "glasbruch",
        "kostenvoranschlag",
        "rechnung",
        "erstattung",
        "polizei",
        "gutachten",
        "krankenhaus",
        "verletzung",
    ],
    "Beschwerde": [
        "beschwerde",
        "unzufrieden",
        "enttäuscht",
        "verärgert",
        "keine antwort",
        "schlechter service",
        "warte",
        "seit wochen",
        "eskalation",
        "aufsicht",
        "anwalt",
        "nicht akzeptabel",
        "fristlos",
    ],
    "Vertragsänderung": [
        "vertrag ändern",
        "vertragsänderung",
        "adresse",
        "bankverbindung",
        "iban",
        "tarif",
        "leistung",
        "beitrag",
        "versicherungsnehmer",
        "familienstand",
        "umzug",
        "nachtrag",
        "deckung",
    ],
    "Kündigung": [
        "kündigung",
        "kündigen",
        "beenden",
        "vertragsende",
        "widerruf",
        "sonderkündigung",
        "ablauf",
        "wechseln",
        "nicht verlängern",
        "bestätigung der kündigung",
    ],
}

DRINGLICHKEITS_TERME = [
    "dringend",
    "sofort",
    "eilig",
    "heute",
    "morgen",
    "frist",
    "frist läuft",
    "unverzüglich",
    "mahnbescheid",
    "klage",
    "gericht",
    "anwalt",
]

SCHADENSSCHWERE_TERME = [
    "personenschaden",
    "verletzung",
    "krankenhaus",
    "notaufnahme",
    "brand",
    "einbruch",
    "diebstahl",
    "wohngebäude unbewohnbar",
    "existenz",
    "hoher betrag",
    "totalschaden",
]

NEGATIVE_TERME = [
    "beschwerde",
    "unzufrieden",
    "enttäuscht",
    "wütend",
    "verärgert",
    "inakzeptabel",
    "problem",
    "ärger",
    "keine reaktion",
    "nicht zufrieden",
    "schlecht",
    "katastrophe",
]

POSITIVE_TERME = [
    "danke",
    "vielen dank",
    "zufrieden",
    "freundlich",
    "hilfreich",
    "gut",
    "reibungslos",
    "schnell",
]

ROUTEN = {
    "Schadensmeldung": "Schadenservice Komposit",
    "Beschwerde": "Beschwerdemanagement",
    "Vertragsänderung": "Vertragsservice Bestand",
    "Kündigung": "Kundenbindung und Vertragsbeendigung",
}

SLA = {
    "Hoch": "4 Stunden",
    "Mittel": "1 Arbeitstag",
    "Niedrig": "3 Arbeitstage",
}


def _geraet() -> int:
    """Ermittelt das Zielgerät für Hugging-Face-Pipelines."""

    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def _sentiment_pipeline() -> Any | None:
    """Lädt das deutsche Sentiment-Modell genau einmal pro Prozess."""

    try:
        return pipeline(
            "text-classification",
            model=SENTIMENT_MODELL,
            tokenizer=SENTIMENT_MODELL,
            device=_geraet(),
        )
    except Exception as exc:  # pragma: no cover - hängt von Modell-Download ab
        LOGGER.exception("Sentiment-Modell konnte nicht geladen werden.")
        raise RuntimeError(f"Sentiment-Modell nicht verfügbar: {exc}") from exc


@lru_cache(maxsize=1)
def _zero_shot_pipeline() -> tuple[Any | None, str, str | None]:
    """Lädt den Zero-Shot-Klassifikator mit robustem Modell-Fallback."""

    fehler: list[str] = []
    kandidaten = [ZERO_SHOT_MODELL, ZERO_SHOT_FALLBACK_MODELL]
    for modell_name in dict.fromkeys(kandidaten):
        try:
            klassifikator = pipeline(
                "zero-shot-classification",
                model=modell_name,
                tokenizer=modell_name,
                device=_geraet(),
            )
            return klassifikator, modell_name, None
        except Exception as exc:  # pragma: no cover - hängt von Modell-Download ab
            fehler.append(f"{modell_name}: {exc}")
            LOGGER.warning("Zero-Shot-Modell %s konnte nicht geladen werden.", modell_name)

    return None, "", " | ".join(fehler)


@lru_cache(maxsize=1)
def _generierungs_pipeline() -> Any:
    """Lädt ein leichtes deutsches GPT-2-Modell für optionale Textverfeinerung."""

    generator = pipeline(
        "text-generation",
        model=GENERIERUNGS_MODELL,
        tokenizer=GENERIERUNGS_MODELL,
        device=_geraet(),
    )
    if getattr(generator, "tokenizer", None) is not None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token
    return generator


def _kuerzen(text: str, max_zeichen: int = 4000) -> str:
    return re.sub(r"\s+", " ", text).strip()[:max_zeichen]


def _zaehle_treffer(text_klein: str, begriffe: list[str]) -> int:
    return sum(1 for begriff in begriffe if begriff in text_klein)


def _regelbasierte_klassifikation(text: str) -> tuple[str, float, dict[str, float], str]:
    """Fallback-Klassifikation, falls das Modell lokal nicht verfügbar ist."""

    text_klein = text.lower()
    roh_scores = {
        kategorie: _zaehle_treffer(text_klein, begriffe)
        for kategorie, begriffe in SCHLUESSELWOERTER.items()
    }

    if sum(roh_scores.values()) == 0:
        roh_scores["Vertragsänderung"] = 1

    gesamt = sum(roh_scores.values())
    scores = {
        kategorie: round(max(0.01, score / gesamt), 4)
        for kategorie, score in roh_scores.items()
    }
    kategorie = max(scores, key=scores.get)
    return kategorie, scores[kategorie], scores, "Regelbasierte Klassifikation aktiv."


def _klassifiziere(text: str) -> tuple[str, float, dict[str, float], list[str]]:
    hinweise: list[str] = []
    klassifikator, modell_name, fehler = _zero_shot_pipeline()

    if klassifikator is None:
        kategorie, score, scores, hinweis = _regelbasierte_klassifikation(text)
        hinweise.append(f"{hinweis} Modellfehler: {fehler}")
        return kategorie, score, scores, hinweise

    try:
        ergebnis = klassifikator(
            _kuerzen(text),
            candidate_labels=KATEGORIEN,
            hypothesis_template=HYPOTHESE_TEMPLATE,
            multi_label=False,
            truncation=True,
        )
        labels = ergebnis.get("labels", [])
        scores_liste = ergebnis.get("scores", [])
        scores = {
            str(label): round(float(score), 4)
            for label, score in zip(labels, scores_liste, strict=False)
        }
        kategorie = str(labels[0]) if labels else "Vertragsänderung"
        score = float(scores_liste[0]) if scores_liste else 0.0
        if modell_name != ZERO_SHOT_MODELL:
            hinweise.append(
                f"Zero-Shot-Fallback genutzt: {modell_name} statt {ZERO_SHOT_MODELL}."
            )
        return kategorie, round(score, 4), scores, hinweise
    except Exception as exc:  # pragma: no cover - hängt vom Modell ab
        kategorie, score, scores, hinweis = _regelbasierte_klassifikation(text)
        hinweise.append(f"{hinweis} Inferenzfehler: {exc}")
        return kategorie, score, scores, hinweise


def _normalisiere_stimmung(label: str) -> str:
    label_klein = label.lower()
    if "negative" in label_klein or "negativ" in label_klein or label_klein == "0":
        return "Negativ"
    if "positive" in label_klein or "positiv" in label_klein or label_klein == "2":
        return "Positiv"
    return "Neutral"


def _regelbasierte_stimmung(text: str) -> tuple[str, float, str]:
    """Fallback-Sentiment für stabile Demo-Nutzung ohne Modellcache."""

    text_klein = text.lower()
    negative = _zaehle_treffer(text_klein, NEGATIVE_TERME)
    positive = _zaehle_treffer(text_klein, POSITIVE_TERME)

    if negative > positive:
        return "Negativ", min(0.98, 0.58 + negative * 0.08), "Regelbasiertes Sentiment aktiv."
    if positive > negative:
        return "Positiv", min(0.98, 0.58 + positive * 0.08), "Regelbasiertes Sentiment aktiv."
    return "Neutral", 0.56, "Regelbasiertes Sentiment aktiv."


def _analysiere_stimmung(text: str) -> tuple[str, float, list[str]]:
    hinweise: list[str] = []
    try:
        sentiment = _sentiment_pipeline()
        ergebnis = sentiment(_kuerzen(text, 1200), truncation=True)
        zeile = ergebnis[0] if isinstance(ergebnis, list) else ergebnis
        label = _normalisiere_stimmung(str(zeile.get("label", "neutral")))
        score = round(float(zeile.get("score", 0.0)), 4)
        return label, score, hinweise
    except Exception as exc:  # pragma: no cover - hängt von Modell-Download ab
        label, score, hinweis = _regelbasierte_stimmung(text)
        hinweise.append(f"{hinweis} Modellfehler: {exc}")
        return label, round(score, 4), hinweise


def _extrahiere_metadaten(text: str) -> dict[str, Any]:
    """Extrahiert einfache operative Merkmale für die Eingangsbearbeitung."""

    vertragsnummern = re.findall(
        r"\b(?:VS|VNR|Vertrag|Versicherungsschein)[-:\s]*([A-Z0-9]{5,18})\b",
        text,
        flags=re.IGNORECASE,
    )
    schadennummern = re.findall(
        r"\b(?:SN|Schaden)[-:\s]*([A-Z0-9]{5,18})\b",
        text,
        flags=re.IGNORECASE,
    )
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    geldbetraege = re.findall(r"\b\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s?(?:€|EUR)\b", text)
    daten = re.findall(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", text)

    return {
        "vertragsnummern": sorted(set(vertragsnummern)),
        "schadennummern": sorted(set(schadennummern)),
        "emails": sorted(set(emails)),
        "geldbetraege": sorted(set(geldbetraege)),
        "daten": sorted(set(daten)),
        "zeichen": len(text),
        "woerter": len(re.findall(r"\w+", text, flags=re.UNICODE)),
    }


def _berechne_prioritaet(
    text: str, kategorie: str, stimmung: str
) -> tuple[str, int, str, str, list[str]]:
    """Kombiniert Modelloutput mit fachlicher Routing-Logik."""

    basis_score = {
        "Schadensmeldung": 55,
        "Beschwerde": 68,
        "Vertragsänderung": 42,
        "Kündigung": 62,
    }.get(kategorie, 45)
    score = basis_score
    gruende = [f"Basisscore aus Kategorie: {kategorie}."]

    text_klein = text.lower()
    dringlichkeit = _zaehle_treffer(text_klein, DRINGLICHKEITS_TERME)
    schadensschwere = _zaehle_treffer(text_klein, SCHADENSSCHWERE_TERME)

    if stimmung == "Negativ":
        score += 14
        gruende.append("Negative Stimmung erhöht Eskalationsrisiko.")
    elif stimmung == "Positiv":
        score -= 4
        gruende.append("Positive Stimmung senkt Eskalationsrisiko leicht.")

    if dringlichkeit:
        score += min(20, 8 + dringlichkeit * 4)
        gruende.append("Dringlichkeitsbegriffe im Schreiben erkannt.")

    if schadensschwere and kategorie == "Schadensmeldung":
        score += min(22, 10 + schadensschwere * 4)
        gruende.append("Hinweise auf hohe Schadenschwere erkannt.")

    if kategorie == "Kündigung" and stimmung == "Negativ":
        score += 8
        gruende.append("Kündigung mit negativer Stimmung an Kundenbindung priorisiert.")

    score = max(0, min(100, int(round(score))))
    if score >= 75:
        prioritaet = "Hoch"
    elif score >= 50:
        prioritaet = "Mittel"
    else:
        prioritaet = "Niedrig"

    basis_route = ROUTEN.get(kategorie, "Zentrales Inputmanagement")
    if prioritaet == "Hoch":
        route = f"{basis_route} - Expressprüfung"
    else:
        route = basis_route

    return prioritaet, score, route, SLA[prioritaet], gruende


def _antwort_bausteine(kategorie: str) -> dict[str, str]:
    bausteine = {
        "Schadensmeldung": {
            "betreff": "Ihre Schadensmeldung",
            "kern": (
                "wir haben Ihre Schadensmeldung erhalten und leiten die Unterlagen "
                "an den zuständigen Schadenservice weiter. Bitte reichen Sie, sofern "
                "noch nicht geschehen, Fotos, Rechnungen und vorhandene Nachweise ein."
            ),
            "abschluss": (
                "Nach erster Prüfung melden wir uns mit den nächsten Schritten zur Regulierung."
            ),
        },
        "Beschwerde": {
            "betreff": "Ihre Rückmeldung zu unserem Service",
            "kern": (
                "vielen Dank für Ihre offene Rückmeldung. Wir bedauern, dass Ihr Anliegen "
                "nicht zu Ihrer Zufriedenheit bearbeitet wurde, und geben den Vorgang an "
                "das Beschwerdemanagement."
            ),
            "abschluss": (
                "Ihr Anliegen wird priorisiert geprüft; Sie erhalten eine sachliche Antwort "
                "mit nachvollziehbarer Begründung."
            ),
        },
        "Vertragsänderung": {
            "betreff": "Ihre gewünschte Vertragsänderung",
            "kern": (
                "wir haben Ihre Anfrage zur Vertragsänderung erhalten und prüfen die "
                "übermittelten Angaben im Vertragsservice."
            ),
            "abschluss": (
                "Falls weitere Nachweise erforderlich sind, informieren wir Sie kurzfristig."
            ),
        },
        "Kündigung": {
            "betreff": "Ihre Kündigungsanfrage",
            "kern": (
                "wir haben Ihre Kündigungsanfrage erhalten und prüfen Vertragslaufzeit, "
                "Kündigungsfrist und mögliche Alternativen."
            ),
            "abschluss": (
                "Sie erhalten eine schriftliche Bestätigung oder einen Hinweis, falls "
                "noch Angaben fehlen."
            ),
        },
    }
    return bausteine.get(kategorie, bausteine["Vertragsänderung"])


def _template_antwort(kategorie: str, prioritaet: str, route: str, sla: str) -> str:
    baustein = _antwort_bausteine(kategorie)
    return (
        f"Betreff: {baustein['betreff']}\n\n"
        "Sehr geehrte Damen und Herren,\n\n"
        f"{baustein['kern']} Der Vorgang wurde der Einheit "
        f"\"{route}\" zugeordnet und mit der Priorität \"{prioritaet}\" versehen.\n\n"
        f"Unser internes Serviceziel für die nächste Bearbeitung beträgt {sla}. "
        f"{baustein['abschluss']}\n\n"
        "Mit freundlichen Grüßen\n"
        "Ihr Inputmanagement-Team der BarmeniaGothaer"
    )


def _baue_prompt(text: str, kategorie: str, prioritaet: str, route: str) -> str:
    textauszug = _kuerzen(text, 700)
    return (
        "Schreibe eine kurze, formelle Antwort einer deutschen Versicherung in Sie-Form.\n"
        f"Kategorie: {kategorie}\n"
        f"Priorität: {prioritaet}\n"
        f"Routing: {route}\n"
        f"Kundenschreiben: {textauszug}\n"
        "Antwort:\nSehr geehrte Damen und Herren,\n\n"
    )


def _bereinige_generierten_text(generierter_text: str, prompt: str) -> str:
    antwort = generierter_text.replace(prompt, "").strip()
    antwort = re.split(r"\n\s*(Kundenschreiben|Kategorie|Antwort):", antwort)[0].strip()
    antwort = re.sub(r"\n{3,}", "\n\n", antwort)
    antwort = antwort[:900].strip()
    if not antwort.startswith("Sehr geehrte"):
        antwort = f"Sehr geehrte Damen und Herren,\n\n{antwort}"
    if "Mit freundlichen Grüßen" not in antwort:
        antwort += "\n\nMit freundlichen Grüßen\nIhr Inputmanagement-Team der BarmeniaGothaer"
    return antwort


def generiere_antwort(
    text: str,
    kategorie: str,
    prioritaet: str,
    route: str,
    sla: str,
    nutze_sprachmodell: bool = False,
) -> tuple[str, list[str]]:
    """Erzeugt einen formellen Antwortentwurf mit Template und optionalem GPT-2."""

    hinweise: list[str] = []
    standardantwort = _template_antwort(kategorie, prioritaet, route, sla)

    if not nutze_sprachmodell:
        return standardantwort, hinweise

    try:
        generator = _generierungs_pipeline()
        prompt = _baue_prompt(text, kategorie, prioritaet, route)
        output = generator(
            prompt,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=generator.tokenizer.eos_token_id,
        )
        generierter_text = output[0].get("generated_text", "")
        antwort = _bereinige_generierten_text(generierter_text, prompt)
        if len(antwort) < 180:
            hinweise.append("GPT-2-Ausgabe war zu kurz; Template-Antwort verwendet.")
            return standardantwort, hinweise
        hinweise.append(f"Antwort mit {GENERIERUNGS_MODELL} verfeinert.")
        return antwort, hinweise
    except Exception as exc:  # pragma: no cover - hängt von Modell-Download ab
        hinweise.append(f"GPT-2-Verfeinerung nicht verfügbar: {exc}")
        return standardantwort, hinweise


def analysiere_schreiben(text: str, nutze_sprachmodell: bool = False) -> dict[str, Any]:
    """Analysiert ein deutsches Eingangsschreiben Ende-zu-Ende."""

    bereinigter_text = text.strip()
    if not bereinigter_text:
        raise ValueError("Bitte geben Sie ein Eingangsschreiben ein.")

    kategorie, klassifikation_score, kategorien_scores, klassifikation_hinweise = _klassifiziere(
        bereinigter_text
    )
    stimmung, stimmung_score, stimmung_hinweise = _analysiere_stimmung(bereinigter_text)
    prioritaet, prioritaet_score, route, sla, gruende = _berechne_prioritaet(
        bereinigter_text, kategorie, stimmung
    )
    antwort, antwort_hinweise = generiere_antwort(
        bereinigter_text,
        kategorie,
        prioritaet,
        route,
        sla,
        nutze_sprachmodell=nutze_sprachmodell,
    )
    metadaten = _extrahiere_metadaten(bereinigter_text)

    ergebnis = AnalyseErgebnis(
        kategorie=kategorie,
        klassifikation_score=klassifikation_score,
        kategorien_scores=kategorien_scores,
        stimmung=stimmung,
        stimmung_score=stimmung_score,
        prioritaet=prioritaet,
        prioritaet_score=prioritaet_score,
        route=route,
        sla=sla,
        antwort=antwort,
        gruende=gruende,
        metadaten=metadaten,
        modellhinweise=klassifikation_hinweise + stimmung_hinweise + antwort_hinweise,
    )
    return asdict(ergebnis)
