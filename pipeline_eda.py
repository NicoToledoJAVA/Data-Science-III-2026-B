"""Pipeline de Preprocesamiento y Diagnostico (PPD) - AG News (Modulo 2)."""
import re, json, html
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spacy
from sklearn.feature_extraction.text import CountVectorizer

from cargar_dataset import cargar

OUT = Path(__file__).parent / "salidas"
OUT.mkdir(exist_ok=True)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
STOP = set(nlp.Defaults.stop_words)
# Palabras con carga semantica que NO se eliminan (negacion / polaridad)
KEEP = {"no", "not", "nor", "never", "n't", "against", "up", "down", "off", "over"}
STOP_FINAL = STOP - KEEP

# --- Limpieza Regex (Unidad 2) ---
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org)\b)")
RE_EMAIL = re.compile(r"\S+@\S+")
RE_AGENCY = re.compile(r"\((?:reuters|ap|afp|ap photo|washingtonpost\.com|sportsnetwork\.com|"
                       r"cp|canadian press|web host industry review)\)", re.I)
RE_ENTITY = re.compile(r"&[a-z]+;|#\d+;|\b(?:quot|amp|lt|gt|nbsp);", re.I)
RE_APOS = re.compile(r"\s?(?:&#39;|#39;)")          # "driver #39;s" -> "driver's"
RE_TICKER = re.compile(r"\([^()]*?:\s*quote,\s*profile,\s*research\)", re.I)
RE_DATELINE = re.compile(r"\b(?:AP|AFP|Reuters|SPACE\.com|Canadian Press|CP)\s+-\s+")
RE_NONALPHA = re.compile(r"[^a-z\s']")
RE_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = RE_APOS.sub("'", str(text))
    text = html.unescape(text)
    text = RE_HTML.sub(" ", text)
    text = RE_TICKER.sub(" ", text)
    text = RE_DATELINE.sub(" ", text)
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    text = RE_AGENCY.sub(" ", text)
    text = RE_ENTITY.sub(" ", text)
    text = text.lower()
    text = text.replace("\\", " ")
    text = RE_NONALPHA.sub(" ", text)      # numeros, puntuacion, simbolos
    return RE_SPACES.sub(" ", text).strip()


def _lemmas(doc, drop_stop=True):
    out = []
    for t in doc:
        lem = t.lemma_.strip("'").strip()
        if not lem or t.is_space or t.is_punct:
            continue
        if t.text == "n't":
            lem = "not"
        if drop_stop and (lem in STOP_FINAL or t.lower_ in STOP_FINAL):
            continue
        if len(lem) < 2 and lem not in KEEP:
            continue
        out.append(lem)
    return out


def preprocess_text(text: str) -> str:
    """Limpieza Regex + normalizacion + lematizacion SpaCy + stop-words (preservando negaciones)."""
    return " ".join(_lemmas(nlp(clean_text(text))))


def main():
    train_df, test_df = cargar("ag_news")
    stats = {"train_shape": train_df.shape, "test_shape": test_df.shape,
             "nulls": int(train_df.isna().sum().sum()),
             "dups": int(train_df.duplicated("text").sum())}
    # Ruido detectado en crudo
    raw = train_df["text"]
    stats["noise"] = {
        "html_tags": int(raw.str.contains(r"<[^>]+>").sum()),
        "html_entities": int(raw.str.contains(r"&[a-z]+;|#\d+;").sum()),
        "urls": int(raw.str.contains(r"https?://|www\.|\.com\b").sum()),
        "agency_tags": int(raw.str.contains(RE_AGENCY).sum()),
        "entities_sin_amp": int(raw.str.contains(r"#39;|\bquot;").sum()),
        "ticker_boilerplate": int(raw.str.contains(RE_TICKER).sum()),
        "datelines": int(raw.str.contains(RE_DATELINE).sum()),
    }

    cleaned = [clean_text(t) for t in train_df["text"]]
    proc, with_stop = [], []
    for doc in nlp.pipe(cleaned, batch_size=500):
        with_stop.append(_lemmas(doc, drop_stop=False))
        proc.append(" ".join(_lemmas(doc, drop_stop=True)))
    train_df["clean"] = proc
    train_df.to_csv(OUT / "ag_news_train_preprocesado.csv", index=False)
    # El test se procesa con la MISMA funcion (sin ajustar nada sobre el)
    test_df["clean"] = [preprocess_text(t) for t in test_df["text"]]
    test_df.to_csv(OUT / "ag_news_test_preprocesado.csv", index=False)

    # Longitudes
    raw_len = train_df["text"].str.split().str.len()
    proc_len = train_df["clean"].str.split().str.len()
    p95 = int(np.percentile(proc_len, 95))
    p95_raw = int(np.percentile(raw_len, 95))
    stats["len_raw"] = raw_len.describe().round(2).to_dict()
    stats["len_proc"] = proc_len.describe().round(2).to_dict()
    stats["p95_proc"], stats["p95_raw"] = p95, p95_raw
    stats["empty_docs"] = int((proc_len == 0).sum())

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(raw_len, bins=40, alpha=.45, label="Texto crudo (split por espacios)", color="#999")
    ax.hist(proc_len, bins=40, alpha=.8, label="Tras pipeline (lemas sin stop-words)", color="#1f77b4")
    ax.axvline(p95, color="red", ls="--", label=f"P95 procesado = {p95} tokens")
    ax.axvline(p95_raw, color="black", ls=":", label=f"P95 crudo = {p95_raw} tokens")
    ax.set_xlabel("Tokens por documento"); ax.set_ylabel("Cantidad de documentos")
    ax.set_title("Distribucion de longitud de tokens - AG News train (n=8.000)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "hist_longitud.png", dpi=150); plt.close(fig)

    # Longitud por clase
    fig, ax = plt.subplots(figsize=(9, 4))
    order = sorted(train_df["label"].unique())
    ax.boxplot([proc_len[train_df.label == c] for c in order], tick_labels=order, showfliers=False)
    ax.set_ylabel("Tokens (procesado)"); ax.set_title("Longitud por categoria")
    fig.tight_layout(); fig.savefig(OUT / "box_longitud_clase.png", dpi=150); plt.close(fig)
    stats["len_by_class"] = proc_len.groupby(train_df.label).median().to_dict()

    # Distribucion de clases (train y test)
    ct = train_df.label.value_counts().reindex(order)
    cte = test_df.label.value_counts().reindex(order)
    stats["classes_train"], stats["classes_test"] = ct.to_dict(), cte.to_dict()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(order))
    b1 = ax.bar(x - .2, ct.values, .4, label="train")
    b2 = ax.bar(x + .2, cte.values, .4, label="test")
    ax.bar_label(b1); ax.bar_label(b2)
    ax.set_xticks(x, order); ax.set_ylabel("Documentos")
    ax.set_title("Distribucion de clases - AG News"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "dist_clases.png", dpi=150); plt.close(fig)

    # N-gramas con scikit-learn
    def top_ngrams(n, k=20):
        cv = CountVectorizer(ngram_range=(n, n), token_pattern=r"(?u)\b[\w']+\b")
        X = cv.fit_transform(train_df["clean"])
        f = np.asarray(X.sum(0)).ravel()
        idx = f.argsort()[::-1][:k]
        voc = cv.get_feature_names_out()
        return [(voc[i], int(f[i])) for i in idx]
    stats["bigrams"], stats["trigrams"] = top_ngrams(2), top_ngrams(3)

    # Top 50 palabras: antes de filtrar stop-words y tras pipeline
    cnt_ws = Counter(w for d in with_stop for w in d)
    top50_ws = cnt_ws.most_common(50)
    stats["top50_with_stop"] = top50_ws
    stats["top50_with_stop_nstop"] = sum(w in STOP for w, _ in top50_ws)
    cnt = Counter(w for d in train_df["clean"] for w in d.split())
    top50 = cnt.most_common(50)
    stats["top50_final"] = top50
    stats["top50_final_nstop"] = sum(w in STOP for w, _ in top50)
    stats["top50_final_stop_list"] = [w for w, _ in top50 if w in STOP]
    stats["vocab_with_stop"], stats["vocab_final"] = len(cnt_ws), len(cnt)
    stats["hapax_final"] = sum(1 for v in cnt.values() if v == 1)
    stats["tokens_with_stop"] = sum(cnt_ws.values())
    stats["tokens_final"] = sum(cnt.values())
    stats["stop_share_tokens"] = round(1 - stats["tokens_final"] / stats["tokens_with_stop"], 4)

    # Zipf
    fr = np.array(sorted(cnt.values(), reverse=True))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(np.arange(1, len(fr) + 1), fr, ".", ms=2)
    ax.set_xlabel("Rango (log)"); ax.set_ylabel("Frecuencia (log)")
    ax.set_title("Ley de Zipf - vocabulario procesado")
    fig.tight_layout(); fig.savefig(OUT / "zipf.png", dpi=150); plt.close(fig)

    stats["example"] = [(train_df.text[i], train_df.clean[i]) for i in range(3)]
    json.dump(stats, open(OUT / "stats.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
