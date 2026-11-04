"""Genera el informe EDA en PDF a partir de salidas/stats.json y las figuras."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, PageBreak)

AUTOR = "TOLEDO, Alejandro Nicolás"          # <-- completar
ARCHIVO = "EDA_NLP_TOLEDO_NICOLAS.pdf"

BASE = Path(__file__).parent
S = json.load(open(BASE / "salidas" / "stats.json", encoding="utf-8"))
st = getSampleStyleSheet()
body = ParagraphStyle("b", parent=st["BodyText"], fontSize=9.5, leading=13, alignment=4)
small = ParagraphStyle("s", parent=body, fontSize=8, leading=10, alignment=0)
H1, H2 = st["Heading1"], st["Heading2"]


def tbl(data, widths, fs=8):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("GRID", (0, 0), (-1, -1), .3, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return t


def img(name, w=16):
    from PIL import Image as PI
    p = BASE / "salidas" / name
    iw, ih = PI.open(p).size
    return Image(str(p), width=w * cm, height=w * cm * ih / iw)


n = S["noise"]; lp = S["len_proc"]; lr = S["len_raw"]
E = []
E += [Paragraph("Informe de EDA Técnico – Corpus AG News", st["Title"]),
      Paragraph(f"Pre-entrega Módulo 2 · Data Science III: NLP &amp; Deep Learning · {AUTOR}", st["Normal"]),
      Spacer(1, 10)]

E += [Paragraph("0. Dataset y splits", H2), Paragraph(
    f"Se utiliza el dataset provisto <b>AG News</b> (4 clases: World, Sports, Business, Sci_Tech), cargado con "
    f"<i>cargar_dataset.py</i>. Train: <b>{S['train_shape'][0]:,}</b> documentos; test: <b>{S['test_shape'][0]:,}</b>. "
    f"Sin nulos ({S['nulls']}) ni duplicados ({S['dups']}). Todo el EDA (frecuencias, N-gramas, percentiles) se calcula "
    "<b>exclusivamente sobre train</b>; al test sólo se le aplica la misma función <i>preprocess_text</i>, sin ajustar "
    "nada sobre él, para evitar data leakage.", body)]

E += [Paragraph("1. Pasos de preprocesamiento aplicados", H2), Paragraph(
    "Función <b>preprocess_text(text)</b> = limpieza Regex → normalización → lematización SpaCy "
    "(<i>en_core_web_sm</i>, parser y NER desactivados) → filtrado de stop-words con lista de excepciones. "
    "Estrategia de tokenización: <b>word-level</b> (SpaCy), coherente con el baseline TF-IDF del Módulo 3; "
    "para el Módulo 4 el texto se re-tokenizará con el tokenizer subword de DistilBERT.", body), Spacer(1, 4)]
steps = [["#", "Paso", "Detalle / Ruido detectado en train crudo"],
         ["1", "Entidades HTML rotas", f"'#39;' → apóstrofo y 'quot;' / '&amp;amp;' eliminados ({n['entities_sin_amp']:,} docs con entidades sin '&amp;')"],
         ["2", "Tags HTML", f"Regex &lt;[^&gt;]+&gt; (control de calidad: {n['html_tags']} docs con tags)"],
         ["3", "Boilerplate de tickers", f"'(XXX.N: Quote, Profile, Research)' de Reuters ({n['ticker_boilerplate']} docs)"],
         ["4", "Agencias / datelines", f"'(AP)', '(Reuters)', 'AP - ', 'AFP - '... ({n['agency_tags']:,} con tag, {n['datelines']} con dateline)"],
         ["5", "URLs y e-mails", f"http, www, dominios .com/.net/.org ({n['urls']} docs)"],
         ["6", "Normalización", "lowercase; remoción de números, puntuación y símbolos; colapso de espacios"],
         ["7", "Lematización", "SpaCy: 'announced'→'announce', 'prices'→'price', \"n't\"→'not'"],
         ["8", "Stop-words", "Lista SpaCy (326) MENOS {no, not, nor, never, against, up, down, off, over}: "
                             "negación y direccionalidad tienen carga semántica (ej. 'shares up', 'win over')"],
         ["9", "Tokens residuales", "se descartan tokens de 1 carácter"]]
E += [tbl([[Paragraph(c, small) for c in r] for r in steps], [0.7*cm, 3.6*cm, 12.2*cm])]
ex = S["example"][0]
E += [Spacer(1, 6), Paragraph(f"<b>Ejemplo.</b> Crudo: <i>{ex[0]}</i><br/>Procesado: <font face='Courier'>{ex[1]}</font>", small)]
E += [Spacer(1, 4), Paragraph(
    f"<b>Impacto global:</b> los tokens pasan de {S['tokens_with_stop']:,} (lematizados, con stop-words) a "
    f"{S['tokens_final']:,} (−{S['stop_share_tokens']*100:.1f}%). Vocabulario final: {S['vocab_final']:,} lemas, "
    f"de los cuales {S['hapax_final']:,} ({S['hapax_final']/S['vocab_final']*100:.0f}%) son hapax (frecuencia 1).", body)]

E += [PageBreak(), Paragraph("2. Distribución de longitud de tokens", H2), img("hist_longitud.png"), Paragraph(
    f"Texto crudo: media {lr['mean']} palabras, mediana {lr['50%']:.0f}, máx. {lr['max']:.0f}, P95 = <b>{S['p95_raw']}</b>. "
    f"Tras el pipeline: media {lp['mean']}, mediana {lp['50%']:.0f}, rango {lp['min']:.0f}–{lp['max']:.0f}, "
    f"<b>P95 = {S['p95_proc']} tokens</b>, sin documentos vacíos. La distribución es unimodal, compacta y con cola "
    "derecha corta: el 95% de las noticias cabe en una secuencia muy corta. Para modelos word-level se fija "
    f"<b>max_len = {S['p95_proc']}</b>; para DistilBERT (subwords, con stop-words) conviene partir de ~{S['p95_raw']} palabras "
    "→ max_length ≈ 64–80 subtokens, lo que trunca sólo ~5% de documentos y minimiza padding.", body),
    Spacer(1, 6), img("box_longitud_clase.png", 13), Paragraph(
    "Medianas por clase: " + ", ".join(f"{k} {v:.0f}" for k, v in S["len_by_class"].items()) +
    ". La longitud no discrimina clases, por lo que el modelo deberá apoyarse en el contenido léxico.", body)]

E += [PageBreak(), Paragraph("3. Top 20 bi-gramas y tri-gramas (CountVectorizer, scikit-learn)", H2)]
rows = [["#", "Bi-grama", "Frec.", "Tri-grama", "Frec."]]
for i, (b, t) in enumerate(zip(S["bigrams"], S["trigrams"]), 1):
    rows.append([i, b[0], b[1], t[0], t[1]])
E += [tbl(rows, [0.8*cm, 4.5*cm, 1.4*cm, 6*cm, 1.4*cm], 8.5), Spacer(1, 6), Paragraph(
    "<b>Interpretación.</b> (a) <b>Calidad de limpieza:</b> no aparecen tags HTML, entidades ('quot', '39'), "
    "nombres de agencias ni el boilerplate 'quote profile research', que sí dominaban una primera versión del "
    "pipeline (ver §5). (b) <b>Entidades multi-palabra:</b> la mayoría son nombres propios (new york, los angeles, "
    "osama bin laden, international space station) que una representación unigram fragmenta; justifica usar "
    "ngram_range=(1,2) en TF-IDF. (c) <b>Señal de clase clara:</b> World (prime minister ariel sharon, colin powell, "
    "vladimir putin, european union, kill people), Sports (red sox, new york yankees, world cup, victory/win over), "
    "Business (oil price, wall street, security exchange commission, initial public offering, new york stock) y "
    "Sci_Tech (mobile phone, microsoft corp, international space station). (d) <b>Valor de las excepciones de "
    "stop-words:</b> 'victory over' y 'win over' sobreviven sólo porque 'over' se preservó, y son marcadores "
    "típicos de Sports. (e) Corpus fechado (2004): bush, sharon, powell; anticipa sesgo temporal.", body)]

top = S["top50_final"]; tws = S["top50_with_stop"]
E += [PageBreak(), Paragraph("4. Top 50 palabras y control de stop-words", H2), Paragraph(
    f"Antes del filtrado, <b>{S['top50_with_stop_nstop']} de las 50</b> palabras más frecuentes son stop-words "
    f"(the {tws[0][1]:,}, to, of, in, be...). Tras el pipeline quedan <b>{S['top50_final_nstop']} de 50</b>: "
    f"{', '.join(S['top50_final_stop_list'])} — exactamente las excepciones conservadas a propósito por su carga "
    "semántica. El resto son términos de contenido.", body), Spacer(1, 4)]
r = [["#", "Palabra", "Frec."] * 3]
for i in range(17):
    row = []
    for j in range(3):
        k = i + 17 * j
        row += [k + 1, top[k][0], top[k][1]] if k < 50 else ["", "", ""]
    r.append(row)
E += [tbl(r, [0.8*cm, 2.7*cm, 1.4*cm] * 3, 8), Spacer(1, 6), Paragraph(
    "Se observan días de la semana (monday…friday), 'yesterday', 'year', 'report': vocabulario periodístico "
    "transversal a las 4 clases y de bajo poder discriminativo. No se eliminan manualmente: TF-IDF (IDF bajo) los "
    "penalizará y en el Transformer aportan contexto. El gráfico log-log confirma la Ley de Zipf: pocos términos "
    "concentran la masa y una cola larga de hapax.", body), img("zipf.png", 9)]

E += [PageBreak(), Paragraph("5. Distribución de clases", H2), img("dist_clases.png"), Paragraph(
    "Train: 2.000 documentos por clase; test: 500 por clase. El corpus está <b>perfectamente balanceado</b>: "
    "accuracy y F1-macro serán equivalentes, no se requieren class_weight ni re-muestreo, y un clasificador "
    "trivial obtiene 25% (piso de referencia).", body)]

concl = (
    "<b>Desafíos en el texto.</b> El corpus no está tan limpio como aparenta: el EDA reveló entidades HTML "
    "rotas sin '&amp;' ('#39;', 'quot;') en ~27% de las noticias, firmas de agencia ('(AP)', 'Reuters - ') y "
    "boilerplate de tickers ('Quote, Profile, Research'). En una primera iteración, 'quot', 'ap' y 'quote profile "
    "research' aparecían entre los términos y N-gramas más frecuentes; un modelo podría haber aprendido la "
    "agencia en lugar del tema. El pipeline los elimina con Regex específicas antes de normalizar, y los N-gramas "
    "finales están libres de ruido. "
    "El segundo desafío es la cola larga: 35% de los tokens son stop-words y 40% del vocabulario son hapax. Se "
    "filtran stop-words, pero preservando negaciones y partículas direccionales ('not', 'no', 'up', 'over', "
    "'against'), que aportan señal ('win over' en Sports, 'shares up' en Business). La lematización reduce la "
    "dispersión morfológica; los hapax restantes se controlarán con min_df en TF-IDF y con subwords en el Transformer. "
    "<b>Longitud.</b> Documentos cortos y homogéneos (P95 = 32 tokens procesados, 53 palabras crudas): max_len=32 "
    "para modelos word-level y ≈64–80 subtokens para DistilBERT, con truncado mínimo y poco padding. "
    "<b>Clases.</b> Balance perfecto (2.000/500 por clase en train/test): no hace falta reponderar y "
    "accuracy y F1-macro son comparables. El reto real es semántico: solapamiento Business–Sci_Tech "
    "(Microsoft, Google, acciones tecnológicas), que los bi-gramas ayudan a desambiguar. "
    "<b>Conclusión.</b> El corpus queda listo: sin nulos ni duplicados, sin ruido en los N-gramas, con max_len "
    "justificado y splits respetados (EDA sólo sobre train). Se exporta la columna 'clean' para TF-IDF (Módulo 3), "
    "mientras que el texto original con limpieza mínima se reserva para el Transformer (Módulo 4), que se "
    "beneficia de stop-words y mayúsculas para el contexto.")
E += [Spacer(1, 10), Paragraph("6. Conclusión sobre la preparación del dato", H2), Paragraph(concl, body)]

SimpleDocTemplate(str(BASE / ARCHIVO), pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                  topMargin=1.8*cm, bottomMargin=1.8*cm, title="EDA NLP - AG News").build(E)
print("words conclusion:", len(Paragraph(concl, body).getPlainText().split()))
