"""Genera el informe EDA en Word (.docx) a partir de salidas/stats.json y las figuras."""
import json
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

AUTOR = "TOLEDO, Alejandro Nicolás"          # <-- completar
ARCHIVO = "EDA_NLP_ToledoNicolas.docx"

BASE = Path(__file__).parent
S = json.load(open(BASE / "salidas" / "stats.json", encoding="utf-8"))
n = S["noise"]; lp = S["len_proc"]; lr = S["len_raw"]

doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2); s.top_margin = s.bottom_margin = Cm(1.8)
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)


def para(text, italic=False, size=None, justify=True):
    """Texto con **negrita** simple."""
    p = doc.add_paragraph()
    for i, chunk in enumerate(text.split("**")):
        r = p.add_run(chunk); r.bold = i % 2 == 1; r.italic = italic
        if size: r.font.size = Pt(size)
    if justify: p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def table(rows, widths, fs=9):
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.width = Cm(widths[j]); c.text = ""
            r = c.paragraphs[0].add_run(str(val)); r.font.size = Pt(fs)
            if i == 0:
                r.bold = True; r.font.color.rgb = RGBColor(255, 255, 255); shade(c, "1F4E79")
            elif i % 2 == 0:
                shade(c, "EEF3F8")
    doc.add_paragraph()
    return t


def img(name, w=16):
    doc.add_picture(str(BASE / "salidas" / name), width=Cm(w))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


doc.add_heading("Informe de EDA Técnico – Corpus AG News", 0)
para(f"Pre-entrega Módulo 2 · Data Science III: NLP & Deep Learning · {AUTOR}", italic=True, justify=False)

doc.add_heading("0. Dataset y splits", 1)
para(f"Se utiliza el dataset provisto **AG News** (4 clases: World, Sports, Business, Sci_Tech), cargado con "
     f"cargar_dataset.py. Train: **{S['train_shape'][0]:,}** documentos; test: **{S['test_shape'][0]:,}**. "
     f"Sin nulos ({S['nulls']}) ni duplicados ({S['dups']}). Todo el EDA (frecuencias, N-gramas, percentiles) se "
     "calcula **exclusivamente sobre train**; al test sólo se le aplica la misma función preprocess_text, sin ajustar "
     "nada sobre él, para evitar data leakage.")

doc.add_heading("1. Pasos de preprocesamiento aplicados", 1)
para("Función **preprocess_text(text)** = limpieza Regex → normalización → lematización SpaCy (en_core_web_sm, "
     "parser y NER desactivados) → filtrado de stop-words con lista de excepciones. Estrategia de tokenización: "
     "**word-level** (SpaCy), coherente con el baseline TF-IDF del Módulo 3; para el Módulo 4 el texto se "
     "re-tokenizará con el tokenizer subword de DistilBERT.")
steps = [["#", "Paso", "Detalle / Ruido detectado en train crudo"],
         ["1", "Entidades HTML rotas", f"'#39;' → apóstrofo y 'quot;' / '&amp;' eliminados ({n['entities_sin_amp']:,} docs con entidades sin '&')"],
         ["2", "Tags HTML", f"Regex <[^>]+> (control de calidad: {n['html_tags']} docs con tags)"],
         ["3", "Boilerplate de tickers", f"'(XXX.N: Quote, Profile, Research)' de Reuters ({n['ticker_boilerplate']} docs)"],
         ["4", "Agencias / datelines", f"'(AP)', '(Reuters)', 'AP - ', 'AFP - '... ({n['agency_tags']:,} con tag, {n['datelines']} con dateline)"],
         ["5", "URLs y e-mails", f"http, www, dominios .com/.net/.org ({n['urls']} docs)"],
         ["6", "Normalización", "lowercase; remoción de números, puntuación y símbolos; colapso de espacios"],
         ["7", "Lematización", "SpaCy: 'announced'→'announce', 'prices'→'price', \"n't\"→'not'"],
         ["8", "Stop-words", "Lista SpaCy (326) MENOS {no, not, nor, never, against, up, down, off, over}: "
                             "negación y direccionalidad tienen carga semántica (ej. 'shares up', 'win over')"],
         ["9", "Tokens residuales", "se descartan tokens de 1 carácter"]]
table(steps, [0.8, 3.8, 11.9])
ex = S["example"][0]
para(f"**Ejemplo.** Crudo: {ex[0]}", size=9)
para(f"Procesado: {ex[1]}", size=9)
para(f"**Impacto global:** los tokens pasan de {S['tokens_with_stop']:,} (lematizados, con stop-words) a "
     f"{S['tokens_final']:,} (−{S['stop_share_tokens']*100:.1f}%). Vocabulario final: {S['vocab_final']:,} lemas, "
     f"de los cuales {S['hapax_final']:,} ({S['hapax_final']/S['vocab_final']*100:.0f}%) son hapax (frecuencia 1).")

doc.add_page_break()
doc.add_heading("2. Distribución de longitud de tokens", 1)
img("hist_longitud.png")
para(f"Texto crudo: media {lr['mean']} palabras, mediana {lr['50%']:.0f}, máx. {lr['max']:.0f}, P95 = **{S['p95_raw']}**. "
     f"Tras el pipeline: media {lp['mean']}, mediana {lp['50%']:.0f}, rango {lp['min']:.0f}–{lp['max']:.0f}, "
     f"**P95 = {S['p95_proc']} tokens**, sin documentos vacíos. La distribución es unimodal, compacta y con cola "
     "derecha corta: el 95% de las noticias cabe en una secuencia muy corta. Para modelos word-level se fija "
     f"**max_len = {S['p95_proc']}**; para DistilBERT (subwords, con stop-words) conviene partir de ~{S['p95_raw']} "
     "palabras → max_length ≈ 64–80 subtokens, lo que trunca sólo ~5% de documentos y minimiza padding.")
img("box_longitud_clase.png", 13)
para("Medianas por clase: " + ", ".join(f"{k} {v:.0f}" for k, v in S["len_by_class"].items()) +
     ". La longitud no discrimina clases, por lo que el modelo deberá apoyarse en el contenido léxico.")

doc.add_page_break()
doc.add_heading("3. Top 20 bi-gramas y tri-gramas (CountVectorizer, scikit-learn)", 1)
rows = [["#", "Bi-grama", "Frec.", "Tri-grama", "Frec."]]
for i, (b, t) in enumerate(zip(S["bigrams"], S["trigrams"]), 1):
    rows.append([i, b[0], b[1], t[0], t[1]])
table(rows, [0.8, 4.5, 1.5, 6.0, 1.5])
para("**Interpretación.** (a) **Calidad de limpieza:** no aparecen tags HTML, entidades ('quot', '39'), nombres de "
     "agencias ni el boilerplate 'quote profile research', que sí dominaban una primera versión del pipeline. "
     "(b) **Entidades multi-palabra:** la mayoría son nombres propios (new york, los angeles, osama bin laden, "
     "international space station) que una representación unigram fragmenta; justifica usar ngram_range=(1,2) en "
     "TF-IDF. (c) **Señal de clase clara:** World (prime minister ariel sharon, colin powell, vladimir putin, european "
     "union, kill people), Sports (red sox, new york yankees, world cup, victory/win over), Business (oil price, wall "
     "street, security exchange commission, initial public offering, new york stock) y Sci_Tech (mobile phone, "
     "microsoft corp, international space station). (d) **Valor de las excepciones de stop-words:** 'victory over' y "
     "'win over' sobreviven sólo porque 'over' se preservó, y son marcadores típicos de Sports. (e) Corpus fechado "
     "(2004): bush, sharon, powell; anticipa sesgo temporal.")

top = S["top50_final"]; tws = S["top50_with_stop"]
doc.add_page_break()
doc.add_heading("4. Top 50 palabras y control de stop-words", 1)
para(f"Antes del filtrado, **{S['top50_with_stop_nstop']} de las 50** palabras más frecuentes son stop-words "
     f"(the {tws[0][1]:,}, to, of, in, be...). Tras el pipeline quedan **{S['top50_final_nstop']} de 50**: "
     f"{', '.join(S['top50_final_stop_list'])} — exactamente las excepciones conservadas a propósito por su carga "
     "semántica. El resto son términos de contenido.")
r = [["#", "Palabra", "Frec."] * 3]
for i in range(17):
    row = []
    for j in range(3):
        k = i + 17 * j
        row += [k + 1, top[k][0], top[k][1]] if k < 50 else ["", "", ""]
    r.append(row)
table(r, [0.8, 2.8, 1.5] * 3)
para("Se observan días de la semana (monday…friday), 'yesterday', 'year', 'report': vocabulario periodístico "
     "transversal a las 4 clases y de bajo poder discriminativo. No se eliminan manualmente: TF-IDF (IDF bajo) los "
     "penalizará y en el Transformer aportan contexto. El gráfico log-log confirma la Ley de Zipf: pocos términos "
     "concentran la masa y una cola larga de hapax.")
img("zipf.png", 9)

doc.add_page_break()
doc.add_heading("5. Distribución de clases", 1)
img("dist_clases.png")
para("Train: 2.000 documentos por clase; test: 500 por clase. El corpus está **perfectamente balanceado**: accuracy "
     "y F1-macro serán equivalentes, no se requieren class_weight ni re-muestreo, y un clasificador trivial obtiene "
     "25% (piso de referencia).")

doc.add_heading("6. Conclusión sobre la preparación del dato", 1)
para("**Desafíos en el texto.** El corpus no está tan limpio como aparenta: el EDA reveló entidades HTML rotas sin "
     "'&' ('#39;', 'quot;') en ~27% de las noticias, firmas de agencia ('(AP)', 'Reuters - ') y boilerplate de "
     "tickers ('Quote, Profile, Research'). En una primera iteración, 'quot', 'ap' y 'quote profile research' "
     "aparecían entre los términos y N-gramas más frecuentes; un modelo podría haber aprendido la agencia en lugar "
     "del tema. El pipeline los elimina con Regex específicas antes de normalizar, y los N-gramas finales están "
     "libres de ruido. El segundo desafío es la cola larga: 35% de los tokens son stop-words y 40% del vocabulario "
     "son hapax. Se filtran stop-words, pero preservando negaciones y partículas direccionales ('not', 'no', 'up', "
     "'over', 'against'), que aportan señal ('win over' en Sports, 'shares up' en Business). La lematización reduce "
     "la dispersión morfológica; los hapax restantes se controlarán con min_df en TF-IDF y con subwords en el "
     "Transformer.")
para("**Longitud.** Documentos cortos y homogéneos (P95 = 32 tokens procesados, 53 palabras crudas): max_len=32 para "
     "modelos word-level y ≈64–80 subtokens para DistilBERT, con truncado mínimo y poco padding.")
para("**Clases.** Balance perfecto (2.000/500 por clase en train/test): no hace falta reponderar y accuracy y "
     "F1-macro son comparables. El reto real es semántico: solapamiento Business–Sci_Tech (Microsoft, Google, "
     "acciones tecnológicas), que los bi-gramas ayudan a desambiguar.")
para("**Conclusión.** El corpus queda listo: sin nulos ni duplicados, sin ruido en los N-gramas, con max_len "
     "justificado y splits respetados (EDA sólo sobre train). Se exporta la columna 'clean' para TF-IDF (Módulo 3), "
     "mientras que el texto original con limpieza mínima se reserva para el Transformer (Módulo 4), que se beneficia "
     "de stop-words y mayúsculas para el contexto.")

doc.save(BASE / ARCHIVO)
print("OK:", ARCHIVO)
