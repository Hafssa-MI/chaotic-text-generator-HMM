"""Lab 03 - Chaotic Text Generator, test bench.   Run:  streamlit run app.py"""
import html
import json
import os
import random
import re

import altair as alt
import pandas as pd
import streamlit as st

from src.pipeline import HMM, NGramGenerator, coarse, generation_metrics, load_tagged, split_by_document

st.set_page_config(page_title="Chaotic Text Generator", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Hanken+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--paper:#F3F0E8;--card:#FBF9F4;--ink:#15171A;--muted:#6B6A64;--rule:#D9D4C7;--accent:#C2410C;}
.stApp{background:var(--paper);color:var(--ink);font-family:'Hanken Grotesk',sans-serif;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.4rem;max-width:1180px;}
h1,h2,h3{font-family:'Instrument Serif',serif!important;font-weight:400!important;letter-spacing:-0.01em;color:var(--ink);}
h3{font-size:1.7rem!important;margin-top:.4rem;}
p,li,label,span,div{font-family:'Hanken Grotesk',sans-serif;}
code,pre{font-family:'JetBrains Mono',monospace!important;}

/* masthead */
.eyebrow{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:.2rem;}
.masthead{font-family:'Instrument Serif',serif;font-size:4rem;line-height:1;letter-spacing:-0.02em;margin:.1rem 0 .7rem 0;}
.masthead em{color:var(--accent);font-style:italic;}
.lede{max-width:46rem;color:#3a3b3d;font-size:1.02rem;line-height:1.55;margin-bottom:1.1rem;}
.notice{border-left:3px solid var(--accent);background:var(--card);padding:.8rem 1rem;font-size:.9rem;line-height:1.5;margin:0 0 1.6rem 0;color:#2c2d30;}
.notice b{font-family:'JetBrains Mono',monospace;font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);display:block;margin-bottom:.15rem;}

/* sidebar */
[data-testid="stSidebar"]{background:#ECE8DD;border-right:1px solid var(--rule);}
.side-title{font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:.2rem 0 1rem 0;}

/* tabs */
[data-baseweb="tab-list"]{gap:1.6rem;border-bottom:1px solid var(--rule);}
button[data-baseweb="tab"]{padding:.5rem 0;background:transparent;}
button[data-baseweb="tab"] p{font-size:.95rem;font-weight:500;color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--ink);}
[data-baseweb="tab-highlight"]{background:var(--accent)!important;height:2px!important;}
[data-baseweb="tab-border"]{display:none;}

/* buttons */
.stButton>button,[data-testid="stDownloadButton"]>button{width:100%;border-radius:2px;font-weight:500;letter-spacing:.02em;padding:.6rem 1rem;}
.stButton>button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:var(--ink);color:var(--paper);border:1px solid var(--ink);}
.stButton>button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{background:var(--accent);border-color:var(--accent);color:#fff;}
[data-testid="stDownloadButton"]>button,button[data-testid="stBaseButton-secondary"]{background:transparent;color:var(--ink);border:1px solid var(--ink);}
[data-testid="stDownloadButton"]>button:hover,button[data-testid="stBaseButton-secondary"]:hover{background:var(--ink);color:var(--paper);}

/* metrics */
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--rule);border-radius:3px;padding:1rem 1.1rem;}
[data-testid="stMetricLabel"] p{font-family:'JetBrains Mono',monospace;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);}
[data-testid="stMetricValue"]{font-family:'Instrument Serif',serif;font-size:2.6rem;line-height:1.1;}

/* tagger */
.toks{display:flex;flex-wrap:wrap;gap:.5rem .35rem;margin:1rem 0 .5rem 0;}
.tok{display:flex;flex-direction:column;align-items:center;gap:.35rem;padding:.55rem .7rem .5rem .7rem;background:var(--card);border:1px solid var(--rule);border-radius:3px;min-width:3.2rem;}
.tok .w{font-family:'Instrument Serif',serif;font-size:1.55rem;line-height:1;}
.tok.unk{border:1px dashed var(--accent);}
.tok.unk .w{text-decoration:underline wavy var(--accent);text-underline-offset:5px;}
.tag{font-family:'JetBrains Mono',monospace;font-size:.66rem;letter-spacing:.08em;padding:.15rem .45rem;border-radius:2px;}
.tag-DET{background:#E7E1D0;color:#4A4533}.tag-NOUN{background:#DCE6DF;color:#2C4A39}
.tag-VERB{background:#F3D9CC;color:#7A2E0E}.tag-ADJ{background:#E4DCEB;color:#4B3760}
.tag-PREP{background:#D9E2EA;color:#294660}.tag-PUNCT{background:#E6E4DE;color:#55534C}
.empty{border:1px dashed var(--rule);padding:2.2rem;text-align:center;color:var(--muted);}
.small{font-size:.85rem;color:var(--muted);}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ models
MODELS = {"plain": "HMM, plain states", "split": "HMM, role-split states", "ngram": "Trigram"}
TEMPS_CURVE = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]


@st.cache_resource
def get_bench():
    docs = load_tagged()
    train, test = split_by_document(docs)
    return {
        "train": train, "test": test,
        "plain": HMM(train), "split": HMM(train, split_states=True),
        "ngram": NGramGenerator([w for w, _ in train]),
        "texts": {" ".join(w) for w, _ in train},
        "patterns": {" ".join(t) for _, t in train},
    }


B = get_bench()


@st.cache_data(show_spinner=False)
def generate(model, T, seed, n, prefix=()):
    rng = random.Random(int(seed))
    rows, samples = [], []
    for _ in range(n):
        if model == "ngram":
            words = B["ngram"].sample_sentence(rng, T, prefix=list(prefix))
            tags = B["split"].viterbi(words)          # tagger used for the trigram, as in evaluate.py
            states = tags
        else:
            words, states = B[model].sample_with_states(rng, T)
            tags = coarse(states)
        sentence = " ".join(words)
        valid = " ".join(tags) in B["patterns"]
        rows.append({"Sentence": sentence, "Hidden states": " ".join(states),
                     "Pattern": "Valid" if valid else "Invalid",
                     "Copied from training": "Yes" if sentence in B["texts"] else "No"})
        samples.append((words, tags))
    return pd.DataFrame(rows), generation_metrics(samples, B["train"], B["split"])


@st.cache_data(show_spinner=False)
def valid_curve(seed, n):
    out = []
    for key, label in MODELS.items():
        for T in TEMPS_CURVE:
            _, m = generate(key, T, seed, n)
            out.append({"Model": label, "Temperature": T, "Valid tag pattern": m["valid_tag_pattern"]})
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def synthetic_accuracy():
    out = {}
    for key in ("plain", "split"):
        c = t = 0
        for words, gold in B["test"]:
            pred = B[key].viterbi(words)
            c += sum(g == p for g, p in zip(gold, pred)); t += len(gold)
        out[key] = c / t
    return out


def tokenize(text):
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9']", text.lower())


def tag_html(tokens, tags, vocab):
    cells = []
    for w, t in zip(tokens, tags):
        unk = " unk" if w not in vocab else ""
        cells.append(f'<div class="tok{unk}"><span class="w">{html.escape(w)}</span>'
                     f'<span class="tag tag-{t}">{t}</span></div>')
    return f'<div class="toks">{"".join(cells)}</div>'


# ------------------------------------------------------------------ header
st.markdown('<div class="eyebrow">NLP module / Lab 03 / Test bench</div>'
            '<div class="masthead">Chaotic <em>Text</em> Generator</div>'
            '<div class="lede">Generate sentences with a hidden Markov model or a trigram model, inspect the hidden '
            'tag sequence behind each one, measure the batch with four metrics, and tag your own sentence with '
            'the Viterbi algorithm.</div>'
            '<div class="notice"><b>Read this first</b>The corpus is synthetic and the grammar is tiny. '
            'Metrics such as valid tag pattern are defined relative to this corpus and do not measure the '
            'quality of real English.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ control panel
with st.sidebar:
    st.markdown('<div class="side-title">Control panel</div>', unsafe_allow_html=True)
    model = st.radio("Model", list(MODELS), format_func=MODELS.get)
    T = st.slider("Temperature", 0.3, 3.0, 1.0, 0.1,
                  help="Sampling weights are count^(1/T). Low is safe and repetitive, high is varied and error-prone.")
    seed = st.number_input("Random seed", 0, 10_000, 42)
    n = st.slider("Number of sentences", 10, 500, 500, 10)
    prefix_text = ""
    if model == "ngram":
        prefix_text = st.text_input("Trigram prompt (optional)", "")
    prefix = tuple(prefix_text.lower().split())
    bad = [w for w in prefix if w not in B["ngram"].vocab]
    if bad:
        st.warning(f"Not in the training vocabulary: {', '.join(bad)}")
    go = st.button("Generate", type="primary", disabled=bool(bad))

if go:
    df, metrics = generate(model, float(T), int(seed), int(n), prefix)
    st.session_state["result"] = {"df": df, "metrics": metrics, "model": model, "T": T, "seed": seed, "n": n}

tab_gen, tab_tag, tab_cmp, tab_hm, tab_real = st.tabs(
    ["Generate", "Tagger", "Compare models", "Transitions", "Real text"])

# ------------------------------------------------------------------ tab: generate
with tab_gen:
    res = st.session_state.get("result")
    if not res:
        st.markdown('<div class="empty">Choose the settings in the control panel, then press Generate.</div>',
                    unsafe_allow_html=True)
    else:
        m, df = res["metrics"], res["df"]
        st.markdown("### Batch metrics")
        st.caption(f'{MODELS[res["model"]]}  |  T = {res["T"]}  |  seed {res["seed"]}  |  {res["n"]} sentences')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valid tag pattern", f'{m["valid_tag_pattern"]:.1%}',
                  help="Share of sentences whose tag sequence appears in the training set.")
        c2.metric("Plausible subject-verb", f'{m["plausible_subject_verb"]:.1%}',
                  help="Share of sentences whose first noun and first verb form a pair seen in training.")
        c3.metric("Novelty", f'{m["novel_sentences"]:.1%}',
                  help="Share of sentences not copied verbatim from the training set.")
        c4.metric("Distinct-2", f'{m["distinct_2"]:.3f}',
                  help="Distinct bigrams divided by total bigrams. Higher means more diverse.")
        copied = int((df["Copied from training"] == "Yes").sum())
        st.caption(f"{copied} of {len(df)} sentences are copied from the training set.")

        st.markdown("### Sentences")
        show = st.radio("Show", ["All", "Valid only", "Invalid only", "Copied only"], horizontal=True)
        view = df
        if show == "Valid only":
            view = df[df["Pattern"] == "Valid"]
        elif show == "Invalid only":
            view = df[df["Pattern"] == "Invalid"]
        elif show == "Copied only":
            view = df[df["Copied from training"] == "Yes"]
        view = view.copy()
        view.insert(0, "No.", view.index + 1)
        st.dataframe(view, hide_index=True, height=440,
                     column_config={"No.": st.column_config.NumberColumn(width="small"),
                                    "Sentence": st.column_config.TextColumn(width="large"),
                                    "Hidden states": st.column_config.TextColumn(width="large"),
                                    "Pattern": st.column_config.TextColumn(width="small"),
                                    "Copied from training": st.column_config.TextColumn(width="small")})
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name=f'generated_{res["model"]}_T{res["T"]}_seed{res["seed"]}.csv',
                           mime="text/csv")

# ------------------------------------------------------------------ tab: tagger
with tab_tag:
    st.markdown("### Viterbi tagger")
    st.markdown('<div class="small">Type a sentence. Each word receives its most probable part-of-speech tag, '
                'computed with log-probabilities and add-k smoothing (k = 0.01).</div>', unsafe_allow_html=True)
    tcol1, tcol2 = st.columns([3, 1])
    text = tcol1.text_input("Sentence", "the farmers watch the garden .")
    tagger_key = tcol2.selectbox("Tagger", ["plain", "split"], format_func=lambda k: MODELS[k])
    tokens = tokenize(text)
    if not tokens:
        st.info("Type a sentence to tag.")
    else:
        try:
            tagger = B[tagger_key]
            tags = tagger.viterbi(tokens)
            unknown = [w for w in tokens if w not in tagger.vocab]
            if unknown:
                st.warning(f"Not in the vocabulary: {', '.join(dict.fromkeys(unknown))}. "
                           "Their tags are a guess from the transitions only and are likely wrong.")
            st.markdown(tag_html(tokens, tags, tagger.vocab), unsafe_allow_html=True)
        except Exception as e:                       # never crash on user input
            st.error(f"Could not tag this sentence: {e}")
    with st.expander(f"Vocabulary ({len(B['plain'].vocab)} words)"):
        st.write(", ".join(sorted(B["plain"].vocab)))

# ------------------------------------------------------------------ tab: compare
with tab_cmp:
    st.markdown("### Three models, same settings")
    st.caption(f"T = {T}  |  seed {int(seed)}  |  {n} sentences per model")
    rows = []
    for key, label in MODELS.items():
        _, mm = generate(key, float(T), int(seed), int(n))
        rows.append({"Model": label, "Valid tag pattern": f'{mm["valid_tag_pattern"]:.1%}',
                     "Plausible subject-verb": f'{mm["plausible_subject_verb"]:.1%}',
                     "Novelty": f'{mm["novel_sentences"]:.1%}', "Distinct-2": f'{mm["distinct_2"]:.3f}'})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.markdown("### Valid tag pattern against temperature")
    if st.button("Compute curves"):
        st.session_state["curve"] = valid_curve(int(seed), min(int(n), 300))
    if "curve" in st.session_state:
        cur = st.session_state["curve"]
        colors = alt.Scale(domain=list(MODELS.values()), range=["#9A9488", "#C2410C", "#15171A"])
        chart = (alt.Chart(cur).mark_line(point=True, strokeWidth=2)
                 .encode(x=alt.X("Temperature:Q"), y=alt.Y("Valid tag pattern:Q", axis=alt.Axis(format="%"),
                                                           scale=alt.Scale(domain=[0, 1])),
                         color=alt.Color("Model:N", scale=colors, legend=alt.Legend(orient="bottom", title=None)))
                 .properties(height=330))
        st.altair_chart(chart)
        st.caption("Curves use at most 300 sentences per point.")

# ------------------------------------------------------------------ tab: transitions
with tab_hm:
    st.markdown("### Learned transitions")
    which = st.radio("States", ["plain", "split"], format_func=lambda k: MODELS[k], horizontal=True)
    tags_, mat = B[which].transition_matrix()
    long = pd.DataFrame([{"current": a, "next": b, "p": mat[i][j]}
                         for i, a in enumerate(tags_) for j, b in enumerate(tags_)])
    base = alt.Chart(long).encode(x=alt.X("next:N", sort=tags_, title="next state"),
                                  y=alt.Y("current:N", sort=tags_, title="current state"))
    rect = base.mark_rect(stroke="#F3F0E8", strokeWidth=2).encode(
        color=alt.Color("p:Q", scale=alt.Scale(domain=[0, 1], range=["#EDE8DA", "#C2410C"]), legend=None))
    txt = (base.mark_text(font="JetBrains Mono", fontSize=11)
           .transform_filter(alt.datum.p > 0.004)
           .encode(text=alt.Text("p:Q", format=".2f"),
                   color=alt.condition(alt.datum.p > 0.55, alt.value("white"), alt.value("#15171A"))))
    st.altair_chart((rect + txt).properties(height=max(300, 34 * len(tags_))))
    st.caption("With plain states, a NOUN is followed by VERB 0.39, PREP 0.22 or PUNCT 0.39 whatever its role. "
               "Switch to role-split states to see the memory that splitting adds.")

# ------------------------------------------------------------------ tab: real text
with tab_real:
    st.markdown("### Synthetic against real text")
    path = "results/ewt_results.json"
    if not os.path.exists(path):
        st.info("Run  python -m src.evaluate_real  and commit results/ewt_results.json to fill this tab.")
    else:
        data = json.load(open(path, encoding="utf-8"))
        rows = []
        for name, r in data["models"].items():
            rows.append({"Corpus": "English EWT (real)", "Model": name, "Accuracy": f'{r["accuracy"]:.1%}',
                         "Known words": f'{r["known_words"]:.1%}', "Unknown words": f'{r["unknown_words"]:.1%}',
                         "Whole sentence": f'{r["whole_sentence"]:.1%}'})
        syn = synthetic_accuracy()
        rows.append({"Corpus": "Synthetic", "Model": "HMM + Viterbi", "Accuracy": f'{syn["plain"]:.1%}',
                     "Known words": "n/a", "Unknown words": "n/a", "Whole sentence": "n/a"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.markdown("### Most frequent HMM errors on real text")
        st.dataframe(pd.DataFrame(data["confusions"][:8], columns=["Gold", "Predicted", "Count"]),
                     hide_index=True)