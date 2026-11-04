# Chaotic Text Generator
Auteur : Hafssa Miftah Idrissi

---
Banc d'essai de générateurs de texte, réalisé pour le Lab 03 du module NLP (M2 SDIA, ENSET Mohammedia).

L'application génère des phrases avec un modèle de Markov caché (HMM) ou un modèle trigramme, affiche la séquence d'étiquettes cachées derrière chaque phrase, mesure la qualité d'un lot avec quatre métriques, et étiquette n'importe quelle phrase saisie avec l'algorithme de Viterbi.

**Application publique :** https://chaotic-text-generator-hmm.streamlit.app/
**Avertissement :** le corpus est synthétique et la grammaire minuscule. Les métriques comme le patron d'étiquettes valide sont définies relativement à ce corpus et ne mesurent pas la qualité de l'anglais réel.

## Fonctionnalités

- Panneau de contrôle : modèle (HMM simple, HMM à états séparés par rôle, trigramme), température, graine aléatoire, nombre de phrases (10 à 500).
- Tableau des phrases générées : mots, états cachés, validité du patron, indicateur de recopie du corpus d'entraînement.
- Quatre métriques par lot : patron d'étiquettes valide, sujet-verbe plausible, nouveauté, distinct-2.
- Tagueur de Viterbi : étiquette sous chaque mot, avertissement visible pour les mots hors vocabulaire, sans plantage.
- Export CSV des phrases générées.
- Onglet de comparaison des trois modèles aux mêmes réglages, avec la courbe du patron valide en fonction de la température.
- Heat-map de la matrice de transitions.
- Onglet "Real text" : résultats sur le treebank réel Universal Dependencies English EWT.

## Principe

- **HMM** : les états cachés sont des étiquettes grammaticales, les observations sont des mots. Trois tables estimées par comptage : probabilités initiales, transitions P(étiquette | précédente) et émissions P(mot | étiquette). Pas de boucle d'entraînement.
- **Génération** : on tire d'abord une séquence d'étiquettes, puis un mot par étiquette. Les poids d'échantillonnage sont `count^(1/T)`.
- **Viterbi** : programmation dynamique en log-probabilités, avec lissage add-k (k = 0,01).
- **États séparés** : NOUN, ADJ et DET sont renommés selon leur rôle (S avant le verbe, O après, P dans le groupe prépositionnel).
- **Base de comparaison** : trigramme avec repli sur bigramme puis unigramme.

## Structure du projet

```
app.py                      application Streamlit
requirements.txt
.streamlit/config.toml      thème
data/corpus.csv             corpus synthétique (1 200 phrases, 6 étiquettes)
src/pipeline.py             HMM, Viterbi, trigramme, métriques
src/evaluate.py             évaluation sur le corpus synthétique
src/evaluate_real.py        évaluation sur English EWT
results/ewt_results.json    résultats réels lus par l'application
```

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pour recalculer les résultats :

```bash
python -m src.evaluate        # corpus synthétique, écrit dans results/
python -m src.evaluate_real   # télécharge English EWT dans data/ewt/, écrit results/ewt_results.json
```

## Résultats à T = 1,0 (500 phrases, graine 42)

| Modèle | Patron valide | Sujet-verbe plausible | Phrases nouvelles | Distinct-2 |
|---|---|---|---|---|
| HMM, états simples | 16,2 % | 14,8 % | 99,8 % | 0,198 |
| HMM, états séparés | 100 % | 60,6 % | 99,4 % | 0,148 |
| Trigramme | 57,0 % | 76,0 % | 91,2 % | 0,105 |

## Texte réel (English EWT, 12 544 phrases d'entraînement, 2 077 de test)

| Modèle | Précision | Mots connus | Mots inconnus | Phrase entière |
|---|---|---|---|---|
| Étiquette la plus fréquente | 86,2 % | 90,8 % | 28,6 % | 31,9 % |
| HMM + Viterbi | 88,1 % | 92,9 % | 28,2 % | 40,4 % |

Les mots inconnus représentent 7,5 % des mots de test. Le HMM ne gagne que 1,9 point sur la base, et la précision de 100 % obtenue sur le corpus synthétique ne se transpose pas au texte réel.

## Déploiement

Streamlit Community Cloud : pousser le dépôt sur GitHub avec `requirements.txt`, `data/corpus.csv` et `results/ewt_results.json`, puis connecter le dépôt sur share.streamlit.io et sélectionner `app.py`. Ne pas committer `data/ewt/`.

## Limites

- Données synthétiques générées par une petite grammaire écrite à la main.
- Test réel sur un seul treebank de texte web anglais, en minuscules. Un HMM d'ordre 1 n'est pas un tagueur de l'état de l'art.
- Un générateur peut reproduire ses données d'entraînement. Un texte généré est fluide mais sans sens ni intention, et ne doit pas être présenté comme écrit par une personne.

## Données

Universal Dependencies English EWT, licence CC BY-SA 4.0.
