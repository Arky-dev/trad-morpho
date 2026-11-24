# Traduction Automatisée via la Morphologie de la Langue (`MorphAlign`)

## Présentation du projet

Ce projet explore une approche expérimentale de **traduction automatique neuronale (NMT)** reposant sur l'intégration explicite de la morphologie de la langue source (ici le français vers l'anglais). 
Ce projet a été réalisé dans le cadre du séminaire de linguistique de 2e année du cycle inégnieur de l'Ecole polytechnique.

Dans les architectures neuronales classiques, les informations morphologiques sont souvent implicites et diluées dans les plongements lexicaux. L'hypothèse centrale de ce travail est qu'une **représentation morphologique explicite** permet d'améliorer les correspondances syntaxiques et sémantiques entre les langues, tout en offrant une meilleure interprétabilité du processus de traduction.

---

## Architecture et Pipeline

Le traitement des données et l'entraînement s'articulent autour d'un pipeline en 5 étapes :

1. **Acquisition du corpus parallèle** : Extraction de paires français-anglais depuis *OpenSubtitles*.
2. **Nettoyage et segmentation lexicale** : Normalisation et tokénisation via `sacremoses`.
3. **Annotation morphologique** : Lemmatisation et étiquetage grammatical via la base *Morphalou 3.1*.
4. **Alignement des tokens** : Mise en correspondance mot à mot / morphème à morphème avec *SimAlign* (`mBERT`).
5. **Entraînement du modèle NMT** : Réseau de neurones séquentiel alimenté par des tenseurs fusionnant le texte et ses traits morphologiques.

---

## Pistes d'amélioration et perspectives

* **Qualité du corpus :** Remplacer *OpenSubtitles* (souvent bruité par des ellipses, traductions libres et découpages asymétriques de répliques) par des corpus plus normés tels que **Europarl** ou **Tatoeba**.
* **Passage à l'échelle :** Entraîner le modèle sur l'intégralité du corpus avec accélération GPU et un plus grand nombre d'époques.
* **Régularisation :** Introduire un échantillonnage aléatoire de sous-séquences ou de tokens isolés lors de l'entraînement pour éviter le surapprentissage de structures figées.

---

## Références

* **Lison, P., & Tiedemann, J. (2016).** *OpenSubtitles2016: Extracting Large Parallel Corpora from Movie and TV Subtitles*. LREC 2016, pp. 923–929.
* **Sabet, M. J., Dufter, P., & Schütze, H. (2020).** *SimAlign: High Quality Word Alignments without Parallel Training Data using Static and Contextualized Embeddings*. arXiv:2004.08728.
* **Saunshi, N., Gupta, A., & Hu, W. (2021).** *A Representation Learning Perspective on the Importance of Train-Validation Splitting in Meta-Learning*. arXiv:2106.15615.
* **Tiedemann, J. (2009).** *News from OPUS - A Collection of Multilingual Parallel Corpora with Tools and Interfaces*. RANLP, Vol. V, pp. 237–248.# MorphAlign
Projet expérimental de traduction automatique explorant le rôle de la morphologie dans l’alignement et la génération de texte en français et en anglais.
L’objectif est de construire un petit modèle de traduction neuronale à partir d’un corpus parallèle annoté morphologiquement.
