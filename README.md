# Word2Vec Text Summarization — Step-by-Step

This repository contains a minimal, readable Jupyter notebook that performs **extractive text summarization** using **Word2Vec**. It trains a tiny Word2Vec model on your input text, scores sentences by similarity, and returns the top-K sentences as the summary.

> File: `Word2vec_text_summarization.ipynb`

---

## What you’ll build

1. Install dependencies.
2. Import libraries and download NLTK resources.
3. Define a reusable `summarize_text_word2vec(text, num_sentences=3)` function.
4. Provide an input paragraph or article.
5. Generate a summary (top-K sentences).
6. Print both the original and the summary with word counts.

---

## Requirements

- Python 3.8+
- Jupyter Notebook / Google Colab

### Python packages
```bash
pip install gensim nltk numpy scikit-learn colorama
```

> If you’re using Google Colab, the notebook already contains `!pip install ...` cells.

---

## Quick Start (Notebook)

1. **Open** `Word2vec_text_summarization.ipynb` in Jupyter or Colab.
2. **Run the installation cell** (skippable if your environment already has the deps).
3. **Run the imports & downloads cell** – this pulls NLTK tokenizers/stopwords:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   # If prompted by newer NLTK versions:
   nltk.download('punkt_tab')
   ```
4. **Run the function cell** to define the summarizer.
5. **Edit the sample `text`** (a multi-sentence paragraph/article).
6. **Execute** the summary cell to see:
   - Original text
   - Summary (top 3 sentences by default)
   - Word counts for both

---

## How it works (inside the function)

1. **Sentence tokenization**  
   Split the input into sentences with `nltk.sent_tokenize`.

2. **Word tokenization & preprocessing**  
   - Lowercase text and tokenize with `nltk.word_tokenize`  
   - Keep alphanumeric tokens only (`.isalnum()`)  
   - Remove English stopwords (`nltk.corpus.stopwords`)

3. **Train Word2Vec (on your text)**  
   - `gensim.models.Word2Vec` with:
     - `vector_size=300`, `window=5`, `min_count=1`, `workers=4`

4. **Sentence embeddings**  
   - For each sentence, average the vectors of its words present in the model.

5. **Sentence similarity matrix**  
   - Use `sklearn.metrics.pairwise.cosine_similarity` on sentence embeddings.

6. **Score & rank sentences**  
   - Sum each sentence’s similarity to all others.
   - Pick the top-`num_sentences` sentences (default 3).
   - Join them in their ranked order to produce the extractive summary.

---

## Usage in plain Python (optional)

If you want to call the function outside the notebook:

```python
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk, numpy as np

nltk.download('punkt')
nltk.download('stopwords')

def summarize_text_word2vec(text: str, num_sentences: int = 3) -> str:
    sentences = nltk.sent_tokenize(text)
    words = [word_tokenize(s.lower()) for s in sentences]
    stop_words = set(stopwords.words('english'))
    words = [[w for w in s if w.isalnum() and w not in stop_words] for s in words]

    model = Word2Vec(words, vector_size=300, window=5, min_count=1, workers=4)

    sentence_embeddings = []
    for s in words:
        vecs = [model.wv[w] for w in s if w in model.wv]
        if vecs:
            sentence_embeddings.append(np.mean(vecs, axis=0))

    sim = cosine_similarity(sentence_embeddings)
    scores = np.sum(sim, axis=1)
    ranked = sorted(((score, i) for i, score in enumerate(scores)), reverse=True)
    summary_sentences = [sentences[i] for _, i in ranked[:num_sentences]]
    return " ".join(summary_sentences)
```

---

## Customization

- **Number of sentences**  
  Change `num_sentences` (e.g., 2, 3, 5) based on how short you want the summary.

- **Word2Vec hyperparameters**  
  - `vector_size`: embedding size (try 100–300)  
  - `window`: context window size  
  - `min_count`: ignore rare words  
  - `workers`: parallelism

- **Preprocessing**  
  Add stemming/lemmatization, keep numbers, or adjust token filters.

- **Reproducibility**  
  Gensim uses multi-threading; for stable results, set seeds and `workers=1`:
  ```python
  import random, numpy as np
  random.seed(42); np.random.seed(42)
  model = Word2Vec(words, vector_size=300, window=5, min_count=1, workers=1, seed=42)
  ```

---

## Notes & Limitations

- **Extractive, not abstractive**: it selects existing sentences; it doesn’t paraphrase.
- **Tiny training data**: training Word2Vec on a single article is simplistic; results improve with a larger domain corpus.
- **Language**: the notebook uses **English stopwords**. For other languages, swap in the relevant tokenizer & stopwords.
- **Short inputs**: if an article has very few sentences, similarities can be less informative.

---

## Troubleshooting

- **`LookupError: Resource 'punkt' not found`**  
  Run:
  ```python
  import nltk
  nltk.download('punkt'); nltk.download('stopwords')
  ```
  Some NLTK versions may also require:
  ```python
  nltk.download('punkt_tab')
  ```

- **`KeyError` / empty embeddings**  
  Ensure each sentence has at least one alphanumeric, non-stopword token.

- **Different results run-to-run**  
  Set seeds and `workers=1` (see Reproducibility).

---

## Project Structure

```
.
├─ Word2vec_text_summarization.ipynb  # The notebook with all steps
└─ README.md                          # This file
```

---

## Next Steps / Ideas

- Use a pre-trained Word2Vec (e.g., GoogleNews vectors) instead of training on the fly.
- Add sentence position bias or TextRank for more robust ranking.
- Add a simple UI (Gradio/Streamlit) to paste text and get summaries.
- Extend to other languages (tokenizers + stopwords, e.g., Arabic).

---

