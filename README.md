# Adaptive-RAG-for-Psychological-Assessment

This repository contains the code related to the work "Are LLMs effective psychological assessors? Leveraging adaptive RAG for interpretable mental health screening through psychometric practice".

<img src="https://github.com/Fede-stack/Adaptive-RAG-for-Psychological-Assessment/blob/main/images/Pipeline.png" alt="" width="900">

## Running the Code with TONYpy

The code in this repository can also be run using the **TONYpy** package, available at:

**https://github.com/Fede-stack/TONYpy**

TONYpy provides an easy-to-use interface for running the psychological assessment pipeline and supports both **OpenRouter** and **Hugging Face** as LLM providers.

After installing TONYpy, the `BDIScorer` can be imported directly from the package:

```python
from TONY.BDI import BDIScorer
```

For example:

```python
from TONY.BDI import BDIScorer

bdi_items = [
    # 1. Sadness
    [
        "I do not feel sad.",
        "I feel sad much of the time.",
        "I am sad all the time.",
        "I am so sad or unhappy that I can't stand it."
    ],
    # 2. Pessimism
    [
        "I am not discouraged about my future.",
        "I feel more discouraged about my future than I used to be.",
        "I do not expect things to work out for me.",
        "I feel my future is hopeless and will only get worse."
    ],
    # ...
    # 21 BDI-II items
]

items_names = [
    'Sadness',
    'Pessimism',
    'Past Failure',
    'Loss of Pleasure',
    'Guilty Feelings',
    'Punishment Feelings',
    'Self-Dislike',
    'Self-Criticalness',
    'Suicidal Thoughts or Wishes',
    'Crying',
    'Agitation',
    'Loss of Interest',
    'Indecisiveness',
    'Worthlessness',
    'Loss of Energy',
    'Changes in Sleeping Pattern',
    'Irritability',
    'Changes in Appetite',
    'Concentration Difficulty',
    'Tiredness or Fatigue',
    'Loss of Interest in Sex'
]

# Each inner list contains all Reddit posts written by a single user
reddit_posts = [
    [
        'I have been feeling empty for weeks',
        'I can barely get out of bed',
        # ...
    ]
]

scorer = BDIScorer(
    retriever_model_name='FritzStack/mpnet_MH_embedding',
    llm_model_name='google/gemma-3-27b-it',
    use_hf=False,
    client=client,
)

response_llms = scorer.score(
    reddit_posts,
    bdi_items,
    items_names
)

# Output: 21-dimensional vector of predicted BDI-II item scores
```

### OpenRouter

TONYpy can be used with models available through **OpenRouter**. In this configuration, `use_hf=False` is used and the appropriate OpenRouter client is passed to `BDIScorer`.

### Hugging Face

TONYpy also supports running the pipeline with **Hugging Face** models. In this case, `use_hf=True` can be used together with the appropriate Hugging Face configuration.

For installation and detailed configuration instructions, please refer to the **[TONYpy repository](https://github.com/Fede-stack/TONYpy)**.

## Citation

To cite this work refer to:

```bibtex
@inproceedings{ravenda-etal-2025-llms,
    title = "Are {LLM}s effective psychological assessors? Leveraging adaptive {RAG} for interpretable mental health screening through psychometric practice",
    author = "Ravenda, Federico  and
      Bahrainian, Seyed Ali  and
      Raballo, Andrea  and
      Mira, Antonietta  and
      Kando, Noriko",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.440/",
    doi = "10.18653/v1/2025.acl-long.440",
    pages = "8975--8991",
    ISBN = "979-8-89176-251-0"
}
```

```bibtex
@inproceedings{ravenda-etal-2026-tony,
    title = "{TONY}: an open-source {TO}olkit for Nlp in ps{Y}chology",
    author = "Ravenda, Federico  and
      Ravenda, Sofia Irene  and
      Karpenko, Volodymyr  and
      Montagnani, Daniele  and
      Raballo, Andrea  and
      Mira, Antonietta",
    editor = "Durrett, Greg  and
      Jian, Ping",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-demo.65/",
    doi = "10.18653/v1/2026.acl-demo.65",
    pages = "660--671",
    ISBN = "979-8-89176-392-0"
}
```
