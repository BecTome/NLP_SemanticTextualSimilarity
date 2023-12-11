# NLP_SemanticTextualSimilarity
SemEval 12's Semantic Textual Similarity task, specifically in paraphrase detection

## Introduction
SemEval (Semantic Evaluation Exercises) are a series of workshops which have the main aim of the evaluation and comparison of semantic analysis systems. The data and corpora provided by them have become a ’de facto’ set of bench- marks for the NLP comunity.

The SemEval event provides data and evaluation frameworks for several tasks. Task 6 is Semantic Textual Similarity (STS), the purpose of this project.

The description of the event is available at:

- http://ixa2.si.ehu.es/starsem/proc/pdf/STARSEM-SEMEVAL051.pdf

and the proceedings of the workshop at:

- http://ixa2.si.ehu.es/starsem/proc/program.semeval.html

# IHLT STS Project

## Statement

This project revolves around utilizing the dataset and task description from SemEval 2012's Semantic Textual Similarity. The primary objective is to implement various approaches for paraphrase detection, focusing on sentence similarity metrics. The project entails:

- Exploring lexical dimensions.
- Exploring the syntactic dimension independently.
- Exploring the combination of both lexical and syntactic dimensions.
- Optionally adding new components.

Pre-generated word or sentence embeddings models, including BERT, are not permitted. The project concludes with a comprehensive comparison and commentary on the achieved results, both internally among the approaches and against official benchmarks.


## Downloading the data

```bash
bash get_data.sh
```
