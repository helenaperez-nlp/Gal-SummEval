### Gal-SummEval


This repository contains the experimental framework and evaluation scripts for the Master's thesis "GalSumm-Eval: A Meta-Evaluation for Abstractive Summarization in Galician".

Gal-SummEval is the first expert-curated benchmark designed to assess the reliability of automated metrics and LLM-as-a-judge frameworks for Galician. Our setup is built upon and extends the codebases of **[SummEval](https://github.com/simonepri/summ-eval)**, **[BASSE](https://github.com/hitz-zentroa/summarization) and **[TFG-de-Leire-Saez-de-Cortazar]([https://github.com/hitz-zentroa/summarization](https://github.com/Leire2303/TFG-de-Leire-Saez-de-Cortazar))**.

# Goal
The goal of this project is to bridge the evaluative gap for minoritized languages by benchmarking how well automated scores correlate with expert human judgments across five dimensions: Coherence, Consistency, Fluency, Relevance, and 5W1H.


# Experimentation

Automated Metrics: implementation of traditional metrics adapted for Galician.

LLM-as-a-Judge: evaluation prompts and scripts for Qwen3 and Prometheus 2.

Cross-lingual Transfer: code for fine-tuning evaluators using Spanish (ES) and Basque (EU) data to evaluate Galician (GL) summaries.

# Data

The Gal-SummEval dataset, comprising 195 news summaries with expert annotations, is available on Hugging Face:
