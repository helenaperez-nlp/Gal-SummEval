import os
os.environ["PYTHONUTF8"] = "1"

import argparse
import functools
import logging
import sys
from collections import defaultdict
from typing import Literal
from pathlib import Path
import pandas as pd
import snowballstemmer

import builtins
import subprocess

original_open = builtins.open
def patched_open(*args, **kwargs):
    is_binary = len(args) > 1 and 'b' in args[1]
    if not is_binary:
        if 'encoding' not in kwargs:
            kwargs['encoding'] = 'utf-8'
        if 'errors' not in kwargs:
            kwargs['errors'] = 'ignore'
    return original_open(*args, **kwargs)
builtins.open = patched_open

original_Popen = subprocess.Popen
class PatchedPopen(original_Popen):
    def __init__(self, args, **kwargs):
        if isinstance(args, list) and len(args) > 0:
            if "ROUGE-1.5.5.pl" in str(args[0]):
                args = ["perl"] + args
        super().__init__(args, **kwargs)
subprocess.Popen = PatchedPopen

from src.util import load_data, ROOT_PATH

rouge_path = ROOT_PATH / 'lib' / 'SummEval' / 'evaluation' / 'summ_eval' / 'ROUGE-1.5.5'
os.environ['ROUGE_HOME'] = str(rouge_path)

try:
    import summ_eval
except ImportError:
    lib_path = ROOT_PATH / 'lib' / 'SummEval' / 'evaluation'
    sys.path.append(str(lib_path))
    import summ_eval



from src.constants import Metric, SummaryConfig
from src.metrics.chrf_metric import ChrfppMetric
from src.metrics.multilingual_meteor_metric import MultilingualMeteorMetric
from src.metrics.multilingual_rouge_we_metric import MultilingualRougeWeMetric

logging.getLogger().setLevel(logging.INFO)

metric_name_map = {
    'rouge_1_f_score': 'ROUGE-1',
    'rouge_2_f_score': 'ROUGE-2',
    'rouge_3_f_score': 'ROUGE-3',
    'rouge_4_f_score': 'ROUGE-4',
    'rouge_l_f_score': 'ROUGE-L',
    'rouge_su*_f_score': 'ROUGE-su*',
    'multilingual_rouge_we_3_f': 'mROUGE-we',
    'bert_score_precision': 'BertScore-p',
    'bert_score_recall': 'BertScore-r',
    'bert_score_f1': 'BertScore-f',
    'summary_length': 'Length',
    'percentage_novel_1-gram': 'Novel unigram',
    'percentage_novel_2-gram': 'Novel bi-gram',
    'percentage_novel_3-gram': 'Novel tri-gram',
    'percentage_repeated_1-gram_in_summ': 'Repeated unigram',
    'percentage_repeated_2-gram_in_summ': 'Repeated bi-gram',
    'percentage_repeated_3-gram_in_summ': 'Repeated tri-gram',
    'coverage': 'Stats-coverage',
    'compression': 'Stats-compression',
    'density': 'Stats-density',
    'bleu': 'BLEU',
    'chrf': 'CHRF',
    'meteor': 'METEOR',
    'm_meteor': 'MultilingualMETEOR',
    'cider': 'CIDEr'
}

summary_configs = list(SummaryConfig)

def handle_error(metric: str):
    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            try:
                logging.info(f'Computing {metric.upper()}')
                return func(*args, **kwargs)
            except Exception as e:
                logging.exception(e)
                return None
        return inner_wrapper
    return outer_wrapper

def compute_simple(metric, ref, pred):
    results = []
    for summary_config in summary_configs:
        if summary_config in pred:
            config_pred = pred[summary_config]
            d = metric.evaluate_batch(config_pred, ref)
            for metric_to_report in d.keys():
                results.append(dict(model=summary_config, metric=metric_to_report, score=d[metric_to_report]))
    return results

@handle_error(Metric.rogue)
def compute_rouge(ref, pred):
    from summ_eval.rouge_metric import RougeMetric  # Import locally
    metric = RougeMetric()
    results = []
    for summary_config in summary_configs:
        if summary_config in pred:
            config_pred = pred[summary_config]
            d = metric.evaluate_batch(config_pred, ref)
            for metric_to_report in (
                'rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score', 'rouge_4_f_score',
                'rouge_l_f_score', 'rouge_su*_f_score'
            ):
                results.append(dict(model=summary_config, metric=metric_to_report, score=d['rouge'][metric_to_report]))
    return results

@handle_error(Metric.m_rouge_we)
def compute_m_rouge_we(ref, pred, vocab: set[str], lang: str):
    emb_path = ROOT_PATH / 'assets' / f'cc.{lang}.300.vec.gz'
    metric = MultilingualRougeWeMetric(emb_path=emb_path, lang=lang, vocab=vocab)
    results = []
    for summary_config in summary_configs:
        if summary_config in pred:
            config_pred = pred[summary_config]
            d = metric.evaluate_batch(config_pred, ref)
            metric_to_report = 'multilingual_rouge_we_3_f'
            results.append(dict(model=summary_config, metric=metric_to_report, score=d[metric_to_report]))
    return results

@handle_error(Metric.m_bert_score)
def compute_m_bert_score(ref, pred):
    from summ_eval.bert_score_metric import BertScoreMetric  # Import locally to prevent PyTorch load in subprocesses
    metric = BertScoreMetric(model_type='bert-base-multilingual-uncased')
    results = []
    for summary_config in summary_configs:
        if summary_config in pred:
            config_pred = pred[summary_config]
            d = metric.evaluate_batch(config_pred, ref)
            for metric_to_report in ('bert_score_precision', 'bert_score_recall', 'bert_score_f1'):
                results.append(dict(model=summary_config, metric=metric_to_report, score=d[metric_to_report]))
    return results

@handle_error(Metric.bleu)
def compute_bleu(ref, pred):
    from summ_eval.bleu_metric import BleuMetric # Import locally
    metric = BleuMetric()
    return compute_simple(metric, ref, pred)

@handle_error(Metric.chrf)
def compute_chrf(ref, pred):
    return compute_simple(ChrfppMetric(), ref, pred)

@handle_error(Metric.m_meteor)
def compute_m_meteor(ref, pred):
    metric = MultilingualMeteorMetric()
    results = []
    for summary_config in summary_configs:
        if summary_config in pred:
            config_pred = pred[summary_config]
            d = metric.evaluate_batch(config_pred, ref)
            results.append(dict(model=summary_config, metric='m_meteor', score=d['meteor']))
    return results

@handle_error(Metric.cider)
def compute_cider(ref, pred, lang):
    from summ_eval.cider_metric import CiderMetric # Import locally
    metric = CiderMetric()
    stemmer_name = 'basque' if lang == 'eu' else 'spanish'
    stemmer = snowballstemmer.stemmer(stemmer_name)
    stemmed_references = [[' '.join(stemmer.stemWords(y.split())) for y in x] for x in ref]
    stemmed_summaries = {k: [' '.join(stemmer.stemWords(x.split())) for x in v] for k, v in pred.items()}
    return compute_simple(metric, stemmed_references, stemmed_summaries)

@handle_error(Metric.stats)
def compute_stats(ref, pred):
    from summ_eval.data_stats_metric import DataStatsMetric # Import locally
    metric = DataStatsMetric(tokenize=False)
    results = []
    for summary_config in summary_configs:
        if summary_config in pred:
            stat = {
                'summary_length': 0, 'compression': 0, 'density': 0, 'coverage': 0,
                'percentage_novel_1-gram': 0, 'percentage_novel_3-gram': 0, 'percentage_novel_2-gram': 0,
                'percentage_repeated_1-gram_in_summ': 0, 'percentage_repeated_2-gram_in_summ': 0,
                'percentage_repeated_3-gram_in_summ': 0
            }
            reference_tokens = [[y.split() for y in x] for x in ref]
            summary_tokens = [x.split() for x in pred[summary_config]]
            
            reference_n = len(reference_tokens[0])
            for i in range(reference_n):
                indv_refs = []
                for r_toks in reference_tokens[:15]:
                    if i < len(r_toks): indv_refs.append(r_toks[i])
                    else: indv_refs.append(r_toks[0])
                for r_toks in reference_tokens[15:]:
                    indv_refs.append(r_toks[0])

                d = metric.evaluate_batch(summary_tokens, indv_refs)
                for metric_to_report in d.keys():
                    stat[metric_to_report] += d[metric_to_report]
            
            for metric_to_report in stat.keys():
                stat[metric_to_report] /= reference_n
            for metric_to_report in stat.keys():
                results.append(dict(model=summary_config, metric=metric_to_report, score=stat[metric_to_report]))
    return results

def dump_result(data: list[dict], lang: str, metric: str):
    if not data: return
    df = pd.DataFrame.from_records(data)
    df['metric'] = df['metric'].map(metric_name_map)
    pred_path = ROOT_PATH / 'pred' / 'metrics' / lang
    pred_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(pred_path / f'{metric}.csv', index=False)

def main(lang: Literal['gl'], metrics: list[str]):
    data = load_data(lang)
    logging.info(f'{len(data)} documents loaded')

    global summary_configs 
    present_keys = set()
    for doc in data: present_keys.update(doc.model_summaries.keys())
    summary_configs[:] = [c for c in summary_configs if c in present_keys]
    logging.info(f"Models found: {[str(c) for c in summary_configs]}")

    logging.info(f'Tokenizing human summaries')
    references, tokenized_references = [], []
    for doc in data:
        references.append(doc.reference_summaries)
        tokenized_references.append([' '.join(x) for x in doc.reference_summary_tokens])

    logging.info(f'Tokenizing automatic summaries')
    summaries, tokenized_summaries = defaultdict(list), defaultdict(list)
    for summary_config in summary_configs:
        for doc in data:
            if summary_config in doc.model_summaries:
                config_summary = doc.model_summaries[summary_config]
                summaries[summary_config].append(config_summary.summ)
                tokenized_summaries[summary_config].append(' '.join(config_summary.tokens))

    if Metric.rogue in metrics:
        result = compute_rouge(tokenized_references, tokenized_summaries)
        dump_result(result, lang, Metric.rogue)

    if Metric.m_rouge_we in metrics:
        vocab = set()
        for doc in data:
            vocab.update(doc.reference_summary_vocab)
            for model_summary in doc.model_summaries.values():
                vocab.update(model_summary.vocab)
        result = compute_m_rouge_we(tokenized_references, tokenized_summaries, vocab, lang)
        dump_result(result, lang, Metric.m_rouge_we)

    if Metric.m_bert_score in metrics:
        result = compute_m_bert_score(references, summaries)
        dump_result(result, lang, Metric.m_bert_score)

    if Metric.bleu in metrics:
        result = compute_bleu(tokenized_references, tokenized_summaries)
        dump_result(result, lang, Metric.bleu)

    if Metric.chrf in metrics:
        result = compute_chrf(references, summaries)
        dump_result(result, lang, Metric.chrf)

    if Metric.m_meteor in metrics:
        result = compute_m_meteor(tokenized_references, tokenized_summaries)
        dump_result(result, lang, Metric.m_meteor)

    if Metric.cider in metrics:
        result = compute_cider(tokenized_references, tokenized_summaries, lang)
        dump_result(result, lang, Metric.cider)

    if Metric.stats in metrics:
        result = compute_stats(tokenized_references, tokenized_summaries)
        dump_result(result, lang, Metric.stats)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['gl'], required=True)
    parser.add_argument('--metrics', choices=['all'] + list(Metric), nargs='+', required=True)
    args = parser.parse_args()
    if 'all' in args.metrics: args.metrics = list(Metric)
    logging.info(args)
    main(args.language, args.metrics)