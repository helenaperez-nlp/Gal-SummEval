import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import scipy.stats as stats

from src.constants import Criterion, SummaryConfig
from src.datamodel import Document
from src.util import load_data, ROOT_PATH

logging.getLogger().setLevel(logging.INFO)

summary_configs = list(SummaryConfig)
criteria = list(Criterion)


def read_human_scores(data: list[Document]):

    human_x = defaultdict(list)
    for summary_config in summary_configs:
        for criterion in criteria:
            annotation = []
            for doc in data:
                annotation.append(np.average(doc.model_summaries[summary_config].anns[criterion]))
            human_x[criterion].append(np.average(annotation))
    return human_x


def read_metric_scores(lang: Literal['eu', 'es', 'gl'], metrics: list[str]):
    results = defaultdict(list)
    pred_path = ROOT_PATH / 'pred' / 'metrics' / lang
    for metric in metrics:
        for summary_config in summary_configs:
            df = pd.read_csv(pred_path / f'{metric}.csv')
            for idx, row in df.loc[df['model'] == summary_config].iterrows():
                results[row['metric']].append(row['score'])
    results = {k: {criterion: v for criterion in criteria} for k, v in results.items()}
    return results



def compute_correlation(ref_x: dict[str, list[float | int]], pred_x: dict[str, dict[str, list[float | int]]]):
    tau_data = np.zeros((len(pred_x), len(criteria)))
    spear_data = np.zeros((len(pred_x), len(criteria)))
    for i, scorer in enumerate(pred_x.keys()):
        for j, criterion in enumerate(criteria):
            ref_score = ref_x[criterion]
            pred_score = pred_x[scorer][criterion]
            tau, p_value = stats.kendalltau(ref_score, pred_score)
            tau_data[i, j] = tau
            s, p_value = stats.spearmanr(ref_score, pred_score)
            spear_data[i, j] = s
    return tau_data, spear_data


def dump_result(results: np.array, output_path: Path, header_name: str, header_values: list):
    df = pd.DataFrame(results, columns=criteria).round(3)
    df.insert(0, header_name, pd.Series(header_values))
    with output_path.open('w') as wf:
        wf.write(df.to_latex(index=False, float_format='%.3f'))


def main(lang: Literal['eu', 'es', 'gl'], metrics: list[str]):

    human_x = read_human_scores(load_data(lang))

    results_path = ROOT_PATH / 'results' / 'correlation' / lang
    results_path.mkdir(parents=True, exist_ok=True)

    if metrics:
        model_x = read_metric_scores(lang, metrics)
        model_tau, model_spear = compute_correlation(human_x, model_x)
        header_name = 'Metrics'
        header_values = list(model_x.keys())
        dump_result(model_tau, results_path / f'metric_tau_corr.{lang}.tex', header_name, header_values)
        dump_result(model_spear, results_path / f'metric_spear_corr.{lang}.tex', header_name, header_values)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['eu', 'es', 'gl'], required=True)
    parser.add_argument('--metrics', nargs='+')
    args = parser.parse_args()

    if args.metrics == ['all']:
        args.metrics = [x.stem for x in (ROOT_PATH / 'pred' / 'metrics' / args.language).glob('*.csv')]
    logging.info(f'Metrics to evaluate: {args.metrics}', )

    if args.metrics:
        main(args.language, args.metrics)