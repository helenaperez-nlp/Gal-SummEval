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
    """
    Reads human annotations and aggregates them to the SYSTEM LEVEL.
    Returns: {criterion: [score_model_1, score_model_2, ...]}
    """
    human_x = defaultdict(list)
    
    for criterion in criteria:
        scores_for_criterion = []
        
        for summary_config in summary_configs:
            annotation = []
            for doc in data:
                if summary_config in doc.model_summaries:
                    anns = doc.model_summaries[summary_config].anns.get(criterion)
                    if anns:
                        annotation.append(np.average(anns))
            
            if annotation:
                scores_for_criterion.append(np.average(annotation))
            else:
                scores_for_criterion.append(np.nan)

        human_x[criterion] = scores_for_criterion
    return human_x


def read_judge_scores(lang: Literal['eu', 'es', 'gl'], judges: list[str]):
    """
    Reads evaluator scores and aggregates them to the SYSTEM LEVEL.
    """
    results = defaultdict(lambda: defaultdict(list))
    pred_path = ROOT_PATH / 'pred' / 'judges' / lang
    
    for judge in judges:
        for criterion in criteria:
            csv_path = pred_path / judge / f'{criterion}.csv'
            
            if not csv_path.exists():
                logging.warning(f"File not found: {csv_path}")
                results[judge][criterion] = [np.nan] * len(summary_configs)
                continue

            df = pd.read_csv(csv_path)
            
            scores_for_criterion = []
            for summary_config in summary_configs:
                model_scores = df.loc[df['model'] == summary_config, 'score']
                
                if not model_scores.empty:
                    scores_for_criterion.append(model_scores.mean())
                else:
                    scores_for_criterion.append(np.nan)
            
            results[judge][criterion] = scores_for_criterion
            
    return results


def compute_metrics(human_data: dict, evaluator_data: dict):
    """
    Computes Kendall's Tau, Spearman, and MAE per criterion AND Overall.
    Returns: DataFrame with columns [Criterion, Kendall Tau, Spearman, MAE]
    """
    results = []
    
    all_human_clean = []
    all_evaluator_clean = []

    for criterion in criteria:
        if criterion not in human_data or criterion not in evaluator_data:
            continue
            
        h_scores = np.array(human_data[criterion])
        e_scores = np.array(evaluator_data[criterion])
        
        valid_mask = ~np.isnan(h_scores) & ~np.isnan(e_scores)
        
        if np.sum(valid_mask) > 1:
            h_clean = h_scores[valid_mask]
            e_clean = e_scores[valid_mask]
            
            tau, _ = stats.kendalltau(h_clean, e_clean)
            spear, _ = stats.spearmanr(h_clean, e_clean)
            mae = np.mean(np.abs(h_clean - e_clean))
            
            all_human_clean.extend(h_clean)
            all_evaluator_clean.extend(e_clean)
        else:
            tau, spear, mae = 0.0, 0.0, 0.0
            logging.warning(f"Not enough valid data for {criterion}")
            
        results.append({
            "Criterion": criterion,
            "Kendall Tau": tau,
            "Spearman": spear,
            "MAE": mae
        })
    
    if all_human_clean:
        overall_tau, _ = stats.kendalltau(all_human_clean, all_evaluator_clean)
        overall_spear, _ = stats.spearmanr(all_human_clean, all_evaluator_clean)
        overall_mae = np.mean(np.abs(np.array(all_human_clean) - np.array(all_evaluator_clean)))
        
        results.append({
            "Criterion": "OVERALL",
            "Kendall Tau": overall_tau,
            "Spearman": overall_spear,
            "MAE": overall_mae
        })
    
    return pd.DataFrame(results)


def dump_result(df: pd.DataFrame, output_path: Path):
    """
    Saves the DataFrame to LaTeX.
    """
    with output_path.open('w') as wf:
        wf.write(df.to_latex(index=False, float_format='%.3f'))
    logging.info(f"Saved: {output_path}")


def main(lang: Literal['eu', 'es', 'gl'], judges: list[str]):
    
    logging.info(f"Loading Human Scores for {lang}...")
    human_x = read_human_scores(load_data(lang))

    results_path = ROOT_PATH / 'results' / 'correlation' / lang / 'prometheus'  
    results_path.mkdir(parents=True, exist_ok=True)

    if judges:
        logging.info(f"Processing Judges: {judges}")
        judge_x_all = read_judge_scores(lang, judges)
        
        for judge in judges:
            judge_data = judge_x_all[judge]
            
            df_results = compute_metrics(human_x, judge_data)
            
            print(f"\nResults for {judge}:")
            print(df_results.to_string(index=False))
            
            dump_result(df_results, results_path / f'{judge}_full_stats.{lang}.tex')
            
            df_results.to_csv(results_path / f'{judge}_full_stats.{lang}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', choices=['eu', 'es', 'gl'], required=True)
    parser.add_argument('--judges', nargs='+')
    args = parser.parse_args()

    if args.judges == ['all']:
        judge_path = ROOT_PATH / 'pred' / 'judges' / args.language 
        if judge_path.exists():
            args.judges = [x.name for x in judge_path.iterdir() if x.is_dir()]
        else:
            args.judges = []
            
    logging.info(f'Judges to evaluate: {args.judges}')

    if args.judges:
        main(args.language, args.judges)