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
    
    all_raw_scores = [] 

    for summary_config in summary_configs:
        for criterion in criteria:
            annotation = []
            
            print(f"\n--- Processing Model: {summary_config} | Criterion: {criterion} ---")
            
            for doc in data:
                if (summary_config not in doc.model_summaries) or \
                   (criterion not in doc.model_summaries[summary_config].anns):
                    continue

                raw_judge_scores = doc.model_summaries[summary_config].anns[criterion]
                
                if raw_judge_scores:
                    summary_score = np.average(raw_judge_scores)
                    
                    
                    doc_id = getattr(doc, 'id', 'Unknown_ID') 
                    print(f"Model: {summary_config} | Score: {summary_score:.2f}")
                    
                    annotation.append(summary_score)
                    all_raw_scores.append(summary_score)

            if annotation:
                system_avg = np.average(annotation)
                human_x[criterion].append(system_avg)
                print(f">> Average for {summary_config} on {criterion}: {system_avg:.4f}")
            else:
                human_x[criterion].append(np.nan)

    return human_x

read_human_scores(load_data('gl'))