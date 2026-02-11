import json
from pathlib import Path
from typing import Literal

from datasets import load_dataset
from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE

from src.datamodel import Document

ROOT_PATH = Path(__file__).parent.parent



def load_data(config: Literal['eu', 'eu-round-0', 'es', 'es-round-0', 'gl'], from_hub: bool = True):
    if config == 'gl':
        from_hub = False

    if from_hub:
        try:
            dataset = load_dataset('HiTZ/BASSE', config, split='test')
            return list(Document.from_hub(dataset))
        except ValueError:
            pass


    file_path = ROOT_PATH / 'data' / config / f'BASSE.{config.replace("-round-0", ".round_0")}.jsonl'
    
    with file_path.open(encoding='utf-8') as f:
        data = [Document.from_json(json.loads(line)) for line in f]
    return data

def load_rubric(criterion):
    with (ROOT_PATH / 'assets' / 'rubrics.json').open() as rf:
        rubrics = json.load(rf)
    rubric_data = rubrics[criterion]
    rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)
    return rubric
