from dataclasses import dataclass

import pandas as pd
from datasets import Dataset
from lingua import Language, LanguageDetectorBuilder
from nltk import tokenize

from src.constants import Criterion

# TODO: loaded even if not necessary
languages = [Language.ENGLISH, Language.SPANISH, Language.BASQUE]
detector = LanguageDetectorBuilder.from_languages(*languages).build()


@dataclass
class Document:
    idx: str
    round: int
    original_document: str
    reference_summaries: list[str]
    model_summaries: dict[str, 'Summary']
    _original_document_tokens: list[str] = None
    _original_document_vocab: set[str] = None
    _reference_summary_tokens: list[list[str]] = None
    _reference_summary_vocab: set[str] = None

    @classmethod
    def from_hub(cls, data: Dataset):
        df =  data.to_pandas()
        df.columns = df.columns.str.lower()
        for url, url_summaries in df.groupby('url'):
            idx = url
            round = int(url_summaries['round'].iloc[0])
            original_document = url_summaries['document'].iloc[0]
            reference_summaries = url_summaries['references'].iloc[0].tolist()
            model_summaries = dict()
            for _, summary_data in url_summaries.iterrows():
                summary_key, summary_value = Summary.from_hub(summary_data)
                model_summaries[summary_key] = summary_value
            yield cls(idx, round, original_document, reference_summaries, model_summaries)

    @classmethod
    def from_json(cls, data: dict):
        for k, v in data['model_summaries'].items():
            data['model_summaries'][k] = Summary.from_json(v)
        return cls(**data)

    def to_json(self):
        return dict(
            idx=self.idx,
            round=self.round,
            original_document=self.original_document,
            reference_summaries=self.reference_summaries,
            model_summaries={k: v.to_json() for k, v in self.model_summaries.items()}
        )

    @property
    def original_document_tokens(self):
        if self._original_document_tokens is None:
            self._original_document_tokens = tokenize.word_tokenize(self.original_document.lower())
        return self._original_document_tokens

    @property
    def original_document_vocab(self):
        if self._original_document_vocab is None:
            self._original_document_vocab = set(self.original_document_tokens)
        return self._original_document_vocab

    @property
    def reference_summary_tokens(self):
        if self._reference_summary_tokens is None:
            self._reference_summary_tokens = [tokenize.word_tokenize(x.lower()) for x in self.reference_summaries]
        return self._reference_summary_tokens

    @property
    def reference_summary_vocab(self):
        if self._reference_summary_vocab is None:
            self._reference_summary_vocab = set(y for x in self.reference_summary_tokens for y in x)
        return self._reference_summary_vocab


@dataclass
class Summary:
    summ: str
    anns: dict[Criterion, list[float]]
    lang: str = None
    _tokens: list[str] = None
    _vocab: set[str] = None

    @classmethod
    def from_hub(cls, data: pd.Series):
        if data['prompt'] is not None:
            summary_key = f'{data["model"]}-{data["prompt"]}'
        else:
            summary_key = data['model']
        anns = {c: data[c.lower()].tolist() for c in Criterion}
        return summary_key, cls(data['summary'], anns)

    @classmethod
    def from_json(cls, data: dict):
        if 'lang' not in data:
            data['lang'] = detector.detect_language_of(data['summ']).iso_code_639_1.name.lower()
        return cls(**data)

    def to_json(self):
        return dict(summ=self.summ, anns=self.anns)

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = tokenize.word_tokenize(self.summ.lower())
        return self._tokens

    @property
    def vocab(self):
        if self._vocab is None:
            self._vocab = set(self.tokens)
        return self._vocab
