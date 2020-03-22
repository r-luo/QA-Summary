import pandas as pd
import jieba as jb
from matplotlib import pylab as plt
from progressbar import ProgressBar
import copy
import pickle
from pathlib import Path
from functools import partial
import logging
from .utils import log_traceback, mprun, CPU_COUNT

LOG = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

jb.load_userdict(Path(__file__).absolute().parent.parent.joinpath('data/all_words.txt').as_posix())


class Counter(object):
    def __init__(self, items=[]):
        '''
        items : list
            list of items to count frequency for
        '''
        self.counter = {}
        if items:
            self.update(items)

    def inc(self, item, freq=1):
        if item in self.counter:
            self[item] += freq
        else:
            self[item] = freq

    def update(self, items):
        for item in items:
            self.inc(item)

    def merge(self, other):
        '''
        Other must be a Counter object
        '''
        assert isinstance(other, Counter), (
            'can only merge other Counter objects')
        for item in other.counter:
            self.inc(item, freq=other[item])

    def __setitem__(self, item, freq):
        self.counter[item] = freq

    def __getitem__(self, item):
        return self.counter[item]

    def __repr__(self):
        return self.counter.__repr__()

    def __str__(self):
        return self.counter.__str__()


def parse_sentence(q, parser='cut', **kwargs):
    '''
    Parses a single sentence using the given parser

    Parameters
    ----------
    q : str
        a single sentence
    parser : function
        a string parser function, defaults to `jieba.cut`
    **kwargs
        additional keyword arguments for the `parser`

    Return
    ------
    list
    '''
    return list(getattr(jb, parser)(q.lower(), **kwargs))


def parse_dialogue(q, parser='cut', **kwargs):
    '''
    Parses dialogues. Split messages by the pipe
    character ('|').

    Parameters
    ----------
    q : str
        dialogue to parse. messages should be separated
        by '|', images and voice recordings are ignored
    parser : function
        a string parser function, defaults to `jieba.cut`
    **kwargs
        additional keyword arguments for the `parser`

    Return
    ------
    Counter
    '''
    log = q.split('|')
    sentence = []
    for msg in log:
        if not ('[语音]' in msg or '[图片]' in msg):
            sentence.extend(parse_sentence(msg, parser=parser, **kwargs))
    return sentence


def filter_words(sentence, filter_list, ):
    return [x for x in sentence if x not in filter_list]


def sentence_cleaning(sentence, replace_list, ):
    cleaned = []
    prev_word = ''
    for word in sentence:
        if word in replace_list:
            word = replace_list[word]
        if not (word == prev_word):
            prev_word = word
            cleaned.append(word)
    return cleaned


class QABase():
    def __init__(self):
        self.Question = None
        self.Dialogue = None
        self.Report = None
        self.stopwords = None
        self.replacements = None
    
    @property
    def question(self):
        return self.Question
    
    @question.setter
    def question(self, question):
        self.Question = question
    
    @property
    def dialogue(self):
        return self.Dialogue
    
    @dialogue.setter
    def dialogue(self, dialogue):
        self.Dialogue = dialogue
    
    @property
    def report(self):
        return self.Report
    
    @report.setter
    def report(self, report):
        self.Report = report
        

class QALoader(QABase):
    def __init__(
        self,
        df_type='train',
        parser='cut',
        parser_args={},
        n_workers=CPU_COUNT,
        stopwords=None,
        replacements=None,
        stopwords_file=None,
        replacements_file=None,
    ):
        super().__init__()
        if df_type not in ('train', 'test'):
            raise ValueError("df_type must be one of 'train' or 'test'")
        self.cols = ['Question', 'Dialogue']
        if df_type == 'train':
            self.cols.append('Report')
        self.stopwords = stopwords
        self.replacements = replacements
        if stopwords_file is not None:
            self.load_stopwords(stopwords_file)
        if replacements_file is not None:
            self.load_replacements(replacements_file)
        self.df_type = df_type
        self.parser = parser
        self.parser_args = parser_args
        self.data = None
        self.n_workers = n_workers
    
    def _load_df(self, file):
        self.data = pd.read_csv(file)
        return self
    
    def _drop_na(self):
        """
        Drop if the record has any missing vlaue in self.cols.
        """
        drop_rows = self.data[self.cols].isnull().any(axis=1)
        self.data = self.data[~drop_rows]
        return self
    
    def _split_data(self):
        for col in self.cols:
            setattr(self, col, self.data[col].values)
        return self
    
    def load_stopwords(self, file):
        self.stopwords = read_yaml_file(file)
        return self
        
    def load_replacements(self, file):
        self.replacements = read_yaml_file(file)
        return self
        
    def parse(self):
        for col in self.cols:
            values = getattr(self, col)
            if values is not None:
                LOG.info(f"Parsing {col}...")
                if col == 'Dialogue':
                    mp_func = partial(parse_dialogue, parser=self.parser, **self.parser_args)
                    self.Dialogue = mprun(
                        mp_func=mp_func,
                        inputs=values,
                        n_workers=self.n_workers
                    )
                else:
                    mp_func = partial(parse_sentence, parser=self.parser, **self.parser_args)
                    parsed = mprun(
                        mp_func=mp_func,
                        inputs=values,
                        n_workers=self.n_workers
                    )
                    setattr(self, col, parsed)
        return self
    
    def remove_stopwords(self):
        if self.stopwords is None:
            LOG.warning('Stopwords not initialized')
        else:
            LOG.info(f"Removing stopwords...")
            for attr in self.cols:
                sentences = getattr(self, attr)
                if sentences is not None:
                    mp_func = partial(filter_words, filter_list=self.stopwords)
                    setattr(self, attr, mprun(
                        mp_func, inputs=sentences, n_workers=self.n_workers))
        return self
    
    def sentence_cleaning(self):
        """
        - replace words with replacement list
        - remove repeated words
        
        """
        if self.replacements is None:
            LOG.warning('Replacements not initialized')
        else:
            LOG.info(f"Cleaning Sentences...")
            for attr in self.cols:
                sentences = getattr(self, attr)
                if sentences is not None:
                    mp_func = partial(sentence_cleaning, replace_list=self.replacements)
                    setattr(self, attr, mprun(
                        mp_func, inputs=sentences, n_workers=self.n_workers))
        return self
    
    def load(self, file):
        self._load_df(file)._drop_na()._split_data()
        self.parse().remove_stopwords().sentence_cleaning()
        return self


class QACounter():
    def __init__(self):
        self.counter = Counter()
        for item in ('<START>', '<END>', '<UNK>', '<PAD>'):
            self.counter.inc(item, freq=100000000)
    
    def add_loader(self, loader):
        for attr in loader.cols:
            sentences = getattr(loader, attr)
            for s in sentences:
                self.counter.merge(Counter(s))
        return self

    def to_df(self):
        return pd.DataFrame.from_dict(
            self.counter.counter, orient='index', columns=['freq']
        ).sort_index().sort_values('freq', ascending=False).reset_index()
    
    def save_df(self, file):
        self.to_df().to_csv(file, sep=' ', header=True, index=True)


class QAVocab():
    def __init__(self, freq_threshold=5):
        self.iw = None
        self.wi = None
        self.freq_threshold = freq_threshold
        
    def load_freq_df(self, df):
        df = df[df['freq'].ge(self.freq_threshold)]
        vocab_list = [row.values for _, row in df.drop('freq', axis=1).iterrows()]
        self.wi = {v[1]: v[0] for v in vocab_list}
        self.iw = {v[0]: v[1] for v in vocab_list}
        return self
        
    def load_freq_file(self, file):
        df = pd.read_csv(file, sep=' ', )
        return self.load_freq_df(df)
        
    def save_vocab(self, file):
        write_yaml_file(self.iw, file)
        return self
    
    def load_vocab(self, file):
        self.iw = read_yaml_file(file)
        self.wi = {self.iw[i]: i for i in self.iw}
        return self

    
def mask_sentence(s, vocab):
    return [vocab.wi[w] if w in vocab.wi else vocab.wi['<UNK>'] for w in s]


def mask_oov(s, vocab):
    return [w if w in vocab.wi else '<UNK>' for w in s]


def standardize_length(s, max_len):
    return ['<START>'] + s[:(max_len - 2)] + ['<END>'] + ['<PAD>'] * max(0, max_len - len(s) - 2)


class QAProcessor(QABase):
    def __init__(self, vocab, len_q=None, len_d=None, len_r=None, n_workers=CPU_COUNT, ):
        """
        
        Parameters
        ----------
        loader: QALoader
        parser: str
            name of the cut module from jieba
        len_q: int
            length of question to standardize to
        len_d: int
            length of dialogue to standardize to
        len_r: int
            length of report to standardize to
            
        Returns
        -------
        
        """
        super().__init__()
        self.cols = []
        self.vocab = vocab
        self.max_len = {
            'Question': len_q,
            'Dialogue': len_d,
            'Report': len_r,
        }
        self.n_workers = n_workers
    
    def load_data(self, loader):
        self.cols = loader.cols.copy()
        for attr in self.cols:
            setattr(self, attr, getattr(loader, attr).copy())
        return self

    def mask_sentence(self):
        """
        mask words with their indices. 
        OOV words are replaced with index for <UNK>
        """
        LOG.info('Masking sentences...')
        for attr in self.cols:
            sentences = getattr(self, attr)
            mp_func = partial(mask_sentence, vocab=self.vocab)
            setattr(self, attr, mprun(
                mp_func, inputs=sentences, n_workers=self.n_workers))
            
        return self
    
    def mask_oov(self):
        """
        OOV words are replaced with <UNK>
        """
        LOG.info('Masking OOV...')
        for attr in self.cols:
            sentences = getattr(self, attr)
            mp_func = partial(mask_oov, vocab=self.vocab)
            setattr(self, attr, mprun(
                mp_func, inputs=sentences, n_workers=self.n_workers))
            
        return self
    
    def standardize_length(self):
        LOG.info('Standardizing Length...')
        for attr in self.cols:
            sentences = getattr(self, attr)
            mp_func = partial(standardize_length, max_len=self.max_len[attr])
            setattr(self, attr, mprun(
                mp_func, inputs=sentences, n_workers=self.n_workers))
            
        return self
    
    def save(self, save_path, prefix=''):
        for attr in self.cols:
            attr_file = save_path.joinpath(f'{prefix}_{attr.lower()}.txt')
            sentences = getattr(self, attr)
            with attr_file.open('w') as file:
                file.writelines([' '.join([str(w) for w in s]) + '\n' for s in sentences])
                