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

def word_freq_sentence(q, parser=jb.cut, **kwargs):
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
    Counter
    '''
    return Counter(list(parser(q.lower(), **kwargs)))


def word_freq_dialogue(q, parser=jb.cut, **kwargs):
    '''
    Parses dialogues. Split messages by the pipe
    character ('|'). Ignores messages containing
    '[语音]' or '[图片]'

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
    counter = Counter()
    for msg in log:
        if '[语音]' or '[图片]' not in msg:
            counter.merge(word_freq_sentence(msg, parser=parser, **kwargs))
    return counter


def word_freq(str_list, parse_func, vocab=None, **kwargs):
    '''
    Runs the provided parse_func on a list of input strings
    (e.g. Question). Adds the results to a vocab set (creates
    one if not provided). A `progressbar.ProgressBar` is used
    to show progress.

    Parameters
    ----------
    str_list : `obj`:list of `obj`:str
        list of strings to parse
    parse_func : function
        function used to parse the strings in the list, one of
        `word_freq_sentence` or `word_freq_dialogue`
    vocab : dict
        dictionary tracking word frequency. an empty dict is
        created if none given
    **kwargs
        additional keyword arguments for the `parse_func`

    Returns
    -------
    Counter
    '''
    if vocab is None:
        vocab = Counter()
    for i, q in enumerate(str_list):
        if not pd.isnull(q):
            vocab.merge(parse_func(q, **kwargs))
    return vocab


def distributed_word_freq(
        str_list,
        parse_func,
        n_workers=4,
        scheduler='threads',
        **kwargs
):
    partitioned_lists = [str_list[i::n_workers] for i in range(n_workers)]
    mp_func = partial(word_freq, parse_func=parse_func)

    results = mprun(
        mp_func=mp_func,
        inputs=partitioned_lists,
        n_workers=n_workers
    )

    agg = Counter()
    for c in results:
        agg.merge(c)
    return agg

def parse_array(str_list, parse_func, concat='append', **kwargs, ):
    '''
    Runs the provided parse_func on a list of input strings
    (e.g. Question). Returns a list of the parsed sentences.

    Parameters
    ----------
    str_list : `obj`:list of `obj`:str
        list of strings to parse
    parse_func : function
        function used to parse the strings in the list, one of
        `parse_sentence` or `parse_dialogue`
    **kwargs
        additional keyword arguments for the `parse_func`

    Returns
    -------
    list
    '''
    res = []
    for i, q in enumerate(str_list):
        if not (pd.isnull(q) or len(q) == 0):
            getattr(res, concat)(parse_func(q, **kwargs))
    return res



def add_start_end_marks(sentence):
    sentence.insert(0, '<START>')
    sentence.append('<END>')
    return sentence

def sentence_mask_low_freq_words(sentence, high_freq_words):
    new_sentence = []
    for word in sentence:
        if word in high_freq_words:
            new_sentence.append(word)
        else:
            new_sentence.append('<UNK>')
    return new_sentence


def sentence_list_processing(
        sentence_list,
        filter_list,
        replace_list,
        high_freq_words,
        words_threshold=2,
):
    res = []
    for sentence in sentence_list:
        sentence = [x for x in sentence if x not in filter_list]
        sentence = add_start_end_marks(sentence)
        cleaned, valid_word_cnt = sentence_cleaning(
            sentence, replace_list=replace_list,
        )
        if valid_word_cnt >= words_threshold:
            cleaned = sentence_mask_low_freq_words(cleaned, high_freq_words)
            res.append(cleaned)
    return res


def distributed_sentence_processing(
        sentence_list,
        filter_list,
        replace_list,
        high_freq_words,
        words_threshold=2,
        n_workers=4,
):

    partitioned_lists = [sentence_list[i::n_workers] for i in range(n_workers)]
    mp_func = partial(
        sentence_list_processing,
        filter_list=filter_list,
        replace_list=replace_list,
        high_freq_words=high_freq_words,
        words_threshold=words_threshold,
    )

    results = mprun(
        mp_func=mp_func,
        inputs=partitioned_lists,
        n_workers=n_workers
    )

    merge = []
    for r in results:
        merge.extend(r)

    return merge


def sentence_word_to_index(sentence, word_to_idx):
    new_sentence = []
    for word in sentence:
        new_sentence.append(word_to_idx[word])
    return new_sentence


def sentence_list_word_to_index(sentence_list, word_to_idx):
    res = []
    for sentence in sentence_list:
        res.append(sentence_word_to_index(sentence, word_to_idx))
    return res


def distributed_sentence_to_idx(
        sentence_list,
        word_to_idx,
        n_workers=4,
):

    partitioned_lists = [sentence_list[i::n_workers] for i in range(n_workers)]
    mp_func = partial(
        sentence_list_word_to_index,
        word_to_idx=word_to_idx,
    )

    results = mprun(
        mp_func=mp_func,
        inputs=partitioned_lists,
        n_workers=n_workers
    )

    merge = []
    for r in results:
        merge.extend(r)

    return merge
