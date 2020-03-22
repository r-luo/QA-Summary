from gensim.models import word2vec
from pathlib import Path
from datetime import datetime
from helpers.utils import read_yaml_file, write_yaml_file

import logging

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def try_mkdir(path):
    """
    Checks if a path exists. If not, attempt to create the path 
    with all its parent directories if not already existing. Returns
    the path as a pathlib.Path object
    
    Parameters
    ----------
    path: str
    
    Returns
    -------
    pathlib.Path
    """
    path = Path(path).absolute()
    if not path.is_dir():
        LOG.warning(f'Directory {path} does not exist, trying to create')
        path.mkdir(parents=True)
    return path

class QAEmbedding():
    def __init__(self, **params):
        """
        
        Parameters
        ----------
        **params: 
            see arguments for gensim.models.word2vec.Word2Vec
            
        Returns
        -------
        None
        """
        self.params = params
        self.model = None
        self.last_trained_time = None
        self.sentences = []
    
    def load_sentences(self, files):
        for file in files:
            lines = Path(file).open().read().splitlines()
            self.sentences.extend([l.split() for l in lines])
    
    def word2vec(self):
        self.model = word2vec.Word2Vec(
            self.sentences,
            **self.params
        )
        self.last_trained_time = datetime.now()
        return self
        
    def get_param(self):
        return self.params

    def set_param(self, param, value):
        self.params.update({param: value})
        return self
    
    def save_model(self, path='models', filename=None):
        path = try_mkdir(path)
        if not filename:
            filename = f"word2vec_{self.last_trained_time.strftime('%Y%m%d_%H%M%S')}.model"
        metadata_filename = f'{filename}.metadata'
        self.model.save(path.joinpath(filename).absolute().as_posix())
        write_yaml_file(self.params, path.joinpath(metadata_filename))
        
    def save_embedding(self, path='data', filename=None):
        path = try_mkdir(path)
        if not filename:
            filename = f"word2vec_embeddings_{self.last_trained_time.strftime('%Y%m%d_%H%M%S')}.txt"
        self.model.wv.save_word2vec_format(path.joinpath(filename).absolute().as_posix())
    
    def load_model(self, path='models', filename='word2vec.model', metadata=None):
        self.model = word2vec.Word2Vec.load(Path(path).joinpath(filename).absolute().as_posix())
        if metadata is None:
            metadata = f'{filename}.metadata'
            if Path(metadata).exists():
                self.params = read_yaml_file()
        
    def top_similar(self, center_word, topn=10):
        LOG.info(f"Top {topn} words similar to {center_word}: ")
        top_similar = self.model.wv.most_similar(center_word, topn=topn)
        return top_similar