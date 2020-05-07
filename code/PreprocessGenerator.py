import gensim

from collections import Iterator
from gensim.models.doc2vec import Doc2Vec


class PreprocessGenerator(Iterator):
    def __init__(self, train_corpus_list, tokens_only=False):
        self.train_corpus_list = train_corpus_list
        self.tokens_only = tokens_only
        self._iter = None

    def iter(self):
        for i, tcd in enumerate(self.train_corpus_list):
            tokens = gensim.utils.simple_preprocess(tcd)
            tcd = tcd.strip()
            if len(tcd) is not 0:
                if self.tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    def __next__(self):
        if self._iter is None:
            self._iter = self.iter()

        try:
            return next(self._iter)
        except StopIteration:
            self._iter = None
            raise StopIteration
