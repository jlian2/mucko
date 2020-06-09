import json
import os
from typing import List, Union


class Vocabulary(object):
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    PAD_INDEX = 0
    UNK_INDEX = 1

    def __init__(self, vocabulary_file):
        with open(vocabulary_file, 'r') as f:
            word_id = json.load(f)
        self.word2index = {}
        self.word2index[self.PAD_TOKEN] = self.PAD_INDEX
        self.word2index[self.UNK_TOKEN] = self.UNK_INDEX

        for word, index in word_id.items():
            self.word2index[word] = index + 2

        self.index2word = {
            index: word
            for word, index in self.word2index.items()
        }

    def to_indices(self, words: List[str]) -> List[int]:
        return [self.word2index.get(word, self.UNK_INDEX) for word in words]

    def to_words(self, indices: List[int]) -> List[str]:
        return [
            self.index2word.get(index, self.UNK_TOKEN) for index in indices
        ]

    # 保存字典文件
    def save(self, save_vocabulary_path: str) -> None:
        with open(save_vocabulary_path, "w") as save_vocabulary_file:
            json.dump(self.word2index, save_vocabulary_file)

    def __len__(self):
        return len(self.index2word)
