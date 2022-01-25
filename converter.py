import os

# import numpy as np
import MeCab


def load_vocab():
    # 単語辞書データを取り出す
    with open("vocab.txt", 'r') as f:
        lines = f.readlines()
    return list(map(lambda s: s.replace("\n", ""), lines))


# データ変換クラスの定義
class DataConverter:
    def __init__(self):
        """
        クラスの初期化
        """
        self.mecab = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
        # 単語辞書の登録
        self.vocab = load_vocab()

        # with open("vocab.txt", "r") as file:
        #     lines = file.read().split("\n")
        #
        # for i, line in enumerate(lines):
        #     if line:  # 空行を弾く
        #         self.vocab[line] = i

    def sentence2words(self, sentence):
        """
        文章を単語の配列にして返却する
        :param sentence: 文章文字列
        """
        sentence_words = []
        for m in self.mecab.parse(sentence).split("\n"):
            w = m.split("\t")[0].lower()
            if (len(w) == 0) or (w == "eos"):
                continue
            sentence_words.append(w)
        sentence_words.append("<eos>")
        return sentence_words

    def sentence2ids(self, sentence):
        """
        文章を単語IDのNumpy配列に変換して返却する
        :param sentence: 文章文字列
        :sentence_type: 学習用でミニバッチ対応のためのサイズ補填方向をクエリー・レスポンスで変更するため"query"or"response"を指定
        :return: 単語IDのNumpy配列
        """
        ids = []  # 単語IDに変換して格納する配列
        sentence_words = self.sentence2words(sentence)  # 文章を単語に分解する
        for word in sentence_words:
            if word in self.vocab:  # 単語辞書に存在する単語ならば、IDに変換する
                ids.append(self.vocab.index(word))
            else:  # 単語辞書に存在しない単語ならば、<unk>のIDに変換する
                ids.append(self.vocab.index("<unk>"))
        return ids

    def ids2words(self, ids):
        """
        予測時に、単語IDのNumpy配列を単語に変換して返却する
        :param ids: 単語IDのNumpy配列
        :return: 単語の配列
        """
        words = []  # 単語を格納する配列
        for i in ids:  # 順番に単語IDを単語辞書から参照して単語に変換する
            words.append(self.vocab[int(i)])  # TODO: これらの変更は不要かも！！！！
        return words
