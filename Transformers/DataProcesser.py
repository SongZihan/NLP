from transformers import BertTokenizerFast
import re
import string
import pandas as pd
import numpy as np
import pickle
from rich import print
from rich.console import Console
from rich.progress import track
from ParameterStorage import ParameterStorage


class Vocabulary:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        self.vocab_save_path = ParameterStorage.vocab_save_path
        #################### 源语言 ####################
        self.text_dict_src = {}  # 词汇表
        self.freq_dict_src = {}  # 词频表
        self.inverse_dict_src = {} # 反向词汇表，用于ids to text
        #################### 目标语言 ####################
        self.text_dict_trt = {}  # 词汇表
        self.freq_dict_trt = {}  # 词频表
        self.inverse_dict_trt = {} # 反向词汇表，用于ids to text


    @staticmethod
    def preprocess(single_text):
        """
        删除标点符号
        """
        single_text = single_text.lower().strip()

        return re.sub(f'[{string.punctuation}\n]', '', single_text)
    def generate_vocabulary(self, src_dataset,trt_dataset):
        """
        根据输入的数据集，生成词汇表
        :param dataset: 以numpy数组存在的嵌套列表
        """
        # 起始，末尾，pad，unkown
        self.text_dict_trt.update({
            '<sos>': 0,
            '<eos>': 1,
            '<pad>': 2,
            '<unk>': 3})
        self.text_dict_src.update({
            '<pad>': 0,
            '<unk>': 1})

        for i in src_dataset:
            # 分词
            this_tokens = self.tokenizer.tokenize(self.preprocess(i))
            # 映射索引
            for token in this_tokens:
                if token not in self.text_dict_src.keys():
                    self.text_dict_src[token] = len(self.text_dict_src)
                    self.freq_dict_src[token] = 0
                else:
                    self.freq_dict_src[token] += 1
        # 更新反向词汇表
        self.inverse_dict_src = {value: key for key, value in self.text_dict_src.items()}

        for i in trt_dataset:
            # 分词
            this_tokens = self.tokenizer.tokenize(self.preprocess(i))
            # 映射索引
            for token in this_tokens:
                if token not in self.text_dict_trt.keys():
                    self.text_dict_trt[token] = len(self.text_dict_trt)
                    self.freq_dict_trt[token] = 0
                else:
                    self.freq_dict_trt[token] += 1
        # 更新反向词汇表
        self.inverse_dict_trt = {value: key for key, value in self.text_dict_trt.items()}

    def text_to_ids(self,dataset,IsDecoderData=False,max_length=256):
        """
        将输入数据集映射为数字索引
        :param dataset: 数据集
        :param add_special_tokens: 是否添加特殊标记，用于decoder数据生成
        :param max_length: 最大句子长度
        """
        if IsDecoderData:
            result = []
            for i in dataset:
                # 分词
                this_tokens = self.tokenizer.tokenize(self.preprocess(i))
                # 映射索引
                sentence_cache = []
                for token in this_tokens:
                    if token not in self.text_dict_trt.keys():
                        sentence_cache.append(self.text_dict_trt['<unk>'])
                    else:
                        sentence_cache.append(self.text_dict_trt[token])

                    if len(sentence_cache) >= max_length:
                        break
                # 最大长度截取
                if len(sentence_cache) < max_length:
                    for i in range(max_length - len(sentence_cache)):
                        sentence_cache.append(self.text_dict_trt['<pad>'])
                # 添加首位
                sentence_cache.insert(0,self.text_dict_trt['<sos>'])
                sentence_cache[-1] = self.text_dict_trt['<eos>']

                result.append(sentence_cache)

        else:
            result = []
            for i in dataset:
                # 分词
                this_tokens = self.tokenizer.tokenize(self.preprocess(i))
                # 映射索引
                sentence_cache = []
                for token in this_tokens:
                    if token not in self.text_dict_src.keys():
                        sentence_cache.append(self.text_dict_src['<unk>'])
                    else:
                        sentence_cache.append(self.text_dict_src[token])

                    if len(sentence_cache) >= max_length:
                        break
                # 最大长度截取
                if len(sentence_cache) < max_length:
                    for i in range(max_length - len(sentence_cache)):
                        sentence_cache.append(self.text_dict_src['<pad>'])

                result.append(sentence_cache)


        return result

    def ids_to_texts(self, ids, remove_special_tokens=False,IsDecoderData=True):
        if IsDecoderData:
            result = []
            for ids_sentence in ids:
                this_sentence = []
                for ids_word in ids_sentence:
                    this_sentence.append(self.inverse_dict_trt[ids_word])
                result.append(this_sentence)
        else:
            result = []
            for ids_sentence in ids:
                this_sentence = []
                for ids_word in ids_sentence:
                    this_sentence.append(self.inverse_dict_src[ids_word])
                result.append(this_sentence)
        return result
    def ids_to_single_text(self,ids, remove_special_tokens=False,IsDecoderData=True):
        if IsDecoderData:
            return [self.inverse_dict_trt[id] for id in ids]
        else:
            return [self.inverse_dict_src[id] for id in ids]


    def to_series(self):
        result = {
            "text_dict_src":self.text_dict_src,
            "inverse_dict_src":self.inverse_dict_src,
            "freq_dict_src":self.freq_dict_src,
            "text_dict_trt": self.text_dict_trt,
            "inverse_dict_trt": self.inverse_dict_trt,
            "freq_dict_trt": self.freq_dict_trt,
        }
        with open(self.vocab_save_path,'wb') as f:
            pickle.dump(result,f)

    def from_series(self):
        with open(self.vocab_save_path,"rb") as f:
            result = pickle.load(f)
        self.freq_dict_src = result['freq_dict_src']
        self.text_dict_src = result['text_dict_src']
        self.inverse_dict_src = result['inverse_dict_src']

        self.freq_dict_trt = result['freq_dict_trt']
        self.text_dict_trt = result['text_dict_trt']
        self.inverse_dict_trt = result['inverse_dict_trt']



if __name__ == '__main__':
    console = Console()
    with console.status("[bold green]Working on processing...") as status:
        df = pd.read_csv(ParameterStorage.data_path)

        result = dict()
        result['split'] = df['split'].tolist()


        train_df = df[df['split'] == "train"]
        valid_df = df[df['split'] == "val"]
        test_df = df[df['split'] == "test"]

        vocab = Vocabulary()


        vocab.generate_vocabulary(train_df['source_language'].tolist(),train_df['target_language'].tolist())

        result['train'] = {}
        result['train']['encoder_input'] = vocab.text_to_ids(train_df['source_language'].tolist(),max_length=ParameterStorage.max_sentence_length,IsDecoderData=False)
        decoder_data = vocab.text_to_ids(train_df['target_language'],True,max_length=ParameterStorage.max_sentence_length)
        decoder_data = np.array(decoder_data)
        result['train']['decoder_input'] = decoder_data[:, :-1].tolist()
        result['train']['decoder_output'] = decoder_data[:, 1:].tolist()

        result['valid'] = {}
        result['valid']['encoder_input'] = vocab.text_to_ids(valid_df['source_language'].tolist(),max_length=ParameterStorage.max_sentence_length,IsDecoderData=False)
        decoder_data = vocab.text_to_ids(valid_df['target_language'],True,max_length=ParameterStorage.max_sentence_length)
        decoder_data = np.array(decoder_data)
        result['valid']['decoder_input'] = decoder_data[:, :-1].tolist()
        result['valid']['decoder_output'] = decoder_data[:, 1:].tolist()


        result['test'] = {}
        result['test']['encoder_input'] = vocab.text_to_ids(test_df['source_language'].tolist(),max_length=ParameterStorage.max_sentence_length,IsDecoderData=False)
        result['test']['encoder_input'] = vocab.text_to_ids(test_df['source_language'].tolist(),max_length=ParameterStorage.max_sentence_length,IsDecoderData=False)
        decoder_data = vocab.text_to_ids(test_df['target_language'],True,max_length=ParameterStorage.max_sentence_length)
        decoder_data = np.array(decoder_data)
        result['test']['decoder_input'] = decoder_data[:, :-1].tolist()
        result['test']['decoder_output'] = decoder_data[:, 1:].tolist()

        #################### 测试正确性 ####################
        print("encoder:")
        print(f"RawText: {train_df['source_language'].tolist()[0]}")
        print(f"TranformText: {vocab.ids_to_single_text(result['train']['encoder_input'][0],IsDecoderData=False)}")
        print("DecoderInput:")
        print(f"RawText: {train_df['target_language'].tolist()[0]}")
        print(f"TranformText: {vocab.ids_to_single_text(result['train']['decoder_input'][0])}")
        print("DecoderOutput:")
        print(f"TranformText: {vocab.ids_to_single_text(result['train']['decoder_output'][0])}")



        # 将字典序列化为pickle文件
        with open(ParameterStorage.storage_path, 'wb') as f:
            pickle.dump(result, f)

        vocab.to_series()