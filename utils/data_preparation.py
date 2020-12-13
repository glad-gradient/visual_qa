import json

import cv2
import numpy as np
import torch
from nltk.tokenize import word_tokenize

from utils.build_vocabs import Vocabulary, AnswerVocabulary


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, data: list, question_tokens: list, answer_tokens: list, image_dir: str, mode: str, transform=None):
        self.data = data
        self.question_vocab = Vocabulary(question_tokens)
        self.answer_vocab = AnswerVocabulary(answer_tokens)
        self.image_dir = image_dir
        self.mode = mode
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]

        image_id = sample['image_id']
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            image = self.transform(image)

        tokens = word_tokenize(sample['question'])
        question_ids = [self.question_vocab.word2idx[t] for t in tokens if t in self.question_vocab.word2idx]
        question_ids = torch.LongTensor(question_ids)

        item = {'image': image, 'question': question_ids}

        if self.mode in ['train', 'validation']:
            answers_ids = [self.answer_vocab.word2idx[ans] for ans in sample['valid_answers']]
            answer_idx = np.random.choice(answers_ids)
            item['answer_label'] = answer_idx

        return item

    @staticmethod
    def prepare(question_file, annotation_file, valid_answers, mode: str):
        qst_id2ann = None
        if mode in ['train', 'validation']:
            with open(annotation_file) as f:
                anns = json.load(f)['annotations']

            qst_id2ann = {ann['question_id']: ann for ann in anns}

        with open(question_file) as f:
            questions = json.load(f)['questions']

        dataset = list()
        for i in range(len(questions)):
            q = questions[i]
            question_id = q['question_id']
            sample = {
                'image_id': q['image_id'],
                'question_id': question_id,
                'question': q['question']
            }

            if qst_id2ann:
                ann = qst_id2ann[question_id]
                all_answers = [answer["answer"] for answer in ann['answers']]
                _valid_answers = [a for a in all_answers if a in valid_answers]
                if len(_valid_answers) == 0:
                    _valid_answers = ['<unk>']

                sample['all_answers'] = all_answers
                sample['valid_answers'] = _valid_answers

            dataset.append(sample)

        return dataset




