import json

import cv2
import numpy as np
import torch
from nltk.tokenize import word_tokenize


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, image_set: str, image_size: int, mode: str,
                 question_vocab, question_file, max_qst_length=30, max_num_ans=10,
                 answer_vocab=None, annotation_file=None, transform=None):
        self.mode = mode
        self.question_vocab = question_vocab
        self.max_qst_length = max_qst_length
        self.answer_vocab = answer_vocab
        self.max_num_ans = max_num_ans
        self.image_set = image_set
        self.image_dir = f'{image_dir}/{image_set}'
        self.image_size = image_size
        self.transform = transform
        self.data = self._prepare(question_file, annotation_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]

        image_id = sample['image_id']
        img_file = f'{self.image_dir}/COCO_{self.image_set}_{image_id:012d}.jpg'
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image /= 255.0
        if self.transform:
            image = self.transform(image)

        tokens = word_tokenize(sample['question'])
        question_ids = np.array([0] * self.max_qst_length)
        question_ids[: len(tokens)] = [self.question_vocab.word2idx[t]
                                       if t in self.question_vocab.word2idx else 0.
                                       for t in tokens]
        question_ids = torch.LongTensor(question_ids)

        item = {'image': image, 'question': question_ids}

        if self.mode in ['train', 'validation']:
            answers_ids = [self.answer_vocab.word2idx[ans] for ans in sample['valid_answers']]
            answer_idx = np.random.choice(answers_ids)
            item['answer'] = answer_idx
            actual_answers = np.array([answer_idx]*self.max_num_ans)
            actual_answers[: len(answers_ids)] = answers_ids
            item['actual_answers'] = actual_answers

        return item

    def _prepare(self, question_file, annotation_file):
        qst_id2ann = None

        if self.mode in ['train', 'validation']:
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
                'question': q['question'].lower()
            }

            if qst_id2ann:
                ann = qst_id2ann[question_id]
                all_answers = [answer["answer"].lower() for answer in ann['answers']]
                _valid_answers = [a for a in all_answers if a in self.answer_vocab.word2idx]
                if len(_valid_answers) == 0:
                    _valid_answers = ['<unk>']

                sample['all_answers'] = all_answers
                sample['valid_answers'] = _valid_answers

            dataset.append(sample)

        return dataset

