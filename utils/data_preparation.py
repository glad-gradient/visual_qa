import json
import os

import cv2
import numpy as np
import torch
from nltk.tokenize import word_tokenize

from utils.misc import configs


def download_vqa(config_file, load_train=True, load_valid=True, load_test=False):
    cfgs = configs(config_file)
    image_dir = cfgs['PATH']['IMAGE_DIR']
    question_dir = cfgs['PATH']['QUESTION_DIR']
    answer_dir = cfgs['PATH']['ANSWER_DIR']
    link = cfgs['LINK']
    image_link = cfgs['IMAGE_LINK']

    # os.makedirs(f'{image_dir}/train/', exist_ok=True)
    # os.makedirs(f'{image_dir}/validation/', exist_ok=True)
    # os.makedirs(f'{image_dir}/test/', exist_ok=True)

    # Download and unzip images
    if load_train and not os.path.exists(f'{image_dir}/train2014.zip'):
        os.system(f'wget {image_link}/train2014.zip -P {image_dir}')
        os.system(f'unzip {image_dir}/train2014.zip -d {image_dir}/')

    if load_valid and not os.path.exists(f'{image_dir}/val2014.zip'):
        os.system(f'wget http://images.cocodataset.org/zips/val2014.zip -P {image_dir}')
        # os.system(f'wget {image_link}/val2014.zip -P {image_dir}')
        os.system(f'unzip {image_dir}/val2014.zip -d {image_dir}/')

    if load_test and not os.path.exists(f'{image_dir}/test2015.zip'):
        os.system(f'wget {image_link}/test2015.zip -P {image_dir}')
        os.system(f'unzip {image_dir}/test2015.zip -d {image_dir}/')

    # Download and unzip the VQA Questions
    if load_train and not os.path.exists(f'{question_dir}/v2_Questions_Train_mscoco.zip'):
        os.system(f'wget {link}/v2_Questions_Train_mscoco.zip -P {question_dir}')
        os.system(f'unzip {question_dir}/v2_Questions_Train_mscoco.zip -d {question_dir}')

    if load_valid and not os.path.exists(f'{question_dir}/v2_Questions_Val_mscoco.zip'):
        os.system(f'wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -P {question_dir}')
        # os.system(f'wget {link}/v2_Questions_Val_mscoco.zip -P {question_dir}')
        os.system(f'unzip {question_dir}/v2_Questions_Val_mscoco.zip -d {question_dir}')

    if load_test and not os.path.exists(f'{question_dir}/v2_Questions_Test_mscoco.zip'):
        os.system(f'wget {link}/v2_Questions_Test_mscoco.zip -P {question_dir}')
        os.system(f'unzip {question_dir}/v2_Questions_Test_mscoco.zip -d {question_dir}')

    # Download and unzip the VQA Annotations
    if load_train and not os.path.exists(f'{answer_dir}/v2_Annotations_Train_mscoco.zip'):
        os.system(f'wget {link}/v2_Annotations_Train_mscoco.zip -P {answer_dir}')
        os.system(f'unzip {answer_dir}/v2_Annotations_Train_mscoco.zip -d {answer_dir}')

    if load_valid and not os.path.exists(f'{answer_dir}/v2_Annotations_Val_mscoco.zip'):
        os.system(f'wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -P {answer_dir}')
        # os.system(f'wget {link}/v2_Annotations_Val_mscoco.zip -P {answer_dir}')
        os.system(f'unzip {answer_dir}/v2_Annotations_Val_mscoco.zip -d {answer_dir}')


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, image_dir: str, image_set: str, image_size: int, mode: str,
                 question_vocab, question_file, max_num_ans=10, answer_vocab=None, annotation_file=None, transform=None):
        self.mode = mode
        self.question_vocab = question_vocab
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
        print(img_file)
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image /= 255.0
        if self.transform:
            image = self.transform(image)

        tokens = word_tokenize(sample['question'])
        question_ids = [self.question_vocab.word2idx[t] for t in tokens if t in self.question_vocab.word2idx]
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
                'question': q['question']
            }

            if qst_id2ann:
                ann = qst_id2ann[question_id]
                all_answers = [answer["answer"] for answer in ann['answers']]
                _valid_answers = [a for a in all_answers if a in self.answer_vocab.word2idx]
                if len(_valid_answers) == 0:
                    _valid_answers = ['<unk>']

                sample['all_answers'] = all_answers
                sample['valid_answers'] = _valid_answers

            dataset.append(sample)

        return dataset

