import json


class Vocabulary:
    def __init__(self, words):
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = dict([(value, key) for key, value in self.word2idx.items()])
        self.size = len(words)


class AnswerVocabulary(Vocabulary):
    def __init__(self, input_dir, no_answers):
        words = self._build(input_dir, no_answers)
        super().__init__(words)

    def _build(self, annotation_file, no_answers):
        vocab_counter = dict()

        with open(annotation_file, 'r') as f:
            annotations = json.load(f)['annotations']
            for ann in annotations:
                for item in ann['answers']:
                    answer = item['answer']
                    vocab_counter[answer] = vocab_counter.get(answer, 0) + 1

        answers = sorted(vocab_counter, key=vocab_counter.get, reverse=True)
        top_answers = ['<unk>'] + answers[:no_answers - 1]
        return top_answers

