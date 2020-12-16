import os
import argparse
import logging
from multiprocessing import cpu_count

from gensim import downloader
import torch
import torchvision

from utils.data_preparation import DataGenerator
from model import VisualQAModel
from train_interface import Trainer
from utils.build_vocabs import Vocabulary, AnswerVocabulary
from utils.misc import configs
from utils.enums import Modes


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Main')


def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    image_size = args.image_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    no_answers = args.no_answers
    num_epochs = args.num_epochs
    verbose_step = args.verbose_step
    verbose = args.verbose
    max_num_ans = args.max_num_ans

    cfgs = configs()
    hidden_dim = cfgs['MODEL_SETTINGS']['HIDDEN_DIM']
    embedding_dim = cfgs['MODEL_SETTINGS']['EMBEDDING_DIM']
    num_layers = cfgs['MODEL_SETTINGS']['NUM_LAYERS']

    lr = cfgs['OPTIMIZER_SETTINGS']['LR']             # learning rate
    momentum = cfgs['OPTIMIZER_SETTINGS']['MOMENTUM']
    weight_decay = cfgs['OPTIMIZER_SETTINGS']['WEIGHT_DECAY']

    logger.info(f'Word embedding model "{cfgs["WORD_EMBEDDING_MODEL_NAME"]}" downloading...')
    word2vec = downloader.load(cfgs['WORD_EMBEDDING_MODEL_NAME'])
    logger.info('Word embedding model downloaded.')

    logger.info('Vocabularies building...')

    word_list = word2vec.index2word
    question_vocab = Vocabulary(word_list)
    train_annotation_file = cfgs['PATH']['ANSWER_DIR'] + 'v2_mscoco_val2014_annotations.json'
    answer_vocab = AnswerVocabulary(train_annotation_file, no_answers)  # only train data

    logger.info('Vocabularies have been builded.')

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                                                 (0.229, 0.224, 0.225))])

    train_dataset = DataGenerator(
        image_dir=cfgs['PATH']['IMAGE_DIR'],
        image_size=image_size,
        mode=Modes.TRAIN,
        question_vocab=question_vocab,
        question_file=cfgs['PATH']['QUESTION_DIR'] + 'v2_OpenEnded_mscoco_train2014_questions.json',
        answer_vocab=answer_vocab,
        annotation_file=train_annotation_file,
        transform=transform,
        max_num_ans=max_num_ans
    )
    logger.info('Train dataset has been created.')

    validation_dataset = DataGenerator(
        image_dir=cfgs['PATH']['IMAGE_DIR'],
        image_size=image_size,
        mode=Modes.VALIDATION,
        question_vocab=question_vocab,
        question_file=cfgs['PATH']['QUESTION_DIR'] + 'v2_OpenEnded_mscoco_val2014_questions.json',
        answer_vocab=answer_vocab,
        annotation_file=cfgs['PATH']['ANSWER_DIR'] + 'v2_mscoco_val2014_annotations.json',
        transform=transform,
        max_num_ans=max_num_ans
    )
    logger.info('Validation dataset has been created.')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    word_embeds = torch.FloatTensor(word2vec.vectors)

    model = VisualQAModel(
        word_embeddings=word_embeds,
        word_embedding_dim=word2vec.vector_size,
        answer_vocab_size=answer_vocab.size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        embedding_dim=embedding_dim
    )
    logger.info('VisualQA model has been created.')
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        mode='min',
        factor=0.1,
        patience=3,
        cooldown=0
    )

    # Print model's state_dict
    logger.info("Model's state_dict:")
    for param_tensor in model.state_dict():
        logger.info(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    logger.info("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        logger.info(var_name, "\t", optimizer.state_dict()[var_name])

    trainer = Trainer(
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        verbose=verbose,
        verbose_step=verbose_step,
        model=model,
        optimizer=optimizer,
        loss_fn=torch.nn.CrossEntropyLoss(),
        lr_scheduler=lr_scheduler
    )

    trainer.fit(train_loader, valid_loader, num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='directory for model checkpoints')

    parser.add_argument('--log_dir', type=str, default='./logs', help='directory for logs')

    parser.add_argument('--image_size', type=int, default=224, help='output image size')

    parser.add_argument('--batch_size', type=int, default=256, help='batch size')

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')

    parser.add_argument('--no_answers', type=int, default=1000, help='the number of answers to be kept in vocab')

    parser.add_argument('--max_num_ans', type=int, default=10, help='maximum number of answers')

    parser.add_argument('--verbose_step', type=int, default=1, help='period of verbose step')

    parser.add_argument('--verbose', type=bool, default=True, help='verbose')

    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='number of processes working on cpu.')

    args = parser.parse_args()

    main(args)
