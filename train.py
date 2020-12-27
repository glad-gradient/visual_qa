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

    image_dir = args.image_dir
    question_dir = args.question_dir
    answer_dir = args.answer_dir
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(question_dir, exist_ok=True)
    os.makedirs(answer_dir, exist_ok=True)

    image_size = args.image_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    no_answers = args.no_answers
    num_epochs = args.num_epochs
    val_size = args.val_size
    verbose_step = args.verbose_step
    verbose = args.verbose
    max_num_ans = args.max_num_ans

    cfgs = configs(config_file=args.config_file)
    hidden_dim = cfgs['MODEL_SETTINGS']['HIDDEN_DIM']
    embedding_dim = cfgs['MODEL_SETTINGS']['EMBEDDING_DIM']
    num_layers = cfgs['MODEL_SETTINGS']['NUM_LAYERS']

    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    word2vec = downloader.load(cfgs['WORD_EMBEDDING_MODEL_NAME'])

    logger.info('Vocabularies building...')

    word_list = word2vec.index_to_key
    question_vocab = Vocabulary(word_list)
    train_annotation_file = answer_dir + '/v2_mscoco_train2014_annotations.json'
    answer_vocab = AnswerVocabulary(train_annotation_file, no_answers)  # only train data

    logger.info('Vocabularies have been built.')

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                                                 (0.229, 0.224, 0.225))])

    train_dataset = DataGenerator(
        image_dir=image_dir,
        image_set='train2014',
        image_size=image_size,
        mode=Modes.TRAIN,
        question_vocab=question_vocab,
        question_file=question_dir + '/v2_OpenEnded_mscoco_train2014_questions.json',
        answer_vocab=answer_vocab,
        annotation_file=train_annotation_file,
        transform=transform,
        max_num_ans=max_num_ans
    )
    logger.info('Train dataset has been created.')

    validation_dataset = DataGenerator(
        image_dir=image_dir,
        image_set='val2014',
        image_size=image_size,
        mode=Modes.VALIDATION,
        question_vocab=question_vocab,
        question_file=question_dir + '/v2_OpenEnded_mscoco_val2014_questions.json',
        answer_vocab=answer_vocab,
        annotation_file=answer_dir + '/v2_mscoco_val2014_annotations.json',
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

    # params = list()
    # params.extend(list(model.image_encoder.model.fc.parameters()))
    # params.extend(list(model.question_encoder.parameters()))
    # params.extend(list(model.fc1.parameters()))
    # params.extend(list(model.fc2.parameters()))

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=momentum,
    #     weight_decay=weight_decay
    # )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        cooldown=0
    )

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

    trainer.fit(train_loader, valid_loader, num_epochs, val_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='directory for model checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='directory for logs')
    parser.add_argument('--config_file', type=str, default='./configs.json', help='path to the config file')

    parser.add_argument('--image_dir', type=str, default='./data/images', help='directory for images')
    parser.add_argument('--question_dir', type=str, default='./data/questions', help='directory for questions')
    parser.add_argument('--answer_dir', type=str, default='./data/answers', help='directory for answers')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay')

    parser.add_argument('--image_size', type=int, default=224, help='output image size')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--val_size', type=float, default=0.3, help='part of the valid dataset')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='number of processes working on cpu')

    parser.add_argument('--no_answers', type=int, default=1000, help='the number of answers to be kept in vocab')
    parser.add_argument('--max_num_ans', type=int, default=10, help='maximum number of answers per question')

    parser.add_argument('--verbose_step', type=int, default=1, help='period of verbose step')
    parser.add_argument('--verbose', type=bool, default=True, help='verbose')

    args = parser.parse_args()

    main(args)
