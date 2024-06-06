import sys

from torch import optim
import torch.backends.cudnn as cudnn
import argparse
from torch.optim import *
from common.TransformerEncoder import *
from common.TransformerDecoder import *
from common.PositionalEmbedding import *
from common.DataParallel import *
from evaluation.Evaluation import *
# from evaluation.Evaluation_intent import *

# 使用正确的绝对路径格式导入模块
from data.WISE.WISE import *
from data.DuConv.DuConv import *
from data.KdConv.KdConv import *
from model.T1 import *
from model.T2 import *
from model.T3 import *
from model.T4 import *
from model.T5 import *
from model.T6 import *
from model.WISE import *
from model.WISE_T1 import *
from model.WISE_T2 import *
from model.WISE_T3 import *
from model.WISE_T4 import *
from model.WISE_T5 import *
from model.DuConv import *
from model.KdConv import *


def makedirs(path):
    if torch.cuda.is_available() and args.local_rank != 0:
        return
    if not os.path.exists(path):
        os.makedirs(path)


def prepare_dataset(args):
    tokenizer = char_tokenizer()
    vocab2id, id2vocab = load_vocab(args.vocab)
    wise_intent2id, wise_id2intent = load_vocab(args.wise_intent)
    wise_action2id, wise_id2action = load_vocab(args.wise_action)

    prepare_wise_dataset(args)
    wise_train_dataset = WISEDataset([args.wise_train_conversation_file], [args.wise_document_file], vocab2id,
                                     wise_intent2id, wise_action2id, tokenizer)
    wise_valid_dataset = WISEDataset([args.wise_valid_conversation_file], [args.wise_document_file], vocab2id,
                                     wise_intent2id, wise_action2id, tokenizer)
    wise_test_dataset = WISEDataset([args.wise_test_conversation_file], [args.wise_document_file], vocab2id,
                                    wise_intent2id, wise_action2id, tokenizer)
    wise_testunseen_dataset = WISEDataset([args.wise_testunseen_conversation_file], [args.wise_document_file], vocab2id,
                                          wise_intent2id, wise_action2id, tokenizer)
    wise_testseen_dataset = WISEDataset([args.wise_testseen_conversation_file], [args.wise_document_file], vocab2id,
                                        wise_intent2id, wise_action2id, tokenizer)
    print('WISE done, ', '\ntrain data size', wise_train_dataset.len, '\nvalid data size', wise_valid_dataset.len,
          '\ntest data size', wise_test_dataset.len, '\ntestunseen data size', wise_testunseen_dataset.len,
          '\ntestseen data size', wise_testseen_dataset.len)


def build_modules(args):
    vocab2id, id2vocab = load_vocab(args.vocab)
    word_embedding = nn.Embedding(len(vocab2id), args.hidden_size, padding_idx=0)
    position_embedding = PositionalEmbedding(args.hidden_size, dropout=args.dropout, max_len=200)
    embedding = nn.Sequential(word_embedding, position_embedding)
    encoder = TransformerEncoder(
        TransformerEncoderLayer(args.hidden_size, args.num_heads, dim_feedforward=2 * args.hidden_size,
                                dropout=args.dropout), args.enc_layers)
    decoder = TransformerDecoder(
        TransformerDecoderLayer(args.hidden_size, args.num_heads, dim_feedforward=2 * args.hidden_size,
                                dropout=args.dropout), args.dec_layers)
    generator = nn.Linear(args.hidden_size, len(id2vocab), bias=False)

    return vocab2id, id2vocab, embedding, encoder, decoder, generator


def build_wise_model(args):
    wise_intent2id, wise_id2intent = load_vocab(args.wise_intent)
    wise_action2id, wise_id2action = load_vocab(args.wise_action)

    vocab2id, id2vocab, embedding, encoder, decoder, generator = build_modules(args)

    t1 = T1(embedding, encoder, args.hidden_size, wise_id2intent)
    t2 = T2(embedding, encoder, args.hidden_size, id2vocab)
    t3 = T3(embedding, encoder, args.hidden_size, wise_id2action)
    t4 = T4(embedding, encoder, args.hidden_size)
    t5 = T5(embedding, encoder, args.hidden_size)
    t6 = T6(embedding, decoder, generator, args.hidden_size, id2vocab)

    wise_model = WISEModel(t1, t2, t3, t4, t5, t6)
    init_params(wise_model)
    return wise_model, vocab2id, wise_intent2id, wise_action2id


def build_pretrained_models(args):
    vocab2id, id2vocab, embedding, encoder, decoder, generator = build_modules(args)

    t1 = T1(embedding, encoder, args.hidden_size, {1: 'None'})
    t2 = T2(embedding, encoder, args.hidden_size, id2vocab)
    t3 = T3(embedding, encoder, args.hidden_size, {1: 'None'})
    t4 = T4(embedding, encoder, args.hidden_size)
    t5 = T5(embedding, encoder, args.hidden_size)
    t6 = T6(embedding, decoder, generator, args.hidden_size, id2vocab)
    duconv_model = DuConvModel(t1, t2, t3, t4, t5, t6)

    t1 = T1(embedding, encoder, args.hidden_size, {1: 'None'})
    t2 = T2(embedding, encoder, args.hidden_size, id2vocab)
    t3 = T3(embedding, encoder, args.hidden_size, {1: 'None'})
    t4 = T4(embedding, encoder, args.hidden_size)
    t5 = T5(embedding, encoder, args.hidden_size)
    t6 = T6(embedding, decoder, generator, args.hidden_size, id2vocab)
    kdconv_model = KdConvModel(t1, t2, t3, t4, t5, t6)

    init_params(duconv_model)
    init_params(kdconv_model)

    return duconv_model, kdconv_model, vocab2id


def pretrain_one(model, dataset, dataset_name, round, collate_fn, trainer, flag=0, folder='pretrained/'):
    for i in range(args.pretrain_epoch):
        print(dataset_name, 'epoch', i + 1)
        trainer.train_epoch('train', dataset, collate_fn, args.batch_size, i + 1)

        if torch.cuda.is_available() and args.local_rank != 0:
            continue
        if not os.path.exists(os.path.join(args.output_path, folder)):
            os.makedirs(os.path.join(args.output_path, folder))
        torch.save(model.t1.embedding.state_dict(), os.path.join(args.output_path, folder, '.'.join(
            [dataset_name, 'embedding', 'round' + str(round), 'epoch' + str((i + 1) + flag * args.epoch), 'model'])))
        torch.save(model.t1.t1_encoder.state_dict(), os.path.join(args.output_path, folder, '.'.join(
            [dataset_name, 'encoder', 'round' + str(round), 'epoch' + str((i + 1) + flag * args.epoch), 'model'])))
        torch.save(model.t6.t6_decoder.state_dict(), os.path.join(args.output_path, folder, '.'.join(
            [dataset_name, 'decoder', 'round' + str(round), 'epoch' + str((i + 1) + flag * args.epoch), 'model'])))
        torch.save(model.t6.t6_generator.state_dict(), os.path.join(args.output_path, folder, '.'.join(
            [dataset_name, 'generator', 'round' + str(round), 'epoch' + str((i + 1) + flag * args.epoch), 'model'])))


def pretrain_nokdconv(args):
    tokenizer = char_tokenizer()
    duconv_model, kdconv_model, vocab2id = build_pretrained_models(args)

    parameters = list(duconv_model.parameters())
    parameters = list(set(parameters))
    optimizer = AdamW(parameters, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100000, T_mult=2, eta_min=1e-7)

    duconv_trainer = DataParallel(duconv_model, optimizer, scheduler, args.local_rank)

    for i in range(20):
        print('DuConv pretraining without kdconv', 'round', i + 1)
        duconv_dataset = DuConvDataset([args.duconv_conversation_file], vocab2id, tokenizer)
        print('DuConv pretraining without kdconv', 'data_size', duconv_dataset.len, 'gpu', args.num_gpus, 'epoch',
              args.pretrain_epoch, 'batch_size', args.batch_size)
        pretrain_one(duconv_model, duconv_dataset, 'DuConv', i, duconv_collate_fn, duconv_trainer,
                     folder='pretrained_nokdconv')
        del duconv_dataset

def pretrain_noduconv(args):
    tokenizer = char_tokenizer()
    duconv_model, kdconv_model, vocab2id = build_pretrained_models(args)

    parameters = list(kdconv_model.parameters())
    parameters = list(set(parameters))
    optimizer = AdamW(parameters, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100000, T_mult=2, eta_min=1e-7)

    kdconv_trainer = DataParallel(kdconv_model, optimizer, scheduler, args.local_rank)

    for i in range(20):
        print('KdConv pretraining without duconv', 'round', i + 1)
        kdconv_dataset = KdConvDataset([args.kdconv_conversation_file], [args.kdconv_document_file], vocab2id,
                                       tokenizer)
        print('KdConv pretraining without duconv', 'data_size', kdconv_dataset.len, 'gpu', args.num_gpus, 'epoch',
              args.pretrain_epoch, 'batch_size', args.batch_size)
        pretrain_one(kdconv_model, kdconv_dataset, 'KdConv', i, kdconv_collate_fn, kdconv_trainer,
                     folder='pretrained_noduconv/')
        del kdconv_dataset


def pretrain(args):
    tokenizer = char_tokenizer()
    duconv_model, kdconv_model, vocab2id = build_pretrained_models(args)

    parameters = list(duconv_model.parameters()) + list(kdconv_model.parameters())
    parameters = list(set(parameters))
    optimizer = AdamW(parameters, lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100000, T_mult=2, eta_min=1e-7)

    duconv_trainer = DataParallel(duconv_model, optimizer, scheduler, args.local_rank)
    kdconv_trainer = DataParallel(kdconv_model, optimizer, scheduler, args.local_rank)

    for i in range(20):
        print('KdConv pretraining', 'round', i + 1)
        kdconv_dataset = KdConvDataset([args.kdconv_conversation_file], [args.kdconv_document_file], vocab2id,
                                       tokenizer)
        print('KdConv pretraining', 'data_size', kdconv_dataset.len, 'gpu', args.num_gpus, 'epoch', args.pretrain_epoch,
              'batch_size', args.batch_size)
        pretrain_one(kdconv_model, kdconv_dataset, 'KdConv', i, kdconv_collate_fn, kdconv_trainer)
        del kdconv_dataset
        print('DuConv pretraining', 'round', i + 1)
        duconv_dataset = DuConvDataset([args.duconv_conversation_file], vocab2id, tokenizer)
        print('DuConv pretraining', 'data_size', duconv_dataset.len, 'gpu', args.num_gpus, 'epoch', args.pretrain_epoch,
              'batch_size', args.batch_size)
        pretrain_one(duconv_model, duconv_dataset, 'DuConv', i, duconv_collate_fn, duconv_trainer)
        del duconv_dataset


def load_pretrained_model(args, prefix='full'):
    wise_intent2id, wise_id2intent = load_vocab(args.wise_intent)
    wise_action2id, wise_id2action = load_vocab(args.wise_action)

    vocab2id, id2vocab, embedding, encoder, decoder, generator = build_modules(args)
    if prefix == 'full':
        file_path = "pretrained_ready"
        print("load pretrain model with full pretrain data...")
    elif prefix == 'noduconv':
        file_path = "pretrained_noduconv_ready"
        print("load pretrain model without duconv data...")
    elif prefix == 'nokdconv':
        file_path = "pretrained_nokdconv_ready"
        print("load pretrain model without kdconv data...")
    elif prefix == 'no':
        file_path = 'no'
        print("do not load pretrain model...")
    else:
        print("load pretrain model faultly")
        raise ValueError

    if os.path.exists(os.path.join(args.output_path, file_path, 'embedding.model')):
        print("load WISE model...")
        embedding.load_state_dict(torch.load(os.path.join(args.output_path,  file_path, 'embedding.model'), map_location='cpu'))
        encoder.load_state_dict(torch.load(os.path.join(args.output_path,  file_path, 'encoder.model'), map_location='cpu'))
        decoder.load_state_dict(torch.load(os.path.join(args.output_path,  file_path, 'decoder.model'), map_location='cpu'))
        generator.load_state_dict(torch.load(os.path.join(args.output_path,  file_path, 'generator.model'), map_location='cpu'))

        freeze_params(embedding)
        freeze_params(encoder)
        freeze_params(decoder)
        freeze_params(generator)
    else:
        print("initial WISE model...")
        init_params(embedding)
        init_params(encoder)
        init_params(decoder)
        init_params(generator)

    t1 = T1(embedding, encoder, args.hidden_size, wise_id2intent)
    init_params(t1.linear)
    t2 = T2(embedding, encoder, args.hidden_size, id2vocab)
    init_params(t2.linear)
    t3 = T3(embedding, encoder, args.hidden_size, wise_id2action)
    init_params(t3.linear)
    t4 = T4(embedding, encoder, args.hidden_size)
    init_params(t4.linear)
    t5 = T5(embedding, encoder, args.hidden_size)
    init_params(t5.linear)
    t6 = T6(embedding, decoder, generator, args.hidden_size, id2vocab)

    wise_model = WISEModel(t1, t2, t3, t4, t5, t6)
    wise_t1model = WISE_T1Model(t1, t2, t3, t4, t5, t6)
    wise_t2model = WISE_T2Model(t1, t2, t3, t4, t5, t6)
    wise_t3model = WISE_T3Model(t1, t2, t3, t4, t5, t6)
    wise_t4model = WISE_T4Model(t1, t2, t3, t4, t5, t6)
    wise_t5model = WISE_T5Model(t1, t2, t3, t4, t5, t6)
    return wise_model, vocab2id, wise_intent2id, wise_action2id, wise_t1model, wise_t2model, wise_t3model, wise_t4model, wise_t5model


def finetune(args):
    tokenizer = char_tokenizer()
    if args.mode == 'finetune-nokdconv':
        prefix = "nokdconv"
        save = "WISE_nokdconv/"
        print("finetune with pretrain model without using kdconv dataset")
    elif args.mode == "finetune-noduconv":
        prefix = "noduconv"
        save = "WISE_noduconv/"
        print("finetune with pretrain model without using duconv dataset")
    elif args.mode == "finetune-none":
        prefix = "no"
        save = "WISE_nopretrain/"
        print("train without using pretrain dataset")
    elif args.mode[:8] == "finetune":
        prefix = "full"
        save = "WISE_withpretrain/"
        print("finetune with full dataset")
    else:
        print("fault args.model")
        raise ValueError
    WISE, vocab2id, wise_intent2id, wise_action2id, w1, w2, w3, w4, w5 = load_pretrained_model(args, prefix)

    if args.mode == 'finetune-not1':
        wise_model = w1
        save = 'WISE_not1/'
        print("finetune WISE-not1")
    elif args.mode == 'finetune-not2':
        wise_model = w2
        save = 'WISE_noT2/'
        print("finetune WISE-not2")
    elif args.mode == 'finetune-not3':
        wise_model = w3
        save = 'WISE_noT3/'
        print("finetune WISE-not3")
    elif args.mode == 'finetune-not4':
        wise_model = w4
        save = 'WISE_noT4/'
        print("finetune WISE-not4")
    elif args.mode == 'finetune-not5':
        wise_model = w5
        save = 'finetune_not5/'
        print("finetune WISE-not5")
    elif args.mode == 'finetune':
        wise_model = WISE
        save = 'WISE_withpretrain/'
        print("finetune WISE")
    elif args.mode == 'finetune-none':
        wise_model = WISE
        save = "WISE_nopretrain/"
        print("train WISE-FULL")
    elif args.mode[:8] == 'finetune':
        wise_model = WISE
        print("finetune WISE-FULL")
    else:
        print("fault args.mode")
        raise ValueError
    print(count_parameters(wise_model))
    print("model save in ", os.path.join(args.output_path, save))
    dataset = WISEDataset([args.wise_train_conversation_file], [args.wise_document_file], vocab2id, wise_intent2id,
                          wise_action2id, tokenizer)
    print('WISE training', 'data_size', dataset.len, 'gpu', args.num_gpus, 'epoch', args.epoch, 'batch_size',
          args.batch_size)
    optimizer = AdamW(wise_model.parameters(), lr=args.lr)
    bp_count = (args.epoch * dataset.len) / (args.num_gpus * args.batch_size)
    print('bp_count', bp_count)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(0.1 * (bp_count + 100)), T_mult=2, eta_min=1e-7)
    trainer = DataParallel(wise_model, optimizer, scheduler, args.local_rank)

    for i in range(args.epoch):
        print('epoch', i + 1)
        trainer.train_epoch('train', dataset, wise_collate_fn, args.batch_size, i + 1)
        if (i + 1) == 10:
            unfreeze_params(wise_model)
        trainer.serialize(i + 1, os.path.join(args.output_path, save))


def infer(args, prefix='valid', epochs=[], folder="withpretrain"):
    tokenizer = char_tokenizer()
    wise_model, vocab2id, wise_intent2id, wise_action2id = build_wise_model(args)

    if prefix == 'valid':
        print("infer valid")
        conversation_file = args.wise_valid_conversation_file
    elif prefix == 'train':
        print("infer train")
        conversation_file = args.wise_train_conversation_file
    elif prefix == 'testseen':
        print("infer testseen")
        conversation_file = args.wise_testseen_conversation_file
    elif prefix == 'testunseen':
        print("infer testunseen")
        conversation_file = args.wise_testunseen_conversation_file
    elif prefix == 'test':
        print("infer test")
        conversation_file = args.wise_test_conversation_file
    else:
        raise ValueError
    dataset = WISEDataset([conversation_file], [args.wise_document_file], vocab2id, wise_intent2id, wise_action2id,
                          tokenizer)

    print("infer data size ", dataset.len)
    trainer = DataParallel(wise_model, None, None, args.local_rank)

    convs = {}
    with codecs.open(conversation_file, encoding='utf-8') as f:
        for line in f:
            conv = json.loads(line)
            convs[conv[-1]['msg_id']] = conv
    # folder = "withpretrain"
    model_path = "".join(["WISE_", folder, "/"])
    file_path = "".join(["WISE_", folder, "_infer_", prefix, "/"])
    print("modle path", model_path)
    print("file path", file_path)
    if not os.path.exists(os.path.join(args.output_path, model_path)):
        print(os.path.join(args.output_path, model_path), "path not exists...")
        makedirs(os.path.join(args.output_path, file_path))
        raise ValueError
    if not os.path.exists(os.path.join(args.output_path, file_path)):
        print(os.path.join(args.output_path, file_path), "path not exists...")
        makedirs(os.path.join(args.output_path, file_path))
    if not epochs:
        epochs = list(range(args.epoch))
    for i in epochs:
        print('epoch', i + 1)
        model_file = os.path.join(os.path.join(args.output_path, model_path), '.'.join([str(i + 1), 'model']))
        if os.path.exists(model_file):
            wise_model.load_state_dict(torch.load(model_file, map_location='cpu'))
            output = trainer.test_epoch('test', dataset, wise_collate_fn, args.batch_size)

            t1_output = wise_model.intent(output['t1_output'])
            t2_output = wise_model.state(output['t2_output'])
            t3_output = wise_model.action(output['t3_output'])
            t4_output = wise_model.query(output['t4_output'])
            t5_output = wise_model.passage(output['t5_output'])
            t6_output = wise_model.response(output['t6_output'])
            for j in range(output['id'].size(0)):
                conv = copy.deepcopy(convs[output['id'][j].item()])
                id = conv[-1]['msg_id']
                for k in range(len(conv)):
                    if conv[-k - 1]['role'] == 'user':
                        conv[-k - 1]['intent'] = t1_output[j][1:-1].split('-')
                        break

                conv[-1]['state'] = t2_output[j]
                conv[-1]['action'] = t3_output[j][1:-1].split('-')

                # conv[-1]['selected_query'] = [dataset.query(id, index) for index in t4_output[j]]
                conv[-1]['selected_query'] = []
                for index in range(len(t4_output[j])):
                    conv[-1]['selected_query'].append((dataset.query(id, index), t4_output[j][index].item()))
                # conv[-1]['selected_query'] = sorted(conv[-1]['selected_query'], key=lambda x: x[1], reverse=True)
                conv[-1]['query_ranking'] = sorted(conv[-1]['selected_query'], key=lambda x: x[1], reverse=True)

                current_directory = os.getcwd()
                output_file_path = os.path.join(current_directory, "t5&passage_out.txt")
                with open(output_file_path, "a") as file2:
                    file2.write(f"t5:\n{t5_output}\n")
                    file2.write(f"pa:\n{dataset}\n")
                with open(output_file_path, "a") as file2:
                    file2.write(f"conv[-1]['selected_passage']old:\n{conv[-1]['selected_passage']}\n")

                conv[-1]['selected_passage'] = []
                for index in range(len(t5_output[j])):
                    conv[-1]['selected_passage'].append((dataset.passage(id, index), t5_output[j][index].item()))
                # conv[-1]['selected_passage'] = sorted(conv[-1]['selected_passage'], key=lambda x: x[1], reverse=True)
                conv[-1]['passage_ranking'] = sorted(conv[-1]['selected_passage'], key=lambda x: x[1], reverse=True)
                conv[-1]['response'] = t6_output[j]


def eval(args, prefix='valid', epochs=[], folder="withpretrain"):
    if args.local_rank != 0:
        return
    tokenizer = char_tokenizer()
    if prefix == 'valid':
        conversation_file = args.wise_valid_conversation_file
    elif prefix == 'train':
        conversation_file = args.wise_train_conversation_file
    elif prefix == 'test':
        conversation_file = args.wise_test_conversation_file
    elif prefix == 'testseen':
        conversation_file = args.wise_testseen_conversation_file
    elif prefix == 'testunseen':
        conversation_file = args.wise_testunseen_conversation_file
    else:
        raise ValueError
    # folder = "withpretrain"
    file_path = "".join(["WISE_", folder, "_infer_", prefix, "/"])
    print("eval file ", file_path)
    print("gt file", conversation_file)
    if not os.path.exists(os.path.join(args.output_path, file_path)):
        print(os.path.join(args.output_path, file_path), "path not exists...")
        raise ValueError
    if not epochs:
        epochs = list(range(args.epoch))
    for i in epochs:
        print('epoch', i + 1)
        if os.path.exists(os.path.join(args.output_path, file_path, '.'.join([str(i + 1), str(0), 'json']))):
            rs_files = [os.path.join(args.output_path, file_path, '.'.join([str(i + 1), str(g), 'json'])) for g in
                        range(4)]
            gt_files = [conversation_file]
            print(rs_files)
            print(gt_files)
            result = evaluate(rs_files, gt_files, tokenizer)
            print(result)


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--mode", type=str, default='finetune')

    parser.add_argument("--kdconv_files", type=list, default=['data/KdConv/film/train.json',
                                                              'data/KdConv/film/dev.json',
                                                              'data/KdConv/film/test.json',
                                                              'data/KdConv/music/train.json',
                                                              'data/KdConv/music/dev.json',
                                                              'data/KdConv/music/test.json',
                                                              'data/KdConv/travel/train.json',
                                                              'data/KdConv/travel/dev.json',
                                                              'data/KdConv/travel/test.json'])
    parser.add_argument("--kdconv_document_file", type=str,
                        default=os.path.join(dir_path, 'data/KdConv/document.json'))
    parser.add_argument("--kdconv_conversation_file", type=str,
                        default=os.path.join(dir_path, 'data/KdConv/KdConv.json'))

    parser.add_argument("--duconv_files", type=list, default=['data/DuConv/train.txt',
                                                              'data/DuConv/dev.txt'])
    parser.add_argument("--duconv_conversation_file", type=str,
                        default=os.path.join(dir_path, 'data/DuConv/DuConv.json'))

    parser.add_argument("--wise_train_file", type=str,
                        default='data/WISE/conversation_train_line.json')
    parser.add_argument("--wise_valid_file", type=str,
                        default='data/WISE/conversation_valid_line.json')
    parser.add_argument("--wise_testunseen_file", type=str,
                        default='data/WISE/conversation_testunseen_line.json')
    parser.add_argument("--wise_testseen_file", type=str,
                        default='data/WISE/conversation_testseen_line.json')
    parser.add_argument("--wise_test_file", type=str,
                        default='data/WISE/conversation_test_line.json')
    parser.add_argument("--wise_intent", type=str,
                        default=os.path.join(dir_path, 'data/WISE/intent.txt'))
    parser.add_argument("--wise_action", type=str,
                        default=os.path.join(dir_path, 'data/WISE/action.txt'))
    parser.add_argument("--wise_document_file", type=str,
                        default=os.path.join(dir_path, 'data/WISE/document_line.json'))
    parser.add_argument("--wise_train_conversation_file", type=str,
                        default=os.path.join(dir_path, 'data/WISE/WISE_train.json'))
    parser.add_argument("--wise_valid_conversation_file", type=str,
                        default=os.path.join(dir_path, 'data/WISE/WISE_valid.json'))
    parser.add_argument("--wise_testunseen_conversation_file", type=str,
                        default=os.path.join(dir_path, 'data/WISE/WISE_testunseen.json'))
    parser.add_argument("--wise_testseen_conversation_file", type=str,
                        default=os.path.join(dir_path, 'data/WISE/WISE_testseen.json'))
    parser.add_argument("--wise_test_conversation_file", type=str,
                        default=os.path.join(dir_path, 'data/WISE/WISE_test.json'))

    parser.add_argument("--vocab", type=str, default=os.path.join(dir_path, 'data/vocab.txt'))

    parser.add_argument("--output_path", type=str, default='./output/')
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--pretrain_epoch", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=2)
    args = parser.parse_args()

    if args.mode == 'data':
        prepare_dataset(args)
        exit(0)

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend='NCCL', init_method='env://')

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)
    print(args.mode)
    if args.mode == 'pretrain':
        pretrain(args)
    elif args.mode[:8] == 'pretrain':
        assert len(args.mode.split('-')) == 2
        folder = args.mode.split('-')[1]
        if folder == 'noduconv':
            pretrain_noduconv(args)
        elif folder == 'nokdconv':
            pretrain_nokdconv(args)
    elif args.mode[:8] == 'finetune':
        finetune(args)
    elif args.mode[:11] == 'infer-valid':
        if args.mode == 'infer-valid':
            folder = 'withpretrain'
        else:
            assert len(args.mode.split('-')) == 3
            folder = args.mode.split("-")[2]
        infer(args, prefix='valid', epochs=list(range(0, args.epoch)), folder=folder)
    elif args.mode[:14] == 'infer-testseen':
        if args.mode == 'infer-testseen':
            folder = 'withpretrain'
        else:
            assert len(args.mode.split('-')) == 3
            folder = args.mode.split("-")[2]
        infer(args, 'testseen', epochs=list(range(0, 50)), folder=folder)
    elif args.mode[:16] == 'infer-testunseen':
        if args.mode == 'infer-testunseen':
            folder = 'withpretrain'
        else:
            assert len(args.mode.split('-')) == 3
            folder = args.mode.split("-")[2]
        infer(args, 'testunseen', epochs=list(range(0, 50)), folder=folder)
    elif args.mode[:10] == 'infer-test':
        if args.mode == 'infer-test':
            folder = 'withpretrain'
        else:
            assert len(args.mode.split('-')) == 3
            folder = args.mode.split("-")[2]
        infer(args, 'test', epochs=list(range(0, 50)), folder=folder)
    elif args.mode[:10] == 'eval-valid':
        if args.mode == 'eval-valid':
            folder = 'withpretrain'
        else:
            assert len(args.mode.split('-')) == 3
            folder = args.mode.split("-")[2]
        eval(args, prefix='valid', epochs=list(range(0, args.epoch)), folder=folder)
    elif args.mode[:13] == 'eval-testseen':
        if args.mode == 'eval-testseen':
            folder = 'withpretrain'
        else:
            assert len(args.mode.split('-')) == 3
            folder = args.mode.split("-")[2]
        eval(args, 'testseen', epochs=list(range(0, 50)), folder=folder)
    elif args.mode[:15] == 'eval-testunseen':
        if args.mode == 'eval-testunseen':
            folder = 'withpretrain'
        else:
            assert len(args.mode.split('-')) == 3
            folder = args.mode.split("-")[2]
        eval(args, 'testunseen', epochs=list(range(0, 50)), folder=folder)
    elif args.mode[:9] == 'eval-test':
        if args.mode == 'eval-test':
            folder = 'withpretrain'
        else:
            assert len(args.mode.split('-')) == 3
            folder = args.mode.split("-")[2]
        eval(args, 'test', epochs=list(range(0, 50)), folder=folder)
