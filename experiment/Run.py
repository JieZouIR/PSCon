import sys

from torch import optim
import torch.backends.cudnn as cudnn
import argparse
from torch.optim import *
from common.TransformerEncoder import *
from common.TransformerDecoder import *
from common.PositionalEmbedding import *
from common.DataParallel import *
from common.Utils import *
from evaluation.Evaluation import *

from data.PSCon.PSCon import *
from data.DuConv.DuConv import *
from data.KdConv.KdConv import *
from data.Utils import *
from model.T1 import *  # Intent Detection model
from model.T2 import *  # Keyword Extraction model
from model.T3 import *  # Action Prediction model
from model.T4 import *  # Query Selection model
from model.T5 import *  # Item Ranking model
from model.T6 import *  # Response Generation model
from model.PSCon_T1 import *
from model.PSCon_T2 import *
from model.PSCon_T3 import *
from model.PSCon_T4 import *
from model.PSCon_T5 import *
from model.PSCon_T6 import *
from model.PSCon import *
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
    PSCon_intent2id, PSCon_id2intent = load_vocab(args.PSCon_intent)
    PSCon_action2id, PSCon_id2action = load_vocab(args.PSCon_action)

    prepare_kdconv_dataset(args)
    kdconv_dataset = KdConvDataset([args.kdconv_conversation_file], [args.kdconv_document_file], vocab2id, tokenizer)
    print('KdConv done, ', 'datasize ', kdconv_dataset.len)
    prepare_duconv_dataset(args)
    duconv_dataset = DuConvDataset([args.duconv_conversation_file], vocab2id, tokenizer)
    print('DuConv done, ', 'datasize ', duconv_dataset.len)
    
    prepare_PSCon_dataset(args)
    PSCon_train_dataset = PSConDataset([args.PSCon_train_conversation_file], [args.PSCon_document_file], vocab2id,
                                     PSCon_intent2id, PSCon_action2id, tokenizer)
    PSCon_test_dataset = PSConDataset([args.PSCon_test_conversation_file], [args.PSCon_document_file], vocab2id,
                                    PSCon_intent2id, PSCon_action2id, tokenizer)
    print('PSCon done, ', '\ntrain data size', PSCon_train_dataset.len, '\ntest data size', PSCon_test_dataset.len)


def build_modules(args):
    """Build core transformer modules for the model"""
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


def build_PSCon_model(args, task='t1'):
    """构建PSCon模型,支持全任务或单任务
    
    Args:
        args: 参数配置
        task: 指定任务(t1-t6),为None时构建完整模型
    """
    PSCon_intent2id, PSCon_id2intent = load_vocab(args.PSCon_intent)
    PSCon_action2id, PSCon_id2action = load_vocab(args.PSCon_action)

    vocab2id, id2vocab, embedding, encoder, decoder, generator = build_modules(args)

    # 初始化所有模型组件
    t1 = T1(embedding, encoder, args.hidden_size, PSCon_id2intent)  
    t2 = T2(embedding, encoder, args.hidden_size, id2vocab)
    t3 = T3(embedding, encoder, args.hidden_size, PSCon_id2action)
    t4 = T4(embedding, encoder, args.hidden_size)
    t5 = T5(embedding, encoder, args.hidden_size)
    t6 = T6(embedding, decoder, generator, args.hidden_size, id2vocab)

    # 根据task参数返回对应的单任务模型
    if task == 't1':
        model = PSCon_T1Model(t1)
    elif task == 't2':
        model = PSCon_T2Model(t2) 
    elif task == 't3':
        model = PSCon_T3Model(t3)
    elif task == 't4':
        model = PSCon_T4Model(t4)
    elif task == 't5':
        model = PSCon_T5Model(t5)
    elif task == 't6':
        model = PSCon_T6Model(t3, t4, t5, t6)
    else:
        # 默认返回完整模型
        model = PSConModel(t1, t2, t3, t4, t5, t6)

    init_params(model)
    return model, vocab2id, PSCon_intent2id, PSCon_action2id


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
    PSCon_intent2id, PSCon_id2intent = load_vocab(args.PSCon_intent)
    PSCon_action2id, PSCon_id2action = load_vocab(args.PSCon_action)

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
        print("load PSCon model...")
        embedding.load_state_dict(torch.load(os.path.join(args.output_path,  file_path, 'embedding.model'), map_location='cpu'))
        encoder.load_state_dict(torch.load(os.path.join(args.output_path,  file_path, 'encoder.model'), map_location='cpu'))
        decoder.load_state_dict(torch.load(os.path.join(args.output_path,  file_path, 'decoder.model'), map_location='cpu'))
        generator.load_state_dict(torch.load(os.path.join(args.output_path,  file_path, 'generator.model'), map_location='cpu'))

        freeze_params(embedding)
        freeze_params(encoder)
        freeze_params(decoder)
        freeze_params(generator)
    else:
        print("initial PSCon model...")
        init_params(embedding)
        init_params(encoder)
        init_params(decoder)
        init_params(generator)

    t1 = T1(embedding, encoder, args.hidden_size, PSCon_id2intent)
    init_params(t1.linear)
    t2 = T2(embedding, encoder, args.hidden_size, id2vocab)
    init_params(t2.linear)
    t3 = T3(embedding, encoder, args.hidden_size, PSCon_id2action)
    init_params(t3.linear)
    t4 = T4(embedding, encoder, args.hidden_size)
    init_params(t4.linear)
    t5 = T5(embedding, encoder, args.hidden_size)
    init_params(t5.linear)
    t6 = T6(embedding, decoder, generator, args.hidden_size, id2vocab)

    PSCon_model = PSConModel(t1, t2, t3, t4, t5, t6)
    PSCon_t1model = PSCon_T1Model(t1)
    PSCon_t2model = PSCon_T2Model(t2)
    PSCon_t3model = PSCon_T3Model(t3)
    PSCon_t4model = PSCon_T4Model(t4)
    PSCon_t5model = PSCon_T5Model(t5)
    PSCon_t6model = PSCon_T6Model(t3, t4, t5, t6)
    return PSCon_model, vocab2id, PSCon_intent2id, PSCon_action2id, PSCon_t1model, PSCon_t2model, PSCon_t3model, PSCon_t4model, PSCon_t5model, PSCon_t6model


def finetune(args):
    tokenizer = char_tokenizer()
    if args.mode == 'finetune-nokdconv':
        prefix = "nokdconv"
        save = "PSCon_nokdconv/"
        print("finetune with pretrain model without using kdconv dataset")
    elif args.mode == "finetune-noduconv":
        prefix = "noduconv"
        save = "PSCon_noduconv/"
        print("finetune with pretrain model without using duconv dataset")
    elif args.mode == "finetune-none":
        prefix = "no"
        save = "PSCon_nopretrain/"
        print("train without using pretrain dataset")
    elif args.mode[:8] == "finetune":
        prefix = "full"
        save = "PSCon_withpretrain/"
        print("finetune with full dataset")
    else:
        print("fault args.model")
        raise ValueError
    PSCon, vocab2id, PSCon_intent2id, PSCon_action2id, t1, t2, t3, t4, t5, t6 = load_pretrained_model(args, prefix)

    if args.mode == 'finetune-t1':
        PSCon_model = t1
        save = 'PSCon_t1/'
        print("finetune PSCon-t1")
    elif args.mode == 'finetune-t2':
        PSCon_model = t2
        save = 'PSCon_t2/'
        print("finetune PSCon-t2")
    elif args.mode == 'finetune-t3':
        PSCon_model = t3
        save = 'PSCon_t3/'
        print("finetune PSCon-t3")
    elif args.mode == 'finetune-t4':
        PSCon_model = t4
        save = 'PSCon_t4/'
        print("finetune PSCon-t4")
    elif args.mode == 'finetune-t5':
        PSCon_model = t5
        save = 'PSCon_t5/'
        print("finetune PSCon-t5")
    elif args.mode == 'finetune-t6':
        PSCon_model = t6
        save = 'PSCon_t6/'
        print("finetune PSCon-t6")
    elif args.mode == 'finetune':
        PSCon_model = PSCon
        save = 'PSCon_withpretrain/'
        print("finetune PSCon")
    elif args.mode == 'finetune-none':
        PSCon_model = PSCon
        save = "PSCon_nopretrain/"
        print("train PSCon-FULL")
    elif args.mode[:8] == 'finetune':
        PSCon_model = PSCon
        print("finetune PSCon-FULL")
    else:
        print("fault args.mode")
        raise ValueError
    print(count_parameters(PSCon_model))
    print("model save in ", os.path.join(args.output_path, save))
    dataset = PSConDataset([args.PSCon_train_conversation_file], [args.PSCon_document_file], vocab2id, PSCon_intent2id,
                          PSCon_action2id, tokenizer)
    print('PSCon training', 'data_size', dataset.len, 'gpu', args.num_gpus, 'epoch', args.epoch, 'batch_size',
          args.batch_size)
    optimizer = AdamW(PSCon_model.parameters(), lr=args.lr)
    bp_count = (args.epoch * dataset.len) / (args.num_gpus * args.batch_size)
    print('bp_count', bp_count)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(0.1 * (bp_count + 100)), T_mult=2, eta_min=1e-7)
    trainer = DataParallel(PSCon_model, optimizer, scheduler, args.local_rank)

    for i in range(args.epoch):
        print('epoch', i + 1)
        trainer.train_epoch('train', dataset, PSCon_collate_fn, args.batch_size, i + 1)
        if (i + 1) == 10:
            unfreeze_params(PSCon_model)
        trainer.serialize(i + 1, os.path.join(args.output_path, save))


def infer(args, prefix='test', epochs=[], folder="withpretrain"):
    """运行指定数据集的推理
    
    Args:
        args: 参数配置 
        prefix: 数据集前缀(train/test等)
        epochs: 指定epoch列表
        folder: 模型保存文件夹
    """
    tokenizer = char_tokenizer()
    
    # 解析任务类型
    task = None
    if args.mode.count('-') == 2:  # 如 infer-test-t1 
        task = args.mode.split('-')[-1]
    print("infer model task", task)
    # 构建对应的模型
    PSCon_model, vocab2id, PSCon_intent2id, PSCon_action2id = build_PSCon_model(args, task)

    # 选择数据集
    if prefix == 'train':
        print("infer train") 
        conversation_file = args.PSCon_train_conversation_file
    elif prefix == 'test':
        print("infer test")
        conversation_file = args.PSCon_test_conversation_file
    else:
        raise ValueError

    dataset = PSConDataset([conversation_file], [args.PSCon_document_file], vocab2id, PSCon_intent2id, PSCon_action2id,
                          tokenizer)

    print("infer data size ", dataset.len)
    trainer = DataParallel(PSCon_model, None, None, args.local_rank)

    convs = {}
    with codecs.open(conversation_file, encoding='utf-8') as f:
        for line in f:
            conv = json.loads(line)
            convs[conv[-1]['msg_id']] = conv

    model_path = "".join(["PSCon_", folder, "/"])
    file_path = "".join(["PSCon_", folder, "_infer_", prefix, "/"])
    if task:
        file_path = file_path[:-1] + f"_{task}/"
        
    print("model path", model_path)
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
            PSCon_model.load_state_dict(torch.load(model_file, map_location='cpu'))
            output = trainer.test_epoch('test', dataset, PSCon_collate_fn, args.batch_size)

            # 根据task类型保存对应的输出
            file_result = codecs.open(os.path.join(args.output_path, file_path, '.'.join([str(i+1), str(args.local_rank), 'json'])), "w", "utf-8")
            
            for j in range(output['id'].size(0)):
                conv = copy.deepcopy(convs[output['id'][j].item()])
                id = conv[-1]['msg_id']
                
                # 根据task保存对应的输出
                if task == 't1':
                    t1_output = PSCon_model.intent(output['t1_output'])
                    for k in range(len(conv)):
                        if conv[-k - 1]['role'] == 'user':
                            conv[-k - 1]['intent'] = t1_output[j][1:-1].split('-')
                            break
                elif task == 't2':
                    t2_output = PSCon_model.state(output['t2_output'])
                    conv[-1]['state'] = t2_output[j]
                elif task == 't3':
                    t3_output = PSCon_model.action(output['t3_output'])
                    conv[-1]['action'] = t3_output[j][1:-1].split('-')
                elif task == 't4':
                    t4_output = PSCon_model.query(output['t4_output'])
                    conv[-1]['selected_query'] = []
                    for index in range(len(t4_output[j])):
                        conv[-1]['selected_query'].append((dataset.query(id, index), t4_output[j][index].item()))
                    conv[-1]['query_ranking'] = sorted(conv[-1]['selected_query'], key=lambda x: x[1], reverse=True)
                elif task == 't5':
                    t5_output = PSCon_model.passage(output['t5_output'])
                    conv[-1]['selected_passage'] = []
                    for index in range(len(t5_output[j])):
                        conv[-1]['selected_passage'].append((dataset.passage(id, index), t5_output[j][index].item()))
                    conv[-1]['passage_ranking'] = sorted(conv[-1]['selected_passage'], key=lambda x: x[1], reverse=True)
                elif task == 't6':
                    t6_output = PSCon_model.response(output['t6_output'])
                    conv[-1]['response'] = t6_output[j]
                else:
                    # 全任务输出
                    t1_output = PSCon_model.intent(output['t1_output'])
                    t2_output = PSCon_model.state(output['t2_output'])
                    t3_output = PSCon_model.action(output['t3_output'])
                    t4_output = PSCon_model.query(output['t4_output'])
                    t5_output = PSCon_model.passage(output['t5_output'])
                    t6_output = PSCon_model.response(output['t6_output'])
                    
                    for k in range(len(conv)):
                        if conv[-k - 1]['role'] == 'user':
                            conv[-k - 1]['intent'] = t1_output[j][1:-1].split('-')
                            break
                            
                    conv[-1]['state'] = t2_output[j]
                    conv[-1]['action'] = t3_output[j][1:-1].split('-')
                    
                    conv[-1]['selected_query'] = []
                    for index in range(len(t4_output[j])):
                        conv[-1]['selected_query'].append((dataset.query(id, index), t4_output[j][index].item()))
                    conv[-1]['query_ranking'] = sorted(conv[-1]['selected_query'], key=lambda x: x[1], reverse=True)
                    
                    conv[-1]['selected_passage'] = []
                    for index in range(len(t5_output[j])):
                        conv[-1]['selected_passage'].append((dataset.passage(id, index), t5_output[j][index].item()))
                    conv[-1]['passage_ranking'] = sorted(conv[-1]['selected_passage'], key=lambda x: x[1], reverse=True)
                    
                    conv[-1]['response'] = t6_output[j]
                    
                file_result.write(json.dumps(conv, ensure_ascii=False) + os.linesep)
            file_result.close()


def eval(args, prefix='test', epochs=[], folder="withpretrain"):
    """评估模型在指定数据集上的表现
    
    Args:
        args: 参数配置
        prefix: 数据集前缀(train/test等) 
        epochs: 指定epoch列表
        folder: 模型保存文件夹
    """
    if args.local_rank != 0:
        return
        
    tokenizer = char_tokenizer()
    
    # 解析任务类型
    task = None 
    if args.mode.count('-') == 2:  # 如 eval-test-t1
        task = args.mode.split('-')[-1]
        
    # 选择评估数据集
    if prefix == 'train':
        conversation_file = args.PSCon_train_conversation_file 
    elif prefix == 'test':
        conversation_file = args.PSCon_test_conversation_file
    else:
        raise ValueError
        
    # 构建文件路径
    if task:
        # 单任务评估,如 PSCon_t1_infer_test_t1/
        file_path = f"PSCon_{task}_infer_{prefix}_{task}/"
    else:
        # 全任务评估
        file_path = f"PSCon_{folder}_infer_{prefix}/"
        
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
            rs_files = [os.path.join(args.output_path, file_path, '.'.join([str(i + 1), str(g), 'json'])) for g in range(4)]
            gt_files = [conversation_file]
            print(rs_files)
            print(gt_files)
            result = evaluate(rs_files, gt_files, tokenizer, task)
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

    parser.add_argument("--PSCon_train_file", type=str,
                        default='data/PSCon/conversation_train_line.json')
    parser.add_argument("--PSCon_test_file", type=str,
                        default='data/PSCon/conversation_test_line.json')
    parser.add_argument("--PSCon_intent", type=str,
                        default=os.path.join(dir_path, 'data/PSCon/intent.txt'))
    parser.add_argument("--PSCon_action", type=str,
                        default=os.path.join(dir_path, 'data/PSCon/action.txt'))
    parser.add_argument("--PSCon_document_file", type=str,
                        default=os.path.join(dir_path, 'data/PSCon/document_line.json'))
    parser.add_argument("--PSCon_train_conversation_file", type=str,
                        default=os.path.join(dir_path, 'data/PSCon/PSCon_train.json'))
    parser.add_argument("--PSCon_test_conversation_file", type=str,
                        default=os.path.join(dir_path, 'data/PSCon/PSCon_test.json'))

    parser.add_argument("--vocab", type=str, default=os.path.join(dir_path, 'data/vocab.txt'))

    parser.add_argument("--output_path", type=str, default='./output/')
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--pretrain_epoch", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=30)
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
    elif args.mode[:10] == 'infer-test':
        if args.mode == 'infer-test':
            folder = 'withpretrain'
        else:
            parts = args.mode.split('-')
            if len(parts) == 3:  # infer-test-folder
                folder = parts[2]
            elif len(parts) == 4:  # infer-test-folder-task
                folder = parts[2]
        infer(args, prefix='test', epochs=list(range(0, args.epoch)), folder=folder)
    elif args.mode[:9] == 'eval-test':
        if args.mode == 'eval-test':
            folder = 'withpretrain'
        else:
            parts = args.mode.split('-')
            if len(parts) == 3:  # eval-test-folder
                folder = parts[2]
            elif len(parts) == 4:  # eval-test-folder-task
                folder = parts[2]
        eval(args, 'test', epochs=list(range(0, args.epoch)), folder=folder)
