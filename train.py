import sys
import os.path
import argparse
import math
import json
# import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import config
import data
if config.model_type == 'gaolingGCN':
    import gaolingGCN_model as model
if config.model_type == 'gaoling':
    import gaoling_model as model
if config.model_type == 'baseline':
    import baseline_model as model
elif config.model_type == 'inter_intra':
    import inter_intra_model as model
elif config.model_type == 'ban':
    import ban_model as model
elif config.model_type == 'counting':
    import counting_model as model
elif config.model_type == 'graph':
    import graph_model as model
elif config.model_type == 'my':
    import my_model as model
import utils
# import os
# os.environ['CUDA_VISBLE_DIVECES']='0'
# torch.cuda.set_device(0)
def run(net, loader, optimizer, scheduler, tracker, train=False, has_answers=True, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    assert not (train and not has_answers)
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    # set learning rate decay policy
    if epoch < len(config.gradual_warmup_steps) and config.schedule_method == 'warm_up':
        utils.set_lr(optimizer, config.gradual_warmup_steps[epoch])
        utils.print_lr(optimizer, prefix, epoch)
    elif (epoch in config.lr_decay_epochs) and train and config.schedule_method == 'warm_up':
        utils.decay_lr(optimizer, config.lr_decay_rate)
        utils.print_lr(optimizer, prefix, epoch)
    else:
        utils.print_lr(optimizer, prefix, epoch)

    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    for v, q, a, b, idx, v_mask, q_mask, q_len in loader:
        var_params = {
            'requires_grad': False,
        }
        v = v.cuda()
        q = q.cuda()
        a = a.cuda()
        b = b.cuda()
        q_len = q_len.cuda()
        v_mask = v_mask.cuda()
        q_mask = q_mask.cuda()

        out = net(v, b, q, v_mask, q_mask, q_len)

        if has_answers:
            answer = utils.process_answer(a)
            loss = utils.calculate_loss(answer, out, method=config.loss_method)
            acc = utils.batch_accuracy(out, answer).data.cpu()
            print(loss)

        if train:
            optimizer.zero_grad()
            loss.backward()
            # print gradient
            if config.print_gradient: 
                utils.print_grad([(n, p) for n, p in net.named_parameters() if p.grad is not None])
            # clip gradient
            clip_grad_norm_(net.parameters(), config.clip_value)
            optimizer.step()
            if (config.schedule_method == 'batch_decay'): 
                scheduler.step()
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            if has_answers:
                accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        if has_answers:
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        if has_answers:
            accs = list(torch.cat(accs, dim=0))
        else:
            accs = []
        idxs = list(torch.cat(idxs, dim=0))
        #print('{} E{:03d}:'.format(prefix, epoch), ' Total num: ', len(accs))
        #print('{} E{:03d}:'.format(prefix, epoch), ' Average Score: ', float(sum(accs) / len(accs)))
        
        # log metrics to wandb
        # wandb.log({"acc": accs, "loss": loss})
        return answ, accs, idxs


def main():

    # torch.multiprocessing.set_start_method('spawn')

    # start a new wandb run to track this script
    # wandb.init(
    # # set the wandb project where this run will be logged
    # project="my-awesome-project",
    # # track hyperparameters and run metadata
    # config={
    # "learning_rate": 0.02,
    # "architecture": "CNN",
    # "dataset": "CIFAR-100",
    # "epochs": 10,}
    # )

    ####以下是train的原始代码

    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='*')
    parser.add_argument('--eval', dest='eval_only', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--trainval', action='store_true')
    parser.add_argument('--resume', nargs='*')
    parser.add_argument('--describe', type=str, default='describe your setting')
    args = parser.parse_args()

    print('-'*50)
    print(args)
    config.print_param()

    # set mannual seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    if args.test:   # False
        args.eval_only = True
    src = open('/home/gaoling/Projects/Projects/VQA2.0-Recent-Approachs-2018/'+ config.model_type + '_model.py').read()
    # model

    # 指定pth文件的名字or以当前时间作为pth文件名
    if args.name:   # args.name:[]
        name = ' '.join(args.name)
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") # time
    # pth文件的路径
    target_name = os.path.join('/home/gaoling/Projects/Projects/VQA2.0-Recent-Approachs-2018/logs', '{}.pth'.format(name))
    
    # 训练or验证，即保存model
    if not args.test:   # 只要不是测试，就保存model
        # target_name won't be used in test mode
        # target_name不会在test模式中使用
        print('will save to {}'.format(target_name))
    
    # 加载模型
    if args.resume:
        print(" loading model")
        logs = torch.load(' '.join(args.resume))
        # hacky way to tell the VQA classes that they should use the vocab without passing more params around
        #告诉VQA classes,他们应该使用vocab而不传递更多参数的hack方法
        data.preloaded_vocab = logs['vocab']
    # 会使得cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法。
    cudnn.benchmark = True  # 让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    
    if args.trainval:
        train_loader = data.get_loader(trainval=True)
    elif not args.eval_only:
        train_loader = data.get_loader(train=True)
    
    if args.trainval:
        pass # since we use the entire train val splits, we don't need val during training
            # 因为我们使用整个train val splits，所以我们在训练期间不需要val
    elif not args.test:
        val_loader = data.get_loader(val=True)
    else:
        val_loader = data.get_loader(test=True)
    
    question_keys = train_loader.dataset.vocab['question'].keys() if args.trainval else val_loader.dataset.vocab['question'].keys()
    # keys():获得字典中所有的键
    net = model.Net(question_keys)
    # print   glove weight shape:  torch.Size([19901, 300])
    #         word embed shape:  torch.Size([19902, 300])
    
    # net = nn.DataParallel(net).cuda()  # Support multiple GPUS  支持多gpu
    net = net.cuda()  # 
    print("Selection optimizer")
    select_optim = optim.Adamax if (config.optim_method == 'Adamax') else optim.Adam
    optimizer = select_optim([p for p in net.parameters() if p.requires_grad], lr=config.initial_lr, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5**(1 / config.lr_halflife))
    print("net.load_state_dict")
    if args.resume: # 加载
        net.load_state_dict(logs['weights'])
    print(net)
    tracker = utils.Tracker()   # 随时跟踪结果，同时可以使用监视器显示有关结果的信息
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        if not args.eval_only:
            run(net, train_loader, optimizer, scheduler, tracker, train=True, prefix='train', epoch=i)
        if not args.trainval:
            r = run(net, val_loader, optimizer, scheduler, tracker, train=False, prefix='val', epoch=i, has_answers=not args.test)
            # r = answ, accs, idxs
        else:
            r = [[-1], [-1], [-1]]  # dummy results 仿真结果
        
        if not args.test:   # 不是测试时
            # log metrics to wandb
            # wandb.log({"acc": r[1], "loss": loss})
            results = {
                'name': name,
                'tracker': tracker.to_dict(),
                'config': config_as_dict,
                # 'weights': net.module.state_dict(),
                'weights': net.state_dict(),
                'eval': {
                    'answers': r[0],
                    'accuracies': r[1],
                    'idx': r[2],
                },
                'vocab': val_loader.dataset.vocab if not args.trainval else train_loader.dataset.vocab,
                'src': src,
            }
            torch.save(results, target_name)
        else:   # 测试时
            # in test mode, save a results file in the format accepted by the submission server
            #在测试模式下，以提交服务器接受的格式保存结果文件
            answer_index_to_string = {a:  s for s, a in val_loader.dataset.answer_to_index.items()}
            results = []
            # r是run的输出，包括：answ, accs, idxs
            for answer, index in zip(r[0], r[2]):
                answer = answer_index_to_string[answer.item()]
                qid = val_loader.dataset.question_ids[index]
                entry = {
                    'question_id': qid,
                    'answer': answer,
                }
                results.append(entry)
            with open(config.result_json_path, 'w') as fd:
                json.dump(results, fd)

        if args.eval_only:
            break
    # [optional] finish the wandb run, necessary in notebooks
    # wandb.finish()

if __name__ == '__main__':
    main()
