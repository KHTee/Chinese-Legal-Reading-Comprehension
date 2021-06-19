import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from tqdm import tqdm as ttqdm
from ujson import load as json_load
from util import *


def train_main(args):
  args.name = 'baseline'
  # Set up logging and devices
  args.save_dir = get_save_dir(args.save_dir, args.name, training=True)
  print(args.save_dir)
  log = get_logger(args.save_dir, args.name)
  tbx = SummaryWriter(args.save_dir)
  device, args.gpu_ids = get_available_devices()
  log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
  args.batch_size *= max(1, len(args.gpu_ids))

  # Set random seed
  log.info('Using random seed {}...'.format(args.seed))
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)

  # Get embeddings
  log.info('Loading embeddings...')
  word_vectors = torch_from_json(args.word_emb_file)
  char_vectors = torch_from_json(args.char_emb_file)
  # Get model
  log.info('Building model...')

  # model = BiDAF(word_vectors=word_vectors,
  #               char_vectors=char_vectors,
  #               hidden_size=args.hidden_size,
  #               drop_prob=args.drop_prob)
  # model = nn.DataParallel(model, args.gpu_ids)

  indicator = None
  while indicator is None:
    try:
      # connect
      model = BiDAF(word_vectors=word_vectors,
                    char_vectors=char_vectors,
                    hidden_size=args.hidden_size,
                    drop_prob=args.drop_prob)
      model = nn.DataParallel(model, args.gpu_ids)
      indicator = True
    except:
      pass

  if args.load_path:
    log.info('Loading checkpoint from {}...'.format(args.load_path))
    model, step = util.load_model(model, args.load_path, args.gpu_ids)
  else:
    step = 0
  model = model.to(device)
  model.train()
  ema = EMA(model, args.ema_decay)

  # Get saver
  saver = CheckpointSaver(args.save_dir,
                          max_checkpoints=args.max_checkpoints,
                          metric_name=args.metric_name,
                          maximize_metric=args.maximize_metric,
                          log=log)

  # Get optimizer and scheduler
  optimizer = optim.Adadelta(model.parameters(),
                             args.lr,
                             weight_decay=args.l2_wd)
  scheduler = sched.LambdaLR(optimizer, lambda s: 1.)    # Constant LR

  # Get data loader
  log.info('Building dataset...')
  train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
  train_loader = data.DataLoader(
      train_dataset,
      batch_size=args.batch_size,
      shuffle=False,    ################################True,
      num_workers=args.num_workers,
      collate_fn=collate_fn)
  dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
  dev_loader = data.DataLoader(dev_dataset,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=args.num_workers,
                               collate_fn=collate_fn)

  # Train
  log.info('Training...')
  steps_till_eval = args.eval_steps
  epoch = step // len(train_dataset)
  while epoch != args.num_epochs:
    epoch += 1
    log.info('Starting epoch {}...'.format(epoch))
    with torch.enable_grad(), ttqdm(
        total=len(train_loader.dataset)) as progress_bar:
      for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
        # Setup for forward
        cw_idxs = cw_idxs.to(device)
        qw_idxs = qw_idxs.to(device)
        cc_idxs = cc_idxs.to(device)
        qc_idxs = qc_idxs.to(device)
        batch_size = cw_idxs.size(0)
        # print('\nids:', ids)
        optimizer.zero_grad()

        # Forward
        log_p1, log_p2 = model(cc_idxs, qc_idxs, cw_idxs, qw_idxs)
        y1, y2 = y1.to(device), y2.to(device)
        loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
        loss_val = loss.item()
        # print('train_loader y1:', y1)
        # print('train_loader y2:', y2)
        # Backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step(step // batch_size)
        ema(model, step // batch_size)
        # print('train_loader loss:', loss)
        # Log info
        step += batch_size
        # progress_bar.update(batch_size)
        # progress_bar.set_postfix(epoch=epoch, NLL=loss_val)
        tbx.add_scalar('train/NLL', loss_val, step)
        tbx.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

        steps_till_eval -= batch_size

        if steps_till_eval <= 0:    ###indent rest
          steps_till_eval = args.eval_steps

          # Evaluate and save checkpoint
          log.info('Evaluating at step {}...'.format(step))
          ema.assign(model)
          results, pred_dict = evaluate(model, dev_loader, device,
                                        args.dev_eval_file, args.max_ans_len,
                                        args.use_squad_v2)
          # results, pred_dict = evaluate(model,
          #                               train_loader, #################dev_loader,
          #                               device,
          #                               args.train_eval_file, #####################args.dev_eval_file,
          #                               args.max_ans_len,
          #                               args.use_squad_v2)
          # print('results:', results)
          # print('pred_dict:', pred_dict)
          saver.save(step, model, results[args.metric_name], device)
          ema.resume(model)

          # Log to console
          results_str = ', '.join(
              '{}: {:05.2f}'.format(k, v) for k, v in results.items())
          log.info('Dev {}'.format(results_str))

          # Log to TensorBoard
          log.info('Visualizing in TensorBoard...')
          for k, v in results.items():
            tbx.add_scalar('dev/{}'.format(k), v, step)
          visualize(
              tbx,
              pred_dict=pred_dict,
              eval_path=args.train_eval_file,    #######args.dev_eval_file,
              step=step,
              split='dev',
              num_visuals=args.num_visuals)


def evaluate(model, data_loader, device, eval_file, max_len, use_squad_v2):
  nll_meter = AverageMeter()

  model.eval()
  pred_dict = {}
  with open(eval_file, 'r') as fh:
    gold_dict = json_load(fh)
  with torch.no_grad(), ttqdm(total=len(data_loader.dataset)) as progress_bar:
    for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
      # Setup for forward
      cw_idxs = cw_idxs.to(device)
      qw_idxs = qw_idxs.to(device)
      cc_idxs = cc_idxs.to(device)
      qc_idxs = qc_idxs.to(device)
      batch_size = cw_idxs.size(0)
      # print('\ndataloader ids', ids)
      # Forward
      log_p1, log_p2 = model(cc_idxs, qc_idxs, cw_idxs, qw_idxs)
      y1, y2 = y1.to(device), y2.to(device)
      # print('dataloader y1:', y1)
      # print('dataloader y2:', y2)
      # print('dataloader y1.shape:', y1.shape)
      # print('dataloader y2.shape:', y2.shape)
      loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
      # print('dataloader loss:', loss)
      nll_meter.update(loss.item(), batch_size)

      # Get F1 and EM scores
      p1, p2 = log_p1.exp(), log_p2.exp()
      # print('dataloader p1:', p1)
      # print('dataloader p2:', p2)
      # print('dataloader max_len:', max_len)
      starts, ends = discretize(p1, p2, max_len, use_squad_v2)
      # print('dataloader pred starts:', starts)
      # print('dataloader pred ends:', ends)

      # Log info
      progress_bar.update(batch_size)
      progress_bar.set_postfix(NLL=nll_meter.avg)

      preds, _ = convert_tokens(gold_dict, ids.tolist(), starts.tolist(),
                                ends.tolist(), use_squad_v2)
      # print('dataloader preds:', preds)
      pred_dict.update(preds)

  model.train()

  results = eval_dicts(gold_dict, pred_dict, use_squad_v2)
  results_list = [('NLL', nll_meter.avg), ('F1', results['F1']),
                  ('EM', results['EM'])]
  if use_squad_v2:
    results_list.append(('AvNA', results['AvNA']))
  results = OrderedDict(results_list)

  return results, pred_dict


# starts, ends = discretize(p1, p2, max_len, use_squad_v2)
def discretize(p_start, p_end, max_len=15, no_answer=False):
  """Discretize soft predictions to get start and end indices.
    Choose the pair `(i, j)` of indices that maximizes `p1[i] * p2[j]`
    subject to `i <= j` and `j - i + 1 <= max_len`.
    Args:
        p_start (torch.Tensor): Soft predictions for start index.
            Shape (batch_size, context_len).
        p_end (torch.Tensor): Soft predictions for end index.
            Shape (batch_size, context_len).
        max_len (int): Maximum length of the discretized prediction.
            I.e., enforce that `preds[i, 1] - preds[i, 0] + 1 <= max_len`.
        no_answer (bool): Treat 0-index as the no-answer prediction. Consider
            a prediction no-answer if `preds[0, 0] * preds[0, 1]` is greater
            than the probability assigned to the max-probability span.
    Returns:
        start_idxs (torch.Tensor): Hard predictions for start index.
            Shape (batch_size,)
        end_idxs (torch.Tensor): Hard predictions for end index.
            Shape (batch_size,)
    """
  if p_start.min() < 0 or p_start.max() > 1 \
          or p_end.min() < 0 or p_end.max() > 1:
    raise ValueError('Expected p_start and p_end to have values in [0, 1]')

  # Compute pairwise probabilities
  p_start = p_start.unsqueeze(dim=2)
  # print('p_start:', p_start)
  p_end = p_end.unsqueeze(dim=1)
  # print('p_end:', p_end)
  p_joint = torch.matmul(p_start, p_end)    # (batch_size, c_len, c_len)
  # print('p_joint:', p_joint.shape)
  # print('p_joint:', p_joint)

  # Restrict to pairs (i, j) such that i <= j <= i + max_len - 1
  c_len, device = p_start.size(1), p_start.device
  is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
  is_legal_pair -= torch.triu(torch.ones((c_len, c_len), device=device),
                              diagonal=max_len)
  if no_answer:
    # Index 0 is no-answer
    p_no_answer = p_joint[:, 0, 0].clone()
    is_legal_pair[0, :] = 0
    is_legal_pair[:, 0] = 0
  else:
    p_no_answer = None
  p_joint *= is_legal_pair

  # Take pair (i, j) that maximizes p_joint
  max_in_row, _ = torch.max(p_joint, dim=2)
  max_in_col, _ = torch.max(p_joint, dim=1)
  start_idxs = torch.argmax(max_in_row, dim=-1)
  # print('\ndiscretize dataloader start_idxs:', start_idxs)
  end_idxs = torch.argmax(max_in_col, dim=-1)
  # print('discretize dataloader end_idxs:', end_idxs)
  # print('discretize dataloader p_no_answer', p_no_answer)

  if no_answer:
    # Predict no-answer whenever p_no_answer > max_prob
    max_prob, _ = torch.max(max_in_col, dim=-1)
    # print('discretize dataloader max_prob:', max_prob)
    start_idxs[p_no_answer > max_prob] = 0
    # if tf.reduce_sum(tf.abs(start_idxs)) !=0:
    # print('start_idxs:', start_idxs)
    # else:
    #     print('\n\n ALL start_idxs ZEROES!!!!!!!!!!!!!!!!!!!!!!!!')
    end_idxs[p_no_answer > max_prob] = 0

  return start_idxs, end_idxs