import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
# import util

from args import get_test_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm as ttqdm
from ujson import load as json_load
from util import *
from train import discretize


def test_main(args):
  args.split = 'test'  ################################################### dev or test
  args.name = 'test'  ######################################################### dev or test
  print('load_path:',
        args.load_path)  # already specified in args.py get_test_args()

  # Set up logging
  args.save_dir = get_save_dir(args.save_dir, args.name, training=False)
  print('save_dir:', args.save_dir)

  log = get_logger(args.save_dir, args.name)
  log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
  device, gpu_ids = get_available_devices()
  args.batch_size *= max(1, len(gpu_ids))

  # Get embeddings
  log.info('Loading embeddings...')
  word_vectors = torch_from_json(args.word_emb_file)
  char_vectors = torch_from_json(args.char_emb_file)

  # Get model
  log.info('Building model...')
  model = BiDAF(word_vectors=word_vectors,
                char_vectors=char_vectors,
                hidden_size=args.hidden_size)
  model = nn.DataParallel(model, gpu_ids)
  log.info('Loading checkpoint from {}...'.format(args.load_path))
  model = load_model(model, args.load_path, gpu_ids, return_step=False)
  model = model.to(device)
  model.eval()

  # Get data loader
  log.info('Building dataset...')
  record_file = vars(args)['{}_record_file'.format(args.split)]
  dataset = SQuAD(record_file, args.use_squad_v2)
  data_loader = data.DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                collate_fn=collate_fn)

  # Evaluate
  log.info('Evaluating on {} split...'.format(args.split))
  nll_meter = AverageMeter()
  pred_dict = {}  # Predictions for TensorBoard
  sub_dict = {}  # Predictions for submission
  eval_file = vars(args)['{}_eval_file'.format(args.split)]
  with open(eval_file, 'r') as fh:
    gold_dict = json_load(fh)

  with torch.no_grad(), ttqdm(total=len(dataset)) as progress_bar:
    for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
      # Setup for forward
      cw_idxs = cw_idxs.to(device)
      qw_idxs = qw_idxs.to(device)
      cc_idxs = cc_idxs.to(device)
      qc_idxs = qc_idxs.to(device)
      batch_size = cw_idxs.size(0)

      # Forward
      log_p1, log_p2 = model(cc_idxs, qc_idxs, cw_idxs, qw_idxs)
      y1, y2 = y1.to(device), y2.to(device)
      loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
      nll_meter.update(loss.item(), batch_size)

      # Get F1 and EM scores
      p1, p2 = log_p1.exp(), log_p2.exp()
      starts, ends = discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

      # Log info
      progress_bar.update(batch_size)
      if args.split != 'test':
        # No labels for the test set, so NLL would be invalid
        progress_bar.set_postfix(NLL=nll_meter.avg)

      idx2pred, uuid2pred = convert_tokens(gold_dict, ids.tolist(),
                                           starts.tolist(), ends.tolist(),
                                           args.use_squad_v2)
      pred_dict.update(idx2pred)
      sub_dict.update(uuid2pred)

  # Log results (except for test set, since it does not come with labels)
  if args.split != 'test':
    results = eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
    results_list = [('NLL', nll_meter.avg), ('F1', results['F1']),
                    ('EM', results['EM'])]
    if args.use_squad_v2:
      results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    # Log to console
    results_str = ', '.join(
        '{}: {:05.2f}'.format(k, v) for k, v in results.items())
    log.info('{} {}'.format(args.split.title(), results_str))

    # Log to TensorBoard
    tbx = SummaryWriter(args.save_dir)
    visualize(tbx,
              pred_dict=pred_dict,
              eval_path=eval_file,
              step=0,
              split=args.split,
              num_visuals=args.num_visuals)

  # Write submission file
  sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
  log.info('Writing submission file to {}...'.format(sub_path))
  with open(sub_path, 'w') as csv_fh:
    csv_writer = csv.writer(csv_fh, delimiter=',')
    csv_writer.writerow(['Id', 'Predicted'])
    for uuid in sorted(sub_dict):
      csv_writer.writerow([uuid, sub_dict[uuid]])
