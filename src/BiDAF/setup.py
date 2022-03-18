import numpy as np
import os
import spacy
import ujson as json
import urllib.request
import jieba

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm as ttqdm
from zipfile import ZipFile
import copy


def word_tokenize(tokens):
  return list(jieba.cut(tokens))


def convert_idx(text, tokens):
  current = 0
  spans = []
  for token in tokens:
    current = text.find(token, current)
    if current < 0:
      print("Token {} cannot be found".format(token))
      raise Exception()
    spans.append((current, current + len(token)))
    current += len(token)
  return spans


def process_file(filename, data_type, word_counter, char_counter):
  print("Pre-processing {} examples...".format(data_type))
  examples = []
  # eval_examples = {}
  eval_examples_intermediate = []
  total = 0
  with open(filename, "r") as fh:
    source = json.load(fh)
    for article in ttqdm(source["data"]):
      for para in article["paragraphs"]:
        # print('para:', para)
        context = para["context"].replace("''", '" ').replace("``", '" ')
        context_tokens = word_tokenize(context)
        context_chars = [list(token) for token in context_tokens]
        spans = convert_idx(context, context_tokens)
        for token in context_tokens:
          word_counter[token] += len(para["qas"])
          for char in token:
            char_counter[char] += len(para["qas"])
        for qa in para["qas"]:
          total += 1
          ques = qa["question"].replace("''", '" ').replace("``", '" ')
          ques_tokens = word_tokenize(ques)
          ques_chars = [list(token) for token in ques_tokens]
          for token in ques_tokens:
            word_counter[token] += 1
            for char in token:
              char_counter[char] += 1
          y1s, y2s = [], []
          answer_texts = []
          ###############################################################################
          # is_impossible = False
          is_impossible = qa['is_impossible']
          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError(
                "For training, each question should have exactly 1 answer.")


###############################################################################
          if is_impossible == 'false':
            for answer in qa["answers"]:
              answer_text = answer["text"]
              answer_span = []
              if answer_text in ['YES', 'NO']:
                # print('context_tokens', total)
                # print('qa:', qa)
                # print('answer:', answer)
                answer_texts.append(answer_text)
                y1 = -1
                y2 = -1
                y1s.append(y1)
                y2s.append(y2)
              elif answer_text == '':
                # print('para:', para)
                # print('context_tokens', total)
                # print('qa:', qa)
                # print('answer:', answer)
                answer_texts.append(answer_text)
                y1 = -1
                y2 = -1
                y1s.append(y1)
                y2s.append(y2)
              else:  # normal answers (i.e. found in context)
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer_text)
                answer_texts.append(answer_text)
                for idx, span in enumerate(spans):
                  if not (answer_end <= span[0] or answer_start >= span[1]):
                    answer_span.append(idx)
                y1, y2 = answer_span[0], answer_span[-1]
                y1s.append(y1)
                y2s.append(y2)
          else:  ### is_impossible == 'true'
            if qa["answers"]:
              for answer in qa[
                  "answers"]:  ## indent rest ####  dev and test have multiple answers
                # print('para:', para)
                # print('context_tokens', total)
                # print('qa:', qa)
                # print('answer:', answer)
                answer_text = ""
                answer_texts.append(answer_text)
                y1 = -1
                y2 = -1
                y1s.append(y1)
                y2s.append(y2)
                # print('y1s:', y1s)
                # print('y1s:', y1s)
            else:  # training can have an empty answer list, can throw error
              answer_text = ""
              answer_texts.append(answer_text)
              y1 = -1
              y2 = -1
              y1s.append(y1)
              y2s.append(y2)

          example = {
              "context_tokens": context_tokens,
              "context_chars": context_chars,
              "ques_tokens": ques_tokens,
              "ques_chars": ques_chars,
              "y1s": y1s,
              "y2s": y2s,
              "id": total
          }
          examples.append(example)
          # print('\npara:', para)
          # print('example:', example)
          eval_example = {
              "context": context,
              "question": ques,
              "spans": spans,
              "answers": answer_texts,
              "uuid": qa["id"],
              ################# newly added
              "impossible": is_impossible
          }
          eval_examples_intermediate.append(eval_example)

    print("{} questions in total".format(len(examples)))
  eval_examples = {
      str(index + 1): value
      for index, value in enumerate(eval_examples_intermediate)
  }
  return examples, eval_examples


def get_embedding(counter,
                  data_type,
                  limit=-1,
                  emb_file=None,
                  vec_size=None,
                  num_vectors=None):
  print("Pre-processing {} vectors...".format(data_type))
  embedding_dict = {}
  filtered_elements = [k for k, v in counter.items() if v > limit]
  if emb_file is not None:
    assert vec_size is not None
    with open(emb_file, "r", encoding="utf-8") as fh:
      for line in ttqdm(fh, total=num_vectors):
        array = line.split()
        word = "".join(array[0:-vec_size])
        vector = list(map(float, array[-vec_size:]))
        if word in counter and counter[word] > limit:
          embedding_dict[word] = vector
    print("{} / {} tokens have corresponding {} embedding vector".format(
        len(embedding_dict), len(filtered_elements), data_type))
  else:
    assert vec_size is not None
    for token in filtered_elements:
      embedding_dict[token] = [
          np.random.normal(scale=0.1) for _ in range(vec_size)
      ]
    print("{} tokens have corresponding {} embedding vector".format(
        len(filtered_elements), data_type))

  NULL = "--NULL--"
  OOV = "--OOV--"
  token2idx_dict = {
      token: idx for idx, token in enumerate(embedding_dict.keys(), 2)
  }
  token2idx_dict[NULL] = 0
  token2idx_dict[OOV] = 1
  embedding_dict[NULL] = [0. for _ in range(vec_size)]
  embedding_dict[OOV] = [0. for _ in range(vec_size)]
  idx2emb_dict = {
      idx: embedding_dict[token] for token, idx in token2idx_dict.items()
  }
  emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
  return emb_mat, token2idx_dict


def trim_context(args,
                 examples_OLD,
                 eval_examples_OLD,
                 is_test=False,
                 is_training=False):
  print("Trimming context...")
  para_limit = args.test_para_limit if is_test else args.para_limit  # 400

  counter = 0
  trim_counter = 0

  examples_NEW = [None] * len(examples_OLD)
  # eval_examples_NEW = {}
  eval_examples_NEW_intermediate = [None] * len(
      examples_OLD
  )  # append as list and then turn into dict using index+1 as key
  # this has to be done because for some reason, the dict can be out of order when running on a large dataset

  for i in range(len(examples_OLD)):
    ############################
    #### Retrieve old fields
    ############################
    # print('\n',i)
    example = examples_OLD[i]
    context_tokens = example['context_tokens']
    context_chars = example['context_chars']
    ques_tokens = example['ques_tokens']
    ques_chars = example['ques_chars']
    y1s = example['y1s']
    y2s = example['y2s']
    id = example['id']
    eval_example = eval_examples_OLD[str(i + 1)]
    old_span = eval_example['spans']
    answer_text_list = eval_example['answers']
    answer_text = answer_text_list[0]
    answer_start = example['y1s'][0]
    answer_end = example['y2s'][0]

    ##############################################################################
    #### Append YES/NO to end of context with < 400 tokens (a token may contain multiple characters)
    #### Train and Dev only
    ##############################################################################
    if len(example["context_tokens"]) < para_limit and not is_test:
      # print('Less than limit')
      # print('context_tokens:', context_tokens[-5:])
      # print('context_chars:', context_chars[-5:])
      # print('answer_start:', answer_start)
      if answer_start == -1 and answer_text in ['YES', 'NO']:
        # print('HERE')
        # print(i)
        # print('answer_text:', answer_text)
        context_tokens_NEW = copy.deepcopy(
            examples_OLD[i]["context_tokens"])  # select all tokens
        context_chars_NEW = [list(token) for token in context_tokens_NEW]
        context_tokens_NEW.append(
            answer_text)  # append 'YES' or 'NO' at end of context
        # print('context_tokens:', context_tokens_NEW[-5:])
        context_chars_NEW.append(
            [answer_text])  # append ['YES'] or ['NO'] at end of context_chars
        # print('context_chars:', context_chars_NEW[-5:])
        answer_text_idx = len(context_tokens_NEW) - 1
        if is_training:
          y1s = [answer_text_idx]
          y2s = [answer_text_idx]
        else:
          y1s = [answer_text_idx] * 3
          y2s = [answer_text_idx] * 3
        span_NEW = copy.deepcopy(old_span)
        span_NEW.append(
            (span_NEW[-1][-1], span_NEW[-1][-1] + 1))  # for 'YES' or 'NO'
        example_NEW = {
            "context_tokens": context_tokens_NEW,
            "context_chars": context_chars_NEW,
            "ques_tokens": examples_OLD[i]["ques_tokens"],
            "ques_chars": examples_OLD[i]["ques_chars"],
            "y1s": y1s,
            "y2s": y2s,
            "id": examples_OLD[i]['id']
        }
        examples_NEW[i] = example_NEW
        context = ''.join(map(str, context_tokens_NEW))
        eval_example_NEW = {
            "context": context,
            "question": eval_examples_OLD[str(i + 1)]['question'],
            "spans": span_NEW,
            "answers": eval_examples_OLD[str(i + 1)]['answers'],
            "uuid": eval_examples_OLD[str(i + 1)]['uuid'],
            "impossible": eval_examples_OLD[str(i + 1)]['impossible']
        }
        eval_examples_NEW_intermediate[i] = eval_example_NEW
        # print('context_tokens:', example_NEW["context_tokens"][-5:])
        # print('context_chars:', example_NEW["context_chars"][-5:])
      else:
        # print('NORMAL')
        example_NEW = {
            "context_tokens": examples_OLD[i]["context_tokens"],
            "context_chars": examples_OLD[i]["context_chars"],
            "ques_tokens": examples_OLD[i]["ques_tokens"],
            "ques_chars": examples_OLD[i]["ques_chars"],
            "y1s": examples_OLD[i]["y1s"],
            "y2s": examples_OLD[i]["y2s"],
            "id": examples_OLD[i]['id']
        }
        examples_NEW[i] = example_NEW
        eval_example_NEW = {
            "context": eval_examples_OLD[str(i + 1)]['context'],
            "question": eval_examples_OLD[str(i + 1)]['question'],
            "spans": eval_examples_OLD[str(i + 1)]['spans'],
            "answers": eval_examples_OLD[str(i + 1)]['answers'],
            "uuid": eval_examples_OLD[str(i + 1)]['uuid'],
            "impossible": eval_examples_OLD[str(i + 1)]['impossible']
        }
        eval_examples_NEW_intermediate[i] = eval_example_NEW
        # print('context_tokens:', example_NEW["context_tokens"][-5:])
        # print('context_chars:', example_NEW["context_chars"][-5:])
    ##############################################################################
    #### Append YES/NO to end of context with = 400 tokens (a token may contain multiple characters)
    #### Train and Dev only
    ##############################################################################
    elif len(example["context_tokens"]) == para_limit and not is_test:
      # print('Equal limit')
      if answer_start == -1 and answer_text in ['YES', 'NO']:
        context_tokens_NEW = copy.deepcopy(
            example["context_tokens"][:-1])  # select 399 tokens
        context_chars_NEW = [list(token) for token in context_tokens_NEW]
        context_tokens_NEW.append(
            answer_text)  # append 'YES' or 'NO' at end of context
        context_chars_NEW.append(
            [answer_text])  # append ['YES'] or ['NO'] at end of context_chars
        answer_text_idx = len(context_tokens_NEW) - 1
        if is_training:
          y1s = [answer_text_idx]
          y2s = [answer_text_idx]
        else:
          y1s = [answer_text_idx] * 3
          y2s = [answer_text_idx] * 3
        span_NEW = copy.deepcopy(old_span)
        span_NEW.append(
            (span_NEW[-1][-1], span_NEW[-1][-1] + 1))  # for 'YES' or 'NO'
        example_NEW = {
            "context_tokens": context_tokens_NEW,
            "context_chars": context_chars_NEW,
            "ques_tokens": examples_OLD[i]["ques_tokens"],
            "ques_chars": examples_OLD[i]["ques_chars"],
            "y1s": y1s,
            "y2s": y2s,
            "id": examples_OLD[i]['id']
        }
        examples_NEW[i] = example_NEW
        context = ''.join(map(str, context_tokens_NEW))
        eval_example_NEW = {
            "context": context,
            "question": eval_examples_OLD[str(i + 1)]['question'],
            "spans": span_NEW,
            "answers": eval_examples_OLD[str(i + 1)]['answers'],
            "uuid": eval_examples_OLD[str(i + 1)]['uuid'],
            "impossible": eval_examples_OLD[str(i + 1)]['impossible']
        }
        eval_examples_NEW_intermediate[i] = eval_example_NEW
      else:
        # print('NORMAL')
        example_NEW = {
            "context_tokens": examples_OLD[i]["context_tokens"],
            "context_chars": examples_OLD[i]["context_chars"],
            "ques_tokens": examples_OLD[i]["ques_tokens"],
            "ques_chars": examples_OLD[i]["ques_chars"],
            "y1s": examples_OLD[i]["y1s"],
            "y2s": examples_OLD[i]["y2s"],
            "id": examples_OLD[i]['id']
        }
        examples_NEW[i] = example_NEW
        eval_example_NEW = {
            "context": eval_examples_OLD[str(i + 1)]['context'],
            "question": eval_examples_OLD[str(i + 1)]['question'],
            "spans": eval_examples_OLD[str(i + 1)]['spans'],
            "answers": eval_examples_OLD[str(i + 1)]['answers'],
            "uuid": eval_examples_OLD[str(i + 1)]['uuid'],
            "impossible": eval_examples_OLD[str(i + 1)]['impossible']
        }
        eval_examples_NEW_intermediate[i] = eval_example_NEW
        # print('context_tokens:', examples_NEW[i]["context_tokens"][-5:-1])
        # print('context_chars:', examples_NEW[i]["context_chars"][-5: -1])
    ##############################################################################
    #### Trim context with more than 400 tokens (a token may contain multiple characters)
    #### Train and Dev only
    ##############################################################################
    elif len(example["context_tokens"]) > para_limit and not is_test:
      # print('NEED TO TRIM!!!!!!!!!!!!!!!!!')
      trim_counter += 1
      original_context_list = copy.deepcopy(examples_OLD[i]["context_tokens"])

      # Select the span containing the answer
      # for Dev, use the first answer to trim context
      if answer_start != -1:
        answer_spans_len = answer_end - answer_start + 1
        # print('answer_doc_spans_len:', answer_spans_len) # Ex: 3
        token_list = list(range(answer_start, answer_end +
                                1))  # generate consecutive numbered list
        # print('token_list:', token_list) # Ex: [60, 61, 62]

        # Trim the context to contain answer.
        # Suppose 400 is the max context token length and answer span is from index 60 to 62.
        if token_list[-1] <= para_limit:
          # print(11111111111111111111)
          # if answer's end index (i.e. 62) is smaller than limit (i.e. 400)
          # no need to re-index context and answers (since answer is already contained within the limit)
          # just select span from 0 to 400
          context_tokens_NEW = copy.deepcopy(examples_OLD[i]["context_tokens"])
          context_tokens_NEW = context_tokens_NEW[0:para_limit]
          context_chars_NEW = [list(token) for token in context_tokens_NEW]

          # some y1s may contain one or two answers with start index beyond the limit
          rechecked_y1s = []
          rechecked_y2s = []
          replacement_answer_y1y2 = []
          for answer in list(
              zip(example['y1s'], example['y2s']
                 )):  # get a list of valid answers [(y1a, y2a), (y1b, y2b)]
            if 0 <= answer[0] < para_limit and 0 < answer[-1] < para_limit:
              replacement_answer_y1y2.append((answer[0], answer[-1]))
          # print('replacement_answer_y1y2:', replacement_answer_y1y2)
          if is_training:
            if 0 <= answer_start < para_limit and 0 <= answer_end < para_limit:  # if initially valid answer
              rechecked_y1s.append(answer_start)
              rechecked_y2s.append(answer_end)
            else:
              rechecked_y1s.append(-1)
              rechecked_y2s.append(-1)
          else:  # dev
            for j in range(0, 3):  # for each of the three answers
              # print('answer:', j)
              if 0 <= example['y1s'][j] < para_limit and 0 <= example['y2s'][
                  j] < para_limit:  # if initially valid answer
                # print('case 1')
                rechecked_y1s.append(example['y1s'][j])
                rechecked_y2s.append(example['y2s'][j])
                # print(rechecked_y1s)
                # print(rechecked_y2s)
              else:
                if not replacement_answer_y1y2:  # if no valid replacement answer, then assume blank answer
                  # print('case 2')
                  rechecked_y1s.append(-1)
                  rechecked_y2s.append(-1)
                  # print(rechecked_y1s)
                  # print(rechecked_y2s)
                else:  # if there is valid replacement answer
                  # print('case 3')
                  rechecked_y1s.append(
                      replacement_answer_y1y2[0]
                      [0])  # replace with the first valid answer's start idx
                  rechecked_y2s.append(
                      replacement_answer_y1y2[0]
                      [-1])  # replace with the first valid answer's end idx
                  # print(rechecked_y1s)
                  # print(rechecked_y2s)
          y1s_NEW = rechecked_y1s
          y2s_NEW = rechecked_y2s
          # print('33 here')
          example_NEW = {
              "context_tokens": context_tokens_NEW,
              "context_chars": context_chars_NEW,
              "ques_tokens": examples_OLD[i]["ques_tokens"],
              "ques_chars": examples_OLD[i]["ques_chars"],
              "y1s": y1s_NEW,
              "y2s": y2s_NEW,
              "id": examples_OLD[i]['id']
          }
          # print('i:', i)
          examples_NEW[i] = example_NEW
          context_NEW = ''.join(map(str, context_tokens_NEW))
          span_NEW = convert_idx(context_NEW, context_tokens_NEW)
          eval_example_NEW = {
              "context": context_NEW,
              "question": eval_examples_OLD[str(i + 1)]['question'],
              "spans": span_NEW,
              "answers": eval_examples_OLD[str(i + 1)]['answers'],
              "uuid": eval_examples_OLD[str(i + 1)]['uuid'],
              "impossible": eval_examples_OLD[str(i + 1)]['impossible']
          }
          eval_examples_NEW_intermediate[i] = eval_example_NEW
          counter += 1
          # print('context_tokens:', context_tokens)
          # print('len(context_tokens):', len(context_tokens))
          # print('context_chars:', context_chars)
          # print('y1s:', y1s)
          # print('y2s:', y2s)
        else:
          # if answer window span is partially or even all located after the first 400 tokens
          ###### need to re-index context AND answers (i.e. the current end index would be re-indexed as 399)
          # print(222222222222222222222)
          # re-index context
          orig_to_tokens = dict(enumerate(example["context_tokens"]))
          # print('orig_to_tokens:', orig_to_tokens)
          new_tokens_indices_list = list(
              range(answer_end - para_limit + 1, answer_end + 1))
          # print('new_tokens_indices_list:', new_tokens_indices_list)
          # print(len(new_tokens_indices_list))

          trimmed_orig_to_tokens = {
              idx: orig_to_tokens[idx] for idx in new_tokens_indices_list
          }
          # print('trimmed_orig_to_tokens:', trimmed_orig_to_tokens)
          reindexed_context = {(k - (answer_end - para_limit + 1)): v
                               for k, v in trimmed_orig_to_tokens.items()}
          # print('reindexed_context:', reindexed_context)

          # re-index all three answers
          reindexed_y1s = []
          reindexed_y2s = []
          for answer in list(zip(example['y1s'], example['y2s'])):
            # print(answer)
            if answer[0] != -1:
              answer_indexed_dict = {
                  idx: orig_to_tokens[idx]
                  for idx in list(range(answer[0], answer[1] + 1))
              }
              # print(answer_indexed_dict)
              reindexed_answer = {(k - (answer_end - para_limit + 1)): v
                                  for k, v in answer_indexed_dict.items()}
              # print(reindexed_answer)
              new_y1s = list(reindexed_answer.keys())[0]
              new_y2s = list(reindexed_answer.keys())[-1]
              reindexed_y1s.append(new_y1s)
              reindexed_y2s.append(new_y2s)
            else:  #answer[0] == -1:
              reindexed_y1s.append(-1)
              reindexed_y2s.append(-1)
          # print('reindexed_y1s:', reindexed_y1s)
          # print('reindexed_y2s:', reindexed_y2s)

          # some y1s may contain one or two answers with start index beyond the limit
          rechecked_y1s = []
          rechecked_y2s = []
          replacement_answer_y1y2 = []
          for answer in list(
              zip(reindexed_y1s, reindexed_y2s
                 )):  # get a list of valid answers [(y1a, y2a), (y1b, y2b)]
            if 0 <= answer[0] < para_limit and 0 < answer[
                -1] < para_limit:  # answer[0] = start idx, answer[-1] = end idx
              replacement_answer_y1y2.append((answer[0], answer[-1]))
          if is_training:
            if 0 <= answer_start < para_limit and 0 <= answer_end < para_limit:  # if initially valid answer
              rechecked_y1s.append(answer_start)
              rechecked_y2s.append(answer_end)
            else:
              rechecked_y1s.append(-1)
              rechecked_y2s.append(-1)
          else:  # dev
            for j in range(0, 3):  # for each of the three answers
              if 0 <= reindexed_y1s[j] < para_limit and 0 <= reindexed_y2s[
                  j] < para_limit:  # if initially valid answer
                rechecked_y1s.append(reindexed_y1s[j])
                rechecked_y2s.append(reindexed_y2s[j])
              else:
                if not replacement_answer_y1y2:  # if no valid replacement answer, then assume blank answer
                  rechecked_y1s.append(-1)
                  rechecked_y2s.append(-1)
                else:  # if there is valid replacement answer
                  rechecked_y1s.append(
                      replacement_answer_y1y2[0]
                      [0])  # replace with the first valid answer's start idx
                  rechecked_y2s.append(
                      replacement_answer_y1y2[0]
                      [-1])  # replace with the first valid answer's end idx

          context_tokens_NEW = list(reindexed_context.values())
          # print('context_token_NEW:', len(context_tokens_NEW))
          context_chars_NEW = [list(token) for token in context_tokens_NEW]
          y1s_NEW = rechecked_y1s
          y2s_NEW = rechecked_y2s
          # print('33 here')
          example_NEW = {
              "context_tokens": context_tokens_NEW,
              "context_chars": context_chars_NEW,
              "ques_tokens": examples_OLD[i]["ques_tokens"],
              "ques_chars": examples_OLD[i]["ques_chars"],
              "y1s": y1s_NEW,
              "y2s": y2s_NEW,
              "id": examples_OLD[i]['id']
          }
          # print('i:', i)
          examples_NEW[i] = example_NEW
          context_NEW = ''.join(map(str, context_tokens_NEW))
          span_NEW = convert_idx(context_NEW, context_tokens_NEW)
          eval_example_NEW = {
              "context": context_NEW,
              "question": eval_examples_OLD[str(i + 1)]['question'],
              "spans": span_NEW,
              "answers": eval_examples_OLD[str(i + 1)]['answers'],
              "uuid": eval_examples_OLD[str(i + 1)]['uuid'],
              "impossible": eval_examples_OLD[str(i + 1)]['impossible']
          }
          eval_examples_NEW_intermediate[i] = eval_example_NEW
          counter += 1
      else:  # if answer start = -1 (first answer)
        answer_text_list = eval_examples_OLD[str(
            i + 1
        )]['answers']  ###################################################################################
        # print('original_context_list:', original_context_list)
        answer_text = answer_text_list[0]
        if answer_text in ['YES', 'NO']:
          # print(3333333333333333)
          context_tokens_NEW = example["context_tokens"][
              0:para_limit - 1]  # select the first 399 tokens
          # print('len(context_tokens):',len(context_tokens))
          context_chars_NEW = [list(token) for token in context_tokens_NEW]
          context_tokens_NEW.append(
              answer_text_list[0]
          )  # append Yes or No at end of context (now 400 tokens [0:399])
          # print('len(context_tokens):',len(context_tokens))
          # print('context_tokens[399]:',context_tokens[399])
          context_chars_NEW.append([answer_text
                                   ])  # append Yes or No at end of context
          # print('context_chars[399]:',context_chars[399])
          if is_training:
            y1s_NEW = [399]
            y2s_NEW = [399]
          else:
            y1s_NEW = [399, 399, 399]
            y2s_NEW = [399, 399, 399]
          span_NEW = copy.deepcopy(
              eval_examples_OLD[str(i + 1)]['spans'][0:para_limit - 1])
          span_NEW.append(
              (span_NEW[-1][-1], span_NEW[-1][-1] + 1))  # for 'YES' or 'NO'

          example_NEW = {
              "context_tokens": context_tokens_NEW,
              "context_chars": context_chars_NEW,
              "ques_tokens": examples_OLD[i]["ques_tokens"],
              "ques_chars": examples_OLD[i]["ques_chars"],
              "y1s": y1s_NEW,
              "y2s": y2s_NEW,
              "id": examples_OLD[i]['id']
          }
          # print('i:', i)
          examples_NEW[i] = example_NEW
          context_NEW = ''.join(map(str, context_tokens_NEW))
          eval_example_NEW = {
              "context": context_NEW,
              "question": eval_examples_OLD[str(i + 1)]['question'],
              "spans": span_NEW,
              "answers": eval_examples_OLD[str(i + 1)]['answers'],
              "uuid": eval_examples_OLD[str(i + 1)]['uuid'],
              "impossible": eval_examples_OLD[str(i + 1)]['impossible']
          }
          eval_examples_NEW_intermediate[i] = eval_example_NEW
          counter += 1
        else:
          # print(4444444444444444444444444)  # blank answer
          context_tokens_NEW = copy.deepcopy(
              example["context_tokens"]
              [0:para_limit])  # select the first 400 tokens
          context_chars_NEW = [list(token) for token in context_tokens_NEW]
          y1s_NEW = copy.deepcopy(example['y1s'])
          y2s_NEW = copy.deepcopy(example['y2s'])
          # print('y2s:', y2s)
          span_NEW = copy.deepcopy(
              eval_examples_OLD[str(i + 1)]['spans'][0:para_limit])
          example_NEW = {
              "context_tokens": context_tokens_NEW,
              "context_chars": context_chars_NEW,
              "ques_tokens": examples_OLD[i]["ques_tokens"],
              "ques_chars": examples_OLD[i]["ques_chars"],
              "y1s": y1s_NEW,
              "y2s": y2s_NEW,
              "id": examples_OLD[i]['id']
          }
          # print('i:', i)
          examples_NEW[i] = example_NEW
          context_NEW = ''.join(map(str, context_tokens_NEW))
          eval_example_NEW = {
              "context": context_NEW,
              "question": eval_examples_OLD[str(i + 1)]['question'],
              "spans": span_NEW,
              "answers": eval_examples_OLD[str(i + 1)]['answers'],
              "uuid": eval_examples_OLD[str(i + 1)]['uuid'],
              "impossible": eval_examples_OLD[str(i + 1)]['impossible']
          }
          eval_examples_NEW_intermediate[i] = eval_example_NEW
          counter += 1
  eval_examples_NEW = {
      str(index + 1): value
      for index, value in enumerate(eval_examples_NEW_intermediate)
  }
  return examples_NEW, eval_examples_NEW


def is_answerable(example):
  return len(example['y2s']) > 0 and len(example['y1s']) > 0


def build_features(args,
                   examples,
                   data_type,
                   out_file,
                   word2idx_dict,
                   char2idx_dict,
                   is_test=False):
  para_limit = args.test_para_limit if is_test else args.para_limit
  ques_limit = args.test_ques_limit if is_test else args.ques_limit
  ans_limit = args.ans_limit
  char_limit = args.char_limit

  def drop_example(ex, is_test_=False):
    if is_test_:
      drop = False
    else:
      drop = len(ex["context_tokens"]) > para_limit or \
             len(ex["ques_tokens"]) > ques_limit or \
             (is_answerable(ex) and
              ex["y2s"][0] - ex["y1s"][0] > ans_limit)

    return drop

  print("Converting {} examples to indices...".format(data_type))
  total = 0
  total_ = 0
  meta = {}
  context_idxs = []
  context_char_idxs = []
  ques_idxs = []
  ques_char_idxs = []
  y1s = []
  y2s = []
  ids = []
  for n, example in ttqdm(enumerate(examples)):
    total_ += 1

    if drop_example(example, is_test):
      continue

    total += 1

    def _get_word(word):
      for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in word2idx_dict:
          return word2idx_dict[each]
      return 1

    def _get_char(char):
      if char in char2idx_dict:
        return char2idx_dict[char]
      return 1

    context_idx = np.zeros([para_limit], dtype=np.int32)
    context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idx = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

    for i, token in enumerate(example["context_tokens"]):
      context_idx[i] = _get_word(token)
    context_idxs.append(context_idx)

    for i, token in enumerate(example["ques_tokens"]):
      ques_idx[i] = _get_word(token)
    ques_idxs.append(ques_idx)

    for i, token in enumerate(example["context_chars"]):
      for j, char in enumerate(token):
        if j == char_limit:
          break
        context_char_idx[i, j] = _get_char(char)
    context_char_idxs.append(context_char_idx)

    for i, token in enumerate(example["ques_chars"]):
      for j, char in enumerate(token):
        if j == char_limit:
          break
        ques_char_idx[i, j] = _get_char(char)
    ques_char_idxs.append(ques_char_idx)

    if is_answerable(example):
      start, end = example["y1s"][-1], example["y2s"][-1]
    else:
      start, end = -1, -1

    y1s.append(start)
    y2s.append(end)
    ids.append(example["id"])

  np.savez(out_file,
           context_idxs=np.array(context_idxs),
           context_char_idxs=np.array(context_char_idxs),
           ques_idxs=np.array(ques_idxs),
           ques_char_idxs=np.array(ques_char_idxs),
           y1s=np.array(y1s),
           y2s=np.array(y2s),
           ids=np.array(ids))
  print("Built {} / {} instances of features in total".format(total, total_))
  meta["total"] = total
  return meta


def append_YesNo_Test(args, examples_OLD, eval_examples_OLD):
  print("Appending YES NO to test...")

  examples_NEW = [None] * len(examples_OLD)
  # eval_examples_NEW = {}
  eval_examples_NEW_intermediate = [None] * len(
      examples_OLD
  )  # append as list and then turn into dict using index+1 as key
  # this has to be done because for some reason, the dict can be out of order when running on a large dataset

  for i in range(len(examples_OLD)):
    # print('\n',i)
    example = examples_OLD[i]
    ##############################
    context_tokens = copy.deepcopy(examples_OLD[i]['context_tokens'])
    # print('context_tokens:', context_tokens)
    # print('original context_tokens:', len(context_tokens))
    context_chars = copy.deepcopy(examples_OLD[i]['context_chars'])
    # print('context_chars:', context_chars)
    # print('original context_chars:', len(context_chars))
    ques_tokens = copy.deepcopy(examples_OLD[i]['ques_tokens'])
    ques_chars = copy.deepcopy(examples_OLD[i]['ques_chars'])
    y1s = copy.deepcopy(examples_OLD[i]['y1s'])
    y2s = copy.deepcopy(examples_OLD[i]['y2s'])
    id = copy.deepcopy(examples_OLD[i]['id'])
    ##############################
    eval_example = copy.deepcopy(eval_examples_OLD[str(i + 1)])
    context = copy.deepcopy(eval_examples_OLD[str(i + 1)]['context'])
    question = copy.deepcopy(eval_examples_OLD[str(i + 1)]['question'])
    span = copy.deepcopy(eval_examples_OLD[str(i + 1)]['spans'])
    answers = copy.deepcopy(eval_examples_OLD[str(i + 1)]['answers'])
    uuid = copy.deepcopy(eval_examples_OLD[str(i + 1)]['uuid'])
    impossible = copy.deepcopy(eval_examples_OLD[str(i + 1)]['impossible'])
    ##############################

    answer_start = example['y1s'][0]
    answer_end = example['y2s'][0]
    answer_text_list = eval_example['answers']
    answer_text = answer_text_list[0]
    # print('answer_text:', answer_text)

    if answer_start == -1 and (answer_text in ['YES', 'NO']):
      # print('HERE')
      # print(answer_text)
      context_tokens_NEW = copy.deepcopy(
          examples_OLD[i]["context_tokens"])  # select all tokens
      # print('len(context_tokens):',len(context_tokens))
      context_chars_NEW = [list(token) for token in context_tokens_NEW]
      context_tokens_NEW.append(
          answer_text)  # append 'YES' or 'NO' at end of context
      # print('len(context_tokens):',len(context_tokens))
      # print('context_tokens_NEW[-1]:',context_tokens_NEW[-5:])
      context_chars_NEW.append(
          [answer_text])  # append ['YES'] or ['NO'] at end of context_chars
      # print('context_chars_NEW[-1]:',context_chars_NEW[-5:])
      answer_text_idx = len(context_tokens_NEW) - 1
      # print('context_tokens[-1]:', context_tokens[answer_text_idx])
      # print('context_chars[-1]:', context_chars[answer_text_idx])
      # print('answer_text_idx:', answer_text_idx)
      y1s_NEW = [None] * 3
      y2s_NEW = [None] * 3
      span_NEW = copy.deepcopy(eval_examples_OLD[str(i + 1)]['spans'])
      for j in range(len(y1s)):
        if answer_text_list[j] in ['YES', 'NO']:
          y1s_NEW[j] = answer_text_idx
          y2s_NEW[j] = answer_text_idx
        else:
          y1s_NEW[j] = y1s[j]
          y2s_NEW[j] = y2s[j]
      span_NEW.append(
          (span_NEW[-1][-1], span_NEW[-1][-1] + 1))  # for 'YES' or 'NO'
      example_NEW = {
          "context_tokens": context_tokens_NEW,
          "context_chars": context_chars_NEW,
          "ques_tokens": example['ques_tokens'],
          "ques_chars": example['ques_chars'],
          "y1s": y1s_NEW,
          "y2s": y2s_NEW,
          "id": example['id']
      }
      # print(i)
      examples_NEW[i] = example_NEW
      # print('final:', examples_NEW[i]["context_tokens"][-5:])
      # print('final:', examples_NEW[i]["context_chars"][-5:])
      context_NEW = ''.join(map(str, context_tokens_NEW))
      eval_example_NEW = {
          "context": context_NEW,
          "question": eval_example['question'],
          "spans": span_NEW,
          "answers": eval_example['answers'],
          "uuid": eval_example['uuid'],
          "impossible": eval_example['impossible']
      }
      eval_examples_NEW_intermediate[i] = eval_example_NEW
    else:  # normal answers: use original example and eval_example
      # print('normal')
      example_NEW = {
          "context_tokens": example['context_tokens'],
          "context_chars": example['context_chars'],
          "ques_tokens": example['ques_tokens'],
          "ques_chars": example['ques_chars'],
          "y1s": example['y1s'],
          "y2s": example['y2s'],
          "id": example['id']
      }
      # print(i)
      examples_NEW[i] = example_NEW
      eval_example_NEW = {
          "context": eval_example['context'],
          "question": eval_example['question'],
          "spans": eval_example['spans'],
          "answers": eval_example['answers'],
          "uuid": eval_example['uuid'],
          "impossible": eval_example['impossible']
      }
      eval_examples_NEW_intermediate[i] = eval_example_NEW

    # example_NEW = {"context_tokens": context_tokens,
    #               "context_chars": context_chars,
    #               "ques_tokens": examples_OLD[i]["ques_tokens"],
    #               "ques_chars": examples_OLD[i]["ques_chars"],
    #               "y1s": y1s,
    #               "y2s": y2s,
    #               "id": examples_OLD[i]['id']}
    # examples_NEW.append(example_NEW)
    # context = ''.join(map(str, context_tokens))
    # eval_example_NEW =  {"context": context,
    #                       "question": eval_examples_OLD[str(i+1)]['question'],
    #                       "spans": old_span,
    #                       "answers": eval_examples_OLD[str(i+1)]['answers'],
    #                       "uuid": eval_examples_OLD[str(i+1)]['uuid'],
    #                       "impossible": eval_examples_OLD[str(i+1)]['impossible']}
    # eval_examples_NEW_intermediate.append(eval_example_NEW)
  eval_examples_NEW = {
      str(index + 1): value
      for index, value in enumerate(eval_examples_NEW_intermediate)
  }
  return examples_NEW, eval_examples_NEW


def save(filename, obj, message=None):
  if message is not None:
    print("Saving {}...".format(message))
    with open(filename, "w") as fh:
      json.dump(obj, fh)


def pre_process(args):
  # Process training set and use it to decide on the word/character vocabularies
  word_counter, char_counter = Counter(), Counter()
  train_examples, train_eval = process_file(args.train_file, "train",
                                            word_counter, char_counter)
  ##############################################################
  train_examples_NEW, train_eval_NEW = trim_context(
      args,
      examples_OLD=train_examples,
      eval_examples_OLD=train_eval,
      is_test=False,
      is_training=True)
  ##############################################################
  word_emb_mat, word2idx_dict = get_embedding(word_counter,
                                              'word',
                                              emb_file=args.glove_file,
                                              vec_size=args.glove_dim,
                                              num_vectors=args.glove_num_vecs)
  char_emb_mat, char2idx_dict = get_embedding(char_counter,
                                              'char',
                                              emb_file=args.glove_file,
                                              vec_size=args.glove_dim)

  # Process dev and test sets
  dev_examples, dev_eval = process_file(args.dev_file, "dev", word_counter,
                                        char_counter)
  ##############################################################
  dev_examples_NEW, dev_eval_NEW = trim_context(args,
                                                examples_OLD=dev_examples,
                                                eval_examples_OLD=dev_eval,
                                                is_test=False,
                                                is_training=False)
  ##############################################################
  # build_features(args, train_examples, "train", args.train_record_file, word2idx_dict, char2idx_dict)
  build_features(args, train_examples_NEW, "train", args.train_record_file,
                 word2idx_dict, char2idx_dict)
  # dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)
  dev_meta = build_features(args, dev_examples_NEW, "dev", args.dev_record_file,
                            word2idx_dict, char2idx_dict)
  if args.include_test_examples:
    test_examples, test_eval = process_file(args.test_file, "test",
                                            word_counter, char_counter)
    ##############################################################
    test_examples_NEW, test_eval_NEW = append_YesNo_Test(
        args, test_examples, test_eval)
    ##############################################################
    save(args.test_eval_file, test_eval_NEW, message="test eval")  # test_eval
    test_meta = build_features(args,
                               test_examples_NEW,
                               "test",
                               args.test_record_file,
                               word2idx_dict,
                               char2idx_dict,
                               is_test=True)
    save(args.test_meta_file, test_meta, message="test meta")

  save(args.word_emb_file, word_emb_mat, message="word embedding")
  save(args.char_emb_file, char_emb_mat, message="char embedding")
  save(args.train_eval_file, train_eval_NEW, message="train eval")  # train_eval
  save(args.dev_eval_file, dev_eval_NEW, message="dev eval")  # dev_eval
  save(args.word2idx_file, word2idx_dict, message="word dictionary")
  save(args.char2idx_file, char2idx_dict, message="char dictionary")
  save(args.dev_meta_file, dev_meta, message="dev meta")
