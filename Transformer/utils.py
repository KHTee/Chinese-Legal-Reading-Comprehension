""" Utils """

import os
import json
import math
import logging
import collections

import torch
from absl import flags, app
import numpy as np
from torch.utils.data import TensorDataset
from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS


class Example(object):
  """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=None,
               q_type=None):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible
    self.q_type = q_type

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (self.qas_id)
    s += ", question_text: %s" % (self.question_text)
    s += ", doc_tokens: [%s]" % ("".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.end_position:
      s += ", end_position: %d" % (self.end_position)
    if self.is_impossible:
      s += ", is_impossible: %r" % (self.is_impossible)
    if self.q_type:
      s += ", question_type: %r" % (self.q_type)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               qas_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               cls_index,
               p_mask,
               paragraph_len,
               start_position=None,
               end_position=None,
               is_impossible=None,
               q_type=None):
    self.unique_id = unique_id
    self.qas_id = qas_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.cls_index = cls_index
    self.p_mask = p_mask
    self.paragraph_len = paragraph_len
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible
    self.q_type = q_type


def read_squad_examples(input_file, is_training):
  """Read a SQuAD json file into a list of Example."""
  with open(input_file, "r", encoding='utf-8') as reader:
    input_data = json.load(reader)["data"]

  examples = []
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      paragraph_text = paragraph["context"]
      doc_tokens = []
      char_to_word_offset = []
      for c in paragraph_text:
        # some text contain whitespace noise. Append without space.
        # eg. paragraph: 赔偿2 300元, answer: 赔偿2300元
        if not is_whitespace(c):
          doc_tokens.append(c)
        char_to_word_offset.append(len(doc_tokens) - 1)

      doc_tokens += ["[YES]", "[NO]"]

      for qa in paragraph["qas"]:
        qas_id = qa["id"]
        question_text = qa["question"]

        # convert is_impossible to bool.
        if qa["is_impossible"] == "true":
          qa["is_impossible"] = True
        else:
          qa["is_impossible"] = False

        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        q_type = None

        if is_training:
          is_impossible = qa["is_impossible"]

          if (len(qa["answers"]) != 1) and (not is_impossible):
            raise ValueError("Each ques should have only 1 ans.")

          if not is_impossible:
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            answer_offset = answer["answer_start"]
            answer_length = len(orig_answer_text)
            if orig_answer_text not in ["YES", "NO"]:
              start_position = char_to_word_offset[answer_offset]
              end_position = char_to_word_offset[answer_offset + answer_length -
                                                 1]

              # Only add answers where the text can be exactly recovered from the document.
              actual_text = "".join(doc_tokens[start_position:(end_position +
                                                               1)])
              cleaned_answer_text = "".join(
                  whitespace_tokenize(orig_answer_text))
              q_type = 0
              if actual_text.find(
                  cleaned_answer_text) == -1 and cleaned_answer_text not in [
                      "YES", "NO"
                  ]:
                logger.warning("Could not find answer: '%s' vs. '%s'",
                               actual_text, cleaned_answer_text)
                logger.warning(doc_tokens[start_position:(end_position + 1)])
                continue
            else:
              if orig_answer_text == "YES":
                orig_answer_text = "[YES]"
                start_position = len(doc_tokens) - 2
                end_position = len(doc_tokens) - 2
                q_type = 1

              if orig_answer_text == "NO":
                orig_answer_text = "[NO]"
                start_position = len(doc_tokens) - 1
                end_position = len(doc_tokens) - 1
                q_type = 2

          else:
            start_position = -1
            end_position = -1
            q_type = 3

        example = Example(qas_id=qas_id,
                          question_text=question_text,
                          doc_tokens=doc_tokens,
                          orig_answer_text=orig_answer_text,
                          start_position=start_position,
                          end_position=end_position,
                          is_impossible=is_impossible,
                          q_type=q_type)
        examples.append(example)
  return examples


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 doc_stride,
                                 max_query_length,
                                 is_training,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
  """Loads a data file into a list of `InputBatch`s."""
  unique_id = 1000000
  features = []
  for (example_index, example) in enumerate(examples):
    # if is_training and example_index > 5:
    #     break

    # unique_id = example.qas_id
    qas_id = example.qas_id
    q_type = example.q_type
    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      tok_to_orig_index.append(i)
      all_doc_tokens.append(token)

    yes_no_ques = example.orig_answer_text in ['YES', 'NO']
    tok_start_position = None
    tok_end_position = None

    if is_training:
      if example.is_impossible or yes_no_ques:
        tok_start_position = -1
        tok_end_position = -1
      else:
        tok_start_position = orig_to_tok_index[example.start_position]

        if example.end_position < len(example.doc_tokens) - 1:
          tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
          tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

    max_tokens_for_doc = max_seq_length - len(query_tokens) - 5

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset

      if length > max_tokens_for_doc:
        length = max_tokens_for_doc

      doc_spans.append(_DocSpan(start=start_offset, length=length))

      if start_offset + length == len(all_doc_tokens):
        break

      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      p_mask = []

      # CLS token at the beginning
      tokens.append(cls_token)
      segment_ids.append(cls_token_segment_id)
      p_mask.append(0)
      cls_index = 0

      # Query
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(sequence_a_segment_id)
        p_mask.append(1)

      # SEP token
      tokens.append(sep_token)
      segment_ids.append(sequence_a_segment_id)
      p_mask.append(1)

      # Paragraph
      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(sequence_b_segment_id)
        p_mask.append(0)
      paragraph_len = doc_span.length

      # SEP token
      tokens.append(sep_token)
      segment_ids.append(sequence_b_segment_id)
      p_mask.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(pad_token)
        input_mask.append(0)
        segment_ids.append(pad_token_segment_id)
        p_mask.append(1)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      span_is_impossible = example.is_impossible
      start_position = None
      end_position = None

      if is_training and not span_is_impossible:
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        if not (tok_start_position >= doc_start
                and tok_end_position <= doc_end):
          start_position = 0
          end_position = 0
          span_is_impossible = True
        else:
          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and span_is_impossible:
        start_position = cls_index
        end_position = cls_index
        doc_start = -1
        doc_offset = -1

      if (example_index < 20) and (FLAGS.verbose):
        logger.info("*** Example ***")

        logger.info("unique_id: %s" % (unique_id))
        logger.info("q_type: %s" % (q_type))
        logger.info("example_index: %s" % (example_index))
        logger.info("doc_span_index: %s" % (doc_span_index))
        logger.info("tokens: %s" % " ".join(tokens))
        logger.info("token_to_orig_map: %s" % " ".join(
            ["%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
        logger.info("token_is_max_context: %s" % " ".join(
            ["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if is_training and span_is_impossible:
          logger.info("impossible example")
        if is_training and not span_is_impossible:
          answer_text = " ".join(tokens[start_position:(end_position + 1)])
          logger.info("start_position: %d" % (start_position))
          logger.info("end_position: %d" % (end_position))
          logger.info("answer: %s" % (answer_text))

        logger.info("\n")

      features.append(
          InputFeatures(unique_id=unique_id,
                        qas_id=qas_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        cls_index=cls_index,
                        p_mask=p_mask,
                        paragraph_len=paragraph_len,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=span_is_impossible,
                        q_type=q_type))
      unique_id += 1

  return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""
  tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def load_and_cache_examples(input_file, tokenizer, is_training=True):
  """Get examples and features. Can either create new or load from cache."""
  features_cache = os.path.splitext(
      input_file)[0] + "_cache_features_{}".format(FLAGS.model_type)
  examples_cache = os.path.splitext(
      input_file)[0] + "_cache_examples_{}".format(FLAGS.model_type)

  if os.path.exists(examples_cache) and not FLAGS.overwrite_cache:
    logger.info("Loading examples from cached file %s", examples_cache)
    examples = torch.load(examples_cache)
  else:
    logger.info("Creating examples from dataset file at %s", input_file)
    examples = read_squad_examples(input_file=input_file,
                                   is_training=is_training)
    logger.info("Saving examples into cached file %s", examples_cache)
    torch.save(examples, examples_cache)

  if os.path.exists(features_cache) and not FLAGS.overwrite_cache:
    logger.info("Loading features from cached file %s", features_cache)
    features = torch.load(features_cache)
  else:
    logger.info("Creating features from dataset file at %s", input_file)
    features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=is_training)
    logger.info("Saving features into cached file %s", features_cache)
    torch.save(features, features_cache)

  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features],
                               dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in features],
                                dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                 dtype=torch.long)
  all_cls_index = torch.tensor([f.cls_index for f in features],
                               dtype=torch.long)
  all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
  if is_training:
    all_start_positions = torch.tensor([f.start_position for f in features],
                                       dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features],
                                     dtype=torch.long)

    all_q_type = torch.tensor([f.q_type for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_start_positions, all_end_positions, all_q_type,
                            all_cls_index, all_p_mask)
  else:
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_example_index, all_cls_index, all_p_mask)

  return dataset, examples, features


RawResult = collections.namedtuple(
    "RawResult", ["unique_id", "start_logits", "end_logits", "q_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      null_threshold):
  """Write final predictions to the json file and log-odds of null if needed."""
  logger.info("Writing predictions to: %s" % (output_prediction_file))
  logger.info("Writing nbest to: %s" % (output_nbest_file))

  _PrelimPrediction = collections.namedtuple("PrelimPrediction", [
      "feature_index", "start_index", "end_index", "start_logit", "end_logit",
      "q_type", "q_logit"
  ])

  _NbestPrediction = collections.namedtuple(
      "NbestPrediction",
      ["qas_id", "text", "start_logit", "end_logit", "q_logit"])

  all_predictions = collections.OrderedDict()
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  for (example_index, example) in enumerate(all_examples):
    features = example_index_to_features[example_index]

    prelim_predictions = []
    score_null = 1000000
    min_null_feature_index = 0    # the paragraph slice with min null score
    null_start_logit = 0    # the start logit at the slice with min null score
    null_end_logit = 0    # the end logit at the slice with min null score
    null_q_logit = 0

    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)

      feature_null_score = result.start_logits[0] + result.end_logits[0]
      if feature_null_score < score_null:
        score_null = feature_null_score
        min_null_feature_index = feature_index
        null_start_logit = result.start_logits[0]
        null_end_logit = result.end_logits[0]

      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=result.start_logits[start_index],
                                end_logit=result.end_logits[end_index],
                                q_type=int(np.argmax(result.q_logits)),
                                q_logit=result.q_logits))

    prelim_predictions.append(
        _PrelimPrediction(feature_index=min_null_feature_index,
                          start_index=0,
                          end_index=0,
                          start_logit=null_start_logit,
                          end_logit=null_end_logit,
                          q_type=int(np.argmax(result.q_logits)),
                          q_logit=result.q_logits))
    prelim_predictions = sorted(prelim_predictions,
                                key=lambda x: (x.start_logit + x.end_logit),
                                reverse=True)

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      qas_id = feature.qas_id
      if pred.start_index > 0:    # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = "".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = "".join(tok_text.split())
        orig_text = "".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text)

        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True

      nbest.append(
          _NbestPrediction(qas_id=qas_id,
                           text=final_text,
                           start_logit=pred.start_logit,
                           end_logit=pred.end_logit,
                           q_logit=pred.q_logit))

    if "" not in seen_predictions:
      nbest.append(
          _NbestPrediction(qas_id="null",
                           text="",
                           start_logit=null_start_logit,
                           end_logit=null_end_logit,
                           q_logit=null_q_logit))

      # In very rare edge cases we could only have single null prediction.
      # So we just create a nonce prediction in this case to avoid failure.
      if len(nbest) == 1:
        nbest.insert(
            0,
            _NbestPrediction(qas_id="empty",
                             text="empty",
                             start_logit=0.0,
                             end_logit=0.0,
                             q_logit=0.0))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(qas_id="empty",
                           text="empty",
                           start_logit=0.0,
                           end_logit=0.0,
                           q_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    # if no prediction, we take it as is_impossible with blank answer.
    if not best_non_null_entry:
      best_non_null_entry = nbest[0]

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["id"] = entry.qas_id
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      output["q_type"] = int(np.argmax(entry.q_logit))
      output["q_logit"] = entry.q_logit
      nbest_json.append(output)

    assert len(nbest_json) >= 1

    score_diff = score_null - best_non_null_entry.start_logit - (
        best_non_null_entry.end_logit)
    scores_diff_json[example.qas_id] = score_diff

    if score_diff > null_threshold:
      all_predictions[example.qas_id] = ("",
                                         int(
                                             np.argmax(
                                                 best_non_null_entry.q_logit)))
    else:
      all_predictions[example.qas_id] = (best_non_null_entry.text,
                                         int(
                                             np.argmax(
                                                 best_non_null_entry.q_logit)))

    all_nbest_json[example.qas_id] = nbest_json

  with open(output_nbest_file, "w") as writer:
    write_data = json.dumps(all_nbest_json, indent=4, ensure_ascii=False)
    writer.write(write_data + "\n")

  with open(output_null_log_odds_file, "w") as writer:
    write_data = json.dumps(scores_diff_json, indent=4, ensure_ascii=False)
    writer.write(write_data + "\n")

  # write JSON file CJRC submission format.
  pred_answer = []
  for k, v in all_predictions.items():
    ans_format = {}
    ans_format["id"] = k
    ans_format["q_type"] = v[1]

    # remove square bracket for YES/NO
    if v[0] == "[YES]":
      ans_format["answer"] = "YES"
    elif v[0] == "[NO]":
      ans_format["answer"] = "NO"
    else:
      ans_format["answer"] = v[0]

    pred_answer.append(ans_format)

  with open(output_prediction_file, "w+") as writer:
    write_data = json.dumps(pred_answer, indent=4, ensure_ascii=False)
    writer.write(write_data + "\n")


def get_final_text(pred_text, orig_text):
  """Project the tokenized prediction back to the original text."""

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if is_whitespace(c):
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = BasicTokenizer(do_lower_case=False)

  tok_text = "".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose:
      logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose:
      logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                  orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in tok_ns_to_s_map.items():
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose:
      logger.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose:
      logger.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


def is_whitespace(c):
  if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    return True
  return False
