# squad_bert
pre-processing and post-processing SQUAD data for BERT
# Question answering in production using BERT fine-tuned on SQUAD 2.0
A lot has already been said about the BERT neural network. If you are reading this article, it is safe to assume that you already are familiar with the pros and cons of it. Instead of repeating those discussions, we jump right into the technical details of implementing a question answering system in a production environment. Specifically, starting with a question and a candidate paragraph that may contain the answer, we want either to extract the answer from the paragraph, or decide that the question cannot be answered with that paragraph.

This process consists of three major steps: preprocessing the input, submitting a request to the server, and postprocessing the response of the server to get the final answer. The motivation for writing this post is that information on the specifics of the required pre and post processing are scarce and hard to come by on the internet. As a result, in many instances these steps are imported from the official code base. While this may be convenient, it hides the details behind several layers of abstraction. In addition, the official code does more than is needed for many practical cases (e.g. it keeps track of metrics that you may not be interested in) and consequently runs slower. Furthermore, the official code relies on the tensorflow library for performing many of its pre and post processing steps, and importing tensorflow can have a significant hit on the run time of your service. The code we present here, in contrast, is independent of tensorflow. We hope that the following discussion can help fill in the missing information.

  

The original BERT released by Google Research was written for Tensorflow 1.14. Some time later Tensorflow was upgraded to 2.0 which was backward incompatible with 1.x. Since then, Google has released a new version of BERT that runs on Tensorflow 2.x. In this writing we focus on this newer version which can be found [here](https://github.com/tensorflow/models/tree/master/official/nlp/bert).

  

We assume that you have successfully setup your BERT network and finetuned it on the SQUAD 2.0 dataset. If not, please follow the instructions on the page linked above to do so.

  

The first thing you need to do is to export the finetuned model so that it can run in serving mode. Although the functionality for exporting is already included in the code, as of this writing there is no mention of it in the accompanying documentation. You can use the following commands to export your trained network:

  

```shell
export SQUAD_VERSION=v2.0
export BERT_DIR=path to pretrained BERT network
export SQUAD_DIR=path to where SQUAD data resides
export MODEL_DIR=path to finetuned model’s checkpoint

python run_squad.py \
--input_meta_data_path=${SQUAD_DIR}/squad_${SQUAD_VERSION}_meta_data \
--predict_file=${SQUAD_DIR}/dev-v2.0.json \
--vocab_file=${BERT_DIR}/vocab.txt \
--bert_config_file=${BERT_DIR}/bert_config.json \
--init_checkpoint=squad2/path to trained checkpoint \
--model_dir=${MODEL_DIR} \
--model_export_path=${MODEL_DIR}/path to save the exported model/1 \
--mode=export_only
```

Note the “1” at the end of the model_export_path. When you launch tensorflow serving, it expects to find the saved model in a directory called “1”, otherwise it will complain.

  

After you have exported the model you can inspect it with

  

```shell
saved_model_cli show --dir model_export_path --all
```

On the output, you should see something similar to

```shell
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_mask'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 384)
        name: serving_default_input_mask:0
    inputs['input_type_ids'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 384)
        name: serving_default_input_type_ids:0
    inputs['input_word_ids'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 384)
        name: serving_default_input_word_ids:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['end_positions'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 384)
        name: StatefulPartitionedCall:0
    outputs['start_positions'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 384)
        name: StatefulPartitionedCall:1
  Method name is: tensorflow/serving/predict
```

  

That tells us that the model expects three tensors on the input: input_mask, input_type_ids, and input_word_ids. Also that the output will consist of two tensors: start_positions and end_positions. Effectively, you need to preprocess the question and the candidate paragraph in order to generate those three tensors which will be fed into the network. Also, once you receive a response to your request, you will have to postprocess the two tensors in order to get the final text of the answer from them.

  

To start the exported model in serving mode use

```shell
tensorflow_model_server --rest_api_port=8503 --model_name=1 --model_base_path=path to saved model
```

  

# Preprocessing

We have two pieces of information to use as input: one question and one paragraph. But to feed them into the network we need to concatenate them into one long string. Since all inputs need to be the same length, we need to insert ‘0’ padding for any string that’s shorter than the expected length. Additionally we need to insert three special tokens into this long string: ‘[CLS]’ at position zero, ‘[SEP]’ in between the question and paragraph, and one more ‘[SEP]’ between the paragraph and the padding.

The preprocessing steps consist of:

1.  Concatenating the paragraph at the end of the question
2. Tokenizing the entire string
    
3.  Inserting the special tokens
    
4.  Assigning unique numbers to the tokens
    
5.  Adding zero padding at the end to compensate for the difference in length from the maximum sequence length
    
6.  Preparing an auxiliary tensor that tells the system which part of the input is the question and which part the paragraph
    
7.  Preparing an auxiliary tensor that tells the system which part of the input is real data and which is just padding and should be ignored
    

  

Take a look at the image below for a visualization of this process:
![enter image description here](https://imgur.com/Vc89Umy.png)
  

The following code accomplishes the required steps. Note that we are importing the tokenizer from the official code to ensure consistency. If you decide to write your own tokenizer, make sure the result is identical to the official one, or you will run the risk of inconsistencies and unexpected behavior from the network.

  

```python
import sys
sys.path.append('path to tokenization.py')
from tokenization import FullTokenizer
import json,requests,string,math
import numpy as np

max_seq_length = 384
vocab_file = 'path to vocab file (contained in the pretrained BERT fodler)'
tokenizer = FullTokenizer(vocab_file)

puncset = set(string.punctuation)
def clean(s):
    s = [w.lower() for w in s.split() if not w in puncset]
    return ' '.join(s)
    
def get_ids(s):
    tokens = tokenizer.tokenize(s)
    return tokens,tokenizer.convert_tokens_to_ids(tokens)

def process_question(question):
    _,qids = get_ids(clean(question))
    return qids

maps = []
def process_paragraph(paragraph,qidslen):
    ptokens,pids = get_ids(clean(paragraph))
    if len(ptokens) + qidslen + 3 > max_seq_length:
        excess = len(ptokens) + qidslen + 3 - max_seq_length
        ptokens = ptokens[:-excess]
        pids = pids[:-excess]
    maps.append({i+qidslen+2:ptoken for i,ptoken in enumerate(ptokens)})
    return pids

def process_pair(qids,pids,qidslen):
    input_word_ids = [101]+qids+[102]+pids+[102]
    input_type_ids = [0]*(qidslen+2)+[1]*(len(pids)+1)
    input_mask = [1]*len(input_word_ids)
    pad = [0]*(max_seq_length - len(input_word_ids))
    input_mask += pad
    input_word_ids += pad
    input_type_ids += pad
    return {
        'input_word_ids': input_word_ids,
        'input_type_ids': input_type_ids,
        'input_mask' : input_mask,
    }

```

# Building and submitting the request

  

This step is straightforward. The preprocessed tensors are stored in instances and submitted to the server:

  

```python
qids = process_question(question)
qidslen = len(qids)
instances = [process_pair(qids,process_paragraph(paragraph,qidslen),qidslen) for paragraph in paragraphs]

data = json.dumps({"signature_name": "serving_default", 'instances':instances})

url = 'http://localhost:8503/v1/models/1:predict'

response = requests.post(url,data=data)
```

# Postprocessing

  

This is where things get a bit tricky. If we were to use BERT in a classification task, then the postprocessing would simply involve getting an argmax over the output probability distribution (i.e. the softmax). But here we expect a different kind of result: a snippet from the paragraph that answers the question. The raw output from BERT contains two probability distributions, one for the start index of the answer in the paragraph, and one for the end index. From these two distributions we need to pick a combination that maximizes the joint probability of start and end indices, $P(\text{start\_index}=s,\text{end\_index}=e)$. Additionally for questions that cannot be answered from the given paragraph, BERT is trained to output index zero as the most likely for both distributions. To tackle this joint probability, we make the naive assumption of conditional independence which gives us $P(\text{start\_index}=s,\text{end\_index}=e)= P(\text{start\_index}=s) \times P(\text{end\_index}=e)$, although since we are working with the logs of the probabilities, the product turns into a sum. From here on, all we need is to take the argmax of each distribution. However, these distributions are over the entire concatenated string of question and answer. So we need to make sure that we are only working with the portion of the distribution that covers the paragraph. More specifically, we need to check for a few conditions:

1.  If the start index falls in the question
    
2.  If the end index falls in the question
    
3.  If the end index falls before the start index
    
4.  If the start and end indices are the same
    
5.  If the start and end indices are specifically zero
    

Note: In the official code, they limit the answer to a maximum size of 32 which makes for one more condition to check. But that is likely just to improve the score of performance over the SQUAD data. We are not going to limit the length of the answer.

Effectively, we need to loop through all combinations of start and end index to validate them according to the above conditions, and then pick the maximum from the validated candidates. To avoid the high price of quadratic complexity for this nested loop, we only pick the top N candidate indices. Once we have a maximum likelihood answer, we need to check it against the null answer corresponding to start and end indices of zero, and decide whether we prefer the null or the non-null answer.

Notice that during preprocessing we kept track of the paragraph tokens in the variable named maps. To get the text of the answer from the indices, we refer to this variable in the function map_back.

  

The code below implements these steps:

```python
def pick_best_indices(logits,topn=5):
    return np.argsort(logits)[-topn:]

def map_back(mapi,sind,eind):
    text = [mapi[i] for i in range(sind,eind+1)]
    text = " ".join(text).replace(" ##", "")
    return text

def logit_to_prob(l):
    e = math.exp(l)
    return e/(e+1)

def find_null_answer(resp,prelims,null_score_diff_threshold):
    slog0,elog0 = resp['start_positions'][0], resp['end_positions'][0]
    if not prelims:
        return [{'sind': 0,
                'slog': slog0,
                'eind': 0,
                'elog': elog0,
                'text': '',
                'probability': logit_to_prob(slog0+elog0)
           }]
    best_non_null_score = prelims[0]['slog'] + prelims[0]['elog']
    null_score = slog0 + elog0
    if null_score - best_non_null_score > null_score_diff_threshold:
        prelims = [{'sind': 0,
                    'slog': slog0,
                    'eind': 0,
                    'elog': elog0,
                    'text': '',
                    'probability': logit_to_prob(slog0+elog0)
                   }] + prelims
    return prelims
    
def postprocess(resp,mapi):
    prelims = []
    for sind in pick_best_indices(resp['start_positions']):
        for eind in pick_best_indices(resp['end_positions']):
            if sind == eind:
	        continue
            if sind not in mapi:
                continue
            if eind not in mapi:
                continue
            if eind < sind:
                continue
            slog = resp['start_positions'][sind]
            elog = resp['end_positions'][eind]
            prelim = {
                'sind': sind,
                'slog': slog,
                'eind': eind,
                'elog': elog,
                'text': map_back(mapi,sind,eind),
                'probability': logit_to_prob(slog+elog)
                           }
            if prelim['text']: prelims.append(prelim)#ensure null answers are not included here          
    prelims = sorted(
        prelims,
        key=lambda x: (x['probability']),
        reverse=True)
    prelims = find_null_answer(resp,prelims,null_score_diff_threshold)
    return prelims
    
for ind,resp in enumerate(eval(response.text)['predictions']):
        result = postprocess(resp,maps[ind])
        print(result)
```
# Batch processing
In practice, to answer a query you will likely start with a number of candidate paragraphs and run each one along with the query to find a suitable answer. The processing steps demonstrated here are capable of processing an entire batch of paragraphs along with a single question. You can simply store the query in the variable called "question" and the candidate paragraphs in "paragraphs", and call the rest of the code.
## Final thoughts
As of this writing, the new question answering part of the CK-12 tutor is days away from going live. This functionality is based on the BERT neural network which according to the score boards has yet to be bettered by a different approach (except of course, for the ensemble systems). In the process of taking the BERT system from prototype to production we found ourselves frustrated by the absence of detailed documentation and practical examples. Our hope is that the present article can ease the burden for others who might set out on a similar path. 
