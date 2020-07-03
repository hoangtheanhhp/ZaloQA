# Zalo AI Challenge 2019 - Vietnamese Wikipedia Question Answering

## General
**Vietnamese Wikipedia Question Answering** task on the **Zalo AI Challenge 2019**.

## Structures
* *QASystem* and *Ultilities* contain source codes, base model as well as fine-tuning models and dataset used in this project. Guide on how to setup and re-produce the result is also provided.
* *Dataset* contains the dataset that is used in this project.

## How to run
Details on how to train/predict using the model is described [here](https://github.com/hoangtheanhhp/ZaloQA/bert_lstm/Readme.md)


## What we have tried

- [x] Apply BERT as baseline for the [QA problem defined by Zalo](https://challenge.zalo.ai/portal/question-answering)
- [x] Data augmented using the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/) by translating
- [x] Improve BERT by trying different approaches ([BERT Embedding + Bi-LSTM](https://github.com/hoangtheanhhp/ZaloQA/blob/bert_lstm/Zalo_AI_anhht.ipynb) but yield no improvements
- [x] Try different loss function for the classification problem ((Squared) Hinge loss, KLD loss & Focal loss) along with label smoothing, but yield no improvements
- [ ] Apply [PhoBERT](https://github.com/VinAIResearch/PhoBERT) for the problem
- [ ] Apply [AlBERT_vi](https://github.com/ngoanpv/albert_vi) for the problem

Our solution yeild an F1 score of *71.84%* on private datasets.
