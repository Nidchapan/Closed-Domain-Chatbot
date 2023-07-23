# Closed Domain Chatbot for Thai Language

we proposed to design the closed-domain or fashion e-commerce Thai chatbot framework for Line social media platform with deep learning models.
This study experiment includes
1) the model with data augmentation and no data augmentation.
2) the model with the data pre-processing by translating the text into English and no data pre-processing.

## Introduction
- Due to the COVID-19 epidemic, several businesses transform their offline businesses into online ones and chatbots become a potential solution for business since chatbots are available around-the-clock for customer services. However, the research that is relevant to Thai chatbots is still uncommon, to fill the gap, we decide to study fashion e-commerce chatbots for the Thai language.
- The chatbot is challenging, with problems including misspelled words, multiple intents in one question, entity extraction, etc. The most popular technique for understanding the intent of users/customers is deep learning models. However, data collection for training deep learning models is difficult because of communicating data privacy policies, which causes the model to overfit when training with a small dataset. 
- This problem can be resolved by using a data augmentation technique that lets the deep learning model be trained on a small dataset while maintaining good model performance. Another way for handling this issue is by applying the pre-trained model. They are trained on a large corpus of text that can be fine-tuned on a smaller dataset specific to a particular task.
- Consequently, in this work, we apply both data augmentation techniques and pre-trained models. However, some pre-trained models deliver excellent performance in English but deliver poor results in other languages. The method for improving the model's performance for the text not in English such as Thai language is the data pre-processing  which first translates from Thai language to English before using the model.
- In this work, we proposed to design the closed-domain  or fashion e-commerce Thai chatbot framework for Line social media platform with deep learning or pre-trained models, 1) study the model with data augmentation and no data augmentation, and 2) study the model with the data pre-processing by translating the text into English and no data pre-processing.


## System Design
The chatbot system design for Line social media platform is shown in Figure 1. Building a chatbot framework with Google Colaboratory - Python and web application with Flask as the backend framework. For message sending and receiving, using Ngrok for creating secure tunnels between the chatbot framework (local development) and the Line social media platforms (public internet).
<p align="center">
<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/6cbf73b39c3085c76267df9eae07c9d5e576317c/image/The%20chatbot%20system%20design%20for%20Line%20social%20media%20platform.png" width="600">
</p>
<p align="center">
Figure 1 The chatbot system design for Line social media platform
</p>

## Model Architectures
After reviewing the research, we decided on the model architecture shown in Figure 2. Firstly, clean the intent text, then classify the intent into 5 classes (greeting, product information, shop information, purchase order and refund, and comment class) which helps the model respond in the proper way for each class. For each class excluding the comment class, using slots labeling technique by extracting intent information to the common use cases (slots) which varies for each class as shown by the example in Figure 3 For the comment class, use sentiment analysis instead of the slot labeling technique. Finally, retrieving appropriate data from the dataset to answer the intent question. If the question isn't related to the product or business, use ChatGPT to answer it.
<p align="center">
<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/22ce4745e6f758ba4ddc5643f118d60e6348fd96/image/The%20chatbot%20model%20architectures.png" width="600">
</p>
<p align="center">
Figure 2 The chatbot model architectures
</p>

### The example of slots labeling
<p align="center">
<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/6b01b7b7b0afbcb48e2ddf270041a8dccb6d55ca/image/The%20example%20of%20slot%20labeling%20of%20product%20information%20class.png" width="600">
</p>
<p align="center">
Figure 3 The example of slots labeling of product information class
</p>

The model that is used for each task is as follows:
1)	Cleaning Intent Text : pythainlp.util.normalize library for text normalization 
(https://pythainlp.github.io/docs/2.0/_modules/pythainlp/util/normalize.html)
  and pythainlp.corpus.thai_stopwords library + own dataset for removing the stopwords
(https://pythainlp.github.io/docs/2.0/api/corpus.html)

2)	Data Augmentation for Intent Classification : Pretraining transformer-based Thai Language Models: WangchanBERTa, trained on assorted Thai text dataset
3)	Intent Classfication : Universal sentence encoder for Multilingual trained with conditional masked language: USE-CMLM
4)	Translator or Slot Labeling Pre-processing : Machine Translator model trained on Open Source Parallel Corpus:Opus-MT
5)	Slot Labeling : Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension: BART, trained on Multi-Genre Natural Language Inference dataset
6)	Sentiment Analysis : Pretraining transformer-based Thai Language Models: WangchanBERTa, trained on wisesight-sentiment dataset
7)	ChatGPT : Generative Pre-trained Transformer 3.5 Turbo: GPT-3.5-Turbo

