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
&emsp;&emsp;The chatbot system design for Line social media platform is shown in Figure 1. Building a chatbot framework with Google Colaboratory - Python and web application with Flask as the backend framework. For message sending and receiving, using Ngrok for creating secure tunnels between the chatbot framework (local development) and the Line social media platforms (public internet).
<p align="center">
<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/6cbf73b39c3085c76267df9eae07c9d5e576317c/image/The%20chatbot%20system%20design%20for%20Line%20social%20media%20platform.png" width="500">
</p>
<p align="center">
<b>Figure 1</b> The chatbot system design for Line social media platform
</p>

## Model Architectures
&emsp;&emsp;After reviewing the research, we decided on the model architecture shown in Figure 2. Firstly, clean the intent text, then classify the intent into 5 classes (greeting, product information, shop information, purchase order and refund, and comment class) which helps the model respond in the proper way for each class. For each class excluding the comment class, using slots labeling technique by extracting intent information to the common use cases (slots) which varies for each class as shown by the example in Figure 3 For the comment class, use sentiment analysis instead of the slot labeling technique. Finally, retrieving appropriate data from the dataset to answer the intent question. If the question isn't related to the product or business, use ChatGPT to answer it.
<p align="center">
<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/22ce4745e6f758ba4ddc5643f118d60e6348fd96/image/The%20chatbot%20model%20architectures.png" width="500">
</p>
<p align="center">
<b>Figure 2</b> The chatbot model architectures
</p>

<p align="center">
<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/6b01b7b7b0afbcb48e2ddf270041a8dccb6d55ca/image/The%20example%20of%20slot%20labeling%20of%20product%20information%20class.png" width="600">
</p>
<p align="center">
<b>Figure 3</b> The example of slots labeling of product information class
</p>

The model that is used for each task is as follows:

**1)	Cleaning Intent Text :** pythainlp.util.normalize library (https://pythainlp.github.io/docs/2.0/_modules/pythainlp/util/normalize.html) for text normalization and pythainlp.corpus.thai_stopwords library (https://pythainlp.github.io/docs/2.0/_modules/pythainlp/corpus/common.html#thai_stopwords) + own dataset for removing the stopwords

**2)	Data Augmentation for Intent Classification :** Pretraining transformer-based Thai Language Models: WangchanBERTa, trained on assorted Thai text dataset by  PyThaiNLP (https://pythainlp.github.io/dev-docs/_modules/pythainlp/augment/lm/wangchanberta.html#Thai2transformersAug)

**3)	Intent Classfication :** Universal sentence encoder for Multilingual trained with conditional masked language: USE-CMLM (https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2, https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1)

**4)	Translator or Slot Labeling Pre-processing :** Machine Translator model trained on Open Source Parallel Corpus:Opus-MT (https://huggingface.co/Helsinki-NLP/opus-mt-th-en)

**5)	Slot Labeling :** Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension: BART, trained on Multi-Genre Natural Language Inference dataset (https://huggingface.co/facebook/bart-large-mnli)

**6)	Sentiment Analysis :** Pretraining transformer-based Thai Language Models: WangchanBERTa (https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased), trained on wisesight-sentiment dataset (https://huggingface.co/datasets/wisesight_sentiment)

**7)	ChatGPT :** Generative Pre-trained Transformer 3.5 Turbo: GPT-3.5-Turbo (https://platform.openai.com/docs/models)

## EXPERIMENTAL RESULTS
**1) Data Augmentation and No-Data Augmentation**
  
&emsp;&emsp;The intent classification model performance between using data augmentation and no-augmentation training on own dataset with 100 epochs as shown in Figure 4

<p align="center">
<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/dd089111c172cc767fe7d95e50f79e7d12f90297/image/no_aug_train.png" width="400"/> 
<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/dd089111c172cc767fe7d95e50f79e7d12f90297/image/aug_train.png" width="400"/>
</p>

<p align="center">
(a) no-augmentation&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(b) augmentation
</p>

<p align="center">
<b>Figure 4</b> (a) F1 score of intent classification with no-data augmentation for 100 epochs,
</p>

<p align="center">
(b) F1 score of intent classification with data augmentation for 100 epochs
</p>
  
&emsp;&emsp;From the graphs presented, the performance of the intent classification model both with data augmentation and no-data augmentation, the F1 score of the training dataset performs well but poorly on the validation dataset or an overfitting issue occurs of both experiments when the epoch exceeds 40. Therefore, we choose the epoch of 40 to compare the results. However, the F1 score of the validation dataset trend of intent classification with data augmentation is higher than the result with no-data augmentation. Table 1 shows that both the accuracy and F1 score of the test dataset for the intent classification model with augmentation is higher than the experiment result with no-data augmentation. While the processing time of experiment results with no-augmentation is lower.

**Table 1** the average 3 times of accuracy, f1 score and processing time of test dataset (152 text data) for intent classification with data augmentation and no-data augmentation at 40 epoch
| results  | no augmentation | augmentation  |
| ------------- | ------------- | ------------- |
| accuracy  | 0.8947 ± 0.0093  | 0.9825 ± 0.0031 |
| F1 score  | 0.8927 ± 0.0110  | 0.9824 ± 0.0031  |
| processing time |	777 ± 541 ms |	1182 ± 544 ms |

**2) Data Pre-Processing and No-Data Pre-Processing**

&emsp;&emsp;The slot labeling model performance between using data pre-processing by translate the text into English and no-data pre-processing as shown in Table 2, when applying the translator model, translate from Thai to English before implementing the model. It improves the model performance by increasing both accuracy and F1 score, while the processing time is slower.

**Table 2** the average 3 times of accuracy, f1 score and processing time of slots labelling model with no-data pre-processing and with data pre-processing
| results  | no pre-processing | pre-processing  |
| ------------- | ------------- | ------------- |
| accuracy  | 0.2553 ± 0.0000 |	0.7449 ± 0.0000 |
| F1 score  | 0.1609 ± 0.0000 |	0.7490 ± 0.0000 |
| processing time |	6.2100 ± 0.1472 s |	7.8400 ± 1.3789 s |


## Conclusion and Discussion
**1)	Data Augmentation and No-Data Augmentation**

&emsp;&emsp;The small dataset leads the deep learning model to the overfitting issue - especially in the closed-domain Thai chatbot task, the data collection is challenging because of communicating data privacy policies. To resolve this issue, we use both a pre-trained model and a data augmentation technique. As a result of the previous section, the experiment comparing between using data augmentation techniques and without data augmentation. The data augmentation technique improves the model's accuracy and F1 score. While requiring a little more processing time.

**2)	Data Pre-Processing and No-Data Pre-Processing**

&emsp;&emsp;Because of the limitations of data collection, using a pre-trained model with zero-shot learning that helps the model can detect classes never seen without fine-tuning. However, these models deliver excellent performance in English but deliver poor results in other languages. The natural language understanding of text in the Thai language is challenging. For example, the Thai language has no spaces between words, making natural language understanding more difficult, and there are fewer datasets in Thai for training the model than English datasets. Data pre-processing by translating the text in Thai into English is the one of technique to improve the model as the result in the previous section. Translating from Thai to English of the text greatly boosted both accuracy and F1 score, while processing time was slightly increased.

**3)	Model Architecture**

&emsp;&emsp;Our solution is the closed-domain or fashion e-commerce Thai chatbot framework as shown in the previous section. The intent classification technique helps the model respond properly for each input question class, and the slot labeling technique makes the model can respond to the answer for multiple questions at once. To deal with the question that isn't included in the product or business. ChatGPT is implemented for handling it, which overcomes the model's vulnerability.

## Example of our results

&emsp;&emsp;We showed the example of a Thai conversation between a customer and our chatbot, the left side is our chatbot, and the right side is the user. The conversation includes questions about shop and product information, multiple intents in one question, the question that isn't included in the product or business, and customer comments.

<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/c0020ea0fa4aac7e37e95a13c70521e946517722/image/the%20example%20of%20conversation%20between%20customer%20and%20our%20chatbot.png" width="500"/> 

## Required Key

1. OpenAI API (https://platform.openai.com/account/api-keys)
2. Channel access tokens - LINE Developers
3. Channel Secret - LINE Developers
4. Webhook URL [get from code - after run flask application]

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@misc{Nidchapan2023,
      author = {Nidchapan Nitisukanan and Ekarat Rattagan},
      title = {Closed-Domain Chatbot for Thai Languge},
      year = {2023},
      publisher = {GitHub},
}
```
