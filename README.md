# Closed Domain Chatbot for Thai Language

we proposed to design the closed-domain or fashion e-commerce Thai chatbot framework for Line social media platform with deep learning models.
This study experiment includes
1) the model with data augmentation and no data augmentation.
2) the model with the data pre-processing by translating the text into English and no data pre-processing.

## System Design
The chatbot system design for Line social media platform is shown in Figure 1. Building a chatbot framework with Google Colaboratory - Python and web application with Flask as the backend framework. For message sending and receiving, using Ngrok for creating secure tunnels between the chatbot framework (local development) and the Line social media platforms (public internet).
<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/6cbf73b39c3085c76267df9eae07c9d5e576317c/image/The%20chatbot%20system%20design%20for%20Line%20social%20media%20platform.png" width="600">

Figure 1 The chatbot system design for Line social media platform

## Model Architectures
After reviewing the research, we decided on the model architecture shown in Figure 2. Firstly, clean the intent text, then classify the intent into 5 classes (greeting, product information, shop information, purchase order and refund, and comment class) which helps the model respond in the proper way for each class. For each class excluding the comment class, using slots labeling technique by extracting intent information to the common use cases (slots) which varies for each class as shown by the example in Figure 3 For the comment class, use sentiment analysis instead of the slot labeling technique. Finally, retrieving appropriate data from the dataset to answer the intent question. If the question isn't related to the product or business, use ChatGPT to answer it.

<img src="https://github.com/Nidchapan/Closed-Domain-Chatbot/blob/22ce4745e6f758ba4ddc5643f118d60e6348fd96/image/The%20chatbot%20model%20architectures.png" width="600">
Figure 2 The chatbot model architectures
