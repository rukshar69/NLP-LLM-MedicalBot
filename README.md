# NLP-LLM-MedicalBot
A quantined llama 2 based medical bot using a medical book pdf for vector database and chainlit for ui

## Demo Video
[![Demo Video](https://img.youtube.com/vi/5z5KZz6xZmQ/0.jpg)](https://youtu.be/5z5KZz6xZmQ)

## Features
- The medicalbot uses the pdf file [71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf](https://github.com/rukshar69/NLP-LLM-MedicalBot/blob/main/medical_bot_llm/data/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf) is used to generate a vector database and this vector database is used as a reference when answering user questions.
- The vector database is generated from the pdf file using the script [ingest.py](https://github.com/rukshar69/NLP-LLM-MedicalBot/blob/main/medical_bot_llm/ingest.py)
    - HuggingFaceEmbeddings from langchain utilizes the model **sentence-transformers/all-MiniLM-L6-v2** to generate the vector database
- The main python script for the chainlit chatbot is [model.py](https://github.com/rukshar69/NLP-LLM-MedicalBot/blob/main/medical_bot_llm/model.py)
    - The chatbot LLM model used is [TheBloke/Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML), a quantized version of [Llama 2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf). As such this quantized model takes up less resources.
    - The chatbot responds by displaying both the user-query answer and the source of info. from the pdf file.

## Environment
**Python** version: **3.10**

## Instructions
1. Install dependencies using `pip install -r requirements.txt` (creating a separate virtual environment is recommended)
2. Run the script `python ingest.py` to create a vector database
3. Run the chainlit-langchain script `chainlit run model.py -w` to run the chatbot powered by quantized Llama 2 7B Chat.
    - **Note**: Since this code utilizes the CPU, the reponse from the bot takes some time to generate.