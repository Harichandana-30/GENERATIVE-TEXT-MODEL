COMPANY : CODTECH IT SOLUTIONS

NAME : SAMUDRALA LAKSHMI HARI CHANDANA

INTERN ID : CT06DM1148

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 6 WEEKS

MENTOR : NEELA SANTHOSH


# GENERATIVE-TEXT-MODEL

## About the Project

This is a generic, Python-based text summarization tool developed during an NLP internship. It reduces large text documents into concise, coherent summaries using state-of-the-art transformer models. The tool uses the facebook/bart-large-cnn model from HuggingFace Transformers to generate high-quality summaries that retain the essential meaning of the original text.

This project was built to address a practical need. Long articles, reports, and research papers can be time-consuming to read. This tool helps users quickly grasp the core ideas without reading every word.

## Features

- Accepts input from:
  - Manual text entry via the command line
  - External plain .txt files
- Uses the facebook/bart-large-cnn pre-trained model for summarization
- Displays:
  - Original input text
  - Summarized output
  - Word count and compression ratio
- Optionally saves the summary to a text file

## Why This Was Built

This project was intended to go beyond theory and solve a real problem: how to efficiently process and understand large volumes of text. Manual summarization is labor-intensive, especially for students, researchers, and professionals. Automating this process with a deep learning model makes it more efficient and offers a valuable learning experience in NLP and machine learning. This tool also serves as a foundation for future advanced NLP projects.

## Prerequisites

- Python 3.7 or higher

### Required Libraries

- transformers
- torch

Install all dependencies using the following command:

pip install -r requirements.txt
