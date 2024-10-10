AI Sales Agent with Web Scraping & Product Integration
Project Overview
This project integrates web scraping, RAG (Retrieval-Augmented Generation) pipeline, and an AI-powered sales agent to automate product sales interactions. The system scrapes product data, uses advanced AI models for sales conversation, and interacts with customers to guide them through the sales funnel.

Features
1. Web Scraping
The project starts with scraping product data from the web. The scraped information includes product names, prices, specifications, and discounts, stored in a CSV file.

Technology Used: BeautifulSoup, Requests
Data Output: products_info.csv
Columns: Product Name, Current Price, Original Price, Discount, Specifications
2. RAG Pipeline Integration
Once the data is scraped, it is integrated into a RAG (Retrieval-Augmented Generation) Pipeline. The RAG model pulls relevant product information during the sales conversation to assist the AI agent.

Data Source: Scraped product data
Functionality: On-the-fly product info generation during conversations
3. AI Sales Agent
The heart of the project is the AI-powered sales agent. Using Langchain, Google's Gemini Model, and custom LLM chains, the agent interacts with potential customers, guiding them through different stages of a sales conversation.

Sales Stages:
Introduction
Qualification
Value Proposition
Needs Analysis
Solution Presentation
Objection Handling
Close
AI Features:
Determines conversation stage based on history
Provides product details from the RAG pipeline
Speech recognition (via speech_recognition) for handling voice inputs
Text-to-speech capabilities (using pyttsx3) to verbally respond to the customer
4. Lead Categorization
The AI sales agent can categorize leads based on conversation context, identifying whether a prospect is a Hot Lead, Cold Lead, or a Conversion.

5. Voice Input Handling
This project also incorporates speech recognition to capture and process user input through a microphone, enabling a fully interactive, voice-enabled sales agent.

Technology Used: SpeechRecognition, Pyttsx3

How to use it --

--- Prepare to have the data about the products for the knowledge base of the agent (RAG).
--- Make sure to check for the div's and classes in the file. It is now set towards the most selling Phones
of the flipkark e-commerce website.
--- Adjust the web drivers and the link from there.
--- ""make sure to put the keys into the .env file.""" from google api key centre.
--- after having the csv file move to the RAG folder. You can also initiate from the RAG folder because i have provided the products_info.csv
--- Provide that file to the RAG, data folder
--- Run the sales_with_gemini.py to ensure that it is running fine.
--- now run the sales_pen_git.py to run the agent.
