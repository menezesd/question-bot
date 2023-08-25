import openai
import os
import logging
import json
import pickle
from datasets import load_dataset
import numpy as np
from scipy.spatial.distance import cosine
from datasets import load_dataset
import hashlib

logging.basicConfig(level=logging.DEBUG)

openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"

openai.api_key = "de4d5adbc0af45bca22499dc3847b134"
openai.api_base = "https://ai-proxy.epam-rail.com"

deployment_name = "gpt-35-turbo"
THRESHOLD=0.88
def calculate_checksum(data):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(str(data).encode('utf-8'))
    return sha256_hash.hexdigest()

def get_ada_embeddings(input_text):
    response = openai.Embedding.create(
        engine='text-embedding-ada-002',
        input=input_text
    )
    embedding = response['data'][0]['embedding']
    return embedding

def process_dataset(dataset):
    res = []
    for example in dataset:
        question = example['question']  # Replace with the actual key for questions
        embedding = get_ada_embeddings(question)
        res += [embedding]
    return res

def find_most_similar_embedding(user_embedding, embeddings_list):
    highest_similarity = -1
    most_similar_embedding = None
    most_similar_index = -1

    for i, embedding in enumerate(embeddings_list):
        similarity = 1 - cosine(np.array(user_embedding), np.array(embedding))
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_embedding = embedding
            most_similar_index = i

    return most_similar_embedding, most_similar_index, highest_similarity


def load_and_process_dataset():
    dataset = load_dataset("web_questions")
    trains = dataset['train']
    tests = dataset['test']
    questions = trains['question'] + tests['question']
    answers = trains['answers'] + tests['answers']
    return dataset, questions, answers

def calculate_checksum(data):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(str(data).encode('utf-8'))
    return sha256_hash.hexdigest()

def load_embeddings(embedding_pickle_path):
    if os.path.exists(embedding_pickle_path):
        try:
            with open(embedding_pickle_path, 'rb') as handle:
                saved_checksum, embeddings_list = pickle.load(handle)
        except (pickle.UnpicklingError, ValueError):
            saved_checksum = None
            embeddings_list = None
    else:
        saved_checksum = None
        embeddings_list = None
    return saved_checksum, embeddings_list

def save_embeddings(embedding_pickle_path, dataset_checksum, embeddings_list):
    with open(embedding_pickle_path, 'wb') as handle:
        pickle.dump((dataset_checksum, embeddings_list), handle)
        

def get_user_question():
    try:
        user_question = input("Enter a question: ").strip()
        return user_question
    except EOFError:
        return None

def process_user_input():
    user_question = get_user_question()

    if user_question is None or user_question.upper() == "QUIT":
        return None
    
    user_embedding = get_ada_embeddings(user_question)
    return user_embedding

def display_result(user_embedding, most_similar_index, similarity_score, answers):
    if similarity_score < THRESHOLD: # Adjust the threshold as needed
        print("No information found")
    else:
        merged_answer = " ".join(answers[most_similar_index])
        print("User Embedding:", user_embedding)
        print("The answer is:", merged_answer)
        print("Most Similar Embedding Index:", most_similar_index)
        print("Cosine Similarity Score:", similarity_score)

def load_or_create_embeddings(embedding_pickle_path, dataset_checksum, dataset):
    saved_checksum, embeddings_list = load_embeddings(embedding_pickle_path)

    if dataset_checksum != saved_checksum:
        embeddings_list = process_dataset(dataset['train'])
        embeddings_list += process_dataset(dataset['test'])

        save_embeddings(embedding_pickle_path, dataset_checksum, embeddings_list)

    return embeddings_list

if __name__ == "__main__":
    dataset, questions, answers = load_and_process_dataset()

    dataset_checksum = calculate_checksum(questions + answers)
    embedding_pickle_path = 'embedding.pickle'
    embeddings_list = load_or_create_embeddings(embedding_pickle_path, dataset_checksum, dataset)

    while True:
        user_embedding = process_user_input()
        
        if user_embedding is None:
            break

        most_similar_embedding, most_similar_index, similarity_score = find_most_similar_embedding(user_embedding, embeddings_list)
        display_result(user_embedding, most_similar_index, similarity_score, answers)

            


