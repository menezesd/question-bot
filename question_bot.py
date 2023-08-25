"""Question Answering System

This module provides functions to create and manage a question
answering system.  It includes functions to calculate SHA-256
checksums, generate text embeddings using the ADA model, process
datasets, find similar embeddings, load and save embeddings, and
interactively handle user input.

Functions:
    calculate_checksum(data):
        Calculate the SHA-256 checksum of the provided data.

    get_ada_embeddings(input_text):
        Get embeddings for the provided input text using the ADA text
        embedding model.

    process_dataset(dataset): Process a dataset of examples by
        generating ADA embeddings for the questions.

    find_most_similar_embedding(user_embedding, embeddings_list): Find
        the most similar embedding from a list to a given
        user_embedding.

    load_and_process_dataset(): Load and process a dataset of
        questions and answers.

    load_embeddings(embedding_pickle_path): Load embeddings from a
        pickle file and return saved checksum along with the
        embeddings list.

    save_embeddings(embedding_pickle_path, dataset_checksum,
        embeddings_list): Save embeddings and dataset checksum to a
        pickle file.

    get_user_question():
        Get a user's question from the console input.

    process_user_input():
        Process user input by obtaining an ADA embedding for their question.

    display_result(user_embedding, most_similar_index,
        similarity_score, answers): Display the result based on the
        similarity between user input and embeddings.

    load_or_create_embeddings(embedding_pickle_path, dataset_checksum,
        dataset): Load existing embeddings or create new ones if the
        dataset has changed.

Usage: This script can be run to interactively answer user questions
    based on text embeddings.  It loads and processes a dataset of
    questions and answers, generates embeddings, and compares user
    input embeddings to find the most relevant answer.

"""
import hashlib
import logging
import os
import pickle

import numpy as np
from scipy.spatial.distance import cosine

import openai
from datasets import load_dataset

logging.basicConfig(level=logging.DEBUG)

openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"

openai.api_key = "de4d5adbc0af45bca22499dc3847b134"
openai.api_base = "https://ai-proxy.epam-rail.com"

deployment_name = "gpt-35-turbo"
THRESHOLD = 0.88

def get_similar_question(question):
    """
    Generate a similar question using ChatGPT.

    Args:
        question (str): The input question.

    Returns:
        str: A generated question similar in content to the input question.
    """
    prompt = f"Generate a question similar to the following, but which actually asks something different:\n\n{question}\n\nGenerated Question:"

    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=50,  # You can adjust the number of tokens for the generated question
    )

    generated_question = response.choices[0].text.strip()
    return generated_question


def are_questions_same(question1, question2):
    """
    Check if two questions are asking the same thing using ChatGPT.

    Args:
        question1 (str): The first question to compare.
        question2 (str): The second question to compare.

    Returns:
        bool: True if questions are determined to be the same, False otherwise.
    """
    prompt = f"Are these two questions asking the same thing? Answer YES or NO, only. \n\nQ1: {question1}\nQ2: {question2}\n\nAnswer:"

    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=1,
    )

    answer = response.choices[0].text.strip().lower()
    return 'yes' in answer

def calculate_checksum(data):
    """
    Calculate the SHA-256 checksum of the provided data.

    This function takes an input `data` and calculates its SHA-256 checksum,
    returning the checksum value in hexadecimal format.

    Args:
        data (str or bytes): The input data for which the checksum will be calculated.

    Returns:
        str: The hexadecimal representation of the calculated SHA-256 checksum.
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(str(data).encode("utf-8"))
    return sha256_hash.hexdigest()


def get_ada_embeddings(input_text):
    """
    Get embeddings for the provided input text using the ADA text embedding model.

    This function utilizes the OpenAI API to generate text embeddings for the given input
    text using the ADA text embedding model.

    Args:
        input_text (str): The input text for which embeddings will be generated.

    Returns:
        list: A list containing the text embeddings generated by the ADA model.
    """
    response = openai.Embedding.create(
        engine="text-embedding-ada-002", input=input_text
    )
    embedding = response["data"][0]["embedding"]
    return embedding


def process_dataset(dataset):
    """
    Process a dataset of examples by generating ADA embeddings for the questions.

    This function takes a dataset of examples, where each example is expected to be a
    dictionary containing a "question" key. It generates ADA embeddings for the questions
    using the `get_ada_embeddings` function and returns a list of embeddings.

    Args:
        dataset (list of dict): A list of dictionaries, each containing at least a "question" key.

    Returns:
        list: A list of ADA embeddings generated for the questions in the dataset.
    """
    return [get_ada_embeddings(example["question"]) for example in dataset]



def find_most_similar_embedding(user_embedding, embeddings_list):
    """Find the most similar embedding from a list to a given user_embedding.

    This function compares the user_embedding to a list of embeddings and identifies
    the embedding that is most similar to the user_embedding based on the cosine similarity.
    It returns the most similar embedding, its index in the embeddings_list, and the highest
    cosine similarity value.

    Args:
        user_embedding (list): The embedding representing the user's input or query.
        embeddings_list (list of lists): A list of embeddings to compare against.

    Returns:
        tuple: A tuple containing the following elements:
            - The most similar embedding to the user_embedding.
            - The index of the most similar embedding in the embeddings_list.
            - The highest cosine similarity value between the
    user_embedding and the embeddings_list.

    """
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
    """Load and process a dataset of questions and answers.

    This function loads a dataset named "web_questions" using the
    load_dataset function.  It extracts the training and test sets,
    retrieves the questions and answers from both sets, and returns
    the original dataset along with the concatenated questions and
    answers.  For our purposes we can use all questions in training
    and test datasets because we are using the datasets for the
    answers, and always returning the dataset answer.

    Returns:
        tuple: A tuple containing the following elements:
            - dataset (dict): The original dataset loaded from "web_questions".
            - questions (list): A list containing concatenated
              questions from training and test sets.
            - answers (list): A list containing concatenated answers
              from training and test sets.

    """
    dataset = load_dataset("web_questions")
    trains = dataset["train"]
    tests = dataset["test"]
    questions = trains["question"] + tests["question"]
    answers = trains["answers"] + tests["answers"]
    return dataset, questions, answers

def load_embeddings(embedding_pickle_path):
    """Load embeddings from a pickle file and return saved checksum along
with the embeddings list.

    This function attempts to load embeddings from a provided pickle
    file located at the given path.  If the file exists and can be
    successfully unpickled, it retrieves the saved checksum and the
    embeddings list. If the file doesn't exist or an error occurs
    during unpickling, the function sets saved_checksum and
    embeddings_list to None.

    Args:
        embedding_pickle_path (str): The path to the pickle file containing embeddings.

    Returns:
        tuple: A tuple containing the following elements:
            - saved_checksum (str or None): The checksum value saved in the pickle file,
              or None if the file doesn't exist or couldn't be unpickled.
            - embeddings_list (list or None): The list of embeddings loaded from the pickle file,
              or None if the file doesn't exist or couldn't be unpickled.

    """
    if os.path.exists(embedding_pickle_path):
        try:
            with open(embedding_pickle_path, "rb") as handle:
                saved_checksum, embeddings_list = pickle.load(handle)
        except (pickle.UnpicklingError, ValueError):
            saved_checksum = None
            embeddings_list = None
    else:
        saved_checksum = None
        embeddings_list = None
    return saved_checksum, embeddings_list


def save_embeddings(embedding_pickle_path, dataset_checksum, embeddings_list):
    """Save embeddings and dataset checksum to a pickle file.

    This function takes a path to a pickle file, a dataset checksum,
    and a list of embeddings, and saves them to the pickle file. The
    saved data includes the dataset checksum along with the embeddings
    list.

    Args:
        embedding_pickle_path (str): The path to the pickle file where data will be saved.
        dataset_checksum (str): The checksum value associated with the dataset.
        embeddings_list (list): The list of embeddings to be saved.

    Returns:
        None

    """
    with open(embedding_pickle_path, "wb") as handle:
        pickle.dump((dataset_checksum, embeddings_list), handle)


def get_user_question():
    """
    Get a user's question from the console input.

    This function prompts the user to enter a question through the console input.
    The user's input is stripped of leading and trailing whitespace before being returned.
    If an EOFError (End of File Error) occurs during input (e.g., when input is redirected
    from a file), the function returns None.

    Returns:
        str or None: The user's entered question or None if an EOFError occurs.
    """
    try:
        user_question = input("Enter a question: ").strip()
        return user_question
    except EOFError:
        return None


def process_user_input():
    """
    Process user input by obtaining an ADA embedding for their question.

    This function collects a user's question using the `get_user_question` function.
    If the user enters "QUIT" or encounters an EOFError during input, the function returns None.
    Otherwise, it generates an ADA embedding for the user's question using the `get_ada_embeddings`
    function and returns the embedding.

    Returns:
        list or None: An ADA embedding generated for the user's question or None if the user quits
        or EOFError occurs during input.
    """
    user_question = get_user_question()

    if user_question is None or user_question.upper() == "QUIT":
        return None

    user_embedding = get_ada_embeddings(user_question)
    return user_embedding


def display_result(user_embedding, most_similar_index, similarity_score, answers):
    """
    Display the result based on the similarity between user input and embeddings.

    This function takes a user_embedding, the index of the most similar embedding,
    a similarity score, and a list of answers. It checks if the similarity score is
    above a certain threshold (THRESHOLD) to determine if the answer is reliable. If
    the score is below the threshold, it prints a message indicating no information
    was found. Otherwise, it displays the user embedding, the merged answer,
    the most similar embedding index, and the cosine similarity score.

    Args:
        user_embedding (list): The ADA embedding generated for the user's question.
        most_similar_index (int): The index of the most similar embedding.
        similarity_score (float): The cosine similarity score between user embedding
                                 and the most similar embedding.
        answers (list): A list of answers corresponding to embeddings.

    Returns:
        None
    """
    if similarity_score < THRESHOLD:  # Adjust the threshold as needed
        print("No information found")
    else:
        merged_answer = " ".join(answers[most_similar_index])
#        print("User Embedding:", user_embedding)
        print("The answer is:", merged_answer)
        print("Most Similar Embedding Index:", most_similar_index)
        print("Question Interpretation:", questions[most_similar_index])
        print("Cosine Similarity Score:", similarity_score)


def load_or_create_embeddings(embedding_pickle_path, dataset_checksum, dataset):
    """Load existing embeddings or create new ones if the dataset has changed.

    This function checks if embeddings have been previously saved and
    loaded from a pickle file located at `embedding_pickle_path`. If
    the saved checksum matches the provided `dataset_checksum`, the
    function returns the loaded embeddings. Otherwise, it generates
    new embeddings by processing the training and test sets from the
    provided dataset using the `process_dataset` function. The newly
    generated embeddings are then saved and returned.

    Args:
        embedding_pickle_path (str): The path to the pickle file where
        embeddings are saved or will be saved.
        dataset_checksum (str): The checksum value associated with the dataset.
        dataset (dict): The dataset containing training and test sets.

    Returns:
        list: A list of embeddings either loaded from the pickle file or newly generated.

    """
    saved_checksum, embeddings_list = load_embeddings(embedding_pickle_path)

    if dataset_checksum != saved_checksum:
        embeddings_list = process_dataset(dataset["train"])
        embeddings_list += process_dataset(dataset["test"])

        save_embeddings(embedding_pickle_path, dataset_checksum, embeddings_list)

    return embeddings_list


if __name__ == "__main__":
    dataset, questions, answers = load_and_process_dataset()

    DATASET_CHECKSUM = calculate_checksum(questions + answers)
    EMBEDDING_PICKLE_PATH = "embedding.pickle"
    embeddings_list = load_or_create_embeddings(
        EMBEDDING_PICKLE_PATH, DATASET_CHECKSUM, dataset
    )

    while True:
        user_embedding = process_user_input()

        if user_embedding is None:
            break

        (
            most_similar_embedding,
            most_similar_index,
            similarity_score,
        ) = find_most_similar_embedding(user_embedding, embeddings_list)
        display_result(user_embedding, most_similar_index, similarity_score, answers)
