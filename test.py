"""
Testing Framework

This module provides functions for generating random questions using templates
and for comparing answers for similarity. It also includes an integration test
that evaluates the similarity of answers between a question-answering system
and ChatGPT.

Functions:
    generate_random_question():
        Generate random questions by filling in templates with random words.

    compare_answers(answer1, answer2):
        Compare the similarity between two answers using a chosen similarity metric.

Usage:
    The functions in this module can be used to generate random questions by
    filling in templates, and to compare the similarity of answers using a
    specified similarity metric. The integration test demonstrates the comparison
    of answers between a question-answering system and ChatGPT, providing insight
    into the similarity of their responses.

    For more details on individual function usage, refer to their respective
    docstrings and comments in the code.
"""

import random
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import openai
import question_bot  # Import the module you want to test


def get_similar_question_different(question):
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

def get_similar_question_same(question):
    """
    Generate a similar question using ChatGPT.

    Args:
        question (str): The input question.

    Returns:
        str: A generated question similar in content to the input question.
    """
    prompt = f"Generate a question similar to the following, which asks the same question with different wording:\n\n{question}\n\nGenerated Question:"

    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=50,  # You can adjust the number of tokens for the generated question
    )

    generated_question = response.choices[0].text.strip()
    return generated_question



# List of words for random generation
nouns = ["apple", "dog", "car", "banana", "computer"]
adjectives = ["red", "happy", "fast", "funny", "loud"]
verbs = ["run", "eat", "jump", "dance", "sleep"]

# Template-based generation
templates = [
    "What is the purpose of the {noun} that is {adjective}?",
    "Can you {verb} the {noun} while being {adjective}?",
    "Why does the {noun} look so {adjective} when it {verb}?",
]


def generate_random_question():
    """This function generates random questions by filling in templates
    with randomly selected words.

    The function utilizes predefined lists of nouns, adjectives, and
    verbs to create questions in a template-based manner. It randomly
    selects a template, then picks a random noun, adjective, and verb
    to populate the template. The resulting question is returned as
    output.

    Example:
        generated_question = generate_random_question()
        print(generated_question)

    Output:
        What is the purpose of the dog that is red?

    """

    choice = random.choice(templates)
    question = choice.format(
        noun=random.choice(nouns),
        adjective=random.choice(adjectives),
        verb=random.choice(verbs),
    )
    return question


class TestYourModule(unittest.TestCase):
    def setUp(self):
        self.dataset = {
            "train": [
                {"question": "What is the capital of France?",
                    "answers": ["Paris"]},
                {
                    "question": "Who wrote 'Romeo and Juliet'?",
                    "answers": ["William Shakespeare"],
                },
            ],
            "test": [
                {"question": "What is 2 + 2?", "answers": ["4"]},
                {"question": "What is the largest planet?",
                    "answers": ["Jupiter"]},
            ],
        }
        self.mocked_embeddings = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([0.7, 0.8, 0.9]),
            np.array([0.2, 0.3, 0.4]),
        ]

    def test_calculate_checksum(self):
        data = "test_data"
        expected_checksum = "expected_checksum"

        with patch("hashlib.sha256") as mock_sha256:
            mock_digest = MagicMock()
            mock_digest.hexdigest.return_value = expected_checksum
            mock_sha256.return_value = mock_digest

            result = question_bot.calculate_checksum(data)

            self.assertEqual(result, expected_checksum)


def test_find_most_similar_embedding_integration():
    """Test the integration of the find_most_similar_embedding function
        with a question-answering system.

    This function tests the integration of various components within
    the question-answering system. It calculates the embedding
    checksum, loads or creates embeddings, compares answers between
    the question-answering system and ChatGPT, and performs similarity
    checks on generated bogus questions. The results are reported,
    including the average similarity score, the number and percentage
    of dissimilar answers, and comparison results for generated
    questions.

    Note:
        - The get_chatgpt_answer() and generate_random_question()
    functions are assumed to be defined and used within this context.

        - The compare_answers() function is assumed to be defined
          elsewhere and used for answer similarity comparison.

    Example:
        test_find_most_similar_embedding_integration()

    Output:
        Average similarity: 0.75
        7 out of 20 are dissimilar in ChatGPT vs database
        This equals 35.0 percent that are dissimilar

    """
    dataset, questions, answers = question_bot.load_and_process_dataset()

    dataset_checksum = question_bot.calculate_checksum(questions + answers)
    embedding_pickle_path = "embedding_test.pickle"
    embeddings_list = question_bot.load_or_create_embeddings(
        embedding_pickle_path, dataset_checksum, dataset
    )

    total_similarity = 0
    not_similar_answers = 0
    for i, example in enumerate(questions):
        user_embedding = question_bot.get_ada_embeddings(example)
        (result_embedding,
         result_index,
         similarity_score,
         ) = question_bot.find_most_similar_embedding(user_embedding,
                                                      embeddings_list)

        assert similarity_score >= 0
        assert result_embedding is not None
        assert result_index >= 0
        assert result_index < len(embeddings_list)

        # Get the expected answer from your question-answering system
        expected_answer = answers[i]
        # Generate a ChatGPT answer using the same question
        # Call the function you've defined earlier
        chatgpt_answer = get_chatgpt_answer(example)

        similarity_score = compare_answers(expected_answer, chatgpt_answer)
        is_answer_similar = similarity_score >= 0.80
        total_similarity = 0.0
        if not is_answer_similar:
            print(
                f"ChatGPT and database answers are not similar for question: {example}"
            )
            print(f"Database answer: {expected_answer}")
            print(f"ChatGPT answer: {chatgpt_answer}")
            not_similar_answers += 1
            total_similarity += similarity_score

        for i in range(1000):
            bogus_question = generate_random_question()
            user_embedding = question_bot.get_ada_embeddings(bogus_question)
            (
                result_embedding,
                result_index,
                similarity_score,
            ) = question_bot.find_most_similar_embedding(
                user_embedding, embeddings_list
            )
            assert result_embedding is not None
            assert result_index >= 0
            assert result_index < len(embeddings_list)
            assert result_embedding is not None
            assert result_index >= 0
            assert similarity_score < question_bot.THRESHOLD
        print(f"Average similarity: {total_similarity / len(questions)}")
        print(
            f"{not_similar_answers} out of {len(questions)} are dissimilar in ChatGPT vs database"
        )
        print(
            f"This equals {100*not_similar_answers/len(questions)} per cent that are dissimilar"
        )
        total_tests = len(questions)*2
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for question in questions:
            similar_question = get_similar_question_same(question)
            different_question = get_similar_question_different(question)
            similar_embedding =  question_bot.get_ada_embeddings(similar_embedding)
            different_embedding =  question_bot.get_ada_embeddings(different_embedding)
            _, _, similar_score = question_bot.find_most_similar_embedding(similar_embedding, embeddings_list)
            _, _, different_score = question_bot.find_most_similar_embedding(different_embedding, embeddings_list)
            if similar_score >= question_bot.THRESHOLD:
                true_positive += 1
            else:
                false_negative += 1
            if different_score >= question_bot.THRESHOLD:
                false_positive += 1
            else:
                true_negative += 1
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + true_negative)
            f1_score = 2 * (precision * recall)/(precision + recall)
            print ("compared with questions considered the same or different by an LLM, we get:")
            print (f"Precision: {precision}")
            print (f"Recall: {recall}")
            print (f"F1 score: {f1_score}")


openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"

openai.api_key = "de4d5adbc0af45bca22499dc3847b134"
openai.api_base = "https://ai-proxy.epam-rail.com"

deployment_name = "gpt-35-turbo"


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


def get_chatgpt_answer(question):
    """
    Generate an answer using the ChatGPT model for a given question.

    Args:
        question (str): The question for which an answer is to be generated.

    Returns:
        str: The generated answer from ChatGPT for the given question.
    """
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        temperature=0,
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message["content"]


def compare_answers(answer1, answer2):
    """Compare the similarity between two answers using a chosen
similarity metric.

        This function compares the similarity between two given
        answers using a specified similarity comparison logic. The
        function calculates a similarity score based on the chosen
        metric, which can include techniques like cosine similarity or
        pre-trained embeddings. The example provided here assumes a
        basic string similarity check using cosine similarity between
        the embeddings of the answers.

    Parameters:
        answer1 (str): The first answer for comparison.
        answer2 (str): The second answer for comparison.

        Returns:
        float: The similarity score between the two answers.

        Example:
        answer1 = "The capital of France is Paris."
        answer2 = "Paris is the capital of France."
        similarity = compare_answers(answer1, answer2)
        print(similarity)

        Output:
        0.85  # Example similarity score (not actual output)

    """

    embedding1 = question_bot.get_ada_emdeddings(answer1)
    embedding2 = question_bot.get_ada_emdeddings(answer2)
    similarity_score = 1 - np.cosine(np.array(embedding1), np.array(embedding2))
    return similarity_score


if __name__ == "__main__":
    test_find_most_similar_embedding_integration()
    unittest.main()
