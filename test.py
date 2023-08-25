import unittest
from unittest.mock import MagicMock, patch
import question_bot # Import the module you want to test
import numpy as np
import random

# List of words for random generation
nouns = ["apple", "dog", "car", "banana", "computer"]
adjectives = ["red", "happy", "fast", "funny", "loud"]
verbs = ["run", "eat", "jump", "dance", "sleep"]

# Template-based generation
templates = [
    "What is the purpose of the {noun} that is {adjective}?",
    "Can you {verb} the {noun} while being {adjective}?",
    "Why does the {noun} look so {adjective} when it {verb}?"
]

def generate_random_question():
    choice = random.choice(templates)
    question = choice.format(
        noun=random.choice(nouns),
        adjective=random.choice(adjectives),
        verb=random.choice(verbs)
    )
    return question

class TestYourModule(unittest.TestCase):
    def setUp(self):
        self.dataset = {
            "train": [
                {"question": "What is the capital of France?", "answers": ["Paris"]},
                {"question": "Who wrote 'Romeo and Juliet'?", "answers": ["William Shakespeare"]},
            ],
            "test": [
                {"question": "What is 2 + 2?", "answers": ["4"]},
                {"question": "What is the largest planet?", "answers": ["Jupiter"]},
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
        
        dataset, questions, answers = question_bot.load_and_process_dataset()

        dataset_checksum = question_bot.calculate_checksum(questions + answers)
        embedding_pickle_path = 'embedding_test.pickle'
        embeddings_list = question_bot.load_or_create_embeddings(embedding_pickle_path, dataset_checksum, dataset)

        total_similarity = 0
        not_similar_answers = 0
        for i, example in enumerate(questions):
            user_embedding = question_bot.get_ada_embeddings(example)
            result_embedding, result_index, similarity_score = question_bot.find_most_similar_embedding(
                user_embedding, embeddings_list
            )

            assert similarity_score >= 0
            assert result_embedding is not None
            assert result_index >= 0
            assert result_index < len(embeddings_list)

            # Get the expected answer from your question-answering system
            expected_answer = answers[i]
            # Generate a ChatGPT answer using the same question
            chatgpt_answer = get_chatgpt_answer(example)  # Call the function you've defined earlier

            similarity_score = compare_answers(expected_answer, chatgpt_answer)
            is_answer_similar = similarity_score >= 0.80
            
            if not is_answer_similar:
                print(f"ChatGPT and database answers are not similar for question: {example}")
                print(f"Database answer: {expected_answer}")
                print(f"ChatGPT answer: {chatgpt_answer}")
                not_similar_answers += 1
            total_similarity += similarity_score
            
        for i in range(1000):
            bogus_question = generate_random_question()
            user_embedding = question_bot.get_ada_embeddings(bogus_question)
            result_embedding, result_index, similarity_score = question_bot.find_most_similar_embedding(
                user_embedding, embeddings_list
            )
            assert(result_embedding is not None)
            assert result_index >= 0
            assert result_index < len(embeddings_list)
            assert result_embedding is not None
            assert result_index >= 0
            assert similarity_score < question_bot.THRESHOLD
        print(f"Average similarity: {total_similarilty / len(questions}")
        print(f"{not_similar answers} out of {len(questions)} are dissimilar in ChatGPT vs database")
        print(f"This equals {100*not_similar_answers/len(questions)} per cent that are dissimilar")
        
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"

openai.api_key = "de4d5adbc0af45bca22499dc3847b134"
openai.api_base = "https://ai-proxy.epam-rail.com"

deployment_name = "gpt-35-turbo"

def get_chatgpt_answer(question):
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
    )
    return response.choices[0].message['content']

def compare_answers(expected_answer, chatgpt_answer):
    # Implement your similarity comparison logic here
    # You can use techniques like cosine similarity or pre-trained embeddings
    # For simplicity, let's assume a basic string similarity check
    embedding1 = question_bot.get_ada_emdeddings(expected_answer)
    embedding2 = question_bot.get_ada_emdeddings(chatgpt_answer)
    similarity_score = 1 - cosine(np.array(embedding1), np.array(embedding2))
    return similarity_score



if __name__ == "__main__":
    test_find_most_similar_embedding_integration()
    unittest.main()

