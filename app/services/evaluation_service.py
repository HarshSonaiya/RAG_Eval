from openai import OpenAI
from config.settings import settings
import logging

logger = logging.getLogger("test_set_generator")  # Create a logger for this module

class TestSetGenerator:
    def __init__(self, api_key):
        logger.info("Initializing TestSetGenerator with API key.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        logger.info("OpenAI client initialized successfully.")

    def create_test_set(self, chunks):
        """Generates questions and ground truths from the provided chunks."""
        logger.info("Starting to create test set from provided chunks.")
        question_ground_truth_pairs = {}

        for passage in chunks[:5]:
            logger.debug(f"Generating questions and ground truths for passage: {passage.page_content[:50]}...")  # Log the first 50 chars
            prompt = f'''
            You are an AI assistant for generating questions and ground truths based on the various passages from the user.
            Please generate questions and ground truths clearly labeled as follows:
                - Questions prefixed with "Q:"
                - Ground truths (answers) prefixed with "A:"
            The complexity of the questions should be 2 simple questions and 2 complex questions.
            Generate at least 4 question-ground_truth pairs based on the passage provided.

            Passage: {passage.page_content}
            '''

            try:
                chat_completion = self.client.chat.completions.create(
                    model="nvidia/nemotron-4-340b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    top_p=0.7,
                    max_tokens=1400,
                )
                response = chat_completion.choices[0].message.content
                logger.info("Successfully generated questions and ground truths.")
            except Exception as e:
                logger.error(f"Error generating questions and ground truths: {e}")
                continue  # Skip this passage and continue with the next

            # The response contains both questions and ground truths, let's parse them
            questions = []
            ground_truths = []

            # Split response by lines, looking for lines prefixed with "Q:" and "A:"
            response_lines = response.split("\n")
            for line in response_lines:
                line = line.strip()
                if line.startswith("Q:"):  # Question line
                    question = line.split("Q:", 1)[1].strip()
                    questions.append(question)
                elif line.startswith("A:"):  # Ground truth line
                    ground_truth = line.split("A:", 1)[1].strip()
                    ground_truths.append(ground_truth)

            # Populate the dictionary with questions as keys and ground truths as values
            for q, a in zip(questions, ground_truths):
                question_ground_truth_pairs[q] = a

        logger.info("Completed creating test set.")
        return question_ground_truth_pairs
    
    def evaluate_hybrid_llm(self, validation_set):
        """Evaluates the language model using the provided validation set."""
        logger.info(f"Evaluating LLM with the provided validation set type: {type(validation_set['retrieved_docs'])}")

        # retrieved_docs = "\n\n".join(doc.payload['content'] for doc in validation_set["retrieved_docs"])

        try:
            completion = self.client.chat.completions.create(
                model="nvidia/nemotron-4-340b-reward",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        user_query: {validation_set["question"]} Based on the below context answer the user's query
                        context: {validation_set["retrieved_docs"]}
                        Expected Answer: {validation_set["ground_truth"]}
                        """
                    },
                    {
                        "role": "assistant",
                        "content": validation_set["llm_response"]
                    }
                ],
            )
            response = completion.choices[0].message
            logger.info(f"Successfully evaluated LLM. {response}")
            return response
        except Exception as e:
            logger.error(f"Error evaluating LLM: {e}")
            return None
        
    def evaluate_llm(self, validation_set):
        """Evaluates the language model using the provided validation set."""

        try:
            completion = self.client.chat.completions.create(
                model="nvidia/nemotron-4-340b-reward",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        user_query: {validation_set["question"]} Based on the below context answer the user's query
                        context: {validation_set['retrieved_docs']}
                        Expected Answer: {validation_set["ground_truth"]}
                        """
                    },
                    {
                        "role": "assistant",
                        "content": validation_set["llm_response"]
                    }
                ],
            )
            response = completion.choices[0].message
            logger.info(f"Successfully evaluated LLM. {response}")
            return response
        except Exception as e:
            logger.error(f"Error evaluating LLM: {e}")
            return None
        

        
    def evaluate_retriever(self, validation_set):
        """Evaluates the document retriever using the provided validation set."""
        logger.info("Evaluating document retriever with the provided validation set.")
        
    
        try:
            completion = self.client.chat.completions.create(
                model="nvidia/nemotron-4-340b-reward",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        Question: {validation_set["question"]}
                        Expected Answer: {validation_set["ground_truth"]}
                        """
                    },
                    {
                        "role": "assistant",
                        "content":validation_set["retrieved_docs"]
                    }
                ]
            )
            response = completion.choices[0].message
            logger.info("Successfully evaluated retriever.")
            return response
        except Exception as e:
            logger.error(f"Error evaluating retriever: {e}")
            return None
        
    def evaluate_hybrid_retriever(self, validation_set):
        """Evaluates the document retriever using the provided validation set."""
        logger.info("Evaluating document retriever with the provided validation set.")
        logger.info(f"Question: {validation_set['question']}")
        logger.info(f"Expected Answer: {validation_set['ground_truth']}")
        logger.info(f"Retrieved Docs: {validation_set['retrieved_docs']}")

        # retrieved_docs = "\n\n".join(doc.payload['content'] for doc in validation_set["retrieved_docs"])

        try:
            completion = self.client.chat.completions.create(
                model="nvidia/nemotron-4-340b-reward",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        Question: {validation_set["question"]}
                        Expected Answer: {validation_set["ground_truth"]}
                        """
                    },
                    {
                        "role": "assistant",
                        "content": validation_set["retrieved_docs"]
                    }
                ]
            )
            response = completion.choices[0].message
            logger.info("Successfully evaluated retriever.")
            return response
        except Exception as e:
            logger.error(f"Error evaluating retriever: {e}")
            return None

    
    def generate_ground_truth(self, query: str):
        """Generates ground truth based on the provided query."""
        logger.info("Generating ground truth based on the provided query.")
        prompt = f'''
        You are an AI assistant for generating ground truth based on the user query and your knowledge.
        Please ground truths clearly labeled as follows:
            - Ground truths (answers) prefixed with "A:"
    
        Query: {query}
        '''

        try:
            chat_completion = self.client.chat.completions.create(
                model="nvidia/nemotron-4-340b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                top_p=0.7,
                max_tokens=1400,
            )
            response = chat_completion.choices[0].message.content
            logger.info("Successfully generated ground truth.")
            return response
        except Exception as e:
            logger.error(f"Error generating ground truth: {e}")
            return None

def clean_and_truncate_input(docs, max_tokens=1400):
        """Cleans and truncates the retrieved documents."""
        # Function to count tokens
        def count_tokens(text):
            return len(text.split())

        # Clean up unnecessary text (e.g., remove excessive whitespace)
        cleaned_docs = ' '.join(docs.split()).strip()

        # Check the token count and truncate if needed
        token_count = count_tokens(cleaned_docs)
        if token_count > max_tokens:
            # Truncate to fit within the token limit
            truncated_docs = " ".join(cleaned_docs.split()[:max_tokens])
            return truncated_docs
        else:
            return cleaned_docs
        
async def evaluate_response(retrieved: str, query: str, llm_response):
    """Evaluates the response of the language model and the retriever."""
    logger.info("Evaluating the response of the language model and the retriever.")
    user = TestSetGenerator(api_key=settings.NVIDIA_API_KEY)
    ground_truth = user.generate_ground_truth(query)

    cleaned_docs = clean_and_truncate_input(retrieved)

    validation_set = [{
        "question": query,
        "ground_truth": ground_truth,
        "retrieved_docs": cleaned_docs,
        "llm_response": llm_response
    }]

    llm_eval = user.evaluate_llm(validation_set[0])
    retriever_eval = user.evaluate_retriever(validation_set[0])
    logger.info("Evaluation completed. Returning results.")
    print("Retriever_eval", retriever_eval)
    return llm_eval, retriever_eval


async def evaluate_hybrid_response(retrieved: str, query: str, llm_response):
    """Evaluates the response of the language model and the retriever."""
    logger.info("Evaluating the response of the language model and the retriever.")
    user = TestSetGenerator(api_key=settings.NVIDIA_API_KEY)
    ground_truth = user.generate_ground_truth(query)

    validation_set = [{
        "question": query,
        "ground_truth": ground_truth,
        "retrieved_docs": retrieved,
        "llm_response": llm_response
    }]

    llm_eval = user.evaluate_hybrid_llm(validation_set[0])
    retriever_eval = user.evaluate_hybrid_retriever(validation_set[0])
    logger.info("Evaluation completed. Returning results.")
    print("Retriever_eval", retriever_eval)
    return llm_eval, retriever_eval
