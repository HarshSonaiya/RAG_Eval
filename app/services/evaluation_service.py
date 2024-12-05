from openai import OpenAI
from config.settings import settings
import logging

logger = logging.getLogger("test_set_generator")  # Create a logger for this module

class Evaluation:
    def __init__(self, api_key):
        logger.info("Initializing TestSetGenerator with API key.")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )
        logger.info("OpenAI client initialized successfully.")
            
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
            
async def evaluate_response(retrieved: str, query: str, llm_response: str, ground_truth: str):
    """Evaluates the response of the language model and the retriever."""
    logger.info("Evaluating the response of the language model and the retriever.")

    user = Evaluation(api_key=settings.NVIDIA_API_KEY)

    validation_set = [{
        "question": query,
        "ground_truth": ground_truth,
        "retrieved_docs": retrieved,
        "llm_response": llm_response
    }]

    llm_eval = user.evaluate_llm(validation_set[0])
    retriever_eval = user.evaluate_retriever(validation_set[0])
    logger.info("Evaluation completed. Returning results.")
    print("Retriever_eval", retriever_eval)
    return llm_eval, retriever_eval
