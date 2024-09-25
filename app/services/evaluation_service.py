from openai import OpenAI
from app.config.settings import settings

class TestSetGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://integrate.api.nvidia.com/v1"
        )

    def create_test_set(self, chunks):
        """Generates questions and ground truths from the provided chunks."""
        question_ground_truth_pairs = {}

        for passage in chunks[:5]:
            prompt = f'''
            You are an AI assistant for generating questions and ground truths based on the various passages from the user.
            Please generate questions and ground truths clearly labeled as follows:
                - Questions prefixed with "Q:"
                - Ground truths (answers) prefixed with "A:"
            The complexity of the questions should be 2 simple questions and 2 complex questions.
            Generate at least 4 question-ground_truth pairs based on the passage provided.

            Passage: {passage.page_content}
            '''

            chat_completion = self.client.chat.completions.create(
                model="nvidia/nemotron-4-340b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                top_p=0.7,
                max_tokens=1400,
            )

            response = chat_completion.choices[0].message.content

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

        return question_ground_truth_pairs

    def evaluate_llm(self, validation_set):
        """Evaluates the language model using the provided validation set."""
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
        return response

    def evaluate_retriever(self, validation_set):
        """Evaluates the document retriever using the provided validation set."""
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
        return response

    def generate_ground_truth(self, query: str):
        """Generates ground truth based on the provided query."""
        prompt = f'''
        You are an AI assistant for generating ground truth based on the user query and your knowledge.
        Please ground truths clearly labeled as follows:
            - Ground truths (answers) prefixed with "A:"
    
        Query: {query}
        '''

        chat_completion = self.client.chat.completions.create(
            model="nvidia/nemotron-4-340b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            top_p=0.7,
            max_tokens=1400,
        )

        response = chat_completion.choices[0].message.content
        return response


def evaluate_response(retrieved: str, query: str, hybrid_response):
    """Evaluates the response of the language model and the retriever."""
    user = TestSetGenerator(api_key=settings.NVIDIA_API_KEY)
    ground_truth = user.generate_ground_truth(query)

    validation_set = [{
        "question": query,
        "ground_truth": ground_truth,
        "retrieved_docs": retrieved,
        "llm_response": hybrid_response
    }]

    llm_eval = user.evaluate_llm(validation_set[0])
    retriever_eval = user.evaluate_retriever(validation_set[0])
    print("Retriever_eval", retriever_eval)
    return llm_eval, retriever_eval
