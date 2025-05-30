import os
import sys
import json
import time
import math
from typing import List, Dict
from datasets import Dataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRelevance,
    ContextRecall,
    ContextPrecision
)
from ragas import evaluate

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.retriever import RAG
from src.config import Config

class QAEvaluator:
    def __init__(self, config: Config):
        self.config = config
        self.rag = RAG()
        self.rate_limit_delay = 45  
        
    def load_test_data(self, test_file: str) -> List[Dict]:
        with open(test_file, 'r') as f:
            return json.load(f)

    def get_response_with_retry(self, question: str, course: str, max_retries: int = 3) -> Dict:
        for attempt in range(max_retries):
            try:
                response, contexts = self.rag.get_response_with_context(
                    user_message=question,
                    history=[],
                    selected_course=course
                )
                time.sleep(self.rate_limit_delay)
                return {
                    "answer": response,
                    "contexts": contexts
                }
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                    wait_time = self.rate_limit_delay * (attempt + 1)
                    print(f"Rate limit hit, waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise e

    def prepare_dataset(self, test_data: List[Dict]) -> Dataset:
        questions, ground_truths, contexts, answers, courses = [], [], [], [], []

        total_questions = len(test_data)
        for idx, test_case in enumerate(test_data, 1):
            print(f"Processing question {idx}/{total_questions}")
            question = test_case['user_input']
            ground_truth = test_case['expected_response']
            course = test_case['course']

            response_data = self.get_response_with_retry(question, course)
            answer = response_data["answer"]
            context = response_data["contexts"]

            questions.append(question)
            ground_truths.append(ground_truth)
            contexts.append(context)
            answers.append(answer)
            courses.append(course)

            if idx < total_questions:
                time.sleep(1)

        return Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "contexts": contexts,
            "course": courses
        })

    def convert_metrics_to_dict(self, metrics_result) -> Dict[str, float]:
        print("Raw _repr_dict:", metrics_result._repr_dict)
        return {
            k: float(v) for k, v in metrics_result._repr_dict.items()
            if isinstance(v, (int, float)) and not math.isnan(v)
        }


    def evaluate_all(self, test_data: List[Dict]) -> Dict:
        dataset = self.prepare_dataset(test_data)

        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextRelevance(),
            ContextRecall(),
            ContextPrecision()
        ]

        result = evaluate(dataset, metrics=metrics)
        overall_metrics = self.convert_metrics_to_dict(result)

        course_metrics = {}
        for course in set(dataset['course']):
            course_indices = [i for i, c in enumerate(dataset['course']) if c == course]
            course_dataset = dataset.select(course_indices)
            course_result = evaluate(course_dataset, metrics=metrics)
            course_metrics[course] = self.convert_metrics_to_dict(course_result)

        return {
            'overall_metrics': overall_metrics,
            'course_specific_metrics': course_metrics,
            'detailed_results': dataset.to_dict()
        }

    def save_evaluation_results(self, results: Dict, output_file: str):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    config = Config()
    evaluator = QAEvaluator(config)
    test_data = evaluator.load_test_data('evaluation_benchmark.json')
    results = evaluator.evaluate_all(test_data)
    evaluator.save_evaluation_results(results, 'evaluation_results.json')

    print("\nOverall Evaluation Summary:")
    for metric_name, score in results['overall_metrics'].items():
        print(f"{metric_name}: {score:.3f}")

    print("\nCourse-Specific Metrics:")
    for course, metrics in results['course_specific_metrics'].items():
        print(f"\n{course}:")
        for metric_name, score in metrics.items():
            print(f"  {metric_name}: {score:.3f}")

if __name__ == "__main__":
    main()
