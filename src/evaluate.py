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
        self.max_retries = 5 
        
    def load_test_data(self, test_file: str) -> List[Dict]:
        with open(test_file, 'r') as f:
            return json.load(f)

    def get_response_with_retry(self, question: str, course: str, max_retries: int = 5) -> Dict:
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
                if attempt < max_retries - 1:
                    wait_time = self.rate_limit_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Error occurred: {str(e)}")
                    print(f"Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to get response after {max_retries} attempts: {str(e)}")
                    return {
                        "answer": "ERROR: Failed to get response",
                        "contexts": []
                    }

    def prepare_dataset(self, test_data: List[Dict]) -> Dataset:
        questions, ground_truths, contexts, answers, courses = [], [], [], [], []
        progress_file = 'evaluation_progress.json'

        # Try to load previous progress
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                questions = progress.get('questions', [])
                ground_truths = progress.get('ground_truths', [])
                contexts = progress.get('contexts', [])
                answers = progress.get('answers', [])
                courses = progress.get('courses', [])
                start_idx = len(questions)
        except FileNotFoundError:
            start_idx = 0

        total_questions = len(test_data)
        for idx, test_case in enumerate(test_data[start_idx:], start_idx + 1):
            print(f"Processing question {idx}/{total_questions}")
            question = test_case['user_input']
            ground_truth = test_case['expected_response']
            course = test_case['course']

            try:
                response_data = self.get_response_with_retry(question, course)
                answer = response_data["answer"]
                context = response_data["contexts"]

                questions.append(question)
                ground_truths.append(ground_truth)
                contexts.append(context)
                answers.append(answer)
                courses.append(course)

                # Save progress after each successful question
                with open(progress_file, 'w') as f:
                    json.dump({
                        'questions': questions,
                        'ground_truths': ground_truths,
                        'contexts': contexts,
                        'answers': answers,
                        'courses': courses
                    }, f)

                if idx < total_questions:
                    time.sleep(1)
            except Exception as e:
                print(f"Error processing question {idx}: {str(e)}")
                continue

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
        try:
            dataset = self.prepare_dataset(test_data)

            metrics = [
                Faithfulness(),
                AnswerRelevancy(),
                ContextRelevance(),
                ContextRecall(),
                ContextPrecision()
            ]

            # Evaluate each course separately
            course_metrics = {}
            for course in set(dataset['course']):
                print(f"\nEvaluating {course}...")
                course_indices = [i for i, c in enumerate(dataset['course']) if c == course]
                course_dataset = dataset.select(course_indices)
                course_result = evaluate(course_dataset, metrics=metrics)
                course_metrics[course] = self.convert_metrics_to_dict(course_result)

            # Calculate overall metrics as average of course metrics
            overall_metrics = {}
            for metric in course_metrics[list(course_metrics.keys())[0]].keys():
                values = [metrics[metric] for metrics in course_metrics.values()]
                overall_metrics[metric] = sum(values) / len(values)

            return {
                'overall_metrics': overall_metrics,
                'course_specific_metrics': course_metrics,
                'detailed_results': dataset.to_dict()
            }
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return {
                'overall_metrics': {},
                'course_specific_metrics': {},
                'detailed_results': {},
                'error': str(e)
            }

    def save_evaluation_results(self, results: Dict, output_file: str):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    config = Config()
    evaluator = QAEvaluator(config)
    test_data = evaluator.load_test_data('evaluation/evaluation_benchmark.json')
    results = evaluator.evaluate_all(test_data)
    evaluator.save_evaluation_results(results, 'evaluation/evaluation_results.json')

    if 'error' in results:
        print(f"\nEvaluation failed with error: {results['error']}")
    else:
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
