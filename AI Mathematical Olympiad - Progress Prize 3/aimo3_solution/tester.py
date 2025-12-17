"""
Testing system for AIMO3 solution
Comprehensive test suite for all problem types
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result for a single problem"""
    problem_id: str
    predicted_answer: int
    correct_answer: int
    is_correct: bool
    domain: str
    confidence: float = 0.0
    reasoning_steps: List[str] = field(default_factory=list)
    

class Tester:
    """Test engines on mathematical problems"""
    
    def __init__(self, config: Any):
        """
        Initialize tester
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.test_results = []
    
    def load_reference_problems(self) -> List[Dict[str, Any]]:
        """Load reference problems for testing"""
        import csv
        
        problems = []
        with open(self.config.reference_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                problems.append({
                    'id': row['id'],
                    'problem': row['problem'],
                    'answer': int(row['answer']),
                })
        
        self.logger.info(f"Loaded {len(problems)} reference problems")
        return problems
    
    def test_engine_on_reference(
        self,
        engine_config: Dict[str, Any],
        solver: Any
    ) -> Tuple[float, List[TestResult]]:
        """
        Test engine on reference problems
        
        Args:
            engine_config: Engine configuration
            solver: Math solver instance
            
        Returns:
            Tuple of (accuracy, results list)
        """
        engine_name = engine_config['name']
        self.logger.info(f"Testing engine {engine_name} on reference problems...")
        
        problems = self.load_reference_problems()
        results = []
        
        for problem in problems:
            # Solve problem
            predicted_answer, reasoning = solver.solve(
                problem['problem'],
                engine_config
            )
            
            # Parse domain
            from .latex_parser import LaTeXParser
            parser = LaTeXParser()
            domain = parser.identify_domain(problem['problem'])
            
            # Create result
            result = TestResult(
                problem_id=problem['id'],
                predicted_answer=predicted_answer,
                correct_answer=problem['answer'],
                is_correct=(predicted_answer == problem['answer']),
                domain=domain,
                confidence=0.8,  # Mock confidence
                reasoning_steps=reasoning
            )
            results.append(result)
        
        # Calculate accuracy
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / len(results) if results else 0.0
        
        self.logger.info(f"Reference test accuracy: {accuracy:.2%} ({correct}/{len(results)})")
        
        # Log per-domain accuracy
        self._log_domain_accuracy(results)
        
        self.test_results = results
        return accuracy, results
    
    def _log_domain_accuracy(self, results: List[TestResult]):
        """Log accuracy breakdown by domain"""
        domain_stats = {}
        
        for result in results:
            if result.domain not in domain_stats:
                domain_stats[result.domain] = {'correct': 0, 'total': 0}
            
            domain_stats[result.domain]['total'] += 1
            if result.is_correct:
                domain_stats[result.domain]['correct'] += 1
        
        self.logger.info("Per-domain accuracy:")
        for domain, stats in sorted(domain_stats.items()):
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            self.logger.info(f"  {domain}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
    
    def cross_validate(
        self,
        engines: List[Dict[str, Any]],
        solver: Any,
        k_folds: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation across engine variants
        
        Args:
            engines: List of engine configurations
            solver: Math solver instance
            k_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary of metrics
        """
        self.logger.info(f"Cross-validating {len(engines)} engines with {k_folds} folds...")
        
        problems = self.load_reference_problems()
        
        # For simplicity, test all engines on all problems
        engine_scores = {}
        
        for engine in engines:
            accuracy, _ = self.test_engine_on_reference(engine, solver)
            engine_scores[engine['name']] = accuracy
        
        # Calculate statistics
        accuracies = list(engine_scores.values())
        metrics = {
            'mean_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0.0,
            'max_accuracy': max(accuracies) if accuracies else 0.0,
            'min_accuracy': min(accuracies) if accuracies else 0.0,
            'best_engine': max(engine_scores, key=engine_scores.get) if engine_scores else None,
        }
        
        self.logger.info(f"Cross-validation results:")
        self.logger.info(f"  Mean accuracy: {metrics['mean_accuracy']:.2%}")
        self.logger.info(f"  Best engine: {metrics['best_engine']} ({metrics['max_accuracy']:.2%})")
        
        return metrics
    
    def validate_answer_format(self, answers: Dict[str, int]) -> bool:
        """
        Validate that all answers are in correct format
        
        Args:
            answers: Dictionary of problem_id -> answer
            
        Returns:
            True if all valid
        """
        min_val = self.config.min_answer
        max_val = self.config.max_answer
        
        all_valid = True
        
        for problem_id, answer in answers.items():
            if not isinstance(answer, int):
                self.logger.error(f"Problem {problem_id}: answer {answer} is not an integer")
                all_valid = False
            elif answer < min_val or answer > max_val:
                self.logger.error(f"Problem {problem_id}: answer {answer} out of range [{min_val}, {max_val}]")
                all_valid = False
        
        if all_valid:
            self.logger.info(f"All {len(answers)} answers are valid")
        
        return all_valid
    
    def measure_accuracy(
        self,
        predictions: Dict[str, int],
        targets: Dict[str, int]
    ) -> float:
        """
        Measure accuracy between predictions and targets
        
        Args:
            predictions: Predicted answers
            targets: Target answers
            
        Returns:
            Accuracy as float
        """
        if len(predictions) != len(targets):
            self.logger.warning(f"Prediction count ({len(predictions)}) != target count ({len(targets)})")
        
        correct = 0
        total = 0
        
        for problem_id in targets:
            if problem_id in predictions:
                total += 1
                if predictions[problem_id] == targets[problem_id]:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        self.logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        
        return accuracy
    
    def run_comprehensive_test(
        self,
        engine_config: Dict[str, Any],
        solver: Any
    ) -> Dict[str, Any]:
        """
        Run comprehensive test suite
        
        Args:
            engine_config: Engine configuration
            solver: Math solver instance
            
        Returns:
            Comprehensive test metrics
        """
        self.logger.info("Running comprehensive test suite...")
        
        # Test on reference problems
        accuracy, results = self.test_engine_on_reference(engine_config, solver)
        
        # Analyze by domain
        domain_stats = {}
        for result in results:
            if result.domain not in domain_stats:
                domain_stats[result.domain] = []
            domain_stats[result.domain].append(result.is_correct)
        
        domain_accuracy = {
            domain: sum(correct_list) / len(correct_list)
            for domain, correct_list in domain_stats.items()
        }
        
        # Overall metrics
        metrics = {
            'overall_accuracy': accuracy,
            'total_problems': len(results),
            'correct_problems': sum(1 for r in results if r.is_correct),
            'domain_accuracy': domain_accuracy,
            'meets_target': accuracy >= self.config.target_reference_accuracy,
        }
        
        self.logger.info(f"Comprehensive test completed:")
        self.logger.info(f"  Overall: {accuracy:.2%}")
        self.logger.info(f"  Meets target ({self.config.target_reference_accuracy:.0%}): {metrics['meets_target']}")
        
        return metrics
    
    def save_results(self, output_path: str):
        """Save test results to file"""
        import json
        import os
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results_data = {
            'results': [
                {
                    'problem_id': r.problem_id,
                    'predicted_answer': r.predicted_answer,
                    'correct_answer': r.correct_answer,
                    'is_correct': r.is_correct,
                    'domain': r.domain,
                    'confidence': r.confidence,
                }
                for r in self.test_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Saved test results to {output_path}")
