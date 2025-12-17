"""
Quality checker for AIMO3 solution
Validates 100% quality on reference problems
"""

import logging
import os
from typing import Dict, List, Any, Tuple
import time


logger = logging.getLogger(__name__)


class QualityChecker:
    """Validate solution quality and compliance"""
    
    def __init__(self, config: Any):
        """
        Initialize quality checker
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
    
    def check_reference_accuracy(
        self,
        predictions: Dict[str, int],
        reference_path: str = None
    ) -> Tuple[bool, float, List[str]]:
        """
        Check accuracy on reference problems
        
        Args:
            predictions: Dictionary of problem_id -> answer
            reference_path: Path to reference CSV (optional)
            
        Returns:
            Tuple of (passed, accuracy, failed_problem_ids)
        """
        if reference_path is None:
            reference_path = self.config.reference_csv
        
        self.logger.info("Checking reference problem accuracy...")
        
        # Load reference answers
        import csv
        reference_answers = {}
        with open(reference_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                reference_answers[row['id']] = int(row['answer'])
        
        # Compare predictions
        correct = 0
        total = 0
        failed_ids = []
        
        for problem_id, correct_answer in reference_answers.items():
            total += 1
            predicted = predictions.get(problem_id, None)
            
            if predicted == correct_answer:
                correct += 1
            else:
                failed_ids.append(problem_id)
                self.logger.warning(f"Problem {problem_id}: predicted {predicted}, expected {correct_answer}")
        
        accuracy = correct / total if total > 0 else 0.0
        passed = accuracy >= self.config.target_reference_accuracy
        
        self.logger.info(f"Reference accuracy: {accuracy:.2%} ({correct}/{total})")
        
        if passed:
            self.logger.info("✓ Reference accuracy check PASSED")
        else:
            self.logger.error(f"✗ Reference accuracy check FAILED (target: {self.config.target_reference_accuracy:.0%})")
        
        return passed, accuracy, failed_ids
    
    def validate_answer_format(
        self,
        answers: Dict[str, int]
    ) -> Tuple[bool, List[str]]:
        """
        Validate answer format (integers 0-99999)
        
        Args:
            answers: Dictionary of problem_id -> answer
            
        Returns:
            Tuple of (passed, error_messages)
        """
        self.logger.info("Validating answer format...")
        
        errors = []
        min_val = self.config.min_answer
        max_val = self.config.max_answer
        
        for problem_id, answer in answers.items():
            # Check type
            if not isinstance(answer, int):
                errors.append(f"Problem {problem_id}: answer {answer} is not an integer")
                continue
            
            # Check range
            if answer < min_val or answer > max_val:
                errors.append(f"Problem {problem_id}: answer {answer} out of range [{min_val}, {max_val}]")
        
        passed = len(errors) == 0
        
        if passed:
            self.logger.info(f"✓ All {len(answers)} answers have valid format")
        else:
            self.logger.error(f"✗ Format validation failed with {len(errors)} errors")
            for error in errors[:10]:  # Show first 10 errors
                self.logger.error(f"  {error}")
        
        return passed, errors
    
    def check_runtime_limits(
        self,
        elapsed_time: float,
        use_gpu: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if runtime is within limits
        
        Args:
            elapsed_time: Elapsed time in seconds
            use_gpu: Whether GPU was used
            
        Returns:
            Tuple of (passed, message)
        """
        self.logger.info("Checking runtime limits...")
        
        elapsed_hours = elapsed_time / 3600.0
        
        if use_gpu:
            limit_hours = self.config.max_gpu_time_hours
            limit_type = "GPU"
        else:
            limit_hours = self.config.max_cpu_time_hours
            limit_type = "CPU"
        
        passed = elapsed_hours <= limit_hours
        
        message = f"{limit_type} runtime: {elapsed_hours:.2f}h / {limit_hours}h limit"
        
        if passed:
            self.logger.info(f"✓ {message}")
        else:
            self.logger.error(f"✗ {message} - EXCEEDED")
        
        return passed, message
    
    def check_offline_compatibility(self) -> Tuple[bool, List[str]]:
        """
        Check if solution can run offline
        
        Returns:
            Tuple of (passed, warnings)
        """
        self.logger.info("Checking offline compatibility...")
        
        warnings = []
        
        # Check for network imports (mock check)
        # In real implementation, would scan code for network calls
        
        # Check dependencies are available
        required_packages = ['csv', 'json', 'logging', 'random', 're']
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            warnings.append(f"Missing required packages: {', '.join(missing)}")
        
        passed = len(warnings) == 0
        
        if passed:
            self.logger.info("✓ Solution appears offline-compatible")
        else:
            self.logger.warning("⚠ Potential offline compatibility issues:")
            for warning in warnings:
                self.logger.warning(f"  {warning}")
        
        return passed, warnings
    
    def run_comprehensive_quality_check(
        self,
        predictions: Dict[str, int],
        elapsed_time: float = 0.0,
        use_gpu: bool = False
    ) -> Dict[str, Any]:
        """
        Run comprehensive quality check
        
        Args:
            predictions: Dictionary of predictions
            elapsed_time: Elapsed time in seconds
            use_gpu: Whether GPU was used
            
        Returns:
            Comprehensive quality report
        """
        self.logger.info("=" * 60)
        self.logger.info("COMPREHENSIVE QUALITY CHECK")
        self.logger.info("=" * 60)
        
        results = {}
        
        # 1. Reference accuracy
        ref_passed, ref_accuracy, failed_ids = self.check_reference_accuracy(predictions)
        results['reference_accuracy'] = {
            'passed': ref_passed,
            'accuracy': ref_accuracy,
            'failed_problems': failed_ids,
        }
        
        # 2. Answer format validation
        format_passed, format_errors = self.validate_answer_format(predictions)
        results['answer_format'] = {
            'passed': format_passed,
            'errors': format_errors,
        }
        
        # 3. Runtime limits
        if elapsed_time > 0:
            runtime_passed, runtime_msg = self.check_runtime_limits(elapsed_time, use_gpu)
            results['runtime'] = {
                'passed': runtime_passed,
                'message': runtime_msg,
                'elapsed_hours': elapsed_time / 3600.0,
            }
        
        # 4. Offline compatibility
        offline_passed, offline_warnings = self.check_offline_compatibility()
        results['offline_compatibility'] = {
            'passed': offline_passed,
            'warnings': offline_warnings,
        }
        
        # Overall pass/fail
        all_passed = (
            ref_passed and
            format_passed and
            (runtime_passed if elapsed_time > 0 else True) and
            offline_passed
        )
        
        results['overall_passed'] = all_passed
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("QUALITY CHECK SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Reference Accuracy: {'✓ PASS' if ref_passed else '✗ FAIL'} ({ref_accuracy:.2%})")
        self.logger.info(f"Answer Format: {'✓ PASS' if format_passed else '✗ FAIL'}")
        if elapsed_time > 0:
            self.logger.info(f"Runtime Limits: {'✓ PASS' if runtime_passed else '✗ FAIL'}")
        self.logger.info(f"Offline Compatible: {'✓ PASS' if offline_passed else '⚠ WARNING'}")
        self.logger.info("=" * 60)
        
        if all_passed:
            self.logger.info("✓ ALL QUALITY CHECKS PASSED")
        else:
            self.logger.error("✗ SOME QUALITY CHECKS FAILED")
        
        self.logger.info("=" * 60)
        
        self.validation_results = results
        return results
    
    def save_quality_report(self, output_path: str):
        """
        Save quality report to file
        
        Args:
            output_path: Path to save report
        """
        import json
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        self.logger.info(f"Quality report saved to {output_path}")
