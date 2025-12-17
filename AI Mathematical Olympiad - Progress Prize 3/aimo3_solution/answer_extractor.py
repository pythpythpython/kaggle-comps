"""
Answer extractor for AIMO3 solutions
Extracts integer answers from solution text
"""

import re
from typing import Optional, List, Tuple
import logging


logger = logging.getLogger(__name__)


class AnswerExtractor:
    """Extract integer answers from mathematical solution text"""
    
    def __init__(self, min_answer: int = 0, max_answer: int = 99999):
        """
        Initialize answer extractor
        
        Args:
            min_answer: Minimum valid answer
            max_answer: Maximum valid answer
        """
        self.min_answer = min_answer
        self.max_answer = max_answer
        self.logger = logging.getLogger(__name__)
    
    def extract(self, text: str) -> Optional[int]:
        """
        Extract answer from solution text
        
        Args:
            text: Solution text containing the answer
            
        Returns:
            Extracted integer answer or None if not found
        """
        # Try multiple extraction strategies in order of preference
        strategies = [
            self._extract_explicit_answer,
            self._extract_final_number,
            self._extract_remainder_format,
            self._extract_boxed_answer,
            self._extract_last_equation,
            self._extract_any_valid_number,
        ]
        
        for strategy in strategies:
            answer = strategy(text)
            if answer is not None and self._validate_answer(answer):
                self.logger.debug(f"Extracted answer {answer} using {strategy.__name__}")
                return answer
        
        self.logger.warning(f"Could not extract valid answer from text: {text[:100]}...")
        return None
    
    def extract_with_confidence(self, text: str) -> Tuple[Optional[int], float]:
        """
        Extract answer with confidence score
        
        Args:
            text: Solution text
            
        Returns:
            Tuple of (answer, confidence) where confidence is 0.0 to 1.0
        """
        # Strategy confidence weights
        strategies = [
            (self._extract_explicit_answer, 0.95),
            (self._extract_boxed_answer, 0.90),
            (self._extract_remainder_format, 0.85),
            (self._extract_final_number, 0.75),
            (self._extract_last_equation, 0.60),
            (self._extract_any_valid_number, 0.40),
        ]
        
        for strategy, confidence in strategies:
            answer = strategy(text)
            if answer is not None and self._validate_answer(answer):
                return answer, confidence
        
        return None, 0.0
    
    def _extract_explicit_answer(self, text: str) -> Optional[int]:
        """Extract explicitly marked answers"""
        patterns = [
            r'(?:final\s+)?answer\s*[:\s=]+\s*(\d+)',
            r'(?:the\s+)?solution\s+is\s*[:\s=]?\s*(\d+)',
            r'(?:therefore|thus|hence)[,\s]+(?:the\s+)?answer\s+is\s*[:\s=]?\s*(\d+)',
            r'\\boxed\{(\d+)\}',
            r'\\text\{answer\}\s*[:\s=]+\s*(\d+)',
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Return the last match
                return int(matches[-1])
        
        return None
    
    def _extract_boxed_answer(self, text: str) -> Optional[int]:
        """Extract answer from \\boxed{} command"""
        # Find boxed expressions
        boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', text)
        
        for match in reversed(boxed_matches):  # Start from last
            # Extract number from boxed content
            numbers = re.findall(r'\d+', match)
            if numbers:
                return int(numbers[-1])
        
        return None
    
    def _extract_remainder_format(self, text: str) -> Optional[int]:
        """Extract from remainder/modulo format"""
        patterns = [
            r'remainder\s+(?:is\s+)?(\d+)',
            r'≡\s*(\d+)\s*\(mod',
            r'mod\s+\d+\s*[=:]\s*(\d+)',
            r'divided\s+by\s+\d+\s+(?:is|gives)\s+(\d+)',
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                return int(matches[-1])
        
        return None
    
    def _extract_final_number(self, text: str) -> Optional[int]:
        """Extract the final number in the text"""
        # Look for standalone numbers in the last portion of text
        last_part = text[-500:]  # Last 500 characters
        
        # Find all numbers
        numbers = re.findall(r'\b(\d+)\b', last_part)
        
        if numbers:
            # Return the last number
            num = int(numbers[-1])
            if self._validate_answer(num):
                return num
        
        return None
    
    def _extract_last_equation(self, text: str) -> Optional[int]:
        """Extract from last equation or equality"""
        # Find equations with equals sign
        equations = re.findall(r'=\s*(\d+)\s*(?:[.\n]|$)', text)
        
        if equations:
            return int(equations[-1])
        
        return None
    
    def _extract_any_valid_number(self, text: str) -> Optional[int]:
        """Extract any valid number as last resort"""
        # Find all numbers in valid range
        numbers = re.findall(r'\b(\d+)\b', text)
        
        valid_numbers = []
        for num_str in numbers:
            num = int(num_str)
            if self._validate_answer(num):
                valid_numbers.append(num)
        
        if valid_numbers:
            # Return the last valid number
            return valid_numbers[-1]
        
        return None
    
    def _validate_answer(self, answer: int) -> bool:
        """
        Validate if answer is in correct range
        
        Args:
            answer: Answer to validate
            
        Returns:
            True if valid
        """
        return self.min_answer <= answer <= self.max_answer
    
    def format_answer(self, answer: Optional[int]) -> int:
        """
        Format answer to ensure it's in valid range
        
        Args:
            answer: Answer to format
            
        Returns:
            Valid integer answer (0 if None)
        """
        if answer is None:
            return 0
        
        # Clamp to valid range
        return max(self.min_answer, min(self.max_answer, answer))
    
    def extract_from_multiple_solutions(self, solutions: List[str]) -> Optional[int]:
        """
        Extract answer from multiple solution attempts using voting
        
        Args:
            solutions: List of solution texts
            
        Returns:
            Most common answer or None
        """
        answers = []
        
        for solution in solutions:
            answer = self.extract(solution)
            if answer is not None:
                answers.append(answer)
        
        if not answers:
            return None
        
        # Return most common answer
        from collections import Counter
        counter = Counter(answers)
        most_common = counter.most_common(1)[0][0]
        
        return most_common
    
    def extract_with_voting(
        self, 
        solutions: List[str],
        weights: Optional[List[float]] = None
    ) -> Tuple[Optional[int], float]:
        """
        Extract answer using weighted voting across multiple solutions
        
        Args:
            solutions: List of solution texts
            weights: Optional weights for each solution (default: equal weights)
            
        Returns:
            Tuple of (answer, confidence)
        """
        if not solutions:
            return None, 0.0
        
        if weights is None:
            weights = [1.0] * len(solutions)
        
        if len(weights) != len(solutions):
            raise ValueError("Number of weights must match number of solutions")
        
        # Extract answers with confidence
        answer_votes = {}
        total_weight = 0.0
        
        for solution, weight in zip(solutions, weights):
            answer, confidence = self.extract_with_confidence(solution)
            if answer is not None:
                vote_weight = weight * confidence
                answer_votes[answer] = answer_votes.get(answer, 0.0) + vote_weight
                total_weight += vote_weight
        
        if not answer_votes:
            return None, 0.0
        
        # Get answer with highest vote
        best_answer = max(answer_votes, key=answer_votes.get)
        best_confidence = answer_votes[best_answer] / total_weight if total_weight > 0 else 0.0
        
        return best_answer, best_confidence
