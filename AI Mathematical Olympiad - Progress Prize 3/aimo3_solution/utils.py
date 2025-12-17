"""
Utility functions for AIMO3 solution
"""

import os
import json
import pickle
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import re


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_pickle(filepath: str) -> Any:
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, filepath: str):
    """Save data to pickle file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def ensure_dir(directory: str):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def extract_number_from_text(text: str) -> Optional[int]:
    """
    Extract a number from text, handling various formats
    
    Args:
        text: Input text containing a number
        
    Returns:
        Extracted integer or None if not found
    """
    # Remove LaTeX formatting
    text = text.replace('$', '').replace('\\', '')
    
    # Try to find numbers with various patterns
    patterns = [
        r'\b(\d+)\b',  # Simple integers
        r'answer[:\s]+(\d+)',  # "answer: 123" or "answer 123"
        r'final[:\s]+(\d+)',  # "final: 123"
        r'result[:\s]+(\d+)',  # "result: 123"
        r'=\s*(\d+)\s*$',  # Ends with "= 123"
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the last match (usually the final answer)
            return int(matches[-1])
    
    # If no pattern matches, try to extract any number
    numbers = re.findall(r'\d+', text)
    if numbers:
        return int(numbers[-1])
    
    return None


def validate_answer(answer: Any, min_val: int = 0, max_val: int = 99999) -> bool:
    """
    Validate if answer is in correct format and range
    
    Args:
        answer: Answer to validate
        min_val: Minimum valid value
        max_val: Maximum valid value
        
    Returns:
        True if answer is valid
    """
    if answer is None:
        return False
    
    try:
        answer_int = int(answer)
        return min_val <= answer_int <= max_val
    except (ValueError, TypeError):
        return False


def format_answer(answer: Union[int, float, str]) -> int:
    """
    Format answer to integer in valid range
    
    Args:
        answer: Answer to format
        
    Returns:
        Formatted integer answer
    """
    if isinstance(answer, str):
        answer = extract_number_from_text(answer)
    
    if answer is None:
        return 0
    
    answer_int = int(answer)
    
    # Clamp to valid range
    return max(0, min(99999, answer_int))


def parse_latex_math(text: str) -> str:
    """
    Parse LaTeX math expressions and convert to readable form
    
    Args:
        text: Text containing LaTeX
        
    Returns:
        Parsed text
    """
    # Remove $ delimiters
    text = re.sub(r'\$([^\$]+)\$', r'\1', text)
    
    # Replace common LaTeX commands
    replacements = {
        r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
        r'\\sqrt\{([^}]+)\}': r'sqrt(\1)',
        r'\\log': 'log',
        r'\\sin': 'sin',
        r'\\cos': 'cos',
        r'\\tan': 'tan',
        r'\\cdot': '*',
        r'\\times': '*',
        r'\\le': '<=',
        r'\\ge': '>=',
        r'\\neq': '!=',
        r'\\infty': 'infinity',
        r'\\_': '_',
        r'\\,': ' ',
        r'\\;': ' ',
        r'\\quad': ' ',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text


def count_latex_symbols(text: str) -> Dict[str, int]:
    """
    Count LaTeX symbols in text
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with symbol counts
    """
    symbols = {
        'equations': len(re.findall(r'\$[^\$]+\$', text)),
        'fractions': len(re.findall(r'\\frac', text)),
        'sqrt': len(re.findall(r'\\sqrt', text)),
        'sum': len(re.findall(r'\\sum', text)),
        'product': len(re.findall(r'\\prod', text)),
        'integral': len(re.findall(r'\\int', text)),
        'greek_letters': len(re.findall(r'\\[alpha|beta|gamma|delta|theta|phi|pi|sigma]', text)),
    }
    
    return symbols


def classify_problem_domain(problem_text: str) -> str:
    """
    Classify mathematical problem domain based on keywords
    
    Args:
        problem_text: Problem text
        
    Returns:
        Domain name: 'algebra', 'combinatorics', 'geometry', 'number_theory'
    """
    problem_lower = problem_text.lower()
    
    # Domain-specific keywords
    keywords = {
        'algebra': ['polynomial', 'equation', 'inequality', 'root', 'coefficient', 'solve', 'variable'],
        'combinatorics': ['permutation', 'combination', 'count', 'ways', 'arrangement', 'choose', 'sequence'],
        'geometry': ['triangle', 'circle', 'angle', 'line', 'point', 'area', 'perimeter', 'parallel', 'perpendicular'],
        'number_theory': ['divisor', 'prime', 'modulo', 'remainder', 'integer', 'gcd', 'lcm', 'congruent', 'factorial'],
    }
    
    scores = {}
    for domain, words in keywords.items():
        score = sum(1 for word in words if word in problem_lower)
        scores[domain] = score
    
    # Return domain with highest score
    if max(scores.values()) == 0:
        return 'algebra'  # Default
    
    return max(scores, key=scores.get)


def calculate_accuracy(predictions: List[int], targets: List[int]) -> float:
    """
    Calculate accuracy between predictions and targets
    
    Args:
        predictions: List of predicted answers
        targets: List of target answers
        
    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def load_reference_problems(reference_csv: str) -> List[Dict[str, Any]]:
    """
    Load reference problems from CSV
    
    Args:
        reference_csv: Path to reference CSV file
        
    Returns:
        List of problem dictionaries
    """
    import csv
    
    problems = []
    with open(reference_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.append({
                'id': row['id'],
                'problem': row['problem'],
                'answer': int(row['answer']),
            })
    
    return problems


def create_submission_csv(predictions: Dict[str, int], output_path: str):
    """
    Create submission CSV file
    
    Args:
        predictions: Dictionary mapping problem IDs to answers
        output_path: Path to save submission CSV
    """
    import csv
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer'])
        for problem_id, answer in sorted(predictions.items()):
            writer.writerow([problem_id, answer])
    
    logging.info(f"Submission CSV saved to {output_path}")


def clone_repo_with_pat(repo_url: str, pat: str, target_dir: str) -> bool:
    """
    Clone a repository using Personal Access Token
    
    Args:
        repo_url: Repository URL
        pat: Personal Access Token
        target_dir: Target directory
        
    Returns:
        True if successful
    """
    import subprocess
    
    # Insert PAT into URL
    if repo_url.startswith('https://github.com/'):
        auth_url = repo_url.replace('https://github.com/', f'https://{pat}@github.com/')
    else:
        auth_url = repo_url
    
    try:
        subprocess.run(
            ['git', 'clone', auth_url, target_dir],
            check=True,
            capture_output=True
        )
        logging.info(f"Successfully cloned {repo_url} to {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to clone repository: {e}")
        return False
