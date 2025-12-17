"""
LaTeX parser for AIMO3 mathematical problems
Handles LaTeX notation following AIMO3 conventions
"""

import re
from typing import Dict, List, Optional, Tuple
import logging


logger = logging.getLogger(__name__)


class LaTeXParser:
    """Parser for LaTeX mathematical notation in AIMO3 problems"""
    
    def __init__(self):
        """Initialize LaTeX parser"""
        self.logger = logging.getLogger(__name__)
        
        # LaTeX command patterns
        self.latex_patterns = {
            # Math environments
            'inline_math': r'\$([^\$]+)\$',
            'display_math': r'\$\$([^\$]+)\$\$',
            'equation': r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
            'align': r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
            
            # Common commands
            'frac': r'\\frac\{([^}]+)\}\{([^}]+)\}',
            'sqrt': r'\\sqrt(?:\[([^]]+)\])?\{([^}]+)\}',
            'overline': r'\\overline\{([^}]+)\}',
            'binom': r'\\binom\{([^}]+)\}\{([^}]+)\}',
            'floor': r'\\lfloor([^\\]+)\\rfloor',
            'ceil': r'\\lceil([^\\]+)\\rceil',
            
            # Sets
            'mathbb': r'\\mathbb\{([A-Z])\}',
            'set': r'\\\{([^}]+)\\\}',
            
            # Greek letters
            'alpha': r'\\alpha',
            'beta': r'\\beta',
            'gamma': r'\\gamma',
            'delta': r'\\delta',
            'theta': r'\\theta',
            'pi': r'\\pi',
            'phi': r'\\phi',
            'sigma': r'\\sigma',
            'omega': r'\\omega',
        }
    
    def parse(self, text: str) -> str:
        """
        Parse LaTeX text and convert to readable format
        
        Args:
            text: LaTeX formatted text
            
        Returns:
            Parsed text
        """
        parsed = text
        
        # Extract math environments
        math_blocks = self._extract_math_blocks(parsed)
        
        # Process inline math
        parsed = self._process_inline_math(parsed)
        
        # Replace LaTeX commands
        parsed = self._replace_latex_commands(parsed)
        
        # Clean up
        parsed = self._cleanup(parsed)
        
        return parsed
    
    def extract_variables(self, text: str) -> List[str]:
        """
        Extract mathematical variables from LaTeX text
        
        Args:
            text: LaTeX text
            
        Returns:
            List of variable names
        """
        variables = set()
        
        # Find variables in math mode
        math_matches = re.findall(r'\$([^\$]+)\$', text)
        for match in math_matches:
            # Extract single letters and subscripted variables
            vars_found = re.findall(r'\b([a-zA-Z])(?:_\{?(\w+)\}?)?\b', match)
            for var, subscript in vars_found:
                if subscript:
                    variables.add(f"{var}_{subscript}")
                else:
                    variables.add(var)
        
        return sorted(list(variables))
    
    def extract_equations(self, text: str) -> List[str]:
        """
        Extract equations from LaTeX text
        
        Args:
            text: LaTeX text
            
        Returns:
            List of equations
        """
        equations = []
        
        # Extract from equation environments
        equation_matches = re.findall(
            r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
            text,
            re.DOTALL
        )
        equations.extend(equation_matches)
        
        # Extract from align environments
        align_matches = re.findall(
            r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
            text,
            re.DOTALL
        )
        equations.extend(align_matches)
        
        # Extract inline equations with equality
        inline_matches = re.findall(r'\$([^$]*=[^$]*)\$', text)
        equations.extend(inline_matches)
        
        return equations
    
    def extract_numbers(self, text: str) -> List[int]:
        """
        Extract numerical values from text
        
        Args:
            text: Text to parse
            
        Returns:
            List of numbers found
        """
        # Remove LaTeX formatting
        cleaned = re.sub(r'\\[a-zA-Z]+', '', text)
        cleaned = cleaned.replace('$', '')
        
        # Find all integers
        numbers = re.findall(r'\b\d+\b', cleaned)
        
        return [int(n) for n in numbers]
    
    def identify_domain(self, text: str) -> str:
        """
        Identify mathematical domain of problem
        
        Args:
            text: Problem text
            
        Returns:
            Domain: 'algebra', 'combinatorics', 'geometry', 'number_theory'
        """
        text_lower = text.lower()
        
        # Domain keywords
        domain_keywords = {
            'algebra': [
                'polynomial', 'equation', 'inequality', 'solve', 'root',
                'coefficient', 'variable', 'quadratic', 'linear'
            ],
            'combinatorics': [
                'permutation', 'combination', 'ways', 'count', 'arrange',
                'choose', 'sequence', 'tournament', 'graph'
            ],
            'geometry': [
                'triangle', 'circle', 'angle', 'line', 'point', 'area',
                'perimeter', 'parallel', 'perpendicular', 'tangent',
                'circumcircle', 'incircle'
            ],
            'number_theory': [
                'divisor', 'prime', 'modulo', 'remainder', 'integer',
                'gcd', 'lcm', 'congruent', 'factorial', 'coprime'
            ],
        }
        
        scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[domain] = score
        
        # Return domain with highest score
        if max(scores.values()) == 0:
            return 'algebra'  # Default
        
        return max(scores, key=scores.get)
    
    def _extract_math_blocks(self, text: str) -> List[str]:
        """Extract math environment blocks"""
        blocks = []
        
        # Display math
        blocks.extend(re.findall(r'\$\$([^\$]+)\$\$', text))
        
        # Equation environments
        blocks.extend(re.findall(
            r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}',
            text,
            re.DOTALL
        ))
        
        # Align environments
        blocks.extend(re.findall(
            r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}',
            text,
            re.DOTALL
        ))
        
        return blocks
    
    def _process_inline_math(self, text: str) -> str:
        """Process inline math expressions"""
        # Keep $ notation but simplify content
        def simplify_inline(match):
            content = match.group(1)
            # Keep it simple - just return with markers
            return f"${content}$"
        
        return re.sub(r'\$([^\$]+)\$', simplify_inline, text)
    
    def _replace_latex_commands(self, text: str) -> str:
        """Replace LaTeX commands with readable equivalents"""
        replacements = {
            # Fractions
            r'\\frac\{([^}]+)\}\{([^}]+)\}': r'(\1)/(\2)',
            
            # Square roots
            r'\\sqrt\{([^}]+)\}': r'sqrt(\1)',
            
            # Binomial coefficient
            r'\\binom\{([^}]+)\}\{([^}]+)\}': r'C(\1,\2)',
            
            # Floor/Ceiling
            r'\\lfloor': '⌊',
            r'\\rfloor': '⌋',
            r'\\lceil': '⌈',
            r'\\rceil': '⌉',
            
            # Math sets
            r'\\mathbb\{N\}': 'ℕ',
            r'\\mathbb\{Z\}': 'ℤ',
            r'\\mathbb\{Q\}': 'ℚ',
            r'\\mathbb\{R\}': 'ℝ',
            
            # Operations
            r'\\cdot': '·',
            r'\\times': '×',
            r'\\div': '÷',
            
            # Relations
            r'\\le': '≤',
            r'\\ge': '≥',
            r'\\neq': '≠',
            r'\\approx': '≈',
            r'\\equiv': '≡',
            
            # Greek letters
            r'\\alpha': 'α',
            r'\\beta': 'β',
            r'\\gamma': 'γ',
            r'\\delta': 'δ',
            r'\\theta': 'θ',
            r'\\pi': 'π',
            r'\\phi': 'φ',
            r'\\sigma': 'σ',
            r'\\omega': 'ω',
            
            # Functions
            r'\\log': 'log',
            r'\\ln': 'ln',
            r'\\sin': 'sin',
            r'\\cos': 'cos',
            r'\\tan': 'tan',
            
            # Misc
            r'\\infty': '∞',
            r'\\sum': '∑',
            r'\\prod': '∏',
            r'\\_': '_',
        }
        
        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _cleanup(self, text: str) -> str:
        """Clean up parsed text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove remaining backslashes before common characters
        text = text.replace('\\{', '{').replace('\\}', '}')
        
        return text.strip()
    
    def parse_problem(self, problem_dict: Dict) -> Dict:
        """
        Parse a complete problem dictionary
        
        Args:
            problem_dict: Dictionary with 'id', 'problem', optionally 'answer'
            
        Returns:
            Enhanced problem dictionary with parsed fields
        """
        parsed = problem_dict.copy()
        
        problem_text = problem_dict['problem']
        
        parsed['parsed_text'] = self.parse(problem_text)
        parsed['variables'] = self.extract_variables(problem_text)
        parsed['equations'] = self.extract_equations(problem_text)
        parsed['numbers'] = self.extract_numbers(problem_text)
        parsed['domain'] = self.identify_domain(problem_text)
        
        return parsed
