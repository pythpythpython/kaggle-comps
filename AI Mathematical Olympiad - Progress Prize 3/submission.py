"""
Main Kaggle submission script for AIMO3
Integrates with Kaggle evaluation API for offline problem solving
"""

import os
import sys
import logging
import csv
from typing import Dict, List

# Add aimo3_solution to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'aimo3_solution'))

from aimo3_solution.config import Config
from aimo3_solution.math_solver import MathSolver
from aimo3_solution.latex_parser import LaTeXParser
from aimo3_solution.answer_extractor import AnswerExtractor
from aimo3_solution.engine_breeder import EngineBreeder
from aimo3_solution.utils import setup_logging, create_submission_csv

# Try to import MathOlympiadAGI
try:
    from aimo3_solution.math_olympiad_agi import MathOlympiadAGI
    MATH_OLYMPIAD_AGI_AVAILABLE = True
except ImportError:
    MATH_OLYMPIAD_AGI_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning("MathOlympiadAGI not available, using fallback")


# Setup logging
logger = setup_logging(log_file='aimo3_submission.log', level=logging.INFO)


def load_best_engine_config(config: Config) -> Dict:
    """Load the best trained engine configuration"""
    import json
    
    # Try to load from MathOlympiadAGI first
    agi_path = os.path.join(config.trained_engines_dir, 'math_olympiad_agi.json')
    
    if os.path.exists(agi_path) and MATH_OLYMPIAD_AGI_AVAILABLE:
        logger.info(f"Loading MathOlympiadAGI from {agi_path}")
        try:
            math_agi = MathOlympiadAGI(config)
            math_agi.load(config.trained_engines_dir)
            engine_config = math_agi.get_best_engine_config()
            if engine_config:
                logger.info(f"Using trained MathOlympiadAGI engine: {engine_config['name']}")
                return engine_config
        except Exception as e:
            logger.warning(f"Failed to load MathOlympiadAGI: {e}")
    
    # Try to load best engine from trained_engines
    engine_path = os.path.join(config.trained_engines_dir, 'best_engine.json')
    
    if os.path.exists(engine_path):
        logger.info(f"Loading best engine from {engine_path}")
        with open(engine_path, 'r') as f:
            return json.load(f)
    
    # Fallback: Create default engine config
    logger.info("Creating default engine configuration")
    breeder = EngineBreeder(config)
    top_engines = breeder.select_top_engines(1)
    return breeder.create_engine_config(top_engines[0])


def solve_kaggle_problems(config: Config) -> Dict[str, int]:
    """
    Solve problems using Kaggle evaluation API
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary of problem_id -> answer
    """
    logger.info("Initializing AIMO3 solution...")
    
    # Initialize components
    solver = MathSolver(config)
    engine_config = load_best_engine_config(config)
    
    logger.info(f"Using engine: {engine_config['name']}")
    
    # Import Kaggle evaluation API
    try:
        from kaggle_evaluation.aimo_3_inference_server import AIMO3InferenceServer
        logger.info("Kaggle evaluation API loaded")
    except ImportError:
        logger.warning("Kaggle evaluation API not available, using local test data")
        return solve_local_test_problems(config, solver, engine_config)
    
    # Initialize inference server
    predictions = {}
    
    try:
        # Create inference server
        inference_server = AIMO3InferenceServer()
        
        # Process problems one by one
        logger.info("Starting problem-by-problem evaluation...")
        
        problem_count = 0
        for problem_batch in inference_server.serve():
            problem_count += 1
            
            # Extract problem data
            problem_id = problem_batch['id'].item()
            problem_text = problem_batch['problem'].item()
            
            logger.info(f"Solving problem {problem_count}: {problem_id}")
            
            # Solve problem
            if config.use_ensemble:
                # Use ensemble if configured
                # For simplicity, use single engine here
                answer, reasoning = solver.solve(problem_text, engine_config)
            else:
                answer, reasoning = solver.solve(problem_text, engine_config)
            
            # Store prediction
            predictions[problem_id] = answer
            
            logger.info(f"Problem {problem_id} answer: {answer}")
            
            # Submit answer to inference server
            inference_server.predict(answer)
        
        logger.info(f"Completed solving {problem_count} problems")
        
    except Exception as e:
        logger.error(f"Error during Kaggle evaluation: {e}")
        logger.info("Falling back to local test data")
        return solve_local_test_problems(config, solver, engine_config)
    
    return predictions


def solve_local_test_problems(
    config: Config,
    solver: MathSolver,
    engine_config: Dict
) -> Dict[str, int]:
    """
    Solve problems from local test.csv file
    
    Args:
        config: Configuration object
        solver: MathSolver instance
        engine_config: Engine configuration
        
    Returns:
        Dictionary of problem_id -> answer
    """
    logger.info("Solving problems from local test.csv...")
    
    predictions = {}
    
    # Load test problems
    with open(config.test_csv, 'r') as f:
        reader = csv.DictReader(f)
        problems = list(reader)
    
    logger.info(f"Found {len(problems)} test problems")
    
    # Solve each problem
    for i, problem in enumerate(problems):
        problem_id = problem['id']
        problem_text = problem.get('problem', '')
        
        logger.info(f"Solving problem {i+1}/{len(problems)}: {problem_id}")
        
        if not problem_text:
            logger.warning(f"Problem {problem_id} has no text, using default answer")
            predictions[problem_id] = 0
            continue
        
        # Solve
        answer, _ = solver.solve(problem_text, engine_config)
        predictions[problem_id] = answer
        
        logger.info(f"Problem {problem_id} answer: {answer}")
    
    return predictions


def main():
    """Main entry point for Kaggle submission"""
    logger.info("=" * 60)
    logger.info("AIMO3 KAGGLE SUBMISSION")
    logger.info("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Solve problems
    predictions = solve_kaggle_problems(config)
    
    # Create submission file
    submission_path = 'submission.csv'
    create_submission_csv(predictions, submission_path)
    
    logger.info(f"Submission saved to {submission_path}")
    logger.info(f"Total problems solved: {len(predictions)}")
    logger.info("=" * 60)
    logger.info("SUBMISSION COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
