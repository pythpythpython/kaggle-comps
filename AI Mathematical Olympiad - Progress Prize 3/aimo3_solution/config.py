"""
Configuration and hyperparameters for AIMO3 solution
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Config:
    """Configuration class for AIMO3 solution"""
    
    def __post_init__(self):
        """Initialize paths relative to the config file location"""
        # Get the directory containing this config file
        config_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(config_dir)
        
        # Update paths to be absolute
        if not os.path.isabs(self.data_dir):
            self.data_dir = os.path.join(project_dir, "Data")
        if not os.path.isabs(self.reference_csv):
            self.reference_csv = os.path.join(project_dir, "Data", "reference.csv")
        if not os.path.isabs(self.test_csv):
            self.test_csv = os.path.join(project_dir, "Data", "test.csv")
        if not os.path.isabs(self.sample_submission_csv):
            self.sample_submission_csv = os.path.join(project_dir, "Data", "sample_submission.csv")
        if not os.path.isabs(self.trained_engines_dir):
            self.trained_engines_dir = os.path.join(project_dir, "trained_engines")
        if not os.path.isabs(self.training_data_dir):
            self.training_data_dir = os.path.join(project_dir, "training_data")
    
    # Paths
    data_dir: str = "../Data"
    reference_csv: str = "../Data/reference.csv"
    test_csv: str = "../Data/test.csv"
    sample_submission_csv: str = "../Data/sample_submission.csv"
    trained_engines_dir: str = "../trained_engines"
    training_data_dir: str = "../training_data"
    
    # GitHub configuration
    github_pat: str = "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"
    work_repo_url: str = "https://github.com/pythpythpython/Work.git"
    
    # Answer constraints
    min_answer: int = 0
    max_answer: int = 99999
    
    # Engine configuration
    top_engines: List[str] = field(default_factory=lambda: [
        "LinguaChart-G4-G3-192",  # 99.28% - LaTeX parsing
        "WiseJust-G4-G3-119",     # 99.23% - Knowledge
        "KnowMoral-G4-G3-120",    # 99.22% - Knowledge
        "PlanVoice-G4-G3-203",    # 99.36% - Planning
    ])
    
    num_bred_variants: int = 10
    
    # Training configuration
    training_domains: List[str] = field(default_factory=lambda: [
        "algebra",
        "combinatorics", 
        "geometry",
        "number_theory"
    ])
    
    num_synthetic_problems_per_domain: int = 100
    training_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-5
    
    # Testing configuration
    validation_split: float = 0.2
    target_reference_accuracy: float = 1.0  # 100% on reference problems
    
    # Hyperparameter tuning
    reasoning_depth_range: tuple = (3, 10)
    temperature_range: tuple = (0.1, 1.0)
    num_tuning_trials: int = 50
    
    # Solver configuration
    max_reasoning_steps: int = 10
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    
    # Chain-of-thought prompting
    use_cot: bool = True
    cot_prompt_template: str = """Solve this mathematical olympiad problem step by step.

Problem: {problem}

Approach:
1. Understand what is being asked
2. Identify the mathematical domain (algebra, combinatorics, geometry, number theory)
3. Break down the problem into smaller parts
4. Apply relevant theorems and techniques
5. Perform calculations carefully
6. Verify the answer is in range [0, 99999]

Step-by-step solution:"""
    
    # Ensemble configuration
    use_ensemble: bool = True
    ensemble_size: int = 3
    ensemble_voting_strategy: str = "weighted"  # "majority", "weighted", "confidence"
    
    # Runtime limits
    max_cpu_time_hours: int = 9
    max_gpu_time_hours: int = 5
    
    # Quality assurance
    require_perfect_reference_score: bool = True
    validate_answer_format: bool = True
    check_offline_compatibility: bool = True
    
    def get_engine_config(self) -> Dict[str, Any]:
        """Get engine-specific configuration"""
        return {
            "max_reasoning_steps": self.max_reasoning_steps,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "use_cot": self.use_cot,
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration"""
        return {
            "domains": self.training_domains,
            "num_synthetic_problems": self.num_synthetic_problems_per_domain,
            "epochs": self.training_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }
    
    def get_tuning_config(self) -> Dict[str, Any]:
        """Get hyperparameter tuning configuration"""
        return {
            "reasoning_depth_range": self.reasoning_depth_range,
            "temperature_range": self.temperature_range,
            "num_trials": self.num_tuning_trials,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary"""
        return {
            "data_dir": self.data_dir,
            "reference_csv": self.reference_csv,
            "test_csv": self.test_csv,
            "sample_submission_csv": self.sample_submission_csv,
            "trained_engines_dir": self.trained_engines_dir,
            "training_data_dir": self.training_data_dir,
            "min_answer": self.min_answer,
            "max_answer": self.max_answer,
            "top_engines": self.top_engines,
            "num_bred_variants": self.num_bred_variants,
            "training_domains": self.training_domains,
            "num_synthetic_problems_per_domain": self.num_synthetic_problems_per_domain,
            "training_epochs": self.training_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "target_reference_accuracy": self.target_reference_accuracy,
            "max_reasoning_steps": self.max_reasoning_steps,
            "temperature": self.temperature,
            "use_cot": self.use_cot,
            "use_ensemble": self.use_ensemble,
            "ensemble_size": self.ensemble_size,
        }


# Default configuration instance
default_config = Config()
