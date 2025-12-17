"""
Hyperparameter tuner for AIMO3 solution
Tune parameters to achieve 100% quality on reference problems
"""

import logging
import random
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json


logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration"""
    reasoning_depth: int
    temperature: float
    top_p: float
    use_cot: bool
    cot_depth: int = 3
    ensemble_size: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'reasoning_depth': self.reasoning_depth,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'use_cot': self.use_cot,
            'cot_depth': self.cot_depth,
            'ensemble_size': self.ensemble_size,
        }


class HyperparameterTuner:
    """Tune hyperparameters for maximum accuracy"""
    
    def __init__(self, config: Any):
        """
        Initialize hyperparameter tuner
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.best_config = None
        self.best_score = 0.0
        self.tuning_history = []
    
    def grid_search(
        self,
        engine_config: Dict[str, Any],
        solver: Any,
        tester: Any
    ) -> HyperparameterConfig:
        """
        Perform grid search over hyperparameters
        
        Args:
            engine_config: Engine configuration
            solver: Math solver instance
            tester: Tester instance
            
        Returns:
            Best hyperparameter configuration
        """
        self.logger.info("Starting grid search for hyperparameters...")
        
        # Define search grid
        reasoning_depths = [3, 5, 7, 10]
        temperatures = [0.3, 0.5, 0.7, 0.9]
        top_ps = [0.8, 0.9, 0.95]
        cot_options = [True, False]
        
        best_config = None
        best_accuracy = 0.0
        
        total_trials = len(reasoning_depths) * len(temperatures) * len(top_ps) * len(cot_options)
        trial = 0
        
        for depth in reasoning_depths:
            for temp in temperatures:
                for top_p in top_ps:
                    for use_cot in cot_options:
                        trial += 1
                        
                        # Create config
                        hp_config = HyperparameterConfig(
                            reasoning_depth=depth,
                            temperature=temp,
                            top_p=top_p,
                            use_cot=use_cot,
                            cot_depth=3,
                            ensemble_size=1
                        )
                        
                        self.logger.info(f"Trial {trial}/{total_trials}: depth={depth}, temp={temp:.2f}, top_p={top_p:.2f}, cot={use_cot}")
                        
                        # Update engine config
                        test_engine = engine_config.copy()
                        test_engine['hyperparameters'].update(hp_config.to_dict())
                        
                        # Test
                        accuracy, _ = tester.test_engine_on_reference(test_engine, solver)
                        
                        # Track history
                        self.tuning_history.append({
                            'config': hp_config.to_dict(),
                            'accuracy': accuracy,
                        })
                        
                        # Update best
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_config = hp_config
                            self.logger.info(f"New best accuracy: {accuracy:.2%}")
        
        self.best_config = best_config
        self.best_score = best_accuracy
        
        self.logger.info(f"Grid search completed:")
        self.logger.info(f"  Best accuracy: {best_accuracy:.2%}")
        self.logger.info(f"  Best config: {best_config.to_dict()}")
        
        return best_config
    
    def bayesian_optimization(
        self,
        engine_config: Dict[str, Any],
        solver: Any,
        tester: Any,
        n_trials: int = None
    ) -> HyperparameterConfig:
        """
        Perform Bayesian optimization over hyperparameters
        
        Args:
            engine_config: Engine configuration
            solver: Math solver instance
            tester: Tester instance
            n_trials: Number of trials (default from config)
            
        Returns:
            Best hyperparameter configuration
        """
        if n_trials is None:
            n_trials = self.config.num_tuning_trials
        
        self.logger.info(f"Starting Bayesian optimization with {n_trials} trials...")
        
        # Simplified Bayesian optimization (random search with tracking)
        best_config = None
        best_accuracy = 0.0
        
        depth_range = self.config.reasoning_depth_range
        temp_range = self.config.temperature_range
        
        for trial in range(n_trials):
            # Sample hyperparameters
            hp_config = HyperparameterConfig(
                reasoning_depth=random.randint(depth_range[0], depth_range[1]),
                temperature=random.uniform(temp_range[0], temp_range[1]),
                top_p=random.uniform(0.8, 0.98),
                use_cot=random.choice([True, False]),
                cot_depth=random.randint(2, 5),
                ensemble_size=random.randint(1, 3)
            )
            
            self.logger.info(f"Trial {trial+1}/{n_trials}: {hp_config.to_dict()}")
            
            # Update engine config
            test_engine = engine_config.copy()
            test_engine['hyperparameters'].update(hp_config.to_dict())
            
            # Test
            accuracy, _ = tester.test_engine_on_reference(test_engine, solver)
            
            # Track history
            self.tuning_history.append({
                'trial': trial + 1,
                'config': hp_config.to_dict(),
                'accuracy': accuracy,
            })
            
            # Update best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = hp_config
                self.logger.info(f"New best accuracy: {accuracy:.2%}")
                
                # Early stopping if we hit target
                if accuracy >= self.config.target_reference_accuracy:
                    self.logger.info("Target accuracy reached! Stopping early.")
                    break
        
        self.best_config = best_config
        self.best_score = best_accuracy
        
        self.logger.info(f"Bayesian optimization completed:")
        self.logger.info(f"  Best accuracy: {best_accuracy:.2%}")
        self.logger.info(f"  Best config: {best_config.to_dict()}")
        
        return best_config
    
    def tune_ensemble_voting(
        self,
        engine_configs: List[Dict[str, Any]],
        solver: Any,
        tester: Any
    ) -> Dict[str, float]:
        """
        Tune ensemble voting weights
        
        Args:
            engine_configs: List of engine configurations
            solver: Math solver instance
            tester: Tester instance
            
        Returns:
            Dictionary of engine name -> weight
        """
        self.logger.info(f"Tuning ensemble voting for {len(engine_configs)} engines...")
        
        # Test each engine individually
        engine_accuracies = {}
        for engine in engine_configs:
            accuracy, _ = tester.test_engine_on_reference(engine, solver)
            engine_accuracies[engine['name']] = accuracy
        
        # Normalize to weights
        total_accuracy = sum(engine_accuracies.values())
        weights = {
            name: acc / total_accuracy if total_accuracy > 0 else 1.0 / len(engine_configs)
            for name, acc in engine_accuracies.items()
        }
        
        self.logger.info("Ensemble weights:")
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {name}: {weight:.4f}")
        
        return weights
    
    def optimize_for_runtime(
        self,
        engine_config: Dict[str, Any],
        solver: Any,
        tester: Any,
        target_accuracy: float = 0.9
    ) -> HyperparameterConfig:
        """
        Optimize hyperparameters for runtime while maintaining accuracy
        
        Args:
            engine_config: Engine configuration
            solver: Math solver instance
            tester: Tester instance
            target_accuracy: Minimum target accuracy
            
        Returns:
            Optimized configuration
        """
        self.logger.info(f"Optimizing for runtime with target accuracy {target_accuracy:.0%}...")
        
        # Start with high accuracy config
        best_config = HyperparameterConfig(
            reasoning_depth=10,
            temperature=0.7,
            top_p=0.9,
            use_cot=True,
            cot_depth=5,
            ensemble_size=3
        )
        
        # Test baseline
        test_engine = engine_config.copy()
        test_engine['hyperparameters'].update(best_config.to_dict())
        baseline_accuracy, _ = tester.test_engine_on_reference(test_engine, solver)
        
        self.logger.info(f"Baseline accuracy: {baseline_accuracy:.2%}")
        
        # Try reducing parameters
        candidates = [
            HyperparameterConfig(7, 0.7, 0.9, True, 4, 3),
            HyperparameterConfig(5, 0.7, 0.9, True, 3, 2),
            HyperparameterConfig(5, 0.6, 0.9, True, 3, 1),
            HyperparameterConfig(3, 0.5, 0.9, True, 2, 1),
        ]
        
        for config in candidates:
            test_engine = engine_config.copy()
            test_engine['hyperparameters'].update(config.to_dict())
            accuracy, _ = tester.test_engine_on_reference(test_engine, solver)
            
            if accuracy >= target_accuracy:
                best_config = config
                self.logger.info(f"Found faster config with accuracy {accuracy:.2%}")
                break
        
        return best_config
    
    def save_tuning_history(self, output_path: str):
        """Save tuning history to file"""
        import os
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        data = {
            'best_config': self.best_config.to_dict() if self.best_config else None,
            'best_score': self.best_score,
            'history': self.tuning_history,
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved tuning history to {output_path}")
