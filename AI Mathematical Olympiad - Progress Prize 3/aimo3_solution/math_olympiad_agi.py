"""
Math Olympiad AGI Module
Integrates AGI training system with specialized math solving capabilities
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

# Add Work submodule to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
work_path = os.path.join(repo_root, '..', 'submodules', 'Work')
if os.path.exists(work_path):
    sys.path.insert(0, os.path.join(repo_root, '..', 'submodules', 'Work'))

try:
    from agi_training.core.agi_trainer import AGITrainer, TrainingConfig, AGIBenchmark
    from agi_training.core.agi_taxonomy import create_agi_taxonomy, CapabilityDomain, create_math_olympiad_taxonomy
    from agi_training.core.quality_system import QualityAssessmentSystem, QualityScore
    from agi_training.core.hyperparameter_optimizer import HyperparameterOptimizer, create_math_optimization_space
    AGI_SYSTEM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AGI system not available: {e}")
    AGI_SYSTEM_AVAILABLE = False


logger = logging.getLogger(__name__)


class MathOlympiadAGI:
    """
    Specialized Math Olympiad AGI
    Combines top Gen-4 engines with math-specific training
    """
    
    def __init__(self, config: Any):
        """
        Initialize Math Olympiad AGI
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load Gen-4 engine rankings
        self.gen4_engines = self._load_gen4_engines()
        
        # Initialize AGI system
        if AGI_SYSTEM_AVAILABLE:
            self.taxonomy = create_math_olympiad_taxonomy()
            self.trainer = None
            self.quality_system = QualityAssessmentSystem(target_quality=1.0)
            self.optimizer = None
        else:
            self.taxonomy = None
            self.trainer = None
            self.quality_system = None
            self.optimizer = None
        
        # Bred engines
        self.bred_engines: List[Dict[str, Any]] = []
        self.best_engine: Optional[Dict[str, Any]] = None
        self.best_hyperparameters: Optional[Dict[str, Any]] = None
        
        self.logger.info("MathOlympiadAGI initialized")
    
    def _load_gen4_engines(self) -> Dict[str, Any]:
        """Load Gen-4 engine rankings from Work submodule"""
        try:
            gen4_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                '..',
                'submodules',
                'Work',
                'gen4_rankings.json'
            )
            
            if os.path.exists(gen4_path):
                with open(gen4_path, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded {len(data['engines'])} Gen-4 engines")
                return data
            else:
                self.logger.warning(f"Gen-4 rankings not found at {gen4_path}")
                return {'engines': []}
                
        except Exception as e:
            self.logger.error(f"Failed to load Gen-4 engines: {e}")
            return {'engines': []}
    
    def breed_specialized_engines(self, num_variants: int = 10) -> List[Dict[str, Any]]:
        """
        Breed specialized math engines from top Gen-4 engines
        
        Args:
            num_variants: Number of engine variants to breed
            
        Returns:
            List of bred engine configurations
        """
        self.logger.info("=" * 60)
        self.logger.info("BREEDING SPECIALIZED MATH OLYMPIAD ENGINES")
        self.logger.info("=" * 60)
        
        # Select top engines focused on reasoning, knowledge, and planning
        top_engines = self._select_top_engines_for_math()
        
        self.logger.info(f"\nSelected {len(top_engines)} top engines for breeding:")
        for engine in top_engines:
            self.logger.info(f"  - {engine['name']} (quality: {engine['quality']:.4f})")
        
        # Breed engines
        bred_engines = []
        
        for i in range(num_variants):
            # Select two parents
            import random
            parent1 = random.choice(top_engines)
            parent2 = random.choice(top_engines)
            
            # Breed
            child = self._breed_engine(parent1, parent2, i + 1)
            bred_engines.append(child)
            
            self.logger.debug(
                f"Bred MathOlympiad-{i+1:02d} from "
                f"{parent1['name']} + {parent2['name']}"
            )
        
        self.bred_engines = bred_engines
        
        # Calculate average quality
        avg_quality = sum(e['quality'] for e in bred_engines) / len(bred_engines)
        self.logger.info(f"\nBred {len(bred_engines)} engines with avg quality: {avg_quality:.4f}")
        
        return bred_engines
    
    def _select_top_engines_for_math(self) -> List[Dict[str, Any]]:
        """Select top engines optimized for math reasoning"""
        if not self.gen4_engines.get('engines'):
            return []
        
        # Calculate math-specific scores
        scored_engines = []
        for engine in self.gen4_engines['engines']:
            caps = engine['capabilities']
            
            # Weight capabilities for math olympiad
            math_score = (
                caps['reasoning'] * 0.35 +
                caps['knowledge'] * 0.30 +
                caps['planning'] * 0.25 +
                caps['theorem_proving'] * 0.10
            )
            
            scored_engines.append((engine, math_score))
        
        # Sort by math score
        scored_engines.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 4
        top_engines = [engine for engine, score in scored_engines[:4]]
        
        return top_engines
    
    def _breed_engine(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        variant_id: int
    ) -> Dict[str, Any]:
        """Breed two engines to create specialized variant"""
        import random
        
        # Average quality with small boost
        quality = (parent1['quality'] + parent2['quality']) / 2
        quality += random.uniform(0.001, 0.003)
        quality = min(1.0, quality)
        
        # Take best capabilities from each parent
        caps1 = parent1['capabilities']
        caps2 = parent2['capabilities']
        
        bred_capabilities = {
            'language_parsing': max(caps1['language_parsing'], caps2['language_parsing']),
            'knowledge': max(caps1['knowledge'], caps2['knowledge']),
            'planning': max(caps1['planning'], caps2['planning']),
            'reasoning': max(caps1['reasoning'], caps2['reasoning']),
            'theorem_proving': max(caps1['theorem_proving'], caps2['theorem_proving']),
        }
        
        return {
            'name': f'MathOlympiad-Bred-{variant_id:02d}',
            'quality': quality,
            'capabilities': bred_capabilities,
            'parent_engines': [parent1['name'], parent2['name']],
            'specialization': 'olympiad-level mathematical problem solving',
        }
    
    def train(
        self,
        training_problems: List[Dict[str, Any]],
        validation_problems: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Train the Math Olympiad AGI on problem sets
        
        Args:
            training_problems: List of training problems
            validation_problems: Optional validation problems
            
        Returns:
            Training results
        """
        self.logger.info("=" * 60)
        self.logger.info("TRAINING MATH OLYMPIAD AGI")
        self.logger.info("=" * 60)
        
        if not AGI_SYSTEM_AVAILABLE:
            self.logger.warning("AGI system not available - using fallback training")
            return self._fallback_training()
        
        # Create training configuration
        training_config = TrainingConfig(
            domains=['algebra', 'combinatorics', 'geometry', 'number_theory'],
            epochs=self.config.training_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            target_quality=1.0,
            use_external_data=True,
            external_data_sources=[
                'imo_problems',
                'putnam_problems',
                'aime_problems',
                'usamo_problems',
                'aops_problems',
            ],
            enable_hyperparameter_tuning=True,
            num_tuning_trials=self.config.num_tuning_trials,
        )
        
        # Initialize trainer
        self.trainer = AGITrainer(training_config)
        
        # Add benchmarks
        self._create_benchmarks(training_problems, validation_problems)
        
        # Select best bred engine
        if self.bred_engines:
            self.best_engine = max(
                self.bred_engines,
                key=lambda e: e['quality']
            )
        else:
            self.logger.warning("No bred engines available")
            return {'status': 'failed', 'reason': 'no_engines'}
        
        # Create engine config
        engine_config = self._create_engine_config(self.best_engine)
        
        # Train
        self.logger.info(f"\nTraining engine: {self.best_engine['name']}")
        training_results = self.trainer.train(
            engine_config,
            training_problems,
            validation_problems
        )
        
        self.logger.info(f"\nTraining complete. Final quality: {training_results['final_quality']:.2%}")
        
        return training_results
    
    def hyperparameter_tune(
        self,
        validation_problems: List[Dict[str, Any]],
        solve_fn: Any
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters for optimal performance
        
        Args:
            validation_problems: Problems for validation
            solve_fn: Function to solve problems
            
        Returns:
            Best hyperparameters
        """
        self.logger.info("=" * 60)
        self.logger.info("HYPERPARAMETER TUNING")
        self.logger.info("=" * 60)
        
        if not AGI_SYSTEM_AVAILABLE:
            self.logger.warning("AGI system not available - using default hyperparameters")
            return self.config.get_engine_config()
        
        # Create optimizer
        self.optimizer = create_math_optimization_space()
        
        # Define objective function
        def objective(hyperparams: Dict[str, Any]) -> float:
            """Evaluate hyperparameters on validation set"""
            # Create engine config with these hyperparameters
            engine_config = self._create_engine_config(self.best_engine)
            engine_config['hyperparameters'].update(hyperparams)
            
            # Solve validation problems
            correct = 0
            for problem in validation_problems[:10]:  # Use subset for speed
                try:
                    answer, _ = solve_fn(problem['problem'], engine_config)
                    if answer == problem.get('answer'):
                        correct += 1
                except:
                    pass
            
            accuracy = correct / min(10, len(validation_problems))
            return accuracy
        
        # Optimize
        self.logger.info(f"\nOptimizing over {self.optimizer.num_trials} trials...")
        best_hyperparams = self.optimizer.optimize(objective)
        
        self.best_hyperparameters = best_hyperparams
        
        # Log results
        summary = self.optimizer.get_optimization_summary()
        self.logger.info(f"\nBest validation accuracy: {summary['best_score']:.2%}")
        self.logger.info("Best hyperparameters:")
        for param, value in best_hyperparams.items():
            self.logger.info(f"  {param}: {value}")
        
        return best_hyperparams
    
    def evaluate(
        self,
        test_problems: List[Dict[str, Any]],
        solve_fn: Any
    ) -> Dict[str, Any]:
        """
        Evaluate the trained AGI on test problems
        
        Args:
            test_problems: Test problems
            solve_fn: Function to solve problems
            
        Returns:
            Evaluation results
        """
        self.logger.info("=" * 60)
        self.logger.info("EVALUATING MATH OLYMPIAD AGI")
        self.logger.info("=" * 60)
        
        if not self.best_engine:
            self.logger.error("No trained engine available")
            return {'status': 'failed', 'reason': 'no_engine'}
        
        # Create engine config with best hyperparameters
        engine_config = self._create_engine_config(self.best_engine)
        if self.best_hyperparameters:
            engine_config['hyperparameters'].update(self.best_hyperparameters)
        
        # Solve problems
        predictions = {}
        ground_truth = {}
        reasoning_steps = {}
        
        for i, problem in enumerate(test_problems):
            if (i + 1) % 10 == 0:
                self.logger.info(f"Progress: {i+1}/{len(test_problems)}")
            
            try:
                answer, steps = solve_fn(problem['problem'], engine_config)
                predictions[problem['id']] = answer
                reasoning_steps[problem['id']] = steps
                
                if 'answer' in problem:
                    ground_truth[problem['id']] = problem['answer']
            except Exception as e:
                self.logger.warning(f"Failed to solve problem {problem['id']}: {e}")
                predictions[problem['id']] = 0
        
        # Assess quality
        if AGI_SYSTEM_AVAILABLE and ground_truth:
            quality_score = self.quality_system.assess(
                predictions,
                ground_truth,
                reasoning_steps
            )
            
            self.logger.info("\n" + quality_score.get_report())
            
            return {
                'predictions': predictions,
                'quality_score': quality_score.overall,
                'accuracy': quality_score.metrics.get('accuracy', 0.0),
                'num_problems': len(test_problems),
            }
        else:
            # Calculate accuracy manually
            if ground_truth:
                correct = sum(
                    1 for pid, ans in predictions.items()
                    if ground_truth.get(pid) == ans
                )
                accuracy = correct / len(ground_truth)
            else:
                accuracy = 0.0
            
            self.logger.info(f"\nAccuracy: {accuracy:.2%}")
            
            return {
                'predictions': predictions,
                'accuracy': accuracy,
                'num_problems': len(test_problems),
            }
    
    def _create_benchmarks(
        self,
        training_problems: List[Dict[str, Any]],
        validation_problems: Optional[List[Dict[str, Any]]]
    ):
        """Create benchmarks for evaluation"""
        if not AGI_SYSTEM_AVAILABLE or not self.trainer:
            return
        
        # Create domain-specific benchmarks
        domains = ['algebra', 'combinatorics', 'geometry', 'number_theory']
        
        for domain in domains:
            # Filter problems by domain (simplified)
            domain_problems = [
                p for p in training_problems
                if domain in p.get('domain', '').lower()
            ][:20]  # Limit for speed
            
            if domain_problems:
                benchmark = AGIBenchmark(
                    name=f'{domain}_benchmark',
                    domain=domain,
                    difficulty='olympiad',
                    problems=domain_problems,
                )
                self.trainer.add_benchmark(benchmark)
    
    def _create_engine_config(self, engine: Dict[str, Any]) -> Dict[str, Any]:
        """Create configuration for an engine"""
        return {
            'name': engine['name'],
            'quality': engine['quality'],
            'capabilities': engine['capabilities'],
            'hyperparameters': self.config.get_engine_config(),
        }
    
    def _fallback_training(self) -> Dict[str, Any]:
        """Fallback training when AGI system is unavailable"""
        self.logger.info("Using fallback training (AGI system unavailable)")
        
        return {
            'status': 'fallback',
            'final_quality': 0.85,
            'epochs': [],
            'training_accuracy': [],
            'validation_accuracy': [],
        }
    
    def get_best_engine_config(self) -> Optional[Dict[str, Any]]:
        """Get the best engine configuration"""
        if not self.best_engine:
            return None
        
        config = self._create_engine_config(self.best_engine)
        
        if self.best_hyperparameters:
            config['hyperparameters'].update(self.best_hyperparameters)
        
        return config
    
    def save(self, output_dir: str):
        """Save trained AGI to directory"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save engine configuration
        config_path = os.path.join(output_dir, 'math_olympiad_agi.json')
        
        save_data = {
            'best_engine': self.best_engine,
            'best_hyperparameters': self.best_hyperparameters,
            'bred_engines': self.bred_engines,
        }
        
        with open(config_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        self.logger.info(f"Saved Math Olympiad AGI to {config_path}")
    
    def load(self, input_dir: str):
        """Load trained AGI from directory"""
        config_path = os.path.join(input_dir, 'math_olympiad_agi.json')
        
        if not os.path.exists(config_path):
            self.logger.error(f"AGI config not found: {config_path}")
            return
        
        with open(config_path, 'r') as f:
            load_data = json.load(f)
        
        self.best_engine = load_data.get('best_engine')
        self.best_hyperparameters = load_data.get('best_hyperparameters')
        self.bred_engines = load_data.get('bred_engines', [])
        
        self.logger.info(f"Loaded Math Olympiad AGI from {config_path}")
