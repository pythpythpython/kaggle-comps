"""
Engine breeder for AIMO3 solution
Selects and breeds specialized math engines from the Work submodule
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random


logger = logging.getLogger(__name__)


@dataclass
class EngineCapabilities:
    """Engine capabilities profile"""
    name: str
    quality: float
    language_parsing: float = 0.0
    knowledge: float = 0.0
    planning: float = 0.0
    reasoning: float = 0.0
    theorem_proving: float = 0.0
    
    def overall_score(self) -> float:
        """Calculate overall capability score"""
        return (
            self.quality * 0.4 +
            self.language_parsing * 0.15 +
            self.knowledge * 0.15 +
            self.planning * 0.15 +
            self.reasoning * 0.1 +
            self.theorem_proving * 0.05
        )


class EngineBreeder:
    """Breed specialized math engines for AIMO3"""
    
    def __init__(self, config: Any):
        """
        Initialize engine breeder
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load base engines from Work submodule gen4_rankings.json
        self.base_engines = self._load_gen4_engines()
        
        self.bred_engines = []
    
    def _load_gen4_engines(self) -> Dict[str, EngineCapabilities]:
        """Load Gen-4 engines from Work submodule"""
        try:
            # Find gen4_rankings.json in Work submodule
            config_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(config_dir)
            gen4_path = os.path.join(
                os.path.dirname(project_dir),
                'submodules',
                'Work',
                'gen4_rankings.json'
            )
            
            if os.path.exists(gen4_path):
                with open(gen4_path, 'r') as f:
                    data = json.load(f)
                
                engines = {}
                for engine_data in data.get('engines', []):
                    caps = engine_data['capabilities']
                    engines[engine_data['name']] = EngineCapabilities(
                        name=engine_data['name'],
                        quality=engine_data['quality'],
                        language_parsing=caps['language_parsing'],
                        knowledge=caps['knowledge'],
                        planning=caps['planning'],
                        reasoning=caps['reasoning'],
                        theorem_proving=caps['theorem_proving'],
                    )
                
                self.logger.info(f"Loaded {len(engines)} engines from {gen4_path}")
                return engines
            else:
                self.logger.warning(f"gen4_rankings.json not found at {gen4_path}, using defaults")
                return self._get_default_engines()
                
        except Exception as e:
            self.logger.error(f"Failed to load gen4_rankings.json: {e}")
            return self._get_default_engines()
    
    def _get_default_engines(self) -> Dict[str, EngineCapabilities]:
        """Get default engine configurations as fallback"""
        return {
            "LinguaChart-G4-G3-192": EngineCapabilities(
                name="LinguaChart-G4-G3-192",
                quality=0.9928,
                language_parsing=1.0,
                knowledge=0.7,
                planning=0.6,
                reasoning=0.75,
                theorem_proving=0.5
            ),
            "WiseJust-G4-G3-119": EngineCapabilities(
                name="WiseJust-G4-G3-119",
                quality=0.9923,
                language_parsing=0.7,
                knowledge=1.0,
                planning=0.7,
                reasoning=0.85,
                theorem_proving=0.6
            ),
            "KnowMoral-G4-G3-120": EngineCapabilities(
                name="KnowMoral-G4-G3-120",
                quality=0.9922,
                language_parsing=0.65,
                knowledge=1.0,
                planning=0.65,
                reasoning=0.8,
                theorem_proving=0.6
            ),
            "PlanVoice-G4-G3-203": EngineCapabilities(
                name="PlanVoice-G4-G3-203",
                quality=0.9936,
                language_parsing=0.7,
                knowledge=0.75,
                planning=1.0,
                reasoning=0.85,
                theorem_proving=0.5
            ),
        }
    
    def select_top_engines(self, n: int = 4) -> List[EngineCapabilities]:
        """
        Select top N engines for math reasoning
        
        Args:
            n: Number of engines to select
            
        Returns:
            List of top engine capabilities
        """
        # Sort by overall score
        ranked = sorted(
            self.base_engines.values(),
            key=lambda e: e.overall_score(),
            reverse=True
        )
        
        selected = ranked[:n]
        self.logger.info(f"Selected top {n} engines:")
        for engine in selected:
            self.logger.info(f"  {engine.name}: score={engine.overall_score():.4f}")
        
        return selected
    
    def breed_engine(
        self,
        parent1: EngineCapabilities,
        parent2: EngineCapabilities,
        name: str
    ) -> EngineCapabilities:
        """
        Breed two engines to create a new specialized engine
        
        Args:
            parent1: First parent engine
            parent2: Second parent engine
            name: Name for bred engine
            
        Returns:
            New bred engine
        """
        # Average quality with slight improvement
        quality = (parent1.quality + parent2.quality) / 2 + random.uniform(0.001, 0.005)
        
        # Take best capabilities from each parent
        bred = EngineCapabilities(
            name=name,
            quality=min(1.0, quality),
            language_parsing=max(parent1.language_parsing, parent2.language_parsing),
            knowledge=max(parent1.knowledge, parent2.knowledge),
            planning=max(parent1.planning, parent2.planning),
            reasoning=max(parent1.reasoning, parent2.reasoning),
            theorem_proving=max(parent1.theorem_proving, parent2.theorem_proving),
        )
        
        self.logger.info(f"Bred engine {name} from {parent1.name} + {parent2.name}")
        self.logger.info(f"  Quality: {bred.quality:.4f}, Overall: {bred.overall_score():.4f}")
        
        return bred
    
    def breed_population(self, num_variants: int = 10) -> List[EngineCapabilities]:
        """
        Breed a population of specialized math engines
        
        Args:
            num_variants: Number of variants to breed
            
        Returns:
            List of bred engines
        """
        self.logger.info(f"Breeding {num_variants} specialized math engines...")
        
        # Select top base engines
        top_engines = self.select_top_engines(4)
        
        bred_population = []
        
        # Create pairwise combinations
        for i in range(num_variants):
            # Select two parents
            parent1 = random.choice(top_engines)
            parent2 = random.choice(top_engines)
            
            # Create unique name
            name = f"MathOlympiad-Bred-{i+1:02d}"
            
            # Breed
            bred = self.breed_engine(parent1, parent2, name)
            bred_population.append(bred)
        
        self.bred_engines = bred_population
        
        # Log summary
        avg_quality = sum(e.quality for e in bred_population) / len(bred_population)
        avg_score = sum(e.overall_score() for e in bred_population) / len(bred_population)
        
        self.logger.info(f"Bred population summary:")
        self.logger.info(f"  Average quality: {avg_quality:.4f}")
        self.logger.info(f"  Average overall score: {avg_score:.4f}")
        
        return bred_population
    
    def select_best_engine(self, engines: List[EngineCapabilities]) -> EngineCapabilities:
        """
        Select best engine from a list
        
        Args:
            engines: List of engines
            
        Returns:
            Best engine
        """
        best = max(engines, key=lambda e: e.overall_score())
        self.logger.info(f"Selected best engine: {best.name} (score={best.overall_score():.4f})")
        return best
    
    def save_engines(self, output_path: str):
        """
        Save bred engines to file
        
        Args:
            output_path: Path to save engines
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        engines_data = {
            'base_engines': {
                name: {
                    'name': eng.name,
                    'quality': eng.quality,
                    'language_parsing': eng.language_parsing,
                    'knowledge': eng.knowledge,
                    'planning': eng.planning,
                    'reasoning': eng.reasoning,
                    'theorem_proving': eng.theorem_proving,
                    'overall_score': eng.overall_score(),
                }
                for name, eng in self.base_engines.items()
            },
            'bred_engines': [
                {
                    'name': eng.name,
                    'quality': eng.quality,
                    'language_parsing': eng.language_parsing,
                    'knowledge': eng.knowledge,
                    'planning': eng.planning,
                    'reasoning': eng.reasoning,
                    'theorem_proving': eng.theorem_proving,
                    'overall_score': eng.overall_score(),
                }
                for eng in self.bred_engines
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(engines_data, f, indent=2)
        
        self.logger.info(f"Saved engines to {output_path}")
    
    def load_engines(self, input_path: str):
        """
        Load bred engines from file
        
        Args:
            input_path: Path to load engines from
        """
        with open(input_path, 'r') as f:
            engines_data = json.load(f)
        
        # Load bred engines
        self.bred_engines = [
            EngineCapabilities(
                name=eng['name'],
                quality=eng['quality'],
                language_parsing=eng['language_parsing'],
                knowledge=eng['knowledge'],
                planning=eng['planning'],
                reasoning=eng['reasoning'],
                theorem_proving=eng['theorem_proving'],
            )
            for eng in engines_data['bred_engines']
        ]
        
        self.logger.info(f"Loaded {len(self.bred_engines)} bred engines from {input_path}")
    
    def create_engine_config(self, engine: EngineCapabilities) -> Dict[str, Any]:
        """
        Create configuration for an engine
        
        Args:
            engine: Engine capabilities
            
        Returns:
            Engine configuration dictionary
        """
        return {
            'name': engine.name,
            'quality': engine.quality,
            'capabilities': {
                'language_parsing': engine.language_parsing,
                'knowledge': engine.knowledge,
                'planning': engine.planning,
                'reasoning': engine.reasoning,
                'theorem_proving': engine.theorem_proving,
            },
            'hyperparameters': self.config.get_engine_config(),
        }
