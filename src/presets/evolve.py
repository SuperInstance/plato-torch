"""
Evolve preset — evolutionary / genetic algorithm room.

Population-based optimization: selection, crossover, mutation.
Maps directly to JC1's cuda-genepool concepts (Gene=Tile).
No dependencies — pure Python.
"""

import random
import copy
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

try:
    from room_base import RoomBase
except ImportError:
    from ..room_base import RoomBase


class Genome:
    """A genome is a set of tile weights / strategy parameters."""
    
    def __init__(self, genes: Dict[str, float] = None):
        self.genes = genes or {}
        self.fitness = 0.0
        self.episodes = 0
    
    def mutate(self, rate: float = 0.1, magnitude: float = 0.3):
        """Random mutation of gene values."""
        for key in self.genes:
            if random.random() < rate:
                self.genes[key] += random.gauss(0, magnitude)
    
    def crossover(self, other: "Genome") -> "Genome":
        """Create offspring by combining genes from two parents."""
        child_genes = {}
        for key in set(list(self.genes.keys()) + list(other.genes.keys())):
            if key in self.genes and key in other.genes:
                # Uniform crossover
                child_genes[key] = random.choice([self.genes[key], other.genes[key]])
            elif key in self.genes:
                child_genes[key] = self.genes[key]
            else:
                child_genes[key] = other.genes[key]
        return Genome(child_genes)
    
    def to_dict(self) -> Dict:
        return {"genes": self.genes, "fitness": self.fitness, "episodes": self.episodes}
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Genome":
        g = cls(d.get("genes", {}))
        g.fitness = d.get("fitness", 0)
        g.episodes = d.get("episodes", 0)
        return g


class EvolveRoom(RoomBase):
    """Evolutionary optimization room.
    
    Maintains a population of genomes (tile configurations).
    Agents interact → genomes get fitness scores → selection + breeding.
    Gene = Tile, RNA = Activation, Protein = Behavior, ATP = EV.
    """
    
    def __init__(self, room_id: str, **kwargs):
        super().__init__(room_id, preset="evolve", **kwargs)
        
        self.population_size = kwargs.get("population_size", 50)
        self.mutation_rate = kwargs.get("mutation_rate", 0.1)
        self.crossover_rate = kwargs.get("crossover_rate", 0.7)
        self.elite_fraction = kwargs.get("elite_fraction", 0.1)
        
        self.population: List[Genome] = []
        self.generation = 0
        self._best_ever: Optional[Genome] = None
        
        # Load or initialize population
        self._load_population()
    
    def feed(self, data: Any, **kwargs) -> Dict:
        """Feed a fitness evaluation result."""
        if isinstance(data, dict):
            genome_id = data.get("genome_id", 0)
            fitness = data.get("fitness", 0)
            
            if genome_id < len(self.population):
                self.population[genome_id].fitness = fitness
                self.population[genome_id].episodes += 1
            
            return {"genome_id": genome_id, "fitness": fitness}
        return {"status": "invalid_data"}
    
    def train_step(self, batch: List[Dict]) -> Dict:
        """One generation: evaluate → select → crossover → mutate."""
        # Update fitness from batch
        for tile in batch:
            sh = tile.get("state_hash", "")
            reward = tile.get("reward", 0)
            # Map reward to genome fitness (simplified)
            if self.population:
                # Distribute reward across population members
                idx = hash(sh) % len(self.population)
                g = self.population[idx]
                g.fitness = g.fitness * 0.9 + reward * 0.1  # exponential moving avg
                g.episodes += 1
        
        # Evolve
        return self.evolve()
    
    def predict(self, input: Any) -> Dict:
        """Return the best genome's gene values."""
        if not self.population:
            return {"best_genes": {}, "fitness": 0}
        
        best = max(self.population, key=lambda g: g.fitness)
        return {
            "best_genes": best.genes,
            "fitness": round(best.fitness, 4),
            "generation": self.generation,
            "population_size": len(self.population),
        }
    
    def export_model(self, format: str = "json") -> Optional[bytes]:
        """Export best genome and population stats."""
        best = self._get_best()
        model = {
            "room_id": self.room_id,
            "preset": "evolve",
            "generation": self.generation,
            "best_genome": best.to_dict() if best else None,
            "population_size": len(self.population),
            "population_fitness": [round(g.fitness, 3) for g in self.population[:20]],
        }
        return json.dumps(model, indent=2).encode()
    
    def evolve(self) -> Dict:
        """Run one generation of evolution."""
        if len(self.population) < 2:
            return {"status": "population_too_small"}
        
        self.generation += 1
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Track best ever
        if self._best_ever is None or self.population[0].fitness > self._best_ever.fitness:
            self._best_ever = copy.deepcopy(self.population[0])
        
        # Elite preservation
        n_elite = max(1, int(len(self.population) * self.elite_fraction))
        new_population = [copy.deepcopy(g) for g in self.population[:n_elite]]
        
        # Breeding
        while len(new_population) < self.population_size:
            # Tournament selection
            parent_a = self._tournament_select(k=3)
            parent_b = self._tournament_select(k=3)
            
            if random.random() < self.crossover_rate:
                child = parent_a.crossover(parent_b)
            else:
                child = copy.deepcopy(random.choice([parent_a, parent_b]))
            
            child.mutate(self.mutation_rate)
            child.fitness = 0  # reset — must be re-evaluated
            child.episodes = 0
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
        self._save_population()
        
        best = self.population[0]
        avg_fitness = sum(g.fitness for g in self.population) / len(self.population)
        
        return {
            "status": "evolved",
            "generation": self.generation,
            "best_fitness": round(best.fitness, 4),
            "avg_fitness": round(avg_fitness, 4),
            "population_size": len(self.population),
        }
    
    def _tournament_select(self, k: int = 3) -> Genome:
        """Tournament selection: pick k random, return the fittest."""
        contestants = random.sample(self.population, min(k, len(self.population)))
        return max(contestants, key=lambda g: g.fitness)
    
    def _get_best(self) -> Optional[Genome]:
        if not self.population:
            return self._best_ever
        current_best = max(self.population, key=lambda g: g.fitness)
        if self._best_ever and self._best_ever.fitness > current_best.fitness:
            return self._best_ever
        return current_best
    
    def _load_population(self):
        path = self.ensign_dir / "evolve_population.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            self.generation = data.get("generation", 0)
            self.population = [Genome.from_dict(g) for g in data.get("population", [])]
            if data.get("best_ever"):
                self._best_ever = Genome.from_dict(data["best_ever"])
        else:
            # Initialize random population
            self._init_population()
    
    def _init_population(self):
        """Create initial population with random gene values."""
        gene_keys = ["aggression", "caution", "exploration", "exploitation",
                     "patience", "risk_tolerance", "pattern_weight", "novelty_weight"]
        self.population = []
        for _ in range(self.population_size):
            genes = {k: random.gauss(0, 1) for k in gene_keys}
            self.population.append(Genome(genes))
    
    def _save_population(self):
        path = self.ensign_dir / "evolve_population.json"
        data = {
            "generation": self.generation,
            "population": [g.to_dict() for g in self.population],
            "best_ever": self._best_ever.to_dict() if self._best_ever else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
