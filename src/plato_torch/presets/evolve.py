"""
EvolveRoom — Evolutionary / Genetic Algorithm training room.

Population-based optimization: selection, crossover, mutation.
Maps to cuda-genepool concepts: Gene=Tile, RNA=Activation,
Protein=Behavior, ATP=ExpectedValue.

Pure Python, no dependencies. PyTorch optional for batch fitness.

Usage:
    from presets.evolve import EvolveRoom, Genome
    room = EvolveRoom("evo-room", population_size=50)
    room.seed_population(gene_keys=["aggression", "caution"])
    room.evaluate_fitness(fitness_fn=my_fn)
    room.evolve(generations=10)
    best = room.best_genome()
"""

import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from room_base import RoomBase
except ImportError:
    try:
        from ..room_base import RoomBase
    except ImportError:
        RoomBase = object


class Genome:
    """A genome: dict of float weights (strategy parameters / tile weights).

    Gene=Tile, RNA=Activation, Protein=Behavior, ATP=Fitness.
    """

    def __init__(self, genes=None, genome_id=None):
        self.genes = genes or {}
        self.fitness = 0.0
        self.episodes = 0
        self.id = genome_id or hashlib.md5(
            json.dumps(self.genes, sort_keys=True).encode()).hexdigest()[:8]
        self._active = {}

    def mutate(self, rate=0.1, magnitude=0.3):
        for key in list(self.genes.keys()):
            if random.random() < rate:
                self.genes[key] += random.gauss(0, magnitude)
        self.id = hashlib.md5(json.dumps(self.genes, sort_keys=True).encode()).hexdigest()[:8]
        return self

    def crossover(self, other):
        child_genes = {}
        all_keys = set(self.genes) | set(other.genes)
        for key in all_keys:
            in_s = key in self.genes
            in_o = key in other.genes
            if in_s and in_o:
                if random.random() < 0.5:
                    alpha = random.random()
                    child_genes[key] = alpha * self.genes[key] + (1-alpha) * other.genes[key]
                else:
                    child_genes[key] = random.choice([self.genes[key], other.genes[key]])
            elif in_s and random.random() < 0.5:
                child_genes[key] = self.genes[key]
            elif in_o and random.random() < 0.5:
                child_genes[key] = other.genes[key]
        return Genome(child_genes)

    def activate(self, threshold=0.0):
        self._active = {k: v > threshold for k, v in self.genes.items()}
        return self._active

    def behavior(self):
        if not self._active: self.activate()
        return {k: v for k, v in self.genes.items() if self._active.get(k, True)}

    def distance(self, other):
        all_keys = set(self.genes) | set(other.genes)
        if not all_keys: return 0.0
        return math.sqrt(sum((self.genes.get(k,0)-other.genes.get(k,0))**2 for k in all_keys) / len(all_keys))

    def to_dict(self):
        return {"id": self.id, "genes": self.genes,
                "fitness": round(self.fitness, 6), "episodes": self.episodes}

    @classmethod
    def from_dict(cls, d):
        g = cls(d.get("genes", {}), genome_id=d.get("id"))
        g.fitness = d.get("fitness", 0)
        g.episodes = d.get("episodes", 0)
        return g

    def __repr__(self):
        return f"Genome({self.id}, genes={len(self.genes)}, fit={self.fitness:.3f})"


class EvolveRoom(RoomBase):
    """Evolutionary optimization room.

    Population of Genomes → evaluate fitness → select → breed → repeat.
    Supports elitism, tournament selection, diversity injection.
    """

    def __init__(self, room_id="evolve", population_size=50,
                 mutation_rate=0.1, mutation_magnitude=0.3,
                 crossover_rate=0.7, elite_fraction=0.1,
                 tournament_k=3, diversity_threshold=0.1, **kwargs):
        super().__init__(room_id, preset="evolve", **kwargs)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_magnitude = mutation_magnitude
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.tournament_k = tournament_k
        self.diversity_threshold = diversity_threshold

        self.population: List[Genome] = []
        self.generation = 0
        self._best_ever = None
        self._gene_keys: List[str] = []
        self._fitness_history: List[Dict] = []
        self._load_population()

    def seed_population(self, gene_keys=None, initial_genes=None):
        if gene_keys:
            self._gene_keys = gene_keys
        if initial_genes:
            self.population = [Genome(g) for g in initial_genes]
            while len(self.population) < self.population_size:
                parent = random.choice(self.population)
                child = copy.deepcopy(parent)
                child.mutate(self.mutation_rate, self.mutation_magnitude)
                child.fitness = 0; child.episodes = 0
                self.population.append(child)
        else:
            self._init_random_population()
        self._save_population()

    def _init_random_population(self):
        if not self._gene_keys:
            self._gene_keys = ["aggression", "caution", "exploration",
                "exploitation", "patience", "risk_tolerance",
                "pattern_weight", "novelty_weight", "adaptation_rate", "social_weight"]
        self.population = [Genome({k: random.gauss(0, 1) for k in self._gene_keys})
                          for _ in range(self.population_size)]

    # ── Fitness ──

    def evaluate_fitness(self, fitness_fn=None, episodes_per_genome=1):
        if fitness_fn:
            for g in self.population:
                f = fitness_fn(g)
                g.fitness = g.fitness * 0.7 + f * 0.3
                g.episodes += episodes_per_genome
        else:
            tiles = self._load_tiles()
            if tiles: self._assign_fitness_from_tiles(tiles)
        fitnesses = [g.fitness for g in self.population if g.episodes > 0]
        if not fitnesses: return {"status": "no_fitness_data"}
        s = self._pop_stats()
        self._save_population()
        return s

    def _assign_fitness_from_tiles(self, tiles):
        if not self.population: return
        for tile in tiles:
            idx = int(tile.get("state_hash", "0"), 16) % len(self.population)
            g = self.population[idx]
            g.fitness = g.fitness * 0.9 + tile.get("reward", 0) * 0.1
            g.episodes += 1

    def _population_diversity(self):
        if len(self.population) < 2: return 0.0
        sample = random.sample(self.population, min(20, len(self.population)))
        dists = [sample[i].distance(sample[j])
                 for i in range(len(sample)) for j in range(i+1, len(sample))]
        return sum(dists) / len(dists) if dists else 0.0

    # ── Evolution ──

    def evolve(self, generations=1):
        if len(self.population) < 2:
            return {"status": "population_too_small", "size": len(self.population)}
        for _ in range(generations):
            self.generation += 1
            self.population.sort(key=lambda g: g.fitness, reverse=True)
            if self._best_ever is None or self.population[0].fitness > self._best_ever.fitness:
                self._best_ever = copy.deepcopy(self.population[0])
            n_elite = max(1, int(len(self.population) * self.elite_fraction))
            new_pop = [copy.deepcopy(g) for g in self.population[:n_elite]]
            while len(new_pop) < self.population_size:
                pa = self._tournament_select()
                pb = self._tournament_select()
                child = pa.crossover(pb) if random.random() < self.crossover_rate else copy.deepcopy(random.choice([pa, pb]))
                child.mutate(self.mutation_rate, self.mutation_magnitude)
                child.fitness = 0; child.episodes = 0
                new_pop.append(child)
            div = self._population_diversity()
            if div < self.diversity_threshold:
                n_imm = max(1, int(self.population_size * 0.1))
                for i in range(n_imm):
                    genes = {k: random.gauss(0, 1) for k in self._gene_keys}
                    if len(new_pop) > n_elite + i:
                        new_pop[-(i+1)] = Genome(genes)
            self.population = new_pop[:self.population_size]
        self._save_population()
        return self._gen_stats()

    def _tournament_select(self):
        k = self.tournament_k
        contestants = random.sample(self.population, min(k, len(self.population)))
        return max(contestants, key=lambda g: g.fitness)

    def _pop_stats(self):
        fitnesses = [g.fitness for g in self.population]
        if not fitnesses: return {"status": "empty"}
        n = len(fitnesses)
        mean = sum(fitnesses) / n
        std = (sum((f-mean)**2 for f in fitnesses) / n) ** 0.5
        return {"size": n, "best": round(max(fitnesses), 4),
                "worst": round(min(fitnesses), 4), "mean": round(mean, 4),
                "std": round(std, 4), "diversity": round(self._population_diversity(), 4),
                "generation": self.generation}

    def _gen_stats(self):
        s = self._pop_stats()
        s["status"] = "evolved"
        s["best_ever"] = round(self._best_ever.fitness, 4) if self._best_ever else 0
        self._fitness_history.append(s)
        return s

    # ── RoomBase interface ──

    def feed(self, data, **kwargs):
        if isinstance(data, dict):
            if "genome_id" in data:
                for g in self.population:
                    if g.id == data["genome_id"]:
                        g.fitness = g.fitness * 0.7 + data.get("fitness", 0) * 0.3
                        g.episodes += 1
                        return {"genome_id": g.id, "fitness": round(g.fitness, 4)}
                return {"status": "genome_not_found"}
            return self.observe(data.get("state",""), data.get("action",""),
                               data.get("outcome",""), reward=data.get("reward"))
        return {"status": "invalid_data"}

    def train_step(self, batch=None):
        if batch is None:
            return {"status": "ok", "message": "no batch", "preset": "evolve"}
        self._assign_fitness_from_tiles(batch)
        return self.evolve(generations=1)

    def predict(self, input):
        best = self.best_genome()
        if not best: return {"best_genes": {}, "fitness": 0}
        return {"best_genes": best.genes, "fitness": round(best.fitness, 4),
                "generation": self.generation, "genome_id": best.id}

    def export_model(self, format="json"):
        model = {
            "room_id": self.room_id, "preset": "evolve",
            "generation": self.generation, "gene_keys": self._gene_keys,
            "best_ever": self._best_ever.to_dict() if self._best_ever else None,
            "population": [g.to_dict() for g in self.population],
            "fitness_history": self._fitness_history[-50:],
        }
        return json.dumps(model, indent=2).encode()

    def simulate(self, episodes=100):
        if not self.population: self.seed_population()
        for g in self.population:
            score = sum(g.genes.get(k, 0) * random.gauss(0, 1)
                       for k in g.genes) / max(len(g.genes), 1)
            g.fitness = g.fitness * 0.5 + score * 0.5
            g.episodes += 1
        self._save_population()
        return self.evolve(generations=1)

    def best_genome(self):
        if not self.population and not self._best_ever: return None
        cur = max(self.population, key=lambda g: g.fitness) if self.population else None
        if self._best_ever and cur:
            return self._best_ever if self._best_ever.fitness >= cur.fitness else cur
        return self._best_ever or cur

    def population_stats(self):
        if not self.population: return {"status": "empty"}
        s = self._pop_stats()
        s["best_ever"] = round(self._best_ever.fitness, 4) if self._best_ever else None
        s["gene_keys"] = self._gene_keys
        return s

    # ── Persistence ──

    def _load_population(self):
        p = self.ensign_dir / "evolve_population.json"
        if not p.exists(): return
        try:
            with open(p) as f: d = json.load(f)
            self.generation = d.get("generation", 0)
            self._gene_keys = d.get("gene_keys", self._gene_keys)
            self.population = [Genome.from_dict(g) for g in d.get("population", [])]
            if d.get("best_ever"): self._best_ever = Genome.from_dict(d["best_ever"])
            self._fitness_history = d.get("fitness_history", [])
        except (json.JSONDecodeError, IOError):
            pass

    def _save_population(self):
        p = self.ensign_dir / "evolve_population.json"
        with open(p, "w") as f:
            json.dump({
                "generation": self.generation, "gene_keys": self._gene_keys,
                "population": [g.to_dict() for g in self.population],
                "best_ever": self._best_ever.to_dict() if self._best_ever else None,
                "fitness_history": self._fitness_history[-100:],
            }, f, indent=2)

    def __repr__(self):
        return f"EvolveRoom({self.room_id}, gen={self.generation}, pop={len(self.population)})"


if __name__ == "__main__":
    room = EvolveRoom("demo", ensign_dir="/tmp/evo_ensigns", buffer_dir="/tmp/evo_buffers")
    room.seed_population(gene_keys=["speed", "power", "defense", "stealth", "luck"])
    print(f"Pop: {len(room.population)}")
    for i in range(10):
        room.evaluate_fitness(lambda g: g.genes.get("power",0)*2 + g.genes.get("defense",0)*1.5)
        r = room.evolve()
        print(f"Gen {r['generation']}: best={r['best']:.3f} div={r['diversity']:.3f}")
    best = room.best_genome()
    print(f"\nBest: {best}")
