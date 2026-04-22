"""Task sampling strategies for meta-RL training."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
from .task_distribution import Task, TaskDistribution


class TaskSampler(ABC):
    """Abstract base class for task sampling strategies."""
    
    def __init__(self, task_distribution: TaskDistribution, seed: Optional[int] = None):
        self.task_distribution = task_distribution
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    @abstractmethod
    def sample(self, n_tasks: int = 1) -> List[Task]:
        """Sample n tasks according to the sampling strategy."""
        pass
    
    @abstractmethod
    def update(self, task_performances: Dict[int, float]) -> None:
        """Update sampling strategy based on task performances."""
        pass
    
    def reset(self) -> None:
        """Reset the sampler state."""
        pass


class UniformTaskSampler(TaskSampler):
    """Uniformly sample tasks from the distribution."""
    
    def sample(self, n_tasks: int = 1) -> List[Task]:
        """Sample tasks uniformly at random."""
        all_tasks = self.task_distribution.get_all_tasks()
        if n_tasks > len(all_tasks):
            # Sample with replacement if requesting more tasks than available
            indices = self.rng.choice(len(all_tasks), size=n_tasks, replace=True)
        else:
            indices = self.rng.choice(len(all_tasks), size=n_tasks, replace=False)
        return [all_tasks[i] for i in indices]
    
    def update(self, task_performances: Dict[int, float]) -> None:
        """Uniform sampler doesn't use performance information."""
        pass


class CurriculumTaskSampler(TaskSampler):
    """Sample tasks according to a curriculum based on agent performance."""
    
    def __init__(self, 
                 task_distribution: TaskDistribution,
                 initial_difficulty: float = 0.0,
                 difficulty_step: float = 0.1,
                 performance_threshold: float = 0.7,
                 window_size: int = 10,
                 seed: Optional[int] = None):
        
        self.initial_difficulty = initial_difficulty
        self.difficulty_step = difficulty_step
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        
        super().__init__(task_distribution, seed)
    
    def reset(self) -> None:
        """Reset curriculum state."""
        self.current_difficulty = self.initial_difficulty
        self.task_history: List[int] = []
        self.performance_history: List[float] = []
    
    def sample(self, n_tasks: int = 1) -> List[Task]:
        """Sample tasks based on current difficulty level."""
        all_tasks = self.task_distribution.get_all_tasks()
        
        # Filter tasks by difficulty
        suitable_tasks = []
        difficulty_range = 0.2  # How wide the difficulty band is
        
        for task in all_tasks:
            if abs(task.difficulty - self.current_difficulty) <= difficulty_range:
                suitable_tasks.append(task)
        
        # If no suitable tasks, expand the range
        if not suitable_tasks:
            suitable_tasks = all_tasks
        
        # Sample from suitable tasks
        if n_tasks > len(suitable_tasks):
            indices = self.rng.choice(len(suitable_tasks), size=n_tasks, replace=True)
        else:
            indices = self.rng.choice(len(suitable_tasks), size=n_tasks, replace=False)
        
        sampled_tasks = [suitable_tasks[i] for i in indices]
        
        # Record sampled tasks
        for task in sampled_tasks:
            self.task_history.append(task.task_id)
        
        return sampled_tasks
    
    def update(self, task_performances: Dict[int, float]) -> None:
        """Update curriculum based on recent performance."""
        # Add new performances to history
        for task_id, performance in task_performances.items():
            if task_id in self.task_history[-len(task_performances):]:
                self.performance_history.append(performance)
        
        # Keep only recent history
        if len(self.performance_history) > self.window_size:
            self.performance_history = self.performance_history[-self.window_size:]
        
        # Update difficulty if we have enough data
        if len(self.performance_history) >= self.window_size:
            avg_performance = np.mean(self.performance_history)
            
            if avg_performance >= self.performance_threshold:
                # Increase difficulty
                self.current_difficulty = min(1.0, self.current_difficulty + self.difficulty_step)
            elif avg_performance < self.performance_threshold * 0.5:
                # Decrease difficulty if struggling
                self.current_difficulty = max(0.0, self.current_difficulty - self.difficulty_step * 0.5)


class AdaptiveTaskSampler(TaskSampler):
    """Adaptively sample tasks to maximize learning progress."""
    
    def __init__(self,
                 task_distribution: TaskDistribution,
                 exploration_rate: float = 0.1,
                 performance_weight: float = 0.5,
                 uncertainty_weight: float = 0.5,
                 seed: Optional[int] = None):
        
        self.exploration_rate = exploration_rate
        self.performance_weight = performance_weight
        self.uncertainty_weight = uncertainty_weight
        
        super().__init__(task_distribution, seed)
    
    def reset(self) -> None:
        """Reset adaptive sampler state."""
        all_tasks = self.task_distribution.get_all_tasks()
        n_tasks = len(all_tasks)
        
        # Initialize task statistics
        self.task_counts = np.zeros(n_tasks)
        self.task_performances = np.zeros(n_tasks)
        self.task_performance_vars = np.ones(n_tasks)  # Initialize with high uncertainty
        
        # Learning progress tracking
        self.prev_performances = np.zeros(n_tasks)
        self.learning_progress = np.zeros(n_tasks)
    
    def sample(self, n_tasks: int = 1) -> List[Task]:
        """Sample tasks based on learning progress and uncertainty."""
        all_tasks = self.task_distribution.get_all_tasks()
        n_total_tasks = len(all_tasks)
        
        sampled_tasks = []
        
        for _ in range(n_tasks):
            if self.rng.random() < self.exploration_rate:
                # Explore: sample uniformly
                task_idx = self.rng.choice(n_total_tasks)
            else:
                # Exploit: sample based on scores
                scores = self._compute_task_scores()
                
                # Convert scores to probabilities
                scores = scores - np.max(scores)  # For numerical stability
                probs = np.exp(scores) / np.sum(np.exp(scores))
                
                task_idx = self.rng.choice(n_total_tasks, p=probs)
            
            sampled_tasks.append(all_tasks[task_idx])
            self.task_counts[task_idx] += 1
        
        return sampled_tasks
    
    def _compute_task_scores(self) -> np.ndarray:
        """Compute scores for each task based on learning progress and uncertainty."""
        # Normalize learning progress
        lp_normalized = self.learning_progress / (np.max(np.abs(self.learning_progress)) + 1e-8)
        
        # Compute uncertainty bonus (higher for less sampled tasks)
        uncertainty = 1.0 / (1.0 + self.task_counts)
        
        # Combine scores
        scores = (self.performance_weight * lp_normalized + 
                 self.uncertainty_weight * uncertainty)
        
        return scores
    
    def update(self, task_performances: Dict[int, float]) -> None:
        """Update task statistics and learning progress."""
        for task_id, performance in task_performances.items():
            # Update performance statistics
            old_mean = self.task_performances[task_id]
            old_count = self.task_counts[task_id]
            
            if old_count > 0:
                # Update running mean
                new_mean = (old_mean * (old_count - 1) + performance) / old_count
                self.task_performances[task_id] = new_mean
                
                # Update running variance
                old_var = self.task_performance_vars[task_id]
                new_var = ((old_count - 2) * old_var + 
                          (performance - new_mean) * (performance - old_mean)) / (old_count - 1)
                self.task_performance_vars[task_id] = max(new_var, 1e-6)
                
                # Compute learning progress
                self.learning_progress[task_id] = new_mean - self.prev_performances[task_id]
                self.prev_performances[task_id] = new_mean
            else:
                self.task_performances[task_id] = performance
                self.prev_performances[task_id] = performance