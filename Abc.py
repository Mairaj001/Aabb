import numpy as np
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter space
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4, 5],
    'gamma': ['scale', 'auto']
}

# Define the population size and number of generations
population_size = 10
generations = 5

# Create initial population
def create_individual():
    individual = {
        'C': random.choice(param_grid['C']),
        'kernel': random.choice(param_grid['kernel']),
        'degree': random.choice(param_grid['degree']),
        'gamma': random.choice(param_grid['gamma'])
    }
    return individual

# Evaluate fitness of an individual
def evaluate_individual(individual):
    model = SVC(C=individual['C'], kernel=individual['kernel'], degree=individual['degree'], gamma=individual['gamma'])
    scores = cross_val_score(model, X_train, y_train, cv=5)
    return np.mean(scores)

# Create initial population
population = [create_individual() for _ in range(population_size)]

# Genetic Algorithm
for generation in range(generations):
    print(f"Generation {generation+1}")
    # Evaluate fitness of each individual
    fitness_scores = [evaluate_individual(individual) for individual in population]
    
    # Select parents
    parents = random.choices(population, weights=fitness_scores, k=population_size)
    
    # Create next generation
    next_generation = []
    for i in range(population_size // 2):
        parent1 = parents[2 * i]
        parent2 = parents[2 * i + 1]
        child1, child2 = {}, {}
        
        # Crossover
        for param in param_grid.keys():
            if random.random() > 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        # Mutation
        for child in [child1, child2]:
            if random.random() < 0.1:  # 10% mutation rate
                param_to_mutate = random.choice(list(param_grid.keys()))
                child[param_to_mutate] = random.choice(param_grid[param_to_mutate])
        
        next_generation.extend([child1, child2])
    
    population = next_generation

# Evaluate final population
final_scores = [evaluate_individual(individual) for individual in population]
best_individual = population[np.argmax(final_scores)]
print("Best hyperparameters found:", best_individual)

# Train final model with best hyperparameters
best_model = SVC(C=best_individual['C'], kernel=best_individual['kernel'], degree=best_individual['degree'], gamma=best_individual['gamma'])
best_model.fit(X_train, y_train)

# Evaluate the final model
test_score = best_model.score(X_test, y_test)
print("Test set accuracy:", test_score)
