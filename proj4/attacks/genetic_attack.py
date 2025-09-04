import torch
import numpy as np

def fitness(model, image, label, perturbation, device='cpu'):
    # Fitness function to evaluate how well a perturbation deceives the model
    # Higher fitness means more successful attack (model prediction != true label)
    
    # Move all tensors to specified device
    model = model.to(device)
    image = image.to(device)
    label = label.to(device)
    perturbation = perturbation.to(device)
    
    # Apply perturbation to image and clamp to valid pixel range [0,1]
    perturbed_image = torch.clamp(image + perturbation, 0, 1)
    
    # Get model prediction on perturbed image
    output = model(perturbed_image)
    pred = output.argmax().item()
    
    # Return 1 if attack successful (wrong prediction), 0 otherwise
    return (pred != label.item())

def generate_initial_population(pop_size, image_size, device='cpu'):
    # Generate initial population of random perturbations
    # Each perturbation is sampled from normal distribution with small variance
    return [torch.randn(image_size, device=device) * 0.1 for _ in range(pop_size)]

def mutate(perturbation, mutation_rate=0.1):
    # Mutation operator: randomly alter parts of the perturbation
    # Adds Gaussian noise to a fraction of perturbation values
    
    mutated_perturbation = perturbation.clone()
    # Create mask for which elements to mutate
    mutation_mask = torch.rand_like(mutated_perturbation) < mutation_rate
    # Add random noise to selected elements
    mutated_perturbation[mutation_mask] += torch.randn_like(mutated_perturbation[mutation_mask]) * 0.1
    return mutated_perturbation

def crossover(perturbation1, perturbation2):
    # Crossover operator: combine two parent perturbations
    # Creates offspring by taking part from each parent
    
    # Choose random crossover point
    crossover_point = np.random.randint(0, perturbation1.numel())
    
    # Flatten perturbations for easy manipulation
    pert1_flat = perturbation1.view(-1)
    pert2_flat = perturbation2.view(-1)
    
    # Create child by combining parts from both parents
    new_perturbation = torch.cat([pert1_flat[:crossover_point], pert2_flat[crossover_point:]])
    
    # Reshape back to original dimensions
    return new_perturbation.view(perturbation1.shape)

def genetic_attack(model, image, label, population_size=10, generations=50, mutation_rate=0.1, device='cpu'):
    # Main genetic algorithm for adversarial attack generation
    # Evolves population of perturbations over multiple generations
    
    # Prepare input tensors
    image = image.clone().detach().to(device)
    label = label.clone().detach().to(device)
    
    # Initialize random population of perturbations
    population = generate_initial_population(population_size, image.shape, device)
    
    # Evolution loop over generations
    for generation in range(generations):
        fitness_scores = []
        
        # Evaluate fitness of each perturbation in population
        for perturbation in population:
            fitness_scores.append(fitness(model, image, label, perturbation, device))
        
        # Selection: keep top half of population based on fitness
        selected_population = [population[i] for i in np.argsort(fitness_scores)[-population_size // 2:]]
        
        # Generate new population through crossover and mutation
        new_population = []
        for _ in range(population_size // 2):
            # Select two parents randomly from selected population
            idx1, idx2 = np.random.choice(len(selected_population), size=2, replace=False)
            parent1 = selected_population[idx1]
            parent2 = selected_population[idx2]

            # Create child through crossover and mutation
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        # Update population with selected parents and new children
        population = selected_population + new_population
        
        # Early termination if perfect attack found
        if max(fitness_scores) == 1:
            break
    
    # Return best perturbation from final population
    return population[np.argmax(fitness_scores)]