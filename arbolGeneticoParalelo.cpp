#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <cmath>
#include <limits>

struct Individual {
    std::vector<int> genes;
    double fitness;
};

// Inicialización de la población con mayor diversidad
std::vector<Individual> initialize_population(int population_size, int gene_length) {
    std::vector<Individual> population(population_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (auto& individual : population) {
        individual.genes.resize(gene_length);
        for (auto& gene : individual.genes) {
            gene = dis(gen);
        }
        individual.fitness = 0.0;
    }
    return population;
}

// Convertir genes binarios a valores continuos en el rango [-5.12, 5.12]
std::vector<double> decode_genes(const std::vector<int>& genes) {
    std::vector<double> decoded_genes(genes.size());
    int n = genes.size();
    for (int i = 0; i < n; ++i) {
        decoded_genes[i] = -5.12 + 10.24 * genes[i] / (1 << 10); // Suponiendo genes de 10 bits
    }
    return decoded_genes;
}

// Evaluación de aptitud usando la función Rastrigin
double evaluate_fitness(const Individual& individual) {
    const double A = 10.0;
    std::vector<double> x = decode_genes(individual.genes);
    double fitness = A * x.size();
    for (double xi : x) {
        fitness += xi * xi - A * std::cos(2 * M_PI * xi);
    }
    return fitness;
}

// Selección por torneo
Individual tournament_selection(const std::vector<Individual>& population, int tournament_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, population.size() - 1);

    Individual best_individual = population[dis(gen)];
    for (int i = 1; i < tournament_size; ++i) {
        Individual competitor = population[dis(gen)];
        if (competitor.fitness < best_individual.fitness) { // Minimización de la función Rastrigin
            best_individual = competitor;
        }
    }
    return best_individual;
}

// Cruce de dos puntos
std::pair<Individual, Individual> two_point_crossover(const Individual& parent1, const Individual& parent2) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, parent1.genes.size() - 1);

    int crossover_point1 = dis(gen);
    int crossover_point2 = dis(gen);
    if (crossover_point1 > crossover_point2) std::swap(crossover_point1, crossover_point2);

    Individual offspring1 = parent1;
    Individual offspring2 = parent2;
    for (int i = crossover_point1; i <= crossover_point2; ++i) {
        offspring1.genes[i] = parent2.genes[i];
        offspring2.genes[i] = parent1.genes[i];
    }
    return {offspring1, offspring2};
}

// Mutación adaptativa
void adaptive_mutation(Individual& individual, double& mutation_rate, double best_fitness, double current_fitness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double adapt_factor = 1.0;
    if (current_fitness > best_fitness) adapt_factor = 1.1;
    else adapt_factor = 0.9;

    mutation_rate *= adapt_factor;

    for (auto& gene : individual.genes) {
        if (dis(gen) < mutation_rate) {
            gene = 1 - gene;
        }
    }
}

// Selección de la próxima generación con reinicialización dinámica
std::vector<Individual> select_next_generation(std::vector<Individual>& population, int num_elites, int max_generations_no_improvement, int& no_improvement_generations) {
    // Ordenar la población
    std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
        return a.fitness < b.fitness; // Minimización de la función Rastrigin
    });

    // Mantener los mejores individuos (elitismo)
    std::vector<Individual> next_generation(population.begin(), population.begin() + num_elites);

    // Reinicializar parte de la población para mantener diversidad
    int num_reinitialized = population.size() - num_elites;
    if (no_improvement_generations >= max_generations_no_improvement) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 1);
        for (int i = 0; i < num_reinitialized; ++i) {
            Individual new_individual;
            new_individual.genes.resize(population[0].genes.size());
            for (auto& gene : new_individual.genes) {
                gene = dis(gen);
            }
            new_individual.fitness = 0.0;
            next_generation.push_back(new_individual);
        }
        no_improvement_generations = 0;
    } else {
        for (int i = num_elites; i < population.size(); ++i) {
            next_generation.push_back(population[i]);
        }
    }

    return next_generation;
}

int main() {
    int population_size;
    int generations;
    double mutation_rate = 0.1;
    const int max_generations_no_improvement = 50;
    int no_improvement_generations = 0;

    std::cout << "Seleccione el tamaño de los datos:\n";
    std::cout << "1. pequeno\n";
    std::cout << "2. Mediano\n";
    std::cout << "3. Grande\n";
    int choice;
    std::cin >> choice;

    switch (choice) {
        case 1:
            population_size = 100;
            generations = 100;
            break;
        case 2:
            population_size = 1000;
            generations = 200;
            break;
        case 3:
            population_size = 10000;
            generations = 300;
            break;
        default:
            std::cout << "Opción no válida. Usando tamaño pequeno por defecto.\n";
            population_size = 100;
            generations = 100;
            break;
    }

    int gene_length = 10; // Longitud de genes para cada dimensión de la función Rastrigin

    std::vector<Individual> population = initialize_population(population_size, gene_length);

    auto start_time = std::chrono::high_resolution_clock::now();

    double best_fitness = std::numeric_limits<double>::max();

    for (int generation = 0; generation < generations; ++generation) {
        // Evaluar aptitud
        #pragma omp parallel for
        for (int i = 0; i < population.size(); ++i) {
            population[i].fitness = evaluate_fitness(population[i]);
        }

        // Comprobar mejora
        if (population.front().fitness < best_fitness) {
            best_fitness = population.front().fitness;
            no_improvement_generations = 0;
        } else {
            no_improvement_generations++;
        }

        // Selección y reproducción
        std::vector<Individual> new_population(population_size);
        #pragma omp parallel for
        for (int i = 0; i < population_size / 2; ++i) {
            Individual parent1 = tournament_selection(population, 3); // Selección por torneo con tamaño 3
            Individual parent2 = tournament_selection(population, 3);
            auto [offspring1, offspring2] = two_point_crossover(parent1, parent2); // Cruce de dos puntos
            adaptive_mutation(offspring1, mutation_rate, best_fitness, population[i].fitness);
            adaptive_mutation(offspring2, mutation_rate, best_fitness, population[i].fitness);

            new_population[2 * i] = offspring1;
            new_population[2 * i + 1] = offspring2;
        }

        // Selección de la próxima generación con reinicialización parcial
        population = select_next_generation(new_population, population_size / 4, max_generations_no_improvement, no_improvement_generations);

        // Imprimir la mejor aptitud de la generación actual
        std::cout << "Generación " << generation << " - Mejor aptitud: " << population.front().fitness << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Escribir los resultados en un archivo
    std::ofstream results_file;
    results_file.open("resultsParalized.txt", std::ios::app);
    results_file << "Tamaño de datos: " << (choice == 1 ? "pequeno" : (choice == 2 ? "Mediano" : "Grande")) << "\n";
    results_file << "Tiempo de ejecución: " << elapsed.count() << " segundos\n";
    results_file << "-----------------------------------\n";
    results_file.close();

    return 0;
}
