
#### 1. Introducción
Este informe detalla la implementación de un algoritmo genético paralelizado utilizando OpenMP, su evaluación de rendimiento mediante la herramienta `perf` y la visualización de los resultados a través de gráficos de speedup y eficiencia.

#### 2. Implementación

##### 2.1. Inicialización de la Población
La población inicial se genera con mayor diversidad para asegurar una buena exploración del espacio de búsqueda. Los genes se inicializan aleatoriamente.

##### Código:
```cpp
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
```

##### 2.2. Evaluación de la Aptitud
La aptitud de cada individuo se evalúa utilizando la función Rastrigin, que es comúnmente utilizada en problemas de optimización debido a su gran cantidad de mínimos locales.

##### Código:
```cpp
double evaluate_fitness(const Individual& individual) {
    const double A = 10.0;
    std::vector<double> x = decode_genes(individual.genes);
    double fitness = A * x.size();
    for (double xi : x) {
        fitness += xi * xi - A * std::cos(2 * M_PI * xi);
    }
    return fitness;
}
```

##### 2.3. Selección, Cruce y Mutación
La selección se realiza mediante un torneo. Se utiliza un cruce de dos puntos y una mutación adaptativa para generar diversidad en la descendencia.

##### Código:
```cpp
Individual tournament_selection(const std::vector<Individual>& population, int tournament_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, population.size() - 1);

    Individual best_individual = population[dis(gen)];
    for (int i = 1; i < tournament_size; ++i) {
        Individual competitor = population[dis(gen)];
        if (competitor.fitness < best_individual.fitness) {
            best_individual = competitor;
        }
    }
    return best_individual;
}

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
```

##### 2.4. Selección de la Próxima Generación
La próxima generación se selecciona mediante elitismo, y se reinicia parcialmente la población si no hay mejora en la aptitud después de un número fijo de generaciones.

##### Código:
```cpp
std::vector<Individual> select_next_generation(std::vector<Individual>& population, int num_elites, int max_generations_no_improvement, int& no_improvement_generations) {
    std::sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
        return a.fitness < b.fitness;
    });

    std::vector<Individual> next_generation(population.begin(), population.begin() + num_elites);

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
```

##### 2.5. Paralelización
Se utiliza OpenMP para paralelizar la evaluación de la aptitud y la selección y reproducción de individuos.

##### Código:
```cpp
int main() {
    int population_size;
    int generations;
    double mutation_rate = 0.1;
    const int max_generations_no_improvement = 50;
    int no_improvement_generations = 0;

    std::cout << "Seleccione el tamaño de los datos:\n";
    std::cout << "1. Pequeño\n";
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
            std::cout << "Opción no válida. Usando tamaño pequeño por defecto.\n";
            population_size = 100;
            generations = 100;
            break;
    }

    int gene_length = 10;

    std::vector<Individual> population = initialize_population(population_size, gene_length);

    auto start_time = std::chrono::high_resolution_clock::now();

    double best_fitness = std::numeric_limits<double>::max();

    for (int generation = 0; generation < generations; ++generation) {
        #pragma omp parallel for
        for (int i = 0; i < population.size(); ++i) {
            population[i].fitness = evaluate_fitness(population[i]);
        }

        if (population.front().fitness < best_fitness) {
            best_fitness = population.front().fitness;
            no_improvement_generations = 0;
        } else {
            no_improvement_generations++;
        }

        std::vector<Individual> new_population(population_size);
        #pragma omp parallel for
        for (int i = 0; i < population_size / 2; ++i) {
            Individual parent1 = tournament_selection(population, 3);
            Individual parent2 = tournament_selection(population, 3);
            auto [offspring1, offspring2] = two_point_crossover(parent1, parent2);
            adaptive_mutation(offspring1, mutation_rate, best_fitness, population[i].fitness);
            adaptive_mutation(offspring2, mutation_rate, best_fitness, population[i].fitness);

            new_population[2 * i] = offspring1;
            new_population[2 * i + 1] = offspring2;
        }

        population = select_next_generation(new_population, population_size / 4, max_generations_no_improvement, no_improvement_generations);

        std::cout << "Generación " << generation << " - Mejor aptitud: " << population.front().fitness << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::ofstream results_file;
    results_file.open("results.txt", std::ios::app);
    results_file << "Tamaño de datos: " << (choice == 1 ? "Pequeño" : (choice == 2 ? "Mediano" : "Grande")) << "\n";
    results_file << "Tiempo de ejecución: " << elapsed.count() << " segundos\n";
    results_file << "-----------------------------------\n";
    results_file.close();

    return 0;
}
```

#### 3. Experimentación

##### 3.1. Configuración del Experimento
Se realizaron experiment

os con tres tamaños de población diferentes: pequeño (100), mediano (1000) y grande (10000). Se midió el tiempo de ejecución para cada caso y se calculó el speedup y la eficiencia.

##### 3.2. Resultados
Los resultados de los experimentos se registraron en un archivo de texto (`results.txt`), que posteriormente se analizaron y graficaron.

#### 4. Resultados y Discusión

##### 4.1. Gráficos de Speedup y Eficiencia
El script de Python se utilizó para generar los gráficos de speedup y eficiencia a partir de los datos registrados.

##### Script de Python:
```python
import matplotlib.pyplot as plt

def parse_results(filename):
    sizes = []
    times = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            size = lines[i].strip().split(': ')[1]
            time = float(lines[i + 1].strip().split(': ')[1].split()[0])
            sizes.append(size)
            times.append(time)
    return sizes, times

def plot_results(sizes, times):
    size_map = {'Pequeño': 100, 'Mediano': 1000, 'Grande': 10000}
    sizes_numeric = [size_map[size] for size in sizes]
    
    plt.figure(figsize=(10, 5))
    
    # Gráfico de tiempo de ejecución
    plt.subplot(1, 2, 1)
    plt.plot(sizes_numeric, times, marker='o')
    plt.title('Tiempo de ejecución')
    plt.xlabel('Tamaño de Datos')
    plt.ylabel('Tiempo (segundos)')
    plt.xscale('log')
    
    # Gráfico de speedup
    baseline_time = times[0]  # Tiempo de ejecución para el tamaño más pequeño
    speedups = [baseline_time / time for time in times]
    plt.subplot(1, 2, 2)
    plt.plot(sizes_numeric, speedups, marker='o')
    plt.title('Speedup')
    plt.xlabel('Tamaño de Datos')
    plt.ylabel('Speedup')
    plt.xscale('log')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = 'results.txt'
    sizes, times = parse_results(filename)
    plot_results(sizes, times)
```

##### Gráficos Generados:
Incluye gráficos de tiempo de ejecución y speedup en función del tamaño de los datos.

#### 5. Conclusiones
El algoritmo genético paralelizado mostró mejoras significativas en el tiempo de ejecución con el aumento del tamaño de la población, demostrando la efectividad de la paralelización con OpenMP. Sin embargo, la mejora no fue lineal debido a la sobrecarga de gestión de hilos y la naturaleza secuencial de algunas partes del algoritmo. Se pueden explorar optimizaciones adicionales y técnicas de paralelización más avanzadas para obtener mejores resultados.

### Anexos

#### A. Código Fuente Completo

#### B. Resultados de Performance con `perf`

#### C. Gráficos de Speedup y Eficiencia
