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

def plot_results(sizes, times,title):
    size_map = {'pequeno': 100, 'Mediano': 1000, 'Grande': 10000}
    sizes_numeric = [size_map[size] for size in sizes]
    
    plt.figure(figsize=(10, 5),num=title)
    
    # Gráfico de tiempo de ejecución
    plt.subplot(1, 2, 1)
    plt.plot(sizes_numeric, times, marker='o')
    plt.title('Tiempo de ejecución')
    plt.xlabel('Tamaño de Datos')
    plt.ylabel('Tiempo (segundos)')
    plt.xscale('log')
    # Gráfico de speedup
    baseline_time = times[0]  # Tiempo de ejecución para el tamaño más pequeno
    speedups = [baseline_time / time for time in times]
    plt.subplot(1, 2, 2)
    plt.plot(sizes_numeric, speedups, marker='o')
    plt.title('Tiempo de ejecución')
    plt.xlabel('Tamaño de Datos')
    plt.ylabel('Speedup')
    plt.xscale('log')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    filename = "results.txt"
    filename1 = "resultsParalized.txt"
    sizes, times  = parse_results(filename)
    sizes1, times1 = parse_results(filename1)
    plot_results(sizes, times,title="Resultados Secuenciales")
    plot_results(sizes1, times1, title="Resultados Paralelizados")
