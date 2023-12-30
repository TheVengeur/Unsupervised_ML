from time import time
import numpy as np
import matplotlib.pyplot as plt
 
def Exercice1():

# Define hair colors and corresponding probabilities
    hair_colors = ["Blonde", "Brunette", "Red", "Black"]
    probabilities = [0.2, 0.4, 0.2, 0.2]  # Example probabilities for each hair color

    # Number of points to sample
    n_points = 100

    # Number of iterations for convergence check
    max_n = 1000

    # Expected value calculation
    expected_value = sum(p * i for p, i in zip(hair_colors, probabilities))

    # Function to calculate Euclidean distance
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))

    # Sample n_points from the joint distribution
    samples = np.random.choice(hair_colors, size=n_points, p=probabilities)

    # Plot the sampled points
    colors = ['gold' if color == 'Blonde' else 'brown' if color == 'Brunette' else 'red' if color == 'Red' else 'black' for color in samples]
    plt.scatter(range(1, n_points + 1), np.random.rand(n_points), c=colors, alpha=0.6)
    plt.title('Sampled Points from Joint Distribution')
    plt.xlabel('Sample Number')
    plt.ylabel('Random Value')
    plt.show()

    # Initialize arrays to store results
    n_values = np.arange(1, max_n + 1)
    distances = np.zeros(max_n)

    # Generate samples and calculate empirical averages
    for n in n_values:
        samples = np.random.choice(hair_colors, size=n, p=probabilities)
        empirical_average = np.sum([1 if color == 'Blonde' else 2 if color == 'Brunette' else 3 if color == 'Red' else 4 for color in samples]) / n
        distances[n - 1] = euclidean_distance(np.array([empirical_average]), np.array([expected_value]))

    # Plot the results
    plt.plot(n_values, distances)
    plt.title('Convergence of Empirical Average to Expected Value')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Euclidean Distance')
    plt.show()



if __name__ == '__main__':
    print("[EX1] Calculate Z for propability of having a certain hair color...")
    start = time()
    Exercice1()
    print(f"[EX1] Done - {(time() - start):0.4f}s")