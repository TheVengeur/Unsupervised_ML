from time import time
import numpy as np
import matplotlib.pyplot as plt
 
def Exercice1():
    # Define possible values for X (age) and Y (hair color percentage)
    age_values = np.array([20, 30, 40])
    hair_color_percentages = np.array([0.2, 0.5, 0.3])

    print("here is a representation of the hair color and the percentage of them in the population")
    # Define possible values for Y (hair color percentage)
    hair_color_categories = ['Blonde', 'Brown', 'Other']
    hair_color_percentages = np.array([0.2, 0.5, 0.3])

    # Plot the results with color
    fig, ax = plt.subplots()
    bars = ax.bar(hair_color_categories, hair_color_percentages, color=['#FFD700', '#964B00', '#808080'])
    ax.set_ylabel('Hair Color Percentage')
    ax.set_title('Hair Color Distribution')

    # Add percentages on top of the bars
    for bar, percentage in zip(bars, hair_color_percentages):
        height = bar.get_height()
        ax.annotate(f'{percentage*100:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.show()

    # Create a joint probability distribution
    joint_distribution = np.array([[0.05, 0.1, 0.05],
                                [0.1, 0.3, 0.1],
                                [0.05, 0.1, 0.05]])

    # Function to calculate Euclidean distance between two points
    def euclidean_distance(point1, point2):
        return np.linalg.norm(point1 - point2)

    # Initialize arrays for storing results
    n_values = np.arange(1, 1001)  # Values of n from 1 to 1000
    distances = []

    print("we sample it to 1000")
    # Calculate expected value E[Z]
    expected_value = np.sum(np.outer(age_values, hair_color_percentages) * joint_distribution)

    print("the expected value is :" +str(expected_value))

    # Sample n points and calculate empirical averages
    for n in n_values:
        probabilities_x = np.sum(joint_distribution, axis=1) / np.sum(joint_distribution)
        probabilities_y = np.sum(joint_distribution, axis=0) / np.sum(joint_distribution)
        
        samples_x = np.random.choice(age_values, size=n, p=probabilities_x)
        samples_y = np.random.choice(hair_color_percentages, size=n, p=probabilities_y)
        
        samples = np.column_stack((samples_x, samples_y))
        empirical_average = np.mean(samples, axis=0)
        distance = euclidean_distance(empirical_average, expected_value)
        distances.append(distance)

    # Plot the results with color
    fig, ax = plt.subplots()
    sc = ax.scatter(n_values, distances, c=samples_y, cmap='viridis', marker='.')
    fig.colorbar(sc, label='Hair Color Percentage')
    ax.set_xlabel('Number of Samples (n)')
    ax.set_ylabel('Euclidean Distance')
    ax.set_title('Convergence of Empirical Average to Expected Value')
    plt.show()



if __name__ == '__main__':
    print("[EX1] Calculate Z for propability of having a certain hair color...")
    start = time()
    Exercice1()
    print(f"[EX1] Done - {(time() - start):0.4f}s")