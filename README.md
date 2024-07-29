# Artificial Neural Network for Predicting Flood Index

## Overview
This project involves an Artificial Neural Network (ANN) designed to predict the flood index using the backpropagation algorithm. The ANN has been trained and tested with various improvements to optimize its performance.

## Features
- **Data Preprocessing:** Includes cleaning, splitting, randomizing, and standardizing the dataset.
- **Predictors:** Uses multiple predictors such as AREA, BFIHOST, FARL, FPEXT, LDP, PROPWET, RMED-1D, and SAAR.
- **Training Algorithm:** Implements the backpropagation algorithm with various enhancements like Momentum, Annealing, Bold Driver, Weight Decay, and Batch Processing.
- **Evaluation:** Compares the performance of different models and selects the best model based on Mean Squared Error (MSE).

## Installation and Usage
### Prerequisites
- Java Development Kit (JDK) installed
- IntelliJ IDEA or any other Java IDE

### Installation
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/EmmanuelAdio/ANNCoursework.git
    ```

2. **Open the Project:**
    - Open the cloned repository in your Java IDE.

3. **Compile the Code:**
    - Ensure all dependencies are resolved and compile the project.

### Running the ANN
1. **Prepare the Dataset:**
    - Place the dataset in the specified directory. The dataset should be cleaned and standardized as per the guidelines in the report.

2. **Configure Parameters:**
    - Set the number of hidden nodes and epochs as required. Example:
    ```java
    int hiddenNodes = 11;
    int epochs = 10000;
    ```

3. **Execute the Main Class:**
    - Run the main class to start the training process.
    ```java
    public class Main {
        public static void main(String[] args) {
            // Load dataset and initialize ANN
            ...
            // Train and evaluate the model
            ...
        }
    }
    ```

## Detailed Implementation
The implementation details are as follows:
- **BackPropagation Class:** Implements the core backpropagation algorithm with methods for initialization, forward pass, backward pass, and weight updates.
- **Momentum, Annealing, Bold Driver, Weight Decay, and Batch Processing:** These improvements are implemented as subclasses extending the BackPropagation class, each enhancing the basic algorithm.

## Results
- **Best Model:** The final model combines Momentum, Annealing, and Weight Decay, achieving the lowest MSE of 0.0015618.
- **Comparison:** The ANN significantly outperforms a linear regression model, highlighting its efficacy in predicting the flood index.

## Limitations
- The user must manually configure the dataset path in the main class.
- Error handling for user inputs (number of nodes, epochs) is not implemented.

## Future Work
- Implement a user interface to simplify configuration and dataset management.
- Enhance error handling and logging for better user experience.

## Contributions
Contributions are welcome. Please fork the repository and create a pull request with your changes.

---

By following these instructions, you can set up and use the ANN for predicting flood index. Enjoy experimenting with different configurations and improvements to achieve optimal results!
