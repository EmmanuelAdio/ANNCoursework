import java.io.FileNotFoundException;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws FileNotFoundException {
        PredictorData TrainingData = new PredictorData("src/TrainingDataset.txt");
        PredictorData ValidationData = new PredictorData("src/ValidationDataset.txt");

        Scanner scan = new Scanner(System.in);
        System.out.println("Enter the number of hidden nodes:");
        String nodesS = scan.nextLine();

        int nodes = Integer.parseInt(nodesS);

        System.out.println("Enter the number of hidden Epochs you want to train the neural network on?:");
        String epochsS = scan.nextLine();

        int epochs = Integer.parseInt(epochsS);

        Weights_Biases W_B = new Weights_Biases(TrainingData.getDataset(), ValidationData.getDataset(),nodes);

        System.out.println("BackPropagation:");
        BackPropagation model1 = new BackPropagation(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs,W_B);

        System.out.println("\n\n\n\n Momentum:");
        Momentum model2 = new Momentum(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs, W_B);

        System.out.println("\n\n\n\n Annealing:");
        Annealing model3 = new Annealing(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs, W_B);

        System.out.println("\n\n\n\n WeightDecay:");
        WeightDecay model4 = new WeightDecay(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs, W_B);

        System.out.println("\n\n\n\n BoldDriver:");
        BoldDriver model5 = new BoldDriver(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs, W_B);
    }
}