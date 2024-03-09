import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

import static java.lang.Integer.parseInt;
import static java.lang.Integer.toHexString;

public class Main {
    public static void main(String[] args) throws IOException {
        //Bring the datasets into the program.
        PredictorData TrainingData = new PredictorData("src/Datasets/TrainingDataset.txt");
        PredictorData ValidationData = new PredictorData("src/Datasets/ValidationDataset.txt");
        PredictorData TestingData = new PredictorData("src/Datasets/TestingDataset.txt");

        int inputs = TrainingData.getDataset().get(0).size()-1;
        System.out.println("The number of predictors = "+inputs);

        //get the number of nodes and epochs
        Scanner scan = new Scanner(System.in);
        String nodesS = null;
        String nodeMessage = "Enter the number of hidden nodes" + " (enter between " + Integer.toString(inputs / 2) +
                " and " + Integer.toString(inputs * 2) + ")" + ":";
        System.out.println(nodeMessage);
        nodesS = scan.nextLine();
        if (((inputs/2)>parseInt(nodesS)) || ((inputs*2)<parseInt(nodesS))) {
            System.out.println(nodeMessage);
            nodesS = scan.nextLine();
        }

        int nodes = parseInt(nodesS);

        System.out.println("Enter the number of hidden Epochs you want to train the neural network on?:");
        String epochsS = scan.nextLine();

        int epochs = parseInt(epochsS);

        //if you want to run each model on the same initial weights use this W_B object
        Weights_Biases W_B = new Weights_Biases(TrainingData.getDataset(), ValidationData.getDataset(),nodes);

        BackPropagation test1 = new BackPropagation(TrainingData.getDataset(),ValidationData.getDataset(),7,20000);
        test1.showResults(TestingData.getDataset());

        System.out.println("\n\n\n\n");
        BackProp_tanh test2 = new BackProp_tanh(TrainingData.getDataset(),ValidationData.getDataset(),7,20000);
        test2.showResults(TestingData.getDataset());

//        System.out.println("BackPropagation:");
//        Momentum_WeightDecay model1 = new Momentum_WeightDecay(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs);
//
//        System.out.println("\n\n\n\n Momentum:");
//        Momentum model2 = new Momentum(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs);


        //this is to run mass testing! will br removed in final version
//        int[] epochsList = {1000, 5000, 10000};
//        int[] nodeList = {8, 9, 10, 11, 12};
//
//        for (int n : nodeList){
//            for (int e : epochsList){
//                System.out.println("\n"+Integer.toString(n)+"_"+Integer.toString(e)+":");
//                new BackPropagation(TrainingData.getDataset(), ValidationData.getDataset(),n,e);
//            }
//        }



//
//        System.out.println("\n\n\n\n Annealing:");
//        Annealing model3 = new Annealing(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs);
//
//        System.out.println("\n\n\n\n WeightDecay:");
//        WeightDecay model4 = new WeightDecay(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs);
//
//        System.out.println("\n\n\n\n BoldDriver:");
//        BatchLearning model5 = new BatchLearning(TrainingData.getDataset(), ValidationData.getDataset(),8,10000);
    }
}