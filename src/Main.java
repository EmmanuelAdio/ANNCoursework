import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

import static java.lang.Integer.parseInt;

public class Main {
    public static void main(String[] args) throws IOException {
        //Bring the datasets into the program.
        PredictorData TrainingData = new PredictorData("src/Datasets/TrainingDataset.txt");
        PredictorData ValidationData = new PredictorData("src/Datasets/ValidationDataset.txt");
        PredictorData TestingData = new PredictorData("src/Datasets/TestingDataset.txt");

        PredictorData TrainingData2 = new PredictorData("src/Datasets/TrainingDataset_2.txt");
        PredictorData ValidationData2 = new PredictorData("src/Datasets/ValidationDataset_2.txt");
        PredictorData TestingData2 = new PredictorData("src/Datasets/TestingDataset_2.txt");

        new Momentum_Annealing(TrainingData.getDataset(),ValidationData.getDataset(),5, 5000);
        new Momentum_Annealing(TrainingData.getDataset(),ValidationData.getDataset(),5, 7500);
        new Momentum_Annealing(TrainingData.getDataset(),ValidationData.getDataset(),5, 10000);

        new Momentum_Annealing(TrainingData.getDataset(),ValidationData.getDataset(),11, 5000);
        new Momentum_Annealing(TrainingData.getDataset(),ValidationData.getDataset(),11, 7500);
        new Momentum_Annealing(TrainingData.getDataset(),ValidationData.getDataset(),11, 10000);

        System.out.println("\n\n");

        new Momentum_Annealing_WeightDecay(TrainingData.getDataset(),ValidationData.getDataset(),5, 5000);
        new Momentum_Annealing_WeightDecay(TrainingData.getDataset(),ValidationData.getDataset(),5, 7500);
        new Momentum_Annealing_WeightDecay(TrainingData.getDataset(),ValidationData.getDataset(),5, 10000);

        new Momentum_Annealing_WeightDecay(TrainingData.getDataset(),ValidationData.getDataset(),11, 5000);
        new Momentum_Annealing_WeightDecay(TrainingData.getDataset(),ValidationData.getDataset(),11, 7500);
        new Momentum_Annealing_WeightDecay(TrainingData.getDataset(),ValidationData.getDataset(),11, 10000);

        System.out.println("\n\n");

        new Momentum_BoldDriver(TrainingData.getDataset(),ValidationData.getDataset(),5, 5000);
        new Momentum_BoldDriver(TrainingData.getDataset(),ValidationData.getDataset(),5, 7500);
        new Momentum_BoldDriver(TrainingData.getDataset(),ValidationData.getDataset(),5, 10000);

        new Momentum_BoldDriver(TrainingData.getDataset(),ValidationData.getDataset(),11, 5000);
        new Momentum_BoldDriver(TrainingData.getDataset(),ValidationData.getDataset(),11, 7500);
        new Momentum_BoldDriver(TrainingData.getDataset(),ValidationData.getDataset(),11, 10000);



//        new Annealing_WeightDecay(TrainingData.getDataset(),ValidationData.getDataset(),11, 4500);

        //this is to run mass testing! will br removed in final version

//        int[] nodeList = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15 ,16};
//        for (int n : nodeList){
//            //System.out.println("\n"+Integer.toString(n)+"_"+Integer.toString(e)+":");
//            new BackPropagation(TrainingData.getDataset(), ValidationData.getDataset(),n,50000);
//            new BackProp_tanh(TrainingData.getDataset(), ValidationData.getDataset(),n,50000);
//        }
//
//        int[] nodeList2 = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
//        for (int n : nodeList2){
//            //System.out.println("\n"+Integer.toString(n)+"_"+Integer.toString(e)+":");
//            new BackPropagation(TrainingData2.getDataset(), ValidationData2.getDataset(),n,50000);
//            new BackProp_tanh(TrainingData2.getDataset(), ValidationData2.getDataset(),n,50000);
//        }


//        int inputs = TrainingData2.getDataset().get(0).size()-1;
//        System.out.println("The number of predictors = "+inputs);
//
//        //get the number of nodes and epochs
//        Scanner scan = new Scanner(System.in);
//        String nodesS;
//        String nodeMessage = "Enter the number of hidden nodes" + " (enter between " + inputs / 2 +
//                " and " + inputs * 2 + ")" + ":";
//        System.out.println(nodeMessage);
//        nodesS = scan.nextLine();
//        while (((inputs/2)>parseInt(nodesS)) || ((inputs*2)<parseInt(nodesS))) {
//            System.out.println(nodeMessage);
//            nodesS = scan.nextLine();
//        }
//
//        int nodes = parseInt(nodesS);
//
//        System.out.println("Enter the number of hidden Epochs you want to train the neural network on?:");
//        String epochsS = scan.nextLine();
//
//        int epochs = parseInt(epochsS);
//
//        //if you want to run each model on the same initial weights use this W_B object
//        Weights_Biases W_B = new Weights_Biases(TrainingData.getDataset(),nodes);

//        BatchLearning test1 = new BatchLearning(TrainingData2.getDataset(),ValidationData2.getDataset(),8,1000000,70);
//        test1.showResults(TestingData2.getDataset());

//        int i = 0;
//        for (ArrayList<Double> sample : TrainingData.getDataset()){
//            i++;
//            System.out.print("sample : "+ i);
//            System.out.println(test1.forwardPass(sample));
//        }


//        System.out.println("\n\n\n\n");

//        BackProp_tanh model11 = new BackProp_tanh(TrainingData2.getDataset(),ValidationData2.getDataset(),7,50000);
//        model11.showResults(TestingData2.getDataset());


//        model11.showWeights();


//        Momentum test2 = new Momentum(TrainingData.getDataset(),ValidationData.getDataset(),7,20000);
//        test2.showResults(TestingData.getDataset());

//        System.out.println("BackPropagation:");
//        Momentum_WeightDecay model1 = new Momentum_WeightDecay(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs);
//
//        System.out.println("\n\n\n\n Momentum:");
//        Momentum model2 = new Momentum(TrainingData.getDataset(), ValidationData.getDataset(),nodes,epochs);

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