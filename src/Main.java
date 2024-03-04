import java.io.FileNotFoundException;

public class Main {
    public static void main(String[] args) throws FileNotFoundException {
        PredictorData TrainingData = new PredictorData("src/TrainingDataset.txt");
        PredictorData ValidationData = new PredictorData("src/ValidationDataset.txt");

        BackPropagation model1 = new BackPropagation(TrainingData.getDataset(), ValidationData.getDataset(),7,1000);
        System.out.println("\n\n\n\n");
        Momentum model2 = new Momentum(TrainingData.getDataset(), ValidationData.getDataset(),7,1000);
        System.out.println("\n\n\n\n");
        Annealing model3 = new Annealing(TrainingData.getDataset(), ValidationData.getDataset(),7,1000);
        System.out.println("\n\n\n\n");
        WeightDecay model4 = new WeightDecay(TrainingData.getDataset(), ValidationData.getDataset(),7,20000);

    }
}