import java.io.File;
import java.io.FileNotFoundException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws FileNotFoundException {
        String file = "src/TrainingDataset.csv";
        PredictorData TrainingData = new PredictorData(file);
        System.out.println(TrainingData.getDataset().toString());
    }
}