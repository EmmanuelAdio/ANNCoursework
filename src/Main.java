import java.io.File;
import java.io.FileNotFoundException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) throws FileNotFoundException {
        PredictorData TrainingData = new PredictorData("src/TrainingDataset.csv");

        //System.out.println(TrainingData.getDataset().toString());
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println();
        System.out.println("After 1 epoch");
        BackPropogation model1 = new BackPropogation(TrainingData.getDataset(),5,10000);



    }
}