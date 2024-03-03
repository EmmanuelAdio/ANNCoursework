import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Scanner; // Import the Scanner class to read text files

public class PredictorData {
    private ArrayList<String> Dataset;
    public static void main(String[] args) throws FileNotFoundException {
        ArrayList<ArrayList<Integer>> Dataset = new ArrayList<ArrayList<Integer>>();
        File DatasetFile = new File("TrainingDataset.txt");
        Scanner scan = new Scanner(DatasetFile);
        while(scan.hasNextLine()){
            ArrayList<Integer> dataSample = new ArrayList<Integer>();
            String[] dataSampleRaw = scan.nextLine().split("|");
            for(String val : dataSampleRaw){
                dataSample.add(Integer.valueOf(val));
            }
            Dataset.add(dataSample);
        }

        System.out.println(Dataset.toString());
    }
}
