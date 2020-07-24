import org.apache.spark.sql.SparkSession;

public class SparkTestingClass {

    public static void SparkSessionTest(){
        SparkSession spark = SparkSession.builder().config("spark.master", "local").getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        System.out.println("WORKING!");
    }

    public static void main(String[] args) {
        SparkSessionTest();
    }
}
