import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
//import org.apache.log4j.{Level, Logger}
/**
  * Created by raghava koundinya on 2/8/2017.
  */
object LinearRegressionwithSGD {

  def main(args: Array[String]): Unit ={


    System.setProperty("hadoop.home.dir","F:\\winutils");

    val sparkConf = new SparkConf().setAppName("ChimpLinearRegModel").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    // Turn off Info Logger for Consolexxx
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    // Load and parse the data
    val Chimpdata = sc.textFile("data\\chimpanzeedata.txt")
    val parsedData = Chimpdata.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    parsedData.take(1).foreach(f=>println(f))

    // Split data into training (95%) and test (5%).
    val Array(train, test) = parsedData.randomSplit(Array(0.65, 0.35))

    // Building the model
    val numofIterations = 110
    val stepSize = 0.00000002
    val model = LinearRegressionWithSGD.train(train, numofIterations, stepSize)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = train.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MeanSquareError = valuesAndPreds.map{ case(x, y) => math.pow((x - y), 2) }.mean()
    println("Train Data Mean Squared Error = " + MeanSquareError)

    // Evaluate model on test examples and compute training error
    val valuesAndPreds2 = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MeanSquareError2 = valuesAndPreds2.map{ case(x, y) => math.pow((x - y), 2) }.mean()
    println("Test Data Mean Squared Error = " + MeanSquareError2)

    // Save and load model
    model.save(sc, "data\\ChimpLinearRegModel")
    val sameModel = LinearRegressionModel.load(sc, "data\\ChimpLinearRegModel")
  }

}
