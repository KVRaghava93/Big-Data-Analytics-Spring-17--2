import org.apache.spark.{SparkContext, SparkConf}
/**
  * Created by raghava koundinya on 2/1/2017.
  */
object sparkAT {


  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir", "F:\\winutils");

    val sparkConf = new SparkConf().setAppName("sparkAT").setMaster("local[*]")

    val sc = new SparkContext(sparkConf)

    val input1 = sc.textFile("input1.txt")
    val input2 = sc.textFile("input2.txt")
    val input = sc.textFile("input")

    //val comb = Seq(input1, input2,input)
    val BigFile = input1.union(input2)
    val BigFile2 = BigFile.union(input)
    val counts = BigFile2.distinct().count()
    val intFile = input1.intersection(input2)
    intFile.collect()
    println(s"number of words in the combined file $counts")
    val SmallFile = BigFile.flatMap(line => {
      line.split(" ")
    }).map(word1 => (word1, 1)).cache()

    val output2 = SmallFile.reduceByKey(_ + _)
    val output = BigFile.filter(word => word == "New York")
    BigFile2.saveAsTextFile("output1")
   // BigFile2.saveAsSequenceFile("output3")
    output.saveAsTextFile("output")
    output2.saveAsTextFile("output2")

    //val c = output.collect()
    //var s:String="Words:Count \n"
    //c.foreach { case (word1, count1) => {

    // s += word1 + " : " + count1 + "\n"
  }



}
