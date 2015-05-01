package distlp

import org.scalatest._
import org.apache.spark.SparkContext

object SparkTest extends org.scalatest.Tag("com.qf.test.tags.SparkTest")

trait SparkTestUtils extends FunSuite with BeforeAndAfter{
  var sc: SparkContext = _

  /**
   * convenience method for tests that use spark. Creates a local spark context,
   * and cleans it up even if your test fails. Also marks the test with the tag
   * SparkTest, so you can turn it off
   *
   * By default, it turn off spark logging, b/c it just clutters up the test output.
   * However, when you are actively debugging one test, you may want to turn the logs on
   *
   * @param name the name of the test
   * @param silenceSpark true to turn off spark logging
   *
   */
  before {
    System.clearProperty("spark.driver.port")
    System.clearProperty("spark.hostPort")
    sc = new SparkContext("local", getClass.getSimpleName)
  }

  after {
    sc.stop
    sc = null
    // To avoid Akka rebinding to the same port,
    // since it doesn't unbind immediately on shutdown
    System.clearProperty("spark.driver.port")
    System.clearProperty("spark.hostPort")
  }
  /*

  def sparkTest(name: String, silenceSpark : Boolean = true)(body: => Unit) {
    test(name, SparkTest){
    //  val origLogLevels = if (silenceSpark) SparkUtil.silenceSpark() else null
      try {
        System.clearProperty("spark.driver.port")
        System.clearProperty("spark.hostPort")
        sc = new SparkContext("local", "test")
        body
      }
      finally {
        sc.stop
        sc = null
        // To avoid Akka rebinding to the same port,
        // since it doesn't unbind immediately on shutdown
        System.clearProperty("spark.driver.port")
        System.clearProperty("spark.hostPort")
      //  if (silenceSpark) Logging.setLogLevels(origLogLevels)
      }
    }
  }
  */
}

object SparkUtil {
  def silenceSpark() {
    //setLogLevels(Level.WARN, Seq("spark", "org.eclipse.jetty", "akka"))
  }

  def setLogLevels(level: org.apache.log4j.Level, loggers: TraversableOnce[String]) = {
    /*
    loggers.map{
      loggerName =>
        val logger = Logger.getLogger(loggerName)
        val prevLevel = logger.getLevel()
        logger.setLevel(level)
        loggerName -> prevLevel
    }.toMap
   */
  }
}
