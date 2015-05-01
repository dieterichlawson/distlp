package distlp.linalg

import org.scalatest.FunSuite
import org.scalatest.Matchers
import distlp.SparkTestUtils
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.Matrix
import distlp.linalg._
import org.apache.spark.vecext._
import org.apache.spark.SparkContext
import scala.util.Random

class DenseVectorExtensionsSuite extends FunSuite{
  test("DV + DV") {
    val a = DenseVector.rand(1000)
    val b = DenseVector.rand(1000)
    assert((norm(a+b) - breeze.linalg.norm(a.toBreeze + b.toBreeze)) <= 1e-8) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = new DenseVector(Array(3.0,4.0))
    assert((c+d) == new DenseVector(Array(4.0,6.0)))
  }

  test("DV - DV") {
    val a = DenseVector.rand(1000)
    val b = DenseVector.rand(1000)
    assert((norm(a-b) - breeze.linalg.norm(a.toBreeze - b.toBreeze)) <= 1e-10) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = new DenseVector(Array(3.0,4.0))
    assert((c-d) == new DenseVector(Array(-2.0,-2.0)))
  }

  test("DV dot DV") {
    val a = DenseVector.rand(1000)
    val b = DenseVector.rand(1000)
    assert((a dot b - a.toBreeze.dot(b.toBreeze)) <= 1e-10)
    val c = new DenseVector(Array(1.0,2.0))
    val d = new DenseVector(Array(3.0,4.0))
    assert((c dot d) == 11.0)

  }
  
  test("norm(DV)") {
    val a = DenseVector.rand(1000)
    assert((norm(a) - breeze.linalg.norm(a.toBreeze)) <= 1e-10)
    val b = new DenseVector(Array(2.0,2.0,2.0,2.0))
    assert(norm(b) == 4.0)
  }
  
  test("DV :* DV") {
    val a = DenseVector.rand(1000)
    val b = DenseVector.rand(1000)
    assert((norm(a :* b) - breeze.linalg.norm(a.toBreeze :* b.toBreeze)) <= 1e-10) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = new DenseVector(Array(3.0,4.0))
    assert((c :* d) == new DenseVector(Array(3.0,8.0)))
  }

  test("DV + Double") {
    val a = DenseVector.rand(1000)
    val b = Random.nextDouble
    assert((norm(a + b) - breeze.linalg.norm(a.toBreeze + b)) <= 1e-10) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = 2.0
    assert((c + d) == new DenseVector(Array(3.0,4.0)))
  }

  test("DV - Double") {
    val a = DenseVector.rand(1000)
    val b = Random.nextDouble
    assert((norm(a - b) - breeze.linalg.norm(a.toBreeze - b)) <= 1e-10) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = 2.0
    assert((c - d) == new DenseVector(Array(-1.0,0.0)))

  }

  test("DV * Double") {
    val a = DenseVector.rand(1000)
    val b = Random.nextDouble
    assert((norm(a * b) - breeze.linalg.norm(a.toBreeze * b)) <= 1e-10) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = 2.0
    assert((c * d) == new DenseVector(Array(2.0,4.0)))
  }

  test("DV / Double") {
    val a = DenseVector.rand(1000)
    val b = Random.nextDouble
    assert((norm(a / b) - breeze.linalg.norm(a.toBreeze / b)) <= 1e-10) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = 2.0
    assert((c / d) == new DenseVector(Array(0.5,1.0)))
  }

  test("Double + DV") {
    val a = DenseVector.rand(1000)
    val b = Random.nextDouble
    assert((norm(b+a) - breeze.linalg.norm(a.toBreeze + b)) <= 1e-10) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = 2.0
    assert((d + c) == new DenseVector(Array(3.0,4.0)))
  }

  test("Double - DV") {
    val a = DenseVector.rand(1000)
    val b = Random.nextDouble
    assert((norm(b-a) - breeze.linalg.norm(BDV.ones[Double](1000)*b - a.toBreeze)) <= 1e-10) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = 2.0
    assert((d - c) == new DenseVector(Array(1.0,0.0)))
  }

  test("Double * DV") {
    val a = DenseVector.rand(1000)
    val b = Random.nextDouble
    assert((norm(b * a) - breeze.linalg.norm(a.toBreeze * b)) <= 1e-10) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = 2.0
    assert((d* c) == new DenseVector(Array(2.0,4.0)))
  }

  test("Double / DV") {
    val a = DenseVector.rand(1000)
    val b = Random.nextDouble
    assert((norm(b / a) - breeze.linalg.norm(BDV.ones[Double](1000)*b :/ a.toBreeze)) <= 1e-10) 
    val c = new DenseVector(Array(1.0,2.0))
    val d = 2.0
    assert((d / c) == new DenseVector(Array(2.0,1.0)))
  }
}
