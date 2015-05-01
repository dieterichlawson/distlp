package distlp.cg

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.SparkContext
import org.apache.spark.vecext._
import org.apache.spark.vecext.norm
import distlp.linalg.matext._

class ConjugateGradientSolver(val A: Function1[DenseVector, DenseVector],
                              val b: DenseVector,
                              val n: Long,
                              val x_0: DenseVector = null,
                              val tol: Double = 1e-6,
                              val verbose: Boolean = false,
                              @transient val sc: SparkContext)
  extends Serializable with Logging {

  var iter: Int = 0
  var x: DenseVector = _
  
  if (x_0 == null){
    x = DenseVector.zeros(n.toInt)
  }else{
    x = x_0.copy
  }

  var r: DenseVector = b.copy
  var p: DenseVector = r.copy

  def solve() {
    var converged = false
    var rs = scala.math.pow(norm(r),2)
    while (!converged) {
      val t0 = System.nanoTime()
      val Ap = A(p)
      val alpha = rs / (p dot Ap)
      x = x + (alpha * p)
      r = r - (alpha * Ap)
      if (norm(r) <= tol) converged = true
      val rs_new = scala.math.pow(norm(r),2)
      val beta = rs_new / rs
      p = r + (beta * p)
      rs = rs_new
      iter = iter + 1
      val err = norm(r)
      val t = (System.nanoTime() - t0)/1e9
      if (verbose) println(s"Iteration $iter, error: $err, time: $t s")
    }
  }
}
