package cg 

import org.apache.spark.Logging

import breeze.linalg._
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.storage._
import org.apache.spark.broadcast._

class ConjugateGradientSolver(
               val A: RowMatrix,
               val b: BDV[Double],
               val tol: Double = 1e-6,
               val verbose: Boolean = false,
               @transient val sc: SparkContext)
               extends Serializable with Logging{

  var iter: Int=0
  var r: BDV[Double] = b
  var p: BDV[Double] = r
  var x: BDV[Double] = BDV.zeros[Double](A.numCols().toInt)

  def solve(){
    var converged = false
    while(!converged){
      val Ap = BDV[Double](A.multiply(Matrices.dense(A.numCols().toInt,1,p.data)).
                             rows.map(x => x(0)).collect())
      val alpha = (r.t*r)/(p.t*Ap)
      x = x + p*alpha
      val r_new = r - Ap*alpha
      if(norm(r_new) <= tol) converged = true
      val beta = (r_new.t*r_new)/(r.t*r)
      r = r_new
      p = r + p*beta
      iter = iter+1
      val err= norm(r_new)
      if(verbose) println(s"Iteration $iter, error: $err")
    }
  }
}
