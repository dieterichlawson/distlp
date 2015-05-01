package distlp

import distlp.cg.ConjugateGradientSolver
import org.apache.spark.Logging

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.storage._
import org.apache.spark.broadcast._
import distlp.linalg.matext._
import org.apache.spark.vecext._

class LPSolver(val A: BlockMatrix,
               val b: DenseVector,
               val c: DenseVector,
               val x_0: DenseVector,
               val y_0: DenseVector,
               val z_0: DenseVector,
               val gamma: Double = 0.5,
               val beta: Double = 0.5,
               val feas_eps: Double = 1e-6,
               val comp_eps: Double = 1e-6,
               val maxIterations: Int = 50,
               @transient val sc: SparkContext)
               extends Serializable with Logging{

  var iter: Int=0
  
  var x = x_0.copy
  var y = y_0.copy
  var z = z_0.copy
  var ystep = DenseVector.zeros(A.numRows.toInt)

  val theta = 0.99;

  def solve = {
    A.blocks.cache
    var done = false
    val total_t0 = System.nanoTime()
    while(!done){
      iter += 1
      val t0 = System.nanoTime()
      val (converged, r1, r2, r_comp, cg_iters) = iterate
      val t1 = System.nanoTime()
      val obj = x dot c
      val time = (t1 - t0)/1e9
      println(f"Iteration $iter: pr: $r1%.6f | dr: $r2%.6f | cr: $r_comp%.6f | obj: $obj%.6f | cg_iters: $cg_iters | t: $time s")
      if(converged){
        println("Converged.")
      }else if(iter >= maxIterations){
        println(s"Max iterations ($maxIterations) reached. Stopping.")
      }
      done = (converged || iter >= maxIterations)
    }
    val total_t1 = System.nanoTime()
    val totaltime = (total_t1 - total_t0)/1e9
    println(s"Elapsed time: $totaltime s")
  }

  def iterate: (Boolean,Double,Double,Double, Int) = {
    val comp_res = x dot z 
    val mu = comp_res/A.numCols;
    val r1 = b - A.multiply(x) // O(# rows) communication cost
    val r2 = c - A.transpose.multiply(y) - z  // TODO: communication cost
    val nr1 = norm(r1)
    val nr2 = norm(r2)
    if(nr1 < feas_eps && nr2 < feas_eps && comp_res < comp_eps) return (true,nr1,nr2,comp_res,0);
    val r3 = gamma*mu - (x :* z) 
    val xinv = 1.0 / x
    val zinv = 1.0 / z
    val r4 = r2 - (xinv :* r3);
    val d = x :* zinv;
    val cg = new ConjugateGradientSolver(x => A.multiply(d :* A.transpose.multiply(x)), 
                                         r1 + A.multiply(d :* r4), 
                                         A.numRows, null,
                                         1e-6, false, sc)
    cg.solve
    ystep = cg.x
    val xstep = d :* (A.transpose.multiply(ystep)-r4); // TODO: communication cost
    val zstep = xinv :* (r3 - (z :* xstep));
    // linesearch
    var alpha = 1.0;
    while((x + alpha*xstep).any_less(0.0) || (z + alpha*zstep).any_less(0.0)){
      alpha = beta*alpha;
    }
    x = x + theta*alpha*xstep
    y = y + theta*alpha*ystep
    z = z + theta*alpha*zstep
    return (false, nr1, nr2, comp_res,cg.iter)
  }
}
