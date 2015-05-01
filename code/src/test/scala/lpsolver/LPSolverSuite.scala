package distlp.cg

import org.scalatest.FunSuite
import org.scalatest.Matchers
import distlp.SparkTestUtils

import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.DenseMatrix
import distlp.linalg._
import distlp.linalg.matext._
import distlp.LPSolver
import org.apache.spark.vecext._
import org.apache.spark.vecext.norm

import org.apache.spark.SparkContext

class LPSolverSuite extends SparkTestUtils{
  
  test("LP Solver correctly solves the system") {
    val m = 100; val n = 200;
    val A = randBlockMatrix(m,n,1,1,sc)
    A.blocks.cache
    val x_0 = DenseVector.rand(n)
    val b = A.multiply(x_0)
    val c = DenseVector.rand(n)
    val y_0 = DenseVector.zeros(m)
    val z_0 = c.copy
    println("Created problem...")
    // assert primal/dual feasibility
    assert(norm(A.multiply(x_0)-b) <= 1e-6)
    assert(norm(A.transpose.multiply(y_0) + z_0-c) <=1e-6)
    assert(!x_0.any_less(0))
    assert(!z_0.any_less(0))
    println("Asserted primal/dual feasibility...")
    println("Solving...")
    val lp = new LPSolver(A, b, c, x_0, y_0, z_0, 0.5, 0.5, 1e-6, 1e-6, 50, sc)
    lp.solve
  }

  def randBlockMatrix(m:Int, n:Int,  nvertblocks: Int, nhorizblocks: Int, sc: SparkContext): BlockMatrix = {
    val inds = for(r <- 0 to nvertblocks-1; c <- 0 to nhorizblocks-1) yield (r,c)
    val blocks = sc.parallelize(inds).map(x => (x, DenseMatrix.rand(m/nvertblocks,n/nhorizblocks)))
    new BlockMatrix(blocks,m/nvertblocks, n/nhorizblocks, m.toLong, n.toLong)
  }
}
