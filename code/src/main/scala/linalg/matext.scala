package distlp.linalg

import org.apache.spark.vecext._
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

import scala.util.Random

package object matext {

  object DenseMatrixExtensions {
    def rand(m:Int, n: Int): Matrix = new DenseMatrix(m,n,Array.fill(m*n)(Random.nextDouble))
  }

  implicit def fromDenseMatrix(objA: DenseMatrix.type) = DenseMatrixExtensions

/*  object BlockMatrixExtensions {
    def rand(m:Int, n:Int,  nvertblocks: Int, nhorizblocks: Int, sc: SparkContext): BlockMatrix = {
      val inds = for(r <- 0 to nvertblocks-1; c <- 0 to nhorizblocks-1) yield (r,c)
      val blocks = sc.parallelize(inds).map(x => (x, DenseMatrix.rand(m/nvertblocks,n/nhorizblocks)))
      new BlockMatrix(blocks,m/nvertblocks, n/nhorizblocks, m.toLong,n.toLong)
    }
  }
  
  implicit def fromBlockMatrix(objA: BlockMatrix.type) = BlockMatrixExtensions
*/
  
  implicit class BlockMatrixExtensions(val A:BlockMatrix) extends Serializable {

    def multiplyBlock(rc: Tuple2[Int,Int], M: Matrix, v: DenseVector): TraversableOnce[(Int,Double)] = {
      val row = rc._1*M.numRows; 
      val col = rc._2*M.numCols;
      (row to row + M.numRows).zip(M.multiply(v(col,col + M.numCols)).toArray)
    }
    
    // multiplies a block matrix by a dense vector
    def multiply(v: DenseVector): DenseVector = {
      val r = A.blocks.flatMap(x => multiplyBlock(x._1,x._2,v)).
                reduceByKey(_ + _).
                collect().
                sortWith(_._1 < _._1).
                map(_._2)
      new DenseVector(r)
    }

    def multiplyBlockDiag(rc: Tuple2[Int,Int], M: Matrix, d: DenseVector): ((Int,Int),Matrix) = {
      val nrows = M.numRows; val ncols = M.numCols;
      val row = rc._1 * nrows; val col = rc._2 * ncols
      val inds = for(i <- Array.range(0,nrows*ncols)) yield col + (i / nrows) 
      val newM = new DenseMatrix(nrows,ncols,inds.zip(M.toArray).map(x => x._2*d(x._1)))
      (rc,newM)
    }

    def multiply_diag(d: DenseVector): BlockMatrix = {
      new BlockMatrix(A.blocks.map(x => multiplyBlockDiag(x._1,x._2,d)), A.rowsPerBlock,A.colsPerBlock,A.numRows,A.numCols)
    }

    // computes A*D*A'
    // where D is diagonal and specified by d
    def weighted_outer_prod(d: DenseVector): BlockMatrix = {
      A.multiply_diag(d).multiply(A.transpose)
    }
  }
}
