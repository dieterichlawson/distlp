package distlp.linalg

import org.apache.spark.mllib.linalg.DenseVector
import com.github.fommil.netlib.{BLAS => NetlibBLAS, F2jBLAS}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}

object BLASExt {

    @transient private var _f2jBLAS: NetlibBLAS = _

    private def f2jBLAS: NetlibBLAS = {
      if (_f2jBLAS == null) {
        _f2jBLAS = new F2jBLAS
      }
      _f2jBLAS
    }

    /**
     * y = alpha*A*x + beta*y
     */
    def sbmv(uplo: String, n: Int, k: Int, alpha: Double, a: DenseVector, lda: Int, x: DenseVector, beta: Double, y: DenseVector): Unit = {
      f2jBLAS.dsbmv(uplo, n, k, alpha, a.values, lda, x.values, 1, beta, y.values, 1)
    }

    /**
     * x = alpha*x
     */
    def scal(x: DenseVector, alpha: Double): Unit = {
      f2jBLAS.dscal(x.size, alpha, x.values, 1)
    }

    /**
     * Euclidean norm of x
     * sqrt(x'x)
     */
    def nrm2(x: DenseVector): Double = {
      f2jBLAS.dnrm2(x.size, x.values, 1)
    }
}
