package org.apache.spark

import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.BLAS
import distlp.linalg.BLASExt
import scala.util.Random
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}

package object vecext {

  def norm(f: DenseVector): Double = BLASExt.nrm2(f)

  object DenseVectorExtensions {
    def rand(m:Int): DenseVector = new DenseVector(Array.fill(m)(Random.nextDouble))
    def zeros(m:Int): DenseVector = new DenseVector(Array.fill(m)(0))
  }

  implicit def fromDenseVector(objA: DenseVector.type) = DenseVectorExtensions

  implicit class DenseVectorExtensions(val v:DenseVector) {

    def apply(f: Int, t:Int): DenseVector = new DenseVector(v.values.slice(f,t))

    def toBreeze: BDV[Double] = new BDV[Double](v.values)

    def unary_-(): DenseVector = {
      v * -1.0
    }

    def +(x: DenseVector): DenseVector = {
      var y = DenseVector.zeros(v.size)
      BLAS.copy(v,y)
      BLAS.axpy(1.0,x,y)
      return y
    }

    def -(b: DenseVector): DenseVector = {
      var a = DenseVector.zeros(b.size)
      BLAS.copy(v,a)
      BLAS.axpy(-1.0,b,a)
      return a
    }
    
    def dot(b: DenseVector): Double = {
      return BLAS.dot(v,b)
    }

    def :*(b: DenseVector): DenseVector = {
      var y = DenseVector.zeros(b.size)
      BLASExt.sbmv("u",v.size,0,1.0,v,1,b,1.0,y)
      return y
    }

    def map(f: (Double) => Double): DenseVector = new DenseVector(v.values.map(f))

    def *(a: Double): DenseVector = {
      var r = DenseVector.zeros(v.size)
      BLAS.copy(v,r)
      BLASExt.scal(r,a)
      return r
    }

    def /(a: Double): DenseVector = map(_/a)
    def +(a: Double): DenseVector = map(_+a)
    def -(a: Double): DenseVector = map(_-a)

    def any_less(y: Double): Boolean = {
      v.values.map(_ < 0).reduce(_ || _)
    }

    def any_greater(y: Double): Boolean = {
      v.values.map(_ > 0).reduce(_ || _)
    }
  }

  implicit class DoubleExtensions(val d:Double){
    def *(v: DenseVector): DenseVector = v*d 
    def /(v: DenseVector): DenseVector = v.map(d/_)
    def +(v: DenseVector): DenseVector = v+d 
    def -(v: DenseVector): DenseVector = v.map(d-_)
  }
}
