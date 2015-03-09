/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package distlp.examples

import cg.ConjugateGradientSolver
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import breeze.linalg.operators._
import scopt.OptionParser

object ConjugateGradientExample {

  case class Params(
    numrows: Int = 100,
    tol: Double = 1e-6
  )

  val defaultParams = Params()

  val parser = new OptionParser[Params]("Conjugate Gradient") {
    head("Conjugate Gradient: An example app for using Conjugate Gradient to solve Ax=b.")
    opt[Int]("numrows")
    .text(s"number of rows, default: ${defaultParams.numrows}")
    .action((x,c) => c.copy(numrows= x))
    opt[Double]("tol")
    .text(s"tolerance, default: ${defaultParams.tol}")
    .action((x,c) => c.copy(tol= x))
  }
 
  def main(args: Array[String]){
    parser.parse(args, defaultParams).map { params =>
      run(params)
    } getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params){
    // Housekeeping for spark
    val conf = new SparkConf().setAppName("Conjugate Gradient")
    val sc = new SparkContext(conf)
    println(sc.master)
    if(sc.master.equals("local")){
      sc.setCheckpointDir("/Users/dlaw/foo")
    }else{
       sc.setCheckpointDir(s"hdfs://${sc.master.substring(8,sc.master.length-5)}:9000/root/scratch")
    }
    val numWorkers = 8
    val coresPerWorker = 4
    val tasksPerCore = 3 
    val parallelism = numWorkers* coresPerWorker*tasksPerCore
  
    // Create a PSD matrix
    val B = BDM.rand[Double](params.numrows,params.numrows)
    val A_loc = B.t*B
    val rows =  sc.parallelize(0 to params.numrows-1).map(x => Vectors.dense(A_loc(x,::).t.toArray))
    val A = new RowMatrix(rows)
    println("Created matrix")
    
    // Create the true solution and a b to go with it
    val truex = BDV.rand[Double](params.numrows)
    val b =  BDV[Double](A.multiply(Matrices.dense(A.numCols().toInt,1,truex.data)).
                           rows.map(x => x(0)).collect())
    // solve the problem
    val cg = new ConjugateGradientSolver(A,b,params.tol,true,sc)
    println("Created solver")
    cg.solve()
    println("Solved")
    print("Final Error: ")
    println(norm(truex- cg.x))
    sc.stop
  }
}
