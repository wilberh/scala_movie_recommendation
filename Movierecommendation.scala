package com.sparkaws

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import org.apache.spark.sql.functions._

class Movierecommendation {
  
  //Rating class
  case class Rating(userId: Int, movieId: Int, rating: Float)
  
  //Passing a line of movies.dat to Rating
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat)
  }
  
  
  //Load map of movie IDs to movie names
  //From disk to memory
  def loadMovieNames() : Map[Int, String] = {
    
    //Handle character encoding issues:
    implicit val codec = Codec("UTF-8")
    codec.onMalformedInput(CodingErrorAction.REPLACE)
    codec.onUnmappableCharacter(CodingErrorAction.REPLACE)
    
    //Create map of Ints to Strings, to populate from movies.dat
    var movieNames:Map[Int, String] = Map()
    
    //Update movies.dat if stored someplace other than current directory
    val lines = Source.fromFile("/home/hadoop/movies.dat").getLines()
    for (line <- lines) {
      var fields = line.split("::")
      if (fields.length > 1) {
        movieNames += (fields(0).toInt -> fields(1))
      }
    }
    return movieNames
  }
  
  
  def main(args: Array[String]) {
    
    //Set SparkSession
    val spark = SparkSession.builder.appName("ALSExample").getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    //print map of movie ID's to movie names in memory
    println("Loading movie names...")
    val nameDict = loadMovieNames()
    
    import spark.implicits._
    
    //  s3://movierecommendation/ml-1m/ratings.dat
    //load ratings or large data to train recommendation model with
    val ratings = spark.read.textFile("s3://movierecommendation/ml-1m/ratings.dat").map(parseRating)
    
    //count number of ratings each movie for later use
    val ratingCounts = ratings.groupBy("movieId").count()
    
    //apply ALS or Alternating Least Squares recommender with fictitious user parameters
    val als = new ALS().setRank(8).setMaxIter(10).setRegParam(0.1).setSeed(1234).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
    
    //making a new user id who likes scifi and classics
    val newUserRatings = Array(
        Rating(0,260,5), //Star Wars
        Rating(0,329,5), //Star Trek Gens
        Rating(0,1356,4), //Star Trek First Cont
        Rating(0,904,5), //Rear Wind
        Rating(0,908,4), //North by NW
        Rating(0,2657,1) //Rocky Picture Show
    )
    
    val newUserRatingsDS = spark.sparkContext.parallelize(newUserRatings).toDS()
    
    //Add user into ratings to train ALS algorithm
    val allRatings = ratings.union(newUserRatingsDS)
    
    //Train ALS recommender model
    val model = als.fit(allRatings)
    
    //Build dataset of movies fictitious user has not seen
    //that have been rated more than 25 times
    val moviesIveSeen = newUserRatings.map(x => x.movieId)
    val unratedMovies = ratings.filter(x => !(moviesIveSeen contains x.movieId))
    
    val myUnratedMovies = unratedMovies.map(x => Rating(0, x.movieId, 0)).distinct()
    
    val myUnratedMoviesWithCounts = myUnratedMovies.join(ratingCounts, "movieId")
    
    val myPopularUnratedMovies = myUnratedMoviesWithCounts.filter(myUnratedMoviesWithCounts("count") > 25)
    
    //Predict ratings on each movie
    val predictions = model.transform(myPopularUnratedMovies)
    
    //Print ratings for the user with movie titles
    println("\nRatings for fictitious user or user ID 0:")
    for (rating <- newUserRatings) {
      println(nameDict(rating.movieId) + ": " + rating.rating)
    }
    
    //taking 10 movies with highest rating predictions and print them
    println("\nTop 10 recommended movies:")
    for (recommendation <- predictions.orderBy(desc("prediction")).take(10)) {
      println( nameDict(recommendation.getAs[Int]("movieId"))
          + " score " + recommendation.getAs[String]("prediction"))
    }
    
    //Stop SPARK session
    spark.stop()
  } 
}
