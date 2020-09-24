# Databricks notebook source
#Camille Anderson
#Assignment 2

from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType, TimestampType
from pyspark.ml.feature import StringIndexer, Bucketizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler


# COMMAND ----------

#dbutils.fs.mkdirs("FileStore/tables/police_stop")
#dbutils.fs.cp("/FileStore/tables/policeStop_Train.csv", "FileStore/tables/police_stop")
#dbutils.fs.cp("/FileStore/tables/policeStop_Test.csv", "FileStore/tables/police_stop")

# COMMAND ----------

#data frame schema
policeStopSchema = StructType( \
 [StructField('OBJECTID', LongType(), True), \
 StructField('problem', StringType(), True), \
 StructField('personSearch', StringType(), True), \
 StructField('vehicleSearch', StringType(), True), \
 StructField('preRace', StringType(), True), \
 StructField('race', StringType(), True), \
 StructField('gender', StringType(), True), \
 StructField('policePrecinct', StringType(), True), \
 ])

#training dataset
policeStopTrain = spark.read.format("csv").option("header", True).schema(policeStopSchema).option("ignoreLeadingWhiteSpace", True).load("FileStore/tables/police_stop/policeStop_Train.csv")

# COMMAND ----------

policeStopTrain.show()

# COMMAND ----------

#create indexers for the pipeline
probIndexer = StringIndexer(inputCol="problem", outputCol="problemIndex")
raceIndexer = StringIndexer(inputCol="race", outputCol="raceIndex")
genderIndexer = StringIndexer(inputCol="gender", outputCol="genderIndex")
precinctIndexer = StringIndexer(inputCol="policePrecinct", outputCol="precinctIndex")
personSearchIndexer = StringIndexer(inputCol="personSearch", outputCol="label")

#logistic regression
lr = LogisticRegression(maxIter=10, regParam=0.01)

# COMMAND ----------

#construct the pipeline
vecAssem = VectorAssembler(inputCols=['problemIndex', 'raceIndex', 'genderIndex', 'precinctIndex'], outputCol='features')
myStages=[probIndexer, raceIndexer, genderIndexer, precinctIndexer, personSearchIndexer, vecAssem, lr]
p = Pipeline(stages=myStages)
pModel = p.fit(policeStopTrain)


# COMMAND ----------

#dbutils.fs.mkdirs("FileStore/tables/police_stop_test")

#testing dataset
policeStopTest = spark.read.format("csv").option("header", True).schema(policeStopSchema).option("ignoreLeadingWhiteSpace", True).load("FileStore/tables/police_stop/policeStop_Test.csv")
policeStopTest.show(20)


# COMMAND ----------

#repartition the data
policeStopTest = policeStopTest.repartition(10)
policeStopTest.persist()
policeStopTest.show()

# COMMAND ----------

dbutils.fs.rm("FileStore/tables/police_stop_test/", True)

#write out the repartitioned data so that we have multiple files to read in for streaming
policeStopTest.write.format("csv").option("header", True).save("FileStore/tables/police_stop_test/")

# COMMAND ----------

#stream in the files
sourceStream = spark.readStream.format("csv").option("header", True).schema(policeStopSchema).option("maxFilesPerTrigger", 1).load("dbfs:///FileStore/tables/police_stop_test/")
#transform with our logistic regression model
pred = pModel.transform(sourceStream)

# COMMAND ----------

#begin streaming
sinkStream = pred.writeStream.format("memory").queryName("police_stop_stream").start()

# COMMAND ----------

current = spark.sql("SELECT * FROM police_stop_stream")
current.select('problem', 'race', 'gender', 'policePrecinct', 'probability', 'prediction').orderBy("probability", ascending=False).show(current.count(), False)

# COMMAND ----------

#results sorted by highest probability
current.select('problem', 'race', 'gender', 'policePrecinct', 'probability', 'prediction').orderBy("probability", ascending=False).show(current.count(), False)
