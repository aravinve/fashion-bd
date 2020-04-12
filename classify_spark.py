# Note run below command to open the spark instances with sparkdl lib
# spark-submit --packages databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11 --executor-cores 3 --num-executors 24 --driver-memory 16g --executor-memory 16g .\classify_spark.py


from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from pyspark.sql import SQLContext
from pyspark import SparkContext,SparkConf
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from functools import reduce
import os

img_dir = ".\\fashion_spark"
dataframes = []
index = 0
for dirs in os.listdir(img_dir):
    print(os.path.join(img_dir, dirs))
    imageSchema = ImageSchema.readImages(os.path.join(img_dir, dirs)).withColumn("label", lit(index))
    dataframes.append(imageSchema)
    index += 1

print(dataframes)
print(len(dataframes))

# merge data frame
df = reduce(lambda first, second: first.union(second), dataframes)
# repartition dataframe 
df = df.repartition(200)
# split the data-frame
train_df, test_df = df.randomSplit([0.8, 0.2], 42)

print(train_df)
print(test_df)

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
p_model = p.fit(train_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

df = p_model.transform(test_df)
df.show()

predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))