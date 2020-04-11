# Note run below command to open the spark instances with sparkdl lib
# spark-submit --packages databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11 --executor-cores 3 --num-executors 24 --driver-memory 8g --executor-memory 8g .\classify_spark.py


from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from pyspark.sql import SQLContext
from pyspark import SparkContext,SparkConf
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from functools import reduce

# I tried to setup sprakconf with the package some errors are thrown
# config = SparkConf().setAll([('spark.executor.memory', '8g'), ('spark.executor.cores', '3'), ('spark.cores.max', '3'), ('spark.driver.memory','8g'), ('spark.jars.packages', 'databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11')])
# sc = SparkContext(conf=config)
# sqlContext = SQLContext(sc)

img_dir = "C:\\Users\\Lenovo\\bigdata\\Fashiondataset\\Apparel\\Topwear"

#Read images and Create training & test DataFrames for transfer learning
zero = ImageSchema.readImages(img_dir + "/Belts").withColumn("label", lit(0))
one = ImageSchema.readImages(img_dir + "/Blazers").withColumn("label", lit(1))
two = ImageSchema.readImages(img_dir + "/Dresses").withColumn("label", lit(2))
three = ImageSchema.readImages(img_dir + "/Dupatta").withColumn("label", lit(3))
four = ImageSchema.readImages(img_dir + "/Kurtas").withColumn("label", lit(4))
five = ImageSchema.readImages(img_dir + "/Kurtis").withColumn("label", lit(5))
six = ImageSchema.readImages(img_dir + "/Nehru Jackets").withColumn("label", lit(6))
seven = ImageSchema.readImages(img_dir + "/Rain Jacket").withColumn("label", lit(7))
eight = ImageSchema.readImages(img_dir + "/Shirts").withColumn("label", lit(8))
nine = ImageSchema.readImages(img_dir + "/Suits").withColumn("label", lit(9))
ten = ImageSchema.readImages(img_dir + "/Sweatshirts").withColumn("label", lit(10))
eleven = ImageSchema.readImages(img_dir + "/Tops").withColumn("label", lit(11))
twelve = ImageSchema.readImages(img_dir + "/Tshirts").withColumn("label", lit(12))
thirteen = ImageSchema.readImages(img_dir + "/Waistcoat").withColumn("label", lit(13))
# accessories_train, accessories_test = accessories_df.randomSplit([0.6, 0.4])
# apparel_train, apparel_test = apparel_df.randomSplit([0.6, 0.4])
# train_df = accessories_train.unionAll(apparel_train)
# test_df = accessories_test.unionAll(apparel_test)

dataframes = [zero, one, two, three,four,
             five, six, seven, eight, nine, ten,eleven,twelve, thirteen]

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