// Databricks notebook source
val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("/FileStore/shared_uploads/salma.romdhani@etudiant-isi.utm.tn/DailyDelhiClimateTrain.csv")

df.show(5)
df.printSchema()

// COMMAND ----------

import org.apache.spark.sql.functions._

val dfClean = df
  .na.drop()
  .withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

dfClean.select("date", "meantemp").show(5)



// COMMAND ----------



val tempParAnnee = spark.sql("""
  SELECT year(date) AS annee, AVG(meantemp) AS temp_moyenne
  FROM climat
  GROUP BY annee
  ORDER BY annee
""")

display(tempParAnnee)


// COMMAND ----------

dfClean.filter(year(col("date")) === 2017).select("date", "meantemp").show(100)



// COMMAND ----------

dfClean.filter(year(col("date")) === 2017).count()


// COMMAND ----------

val dfSans2017 = dfClean.filter(year(col("date")) =!= 2017)


// COMMAND ----------

dfSans2017.createOrReplaceTempView("climat_sans_2017")

val tempParAnnee2 = spark.sql("""
  SELECT year(date) AS annee, AVG(meantemp) AS temp_moyenne
  FROM climat_sans_2017
  GROUP BY annee
  ORDER BY annee
""")

display(tempParAnnee2)
tempParAnnee2.createOrReplaceTempView("tempParAnnee2")



// COMMAND ----------

// MAGIC %python
// MAGIC import matplotlib.pyplot as plt
// MAGIC
// MAGIC df_temp_par_annee = spark.table("tempParAnnee2")
// MAGIC
// MAGIC df_pd = df_temp_par_annee.toPandas()
// MAGIC
// MAGIC
// MAGIC
// MAGIC plt.figure(figsize=(10,6))
// MAGIC bars = plt.bar(df_pd["annee"], df_pd["temp_moyenne"], color='skyblue')
// MAGIC
// MAGIC plt.xlabel("Année")
// MAGIC plt.ylabel("Température moyenne")
// MAGIC plt.title("Température moyenne par année (sans 2017)")
// MAGIC
// MAGIC plt.xticks(rotation=45)
// MAGIC plt.grid(axis='y')
// MAGIC
// MAGIC for bar in bars:
// MAGIC     hauteur = bar.get_height()
// MAGIC     plt.text(bar.get_x() + bar.get_width()/2, hauteur, f'{hauteur:.2f}', 
// MAGIC              ha='center', va='bottom')
// MAGIC
// MAGIC plt.show()
// MAGIC
// MAGIC

// COMMAND ----------

val stats = dfSans2017.agg(
  avg("meantemp").as("moyenne"),
  stddev("meantemp").as("ecart_type")
).first()

val moyenne = stats.getAs[Double]("moyenne")
val ecartType = stats.getAs[Double]("ecart_type")

val seuilHaut = moyenne + 2 * ecartType
val seuilBas = moyenne - 2 * ecartType
println(s"Moyenne = $moyenne")
println(s"Ecart type = $ecartType")
println(s"Seuil haut = $seuilHaut")
println(s"Seuil bas = $seuilBas")

val anomalies = dfClean.filter(col("meantemp") > seuilHaut || col("meantemp") < seuilBas)
anomalies.show(20)
dfClean.createOrReplaceTempView("dfCleanTable")


// COMMAND ----------

// MAGIC %python
// MAGIC import matplotlib.pyplot as plt
// MAGIC import matplotlib.dates as mdates
// MAGIC
// MAGIC
// MAGIC df_clean = spark.table("dfCleanTable")
// MAGIC
// MAGIC df_clean_pd = df_clean.toPandas()
// MAGIC
// MAGIC moyenne = df_clean_pd["meantemp"].mean()
// MAGIC ecart_type = df_clean_pd["meantemp"].std()
// MAGIC
// MAGIC seuilHaut = moyenne + 2 * ecart_type
// MAGIC seuilBas = moyenne - 2 * ecart_type
// MAGIC
// MAGIC anomalies = df_clean_pd[(df_clean_pd["meantemp"] > seuilHaut) | (df_clean_pd["meantemp"] < seuilBas)]
// MAGIC normales = df_clean_pd[~((df_clean_pd["meantemp"] > seuilHaut) | (df_clean_pd["meantemp"] < seuilBas))]
// MAGIC
// MAGIC plt.figure(figsize=(14,6))
// MAGIC
// MAGIC plt.scatter(normales["date"], normales["meantemp"], c='blue', label='Normal')
// MAGIC plt.scatter(anomalies["date"], anomalies["meantemp"], c='red', label='Anomalie')
// MAGIC
// MAGIC plt.legend()
// MAGIC
// MAGIC plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  
// MAGIC plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
// MAGIC
// MAGIC plt.xticks(rotation=45) 
// MAGIC
// MAGIC plt.tight_layout()  
// MAGIC
// MAGIC plt.show()

// COMMAND ----------

val corr = dfSans2017.stat.corr("meantemp", "humidity")
println(s"Corrélation température/humidité : $corr")


// COMMAND ----------

// MAGIC %python
// MAGIC import seaborn as sns
// MAGIC import matplotlib.pyplot as plt
// MAGIC
// MAGIC
// MAGIC
// MAGIC cols = ["meantemp", "humidity", "wind_speed", "meanpressure"]
// MAGIC corr_matrix = df_clean_pd[cols].corr()
// MAGIC
// MAGIC plt.figure(figsize=(8,6))
// MAGIC sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
// MAGIC plt.title("Matrice de corrélation des variables météo")
// MAGIC plt.show()
// MAGIC

// COMMAND ----------

// MAGIC %python
// MAGIC import numpy as np
// MAGIC
// MAGIC plt.figure(figsize=(8,6))
// MAGIC
// MAGIC plt.scatter(df_clean_pd["meantemp"], df_clean_pd["humidity"], alpha=0.5, label="Données")
// MAGIC
// MAGIC coeffs = np.polyfit(df_clean_pd["meantemp"], df_clean_pd["humidity"], deg=1)
// MAGIC poly_eqn = np.poly1d(coeffs)
// MAGIC x_vals = np.linspace(df_clean_pd["meantemp"].min(), df_clean_pd["meantemp"].max(), 100)
// MAGIC y_vals = poly_eqn(x_vals)
// MAGIC
// MAGIC plt.plot(x_vals, y_vals, color='red', label=f"Ligne de tendance: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}")
// MAGIC
// MAGIC plt.xlabel("Température moyenne")
// MAGIC plt.ylabel("Humidité")
// MAGIC plt.title("Nuage de points température vs humidité avec ligne de tendance")
// MAGIC plt.legend()
// MAGIC plt.grid(True)
// MAGIC plt.show()
// MAGIC

// COMMAND ----------

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler

val featuresCols = Array("meantemp", "humidity", "wind_speed", "meanpressure")
val assembler = new VectorAssembler().setInputCols(featuresCols).setOutputCol("features")
val dfVec = assembler.transform(dfSans2017.na.drop())

val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(dfVec)

val clustered = model.transform(dfVec)
clustered.groupBy("prediction").count().show()


// COMMAND ----------

import org.apache.spark.ml.feature.StandardScaler

val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithMean(true)
  .setWithStd(true)

val scalerModel = scaler.fit(dfVec)
val dfScaled = scalerModel.transform(dfVec)
dfScaled.createOrReplaceTempView("dfScaled")


// COMMAND ----------

val kmeans = new KMeans().setK(3).setSeed(1L).setFeaturesCol("scaledFeatures")
val model = kmeans.fit(dfScaled)
val clustered = model.transform(dfScaled)
clustered.createOrReplaceTempView("clustered")
clustered.groupBy("prediction").count().show()


// COMMAND ----------

val centers = model.clusterCenters
centers.zipWithIndex.foreach { case (center, idx) =>
  println(s"Cluster $idx : ${center}")
}


// COMMAND ----------

// MAGIC %python
// MAGIC clustered = spark.table("clustered")
// MAGIC clustered_pd = clustered.select("meantemp", "humidity", "wind_speed", "meanpressure", "prediction").toPandas()
// MAGIC
// MAGIC import matplotlib.pyplot as plt
// MAGIC import seaborn as sns
// MAGIC
// MAGIC plt.figure(figsize=(10, 6))
// MAGIC sns.scatterplot(data=clustered_pd, x="meantemp", y="humidity", hue="prediction", palette="Set1")
// MAGIC plt.title("Clustering météo (température vs humidité)")
// MAGIC plt.xlabel("Température moyenne")
// MAGIC plt.ylabel("Humidité")
// MAGIC plt.legend(title="Cluster")
// MAGIC plt.show()
// MAGIC

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.col

val dataFiltered = tempParAnnee.filter("temp_moyenne > 15")

val assembler = new VectorAssembler()
  .setInputCols(Array("annee"))
  .setOutputCol("features")

val dataML = assembler.transform(dataFiltered)
  .withColumnRenamed("temp_moyenne", "label")
  .select("features", "label")


dataML.show()



// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression()
val model = lr.fit(dataML)


// COMMAND ----------

import spark.implicits._

val predictionData = Seq(2017).toDF("annee")
val predictionFeatures = assembler.transform(predictionData)

val prediction = model.transform(predictionFeatures)
prediction.select("annee", "prediction").show()
dataFiltered.createOrReplaceTempView("data_filtered")
prediction.select("annee", "prediction").createOrReplaceTempView("prediction_2017")



// COMMAND ----------

println(s"Coefficient: ${model.coefficients} Intercept: ${model.intercept}")


// COMMAND ----------

// MAGIC %python
// MAGIC import matplotlib.pyplot as plt
// MAGIC import seaborn as sns
// MAGIC import pandas as pd
// MAGIC
// MAGIC df = spark.table("data_filtered").toPandas()
// MAGIC pred = spark.table("prediction_2017").toPandas()
// MAGIC
// MAGIC plt.figure(figsize=(10, 6))
// MAGIC sns.regplot(data=df, x="annee", y="temp_moyenne", label="Données historiques", scatter_kws={"color": "blue"}, line_kws={"color": "green"})
// MAGIC
// MAGIC plt.scatter(pred["annee"], pred["prediction"], color='red', label="Prédiction 2017", s=100, marker='X')
// MAGIC
// MAGIC plt.text(pred["annee"].values[0], pred["prediction"].values[0], f'{pred["prediction"].values[0]:.2f}°C', color='red', fontsize=12, ha='left')
// MAGIC
// MAGIC plt.title("Régression Linéaire : Température moyenne par année")
// MAGIC plt.xlabel("Année")
// MAGIC plt.ylabel("Température Moyenne (°C)")
// MAGIC plt.legend()
// MAGIC plt.grid(True)
// MAGIC plt.tight_layout()
// MAGIC plt.show()
// MAGIC