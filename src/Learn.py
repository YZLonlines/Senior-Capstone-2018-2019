from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import FloatType

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LinearSVCModel

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
import yaml

# change to be reflective of your environment
data_dir = '/app/TestData'
model_dir = '/app/TestData/models/'

# change to match your environment
output_dir = data_dir + "/merge_data"


def linearSVC(df, feature_list=['BFSIZE', 'HDRSIZE', 'NODETYPE'], maxIter=100, regParam=0.0, threshold=0.0, overwrite_model = False):
    # Checks if there is a SparkContext running if so grab that if not start a new one
    # sc = SparkContext.getOrCreate()
    # sqlContext = SQLContext(sc)
    # sqlContext.setLogLevel('INFO')
    feature_list.sort()
    feature_name = '_'.join(feature_list)
    param_name = '_'.join([str(regParam), str(threshold), str(maxIter)])
    model_path_name = model_dir + 'LinearSVC/' + feature_name + '_' + param_name
    model = None

    vector_assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    df_temp = vector_assembler.transform(df)

    df = df_temp.select(['label', 'features'])

    trainingData, testData = df.randomSplit([0.7, 0.3])

    if os.path.isdir(model_path_name) and not overwrite_model:
        print('Loading model from ' + model_path_name)
        model = LinearSVCModel.load(model_path_name)

    else:
        lsvc = LinearSVC(maxIter=maxIter, regParam=regParam, threshold=threshold)
        model = lsvc.fit(trainingData)

    print('Making predictions on validation data')
    predictions = model.transform(testData)
    evaluator = BinaryClassificationEvaluator()

    evaluator.setMetricName('areaUnderROC')
    print('Evaluating areaUnderROC')
    auc = evaluator.evaluate(predictions)

    evaluator.setMetricName('areaUnderPR')
    print('Evaluating areaUnderPR')
    areaUnderPR = evaluator.evaluate(predictions)


    # test distribution of outputs
    total = df.select('label').count()
    disk = df.filter(df.label == 0).count()
    cloud = df.filter(df.label == 1).count()

    # print outputs
    print('Linear SVC')
    print(feature_list)
    print('Data distribution')
    print('Total Observations {}'.format(total))
    print(' Cloud %{}'.format((cloud/total) * 100))
    print(' Disk %{}'.format((disk/total) * 100))

    print(" Test AUC = {}\n".format(auc * 100))

    print('Error distribution')
    misses = predictions.filter(predictions.label != predictions.prediction)
    # now get percentage of error
    disk_misses = misses.filter(misses.label == 0).count()
    cloud_misses = misses.filter(misses.label == 1).count()

    disk_pred = predictions.filter(predictions.label == 0).count()
    cloud_pred = predictions.filter(predictions.label == 1).count()

    print(' Cloud Misses %{}'.format((cloud_misses/cloud_pred) * 100))
    print(' Disk Misses %{}'.format((disk_misses/disk_pred) * 100))

    if auc > 0.70:
        if os.path.isdir(model_path_name):
            if overwrite_model:
                print('Saving model to ' + model_path_name)
                model.write().overwrite().save(model_path_name)
            else:
                pass
        else:
            print('Saving model to ' + model_path_name)
            model.save(model_path_name)

    metrics = { 'data' : {  'total' : total,
                            'cloud' : (cloud/total) * 100,
                            'disk' :  (disk/total) * 100 },
                'metrics' : {   'Area Under ROC curve' : auc * 100,
                                'Area Under PR curve' : areaUnderPR * 100 },
                'error_percentage' : {  'cloud' : cloud_misses/cloud_pred * 100,
                                        'disk' :  disk_misses/disk_pred * 100 },
                'params' : { 'Regularization Parameter' : regParam,
                             'Maximum Iteration' : maxIter,
                             'Threshold' : threshold },
                'name' : 'Linear SVC',
                'features' : feature_list
            }

    with open('tmp/temp3.yml', 'w') as outfile:
        yaml.dump(metrics, outfile)

    return metrics, model


def multinomialRegression(df, feature_list=['BFSIZE', 'HDRSIZE', 'NODETYPE'], maxIter = 100, regParam = 0.0, elasticNetParam = 0.0, threshold = 0.5, overwrite_model = False):
    # Checks if there is a SparkContext running if so grab that if not start a new one
    # sc = SparkContext.getOrCreate()
    # sqlContext = SQLContext(sc)
    # sqlContext.setLogLevel('INFO')
    feature_list.sort()
    feature_name = '_'.join(feature_list)
    param_name = '_'.join([str(regParam), str(elasticNetParam), str(maxIter), str(threshold)])
    model_path_name = model_dir + 'MultinomialRegression/' + feature_name + '_' + param_name
    model = None

    vector_assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    df_temp = vector_assembler.transform(df)

    df = df_temp.select(['label', 'features'])

    trainingData, testData = df.randomSplit([0.7, 0.3])

    if os.path.isdir(model_path_name) and not overwrite_model:
        print('Loading model from ' + model_path_name)
        model = LogisticRegressionModel.load(model_path_name)

    else:
        lr = LogisticRegression(labelCol="label", maxIter= maxIter, regParam=regParam, elasticNetParam=elasticNetParam)
        model = lr.fit(trainingData)

    print('Making predictions on validation data')
    predictions = model.transform(testData)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

    evaluator.setMetricName('accuracy')
    print('Evaluating accuracy')
    accuracy = evaluator.evaluate(predictions)

    evaluator.setMetricName('f1')
    print('Evaluating f1')
    f1 = evaluator.evaluate(predictions)

    evaluator.setMetricName('weightedPrecision')
    print('Evaluating weightedPrecision')
    weightedPrecision = evaluator.evaluate(predictions)

    evaluator.setMetricName('weightedRecall')
    print('Evaluating weightedRecall')
    weightedRecall = evaluator.evaluate(predictions)

    print('accuracy {}'.format(accuracy))
    print('f1 {}'.format(f1))
    print('weightedPrecision {}'.format(weightedPrecision))
    print('weightedRecall {}'.format(weightedRecall))

    # test distribution of outputs
    total = df.select('label').count()
    tape = df.filter(df.label == 0).count()
    disk = df.filter(df.label == 1).count()
    cloud = df.filter(df.label == 2).count()

    # print outputs
    print('Multinomial Regression Classification')
    print(feature_list)
    print('Data distribution')
    print('Total Observations {}'.format(total))
    print(' Cloud %{}'.format((cloud/total) * 100))
    print(' Disk %{}'.format((disk/total) * 100))
    print(' Tape %{}\n'.format((tape/total) * 100))

    print(" Test Error = {}".format((1.0 - accuracy) * 100))
    print(" Test Accuracy = {}\n".format(accuracy * 100))

    print('Error distribution')
    misses = predictions.filter(predictions.label != predictions.prediction)
    # now get percentage of error
    tape_misses = misses.filter(misses.label == 0).count()
    disk_misses = misses.filter(misses.label == 1).count()
    cloud_misses = misses.filter(misses.label == 2).count()

    tape_pred = predictions.filter(predictions.label == 0).count()
    disk_pred = predictions.filter(predictions.label == 1).count()
    cloud_pred = predictions.filter(predictions.label == 2).count()

    print(' Cloud Misses %{}'.format((cloud_misses/cloud_pred) * 100))
    print(' Disk Misses %{}'.format((disk_misses/disk_pred) * 100))
    print(' Tape Misses %{}'.format((tape_misses/tape_pred) * 100))

    if accuracy > 0.80:
        if os.path.isdir(model_path_name):
            if overwrite_model:
                print('Saving model to ' + model_path_name)
                model.write().overwrite().save(model_path_name)
            else:
                pass
        else:
            print('Saving model to ' + model_path_name)
            model.save(model_path_name)

    metrics = { 'data' : {  'Total' : total,
                            'Cloud' : (cloud/total) * 100,
                            'Disk' :  (disk/total) * 100,
                            'Tape' :  (tape/total) * 100 },
                'metrics' : {   'Accuracy' : accuracy * 100,
                                'f1' : f1 * 100,
                                'Weighted Precision' : weightedPrecision * 100,
                                'Weighted Recall' : weightedRecall * 100},
                'error_percentage' : {  'Cloud' : cloud_misses/cloud_pred * 100,
                                        'Disk' :  disk_misses/disk_pred * 100,
                                        'Tape' :  tape_misses/tape_pred * 100 },
                'params' : { 'Regularization Parameter' : regParam,
                             'Maximum Iteration' : maxIter,
                             'ElasticNet Mixing Parameter' : elasticNetParam,
                             'Threshold' : threshold },
                'name' : 'Multinomial Regression Classification',
                'features' : feature_list
            }

    with open('tmp/temp2.yml', 'w') as outfile:
        yaml.dump(metrics, outfile)

    return metrics, model



def randomForest(df, feature_list=['BFSIZE', 'HDRSIZE', 'NODETYPE'], maxDepth = 5, numTrees = 20, seed=None, overwrite_model = False):
    # Checks if there is a SparkContext running if so grab that if not start a new one
    # sc = SparkContext.getOrCreate()
    # sqlContext = SQLContext(sc)
    # sqlContext.setLogLevel('INFO')
    feature_list.sort()
    feature_name = '_'.join(feature_list)
    param_name = '_'.join([str(maxDepth), str(numTrees)])
    model_path_name = model_dir + 'RandomForest/' + feature_name + '_' + param_name
    model = None

    vector_assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    df_temp = vector_assembler.transform(df)

    df = df_temp.select(['label', 'features'])

    trainingData, testData = df.randomSplit([0.7, 0.3])

    if os.path.isdir(model_path_name) and not overwrite_model:
        print('Loading model from ' + model_path_name)
        model = RandomForestClassificationModel.load(model_path_name)

    else:
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees= numTrees, maxDepth = maxDepth, seed = seed)
        model = rf.fit(trainingData)

    print('Making predictions on validation data')
    predictions = model.transform(testData)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    # f1|weightedPrecision|weightedRecall|accuracy
    evaluator.setMetricName('accuracy')
    print('Evaluating accuracy')
    accuracy = evaluator.evaluate(predictions)

    evaluator.setMetricName('f1')
    print('Evaluating f1')
    f1 = evaluator.evaluate(predictions)

    evaluator.setMetricName('weightedPrecision')
    print('Evaluating weightedPrecision')
    weightedPrecision = evaluator.evaluate(predictions)

    evaluator.setMetricName('weightedRecall')
    print('Evaluating weightedRecall')
    weightedRecall = evaluator.evaluate(predictions)

    print('accuracy {}'.format(accuracy))
    print('f1 {}'.format(f1))
    print('weightedPrecision {}'.format(weightedPrecision))
    print('weightedRecall {}'.format(weightedRecall))

    # test distribution of outputs
    total = df.select('label').count()
    tape = df.filter(df.label == 0).count()
    disk = df.filter(df.label == 1).count()
    cloud = df.filter(df.label == 2).count()

    # print outputs
    print('Random Forests')
    print(feature_list)
    print('Data distribution')
    print('Total Observations {}'.format(total))
    print(' Cloud %{}'.format((cloud/total) * 100))
    print(' Disk %{}'.format((disk/total) * 100))
    print(' Tape %{}\n'.format((tape/total) * 100))

    print(" Test Error = {}".format((1.0 - accuracy) * 100))
    print(" Test Accuracy = {}\n".format(accuracy * 100))

    print('Error distribution')
    misses = predictions.filter(predictions.label != predictions.prediction)
    # now get percentage of error
    tape_misses = misses.filter(misses.label == 0).count()
    disk_misses = misses.filter(misses.label == 1).count()
    cloud_misses = misses.filter(misses.label == 2).count()

    tape_pred = predictions.filter(predictions.label == 0).count()
    disk_pred = predictions.filter(predictions.label == 1).count()
    cloud_pred = predictions.filter(predictions.label == 2).count()


    print(' Cloud Misses %{}'.format((cloud_misses/cloud_pred) * 100))
    print(' Disk Misses %{}'.format((disk_misses/disk_pred) * 100))
    print(' Tape Misses %{}'.format((tape_misses/tape_pred) * 100))

    if accuracy > 0.80:
        if os.path.isdir(model_path_name):
            if overwrite_model:
                print('Saving model to ' + model_path_name)
                model.write().overwrite().save(model_path_name)
            else:
                pass
        else:
            print('Saving model to ' + model_path_name)
            model.save(model_path_name)

    metrics = { 'data' : {  'Total' : total,
                            'Cloud' : (cloud/total) * 100,
                            'Disk' :  (disk/total) * 100,
                            'Tape' :  (tape/total) * 100 },
                'metrics' : {   'Accuracy' : accuracy * 100,
                                'f1' : f1 * 100,
                                'Weighted Precision' : weightedPrecision * 100,
                                'Weighted Recall' : weightedRecall * 100},
                'error_percentage' : {  'Cloud' : cloud_misses/cloud_pred * 100,
                                        'Disk' :  disk_misses/disk_pred * 100,
                                        'Tape' :  tape_misses/tape_pred * 100 },
                'params' : { 'Number of Trees' : model.getNumTrees,
                             'Maximum Depth' : maxDepth },
                'model_debug' : model.toDebugString,
                'name' : 'Random Forest Model',
                'features' : feature_list
            }

    with open('tmp/temp.yml', 'w') as outfile:
        yaml.dump(metrics, outfile)

    return metrics, model


def gradientBoosting(df, feature_list=['BFSIZE', 'HDRSIZE', 'NODETYPE'], maxIter=20, stepSize=0.1, maxDepth=5, overwrite_model = False):
    # Checks if there is a SparkContext running if so grab that if not start a new one
    # sc = SparkContext.getOrCreate()
    # sqlContext = SQLContext(sc)
    # sqlContext.setLogLevel('INFO')
    feature_list.sort()
    feature_name = '_'.join(feature_list)
    param_name = '_'.join([str(maxDepth), str(stepSize), str(maxIter)])
    model_path_name = model_dir + 'GradientBoosting/' + feature_name + '_' + param_name
    model = None

    vector_assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
    df_temp = vector_assembler.transform(df)

    df = df_temp.select(['label', 'features'])

    trainingData, testData = df.randomSplit([0.7, 0.3])

    if os.path.isdir(model_path_name) and not overwrite_model:
        print('Loading model from ' + model_path_name)
        model = GBTClassificationModel.load(model_path_name)

    else:
        gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=maxIter, stepSize=stepSize, maxDepth=maxDepth)
        model = gbt.fit(trainingData)

    print('Making predictions on validation data')
    predictions = model.transform(testData)
    evaluator = BinaryClassificationEvaluator()

    evaluator.setMetricName('areaUnderROC')
    print('Evaluating areaUnderROC')
    auc = evaluator.evaluate(predictions)

    evaluator.setMetricName('areaUnderPR')
    print('Evaluating areaUnderPR')
    areaUnderPR = evaluator.evaluate(predictions)


    # test distribution of outputs
    total = df.select('label').count()
    disk = df.filter(df.label == 0).count()
    cloud = df.filter(df.label == 1).count()

    # print outputs
    print('Gradient-Boosted Tree')
    print(feature_list)
    print('Data distribution')
    print('Total Observations {}'.format(total))
    print(' Cloud %{}'.format((cloud/total) * 100))
    print(' Disk %{}'.format((disk/total) * 100))

    print(" Test AUC = {}\n".format(auc * 100))

    print('Error distribution')
    misses = predictions.filter(predictions.label != predictions.prediction)
    # now get percentage of error
    disk_misses = misses.filter(misses.label == 0).count()
    cloud_misses = misses.filter(misses.label == 1).count()

    disk_pred = predictions.filter(predictions.label == 0).count()
    cloud_pred = predictions.filter(predictions.label == 1).count()

    print(' Cloud Misses %{}'.format((cloud_misses/cloud_pred) * 100))
    print(' Disk Misses %{}'.format((disk_misses/disk_pred) * 100))

    if auc > 0.80:
        if os.path.isdir(model_path_name):
            if overwrite_model:
                print('Saving model to ' + model_path_name)
                model.write().overwrite().save(model_path_name)
            else:
                pass
        else:
            print('Saving model to ' + model_path_name)
            model.save(model_path_name)

    metrics = { 'data' : {  'total' : total,
                            'cloud' : (cloud/total) * 100,
                            'disk' :  (disk/total) * 100 },
                'metrics' : {   'Area Under ROC curve' : auc * 100,
                                'Area Under PR curve' : areaUnderPR * 100 },
                'error_percentage' : {  'cloud' : cloud_misses/cloud_pred * 100,
                                        'disk' :  disk_misses/disk_pred * 100 },
                'params' : { 'Number of Trees' : model.getNumTrees,
                             'Maximum Depth' : maxDepth,
                             'Maximum Number of Iterations' : maxIter,
                             'Step Size' : stepSize },
                'model_debug' : model.toDebugString,
                'name' : 'Gradient Boosted Model',
                'features' : feature_list
            }

    with open('tmp/temp1.yml', 'w') as outfile:
        yaml.dump(metrics, outfile)

    return metrics, model


def compare_algorithms():
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('ERROR')
    binary = [gradientBoosting, linearSVC]
    multiclass = [randomForest, multinomialRegression]
    binary_results = []
    class_results = []
    print('Binary Function Comparisons')
    for f in binary:
        res = f()
        print(res[1])
        binary_results.append(res)
    print('Multiclass Function Comparisons')
    for f in multiclass:
        res = f()
        print(res[1])
        class_results.append(res)

    binary_results.sort(reverse=True)
    class_results.sort(reverse=True)

    print('Binary Results in order:')
    for res, res_string in binary_results:
        print(res_string)

    print('Multiclass Results in order:')
    for res, res_string in class_results:
        print(res_string)

    sc.stop()

    return 0


def main():
    from Process import extract_features
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('ERROR')
    sqlContext = SQLContext(sc)

    feature_list = 'BFSIZE HDRSIZE NODETYPE NODESTATE METADATASIZE'.split()
    merged_data, merged_data_binary = extract_features(feature_list, binary = True, multiclass = True, overwrite = False)
    results = []

    # print('Start Random Forest')
    # results.append(randomForest(merged_data, feature_list = feature_list, maxDepth = 5, numTrees = 20, seed=None))
    print('Start GradientBoosting')
    results.append(gradientBoosting(merged_data_binary, feature_list = feature_list, maxIter=10, stepSize=0.3))

    print('Results:')
    for result in results:
        print(result)

    sc.stop()
    return 0


if __name__ == "__main__":
    main()
