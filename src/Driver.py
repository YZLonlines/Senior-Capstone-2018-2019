import Process
import Learn

from pyspark import SparkContext
from pyspark.sql import SparkSession

import click

@click.command()
@click.argument('features')
@click.argument('algorithm', type=click.Choice(['RandomForest', 'GradientBoosting', 'MultinomialRegression', 'LinearSVC']))
@click.option('--seed', '-s', default=None, type=int)
@click.option('--maxIter',  default=20)
@click.option('--stepSize', default=0.1)
@click.option('--numTrees', default=20)
@click.option('--maxDepth', default=5)
@click.option('--regParam', default=0.0)
@click.option('--elasticNetParam', default=0.0)
@click.option('--threshold', default=0.5)
@click.option('--overwrite/--no-overwrite', default=False)
@click.option('--overwrite_model/--no-overwrite_model', default=False)
def main(features, algorithm, seed, maxiter, stepsize, numtrees, maxdepth, regparam, elasticnetparam, threshold, overwrite, overwrite_model):
    sc = SparkContext.getOrCreate()
    sc.setLogLevel('ERROR')
    feature_list = features.split()
    if algorithm == 'RandomForest':
        # python Driver.py 'BFSIZE HDRSIZE NODETYPE NODESTATE METADATASIZE' RandomForest
        # numTrees = 20
        df, _ = Process.extract_features(feature_list, binary = False, multiclass = True, overwrite = overwrite)
        Learn.randomForest(df, feature_list, maxDepth=maxdepth, numTrees = numtrees, seed=seed, overwrite_model = overwrite_model)
    elif algorithm == 'GradientBoosting':
        _, df = Process.extract_features(feature_list, binary = True, multiclass = False, overwrite = overwrite)
        Learn.gradientBoosting(df, feature_list, maxIter=maxiter, stepSize=stepsize, overwrite_model = overwrite_model)
    elif algorithm == 'MultinomialRegression':
        # python Driver.py 'BFSIZE HDRSIZE NODETYPE NODESTATE METADATASIZE' MultinomialRegression --maxIter 10 --regParam 0.3 --elasticNetParam 0.8 --threshold 0.5
        # without metadatasize
        df, _ = Process.extract_features(feature_list, binary = False, multiclass = True, overwrite = overwrite)
        Learn.multinomialRegression(df, feature_list, maxIter=maxiter, regParam=regparam, elasticNetParam=elasticnetparam, threshold=threshold, overwrite_model = overwrite_model)
    elif algorithm == 'LinearSVC':
        # python Driver.py 'BFSIZE HDRSIZE NODETYPE NODESTATE METADATASIZE' LinearSVC --maxIter 10 --regParam 0.3 --threshold 0.5
        _, df = Process.extract_features(feature_list, binary = True, multiclass = False, overwrite = overwrite)
        Learn.linearSVC(df, feature_list, maxIter=maxiter, regParam=regparam, threshold=threshold, overwrite_model = overwrite_model)

    sc.stop()
if __name__ == "__main__":
    main()
