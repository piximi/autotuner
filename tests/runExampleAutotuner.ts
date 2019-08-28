import { TensorflowlModelAutotuner } from '../src/index';
import { createDataset } from './data/dataset';
import { LossFunction, DataPoint } from '../types/types';
import * as tensorflow from '@tensorflow/tfjs';
import { Classifier } from '@piximi/types';

const runExampleAutotuner = async () => {
        var fs = require("fs");
        var stringContent = fs.readFileSync("C:/Users/m_lev/Projects/BA/piximi/autotuner/tests/data/smallMNISTTest.piximi");
        var classifier = JSON.parse(stringContent) as Classifier;

        const dataset = await createDataset(classifier.categories, classifier.images);

        var autotuner = new TensorflowlModelAutotuner(['accuracy'], dataset.dataSet as DataPoint[], dataset.numberOfCategories);
        
        const testModel = await createModel();

        const parameters = { lossfunction: [LossFunction.categoricalCrossentropy], optimizerAlgorithm: [tensorflow.train.adadelta()], batchSize: [10], epochs: [5,10] };
        autotuner.addModel('testModel', testModel, parameters);

        autotuner.bayesianOptimization();
};

runExampleAutotuner();

const createModel = async () => {
    const model = tensorflow.sequential();
    model.add(tensorflow.layers.conv2d({inputShape: [28,28,3], kernelSize: 3, filters: 16, activation: 'relu'}));
    model.add(tensorflow.layers.maxPooling2d({poolSize: 2, strides: 2}));
    model.add(tensorflow.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
    model.add(tensorflow.layers.maxPooling2d({poolSize: 2, strides: 2}));
    model.add(tensorflow.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));
    model.add(tensorflow.layers.flatten());
    model.add(tensorflow.layers.dense({units: 64, activation: 'relu'}));
    model.add(tensorflow.layers.dense({units: 2, activation: 'softmax'}));
    return model;
}
