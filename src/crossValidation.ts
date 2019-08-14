import * as tensorflow from '@tensorflow/tfjs';
import * as math from 'mathjs';
import { datasetType } from '../types/types';


// evaluate a tensorflow model on a given data set using cross validation
const CVEvaluation = async (model: tensorflow.Sequential, dataset: datasetType, numberOfCategories: number, args: any): Promise<number> => {
    const dataSize = dataset.lables.length;
    const k = math.min(10, math.floor(math.nthRoot(dataSize) as number));

    const dataFolds = Array.from(Array(math.ceil(dataset.data.length/k)), (_,i) => dataset.data.slice(i*k,i*k+k));
    const lableFolds = Array.from(Array(math.ceil(dataset.lables.length/k)), (_,i) => dataset.lables.slice(i*k,i*k+k));

    var error = 0;
    for (let i = 0; i < k; i++) {
        var testData = dataFolds[i];
        var testLables = lableFolds[i];

        var trainData: tensorflow.Tensor<tensorflow.Rank>[] = [];
        var trainLables: number[] = [];

        for (var j = 0; j < k; j++) {
            if (j !== i) {
                trainData = trainData.concat(dataFolds[j]);
                trainLables = trainLables.concat(lableFolds[j]);
            }
        }

        let concatenatedTensorTrainData = tensorflow.tidy(() => tensorflow.concat(trainData));
        let concatenatedTrainLables = tensorflow.tidy(() => tensorflow.oneHot(trainLables, numberOfCategories));
        await model.fit(concatenatedTensorTrainData, concatenatedTrainLables, args);

        let concatenatedTensorTestData = tensorflow.tidy(() => tensorflow.concat(testData));
        let concatenatedTestLables = tensorflow.tidy(() => tensorflow.oneHot(testLables, numberOfCategories));
        const evaluationResult = model.evaluate(concatenatedTensorTestData, concatenatedTestLables) as tensorflow.Tensor[];

        error += evaluationResult[0].dataSync()[0];
    }
    return (error/dataSize);

}