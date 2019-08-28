import * as tensorflow from '@tensorflow/tfjs';
import * as math from 'mathjs';
import { DataPoint, DomainPointValue } from '../types/types';

export class ModelEvaluater{
    trainData: DataPoint[] = [];
    validationData: DataPoint[] = [];
    testData: DataPoint[] = [];
    numberOfCategories: number;

    constructor(dataSet: DataPoint[], numberOfCategories: number, validationSetRatio: number, testSetRatio: number) {
        this.numberOfCategories = numberOfCategories;

        // shuffle the dataset
        tensorflow.util.shuffle(dataSet);

        // create validation dataset
        const numSamplesValidation = Math.max(
            1,
            Math.round(dataSet.length * validationSetRatio)
        );
        this.validationData = dataSet.splice(0, numSamplesValidation);

        // create test dataset
        const numSamplesTest = Math.max(
            1,
            Math.round(dataSet.length * testSetRatio)
        );
        this.testData = dataSet.splice(0, numSamplesTest);
        this.trainData = dataSet;
    }

    ConcatenateTensorData(data: DataPoint[]) {
        const trainData: tensorflow.Tensor<tensorflow.Rank>[] = [];
        const trainLables: number[] = [];
        for (let i = 0; i < data.length; i++) {
            trainData.push(data[i].data);
            trainLables.push(data[i].lables);
        }
        
        let concatenatedTensorData = tensorflow.tidy(() => tensorflow.concat(trainData));
        let concatenatedLables = tensorflow.tidy(() => tensorflow.oneHot(trainLables, this.numberOfCategories));
        return { concatenatedTensorData, concatenatedLables };
    }

    EvaluateSequentialTensorflowModel = async (model: tensorflow.Sequential, args: any): Promise<DomainPointValue> => {
        var trainData = this.ConcatenateTensorData(this.trainData);
        await model.fit(trainData.concatenatedTensorData, trainData.concatenatedLables, args);

        var validationData = this.ConcatenateTensorData(this.validationData);
        const evaluationResult = model.evaluate(validationData.concatenatedTensorData, validationData.concatenatedLables) as tensorflow.Tensor[];

        const error = evaluationResult[0].dataSync()[0];
        const score = evaluationResult[1].dataSync()[0];
        return {error: error, metricScores: [score]} as DomainPointValue;
    }

    EvaluateSequentialTensorflowModelCV = async (model: tensorflow.Sequential, args: any): Promise<DomainPointValue> => {
        const dataSet = this.trainData.concat(this.validationData);
        const dataSize = dataSet.length;
        const k = math.min(10, math.floor(math.nthRoot(dataSize) as number));
    
        const dataFolds: DataPoint[][] = Array.from(Array(math.ceil(dataSet.length/k)), (_,i) => dataSet.slice(i*k,i*k+k));
    
        var error = 0;
        var score = 0;
        for (let i = 0; i < k; i++) {
            var validationData = dataFolds[i];
            var trainData: DataPoint[] = [];
    
            for (var j = 0; j < k; j++) {
                if (j !== i) {
                    trainData = trainData.concat(dataFolds[j]);
                }
            }

            var concatenatedTrainData = this.ConcatenateTensorData(trainData);
            await model.fit(concatenatedTrainData.concatenatedTensorData, concatenatedTrainData.concatenatedLables, args);

            var concatenatedValidationData = this.ConcatenateTensorData(validationData);
            const evaluationResult = model.evaluate(concatenatedValidationData.concatenatedTensorData, concatenatedValidationData.concatenatedLables) as tensorflow.Tensor[];

            const foldError = evaluationResult[0].dataSync()[0];
            const foldScore = evaluationResult[1].dataSync()[0];
            error += foldError;
            score += foldScore;
        }
        return {error: error/dataSize, metricScores: [score/k]} as DomainPointValue;
    
    }

}