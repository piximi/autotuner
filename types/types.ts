import * as math from 'mathjs';
import * as tensorflow from '@tensorflow/tfjs';

export type Model = tensorflow.Sequential;

export type DataPoint = {data: tensorflow.Tensor<tensorflow.Rank>; lables: number}

export type BaysianOptimisationStep = { nextPoint: number, acquisitionFunctionValue: number}

export enum LossFunction {
    'absoluteDifference',
    'cosineDistance',
    'hingeLoss',
    'huberLoss',
    'logLoss',
    'meanSquaredError',
    'sigmoidCrossEntropy',
    'softmaxCrossEntropy',
    'categoricalCrossentropy',
}

export type StringModelParameter = { [identifier: string]: number[]};

export type SequentialModelParameters = { lossfunction: LossFunction[], optimizerAlgorithm: any[], batchSize: number[], epochs: number[] };

export type StringParameters = { [parameterIdentifier: string]: number[] };

export type StringModelParameters = { [modelIdentifier: string]: StringParameters};

export type ModelMapping = { [identifier: string] : StringParameters};

export type Domain = { [identifier: string] : StringParameters};
    
export type ModelsDomain = { [identifier: string] : number[]};

export type ModelDict = { [identifier: string] : Model};

export type DomainPointValue = { error: number, accuracy: number }

export type ObservedValues = { [identifier: string]: number[] }