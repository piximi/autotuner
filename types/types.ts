import * as math from 'mathjs';
import * as tensorflow from '@tensorflow/tfjs';

export type Model = tensorflow.Sequential;

export type CreateModelFunction = (
    classes: number,
    units: number,
    loss: string,
    metrics: string[],
    optimizer: tensorflow.Optimizer) => tensorflow.Sequential;

export type DataSet = {trainingSet: tensorflow.Tensor<tensorflow.Rank>, evaluationSet: tensorflow.Tensor<tensorflow.Rank>, testSet: tensorflow.Tensor<tensorflow.Rank>}

export type BaysianOptimisationStep = { nextPoint: number, expectedImprovement: number}

export type LossFunction =
    'absoluteDifference' |
    'cosineDistance' |
    'hingeLoss' |
    'huberLoss' |
    'logLoss' |
    'meanSquaredError' |
    'sigmoidCrossEntropy' |
    'softmaxCrossEntropy' |
    'categoricalCrossentropy';

export type StringModelParameter = { [identifier: string]: number[]};

export type SequentialModelParameters = { lossfunction: LossFunction[], optimizerAlgorith: tensorflow.Optimizer[], batchSize: number[], epochs: number[] };

type StringParameters = { [parameterIdentifier: string]: number[] };

export type StringModelParameters = { modelIdentifier: string, stringParameters: StringParameters};

export type ModelMapping = { [identifier: string] : StringModelParameters};

export type Domain = { [identifier: string] : StringParameters};
    
export type ModelsDomain = { [identifier: string] : number[]};

export type ModelDict = { [identifier: string] : Model};

export type NullableMatrix = math.Matrix | null;
