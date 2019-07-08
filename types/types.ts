import * as math from 'mathjs';
import * as tensorflow from '@tensorflow/tfjs';

export type Model = tensorflow.Sequential;

export type BaysianOptimisationStep = { nextPoint: number, expectedImprovement: number}

export type ParameterNames = 'lossFunction' | 'optimizer' | 'batchSize' | 'epochs';

export type ParameterTypes = number | string | tensorflow.Optimizer;

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

type LossFunctionParameter = { 'lossFunction' : LossFunction[]};

type OptimizerParameter = { 'optimizer' : tensorflow.Optimizer[]};

type BatchSizeParameter = { 'batchSize' : number[]};

type EpochsParameter = { 'epochs' : number[]};

export type Parameter = LossFunctionParameter | OptimizerParameter | BatchSizeParameter | EpochsParameter;

export type ModelParameters = Parameter[];

export type ParameteSample = { loss: LossFunction, optimizer: tensorflow.Optimizer, batchSize: number, epochs: number};

export type ModelMapping = { [identifier: string] : ModelParameters};

export type Domain = { [identifier: number] : ParameteSample};
    
export type ModelsDomain = { [identifier: string] : number[]};

export type ModelDict = { [identifier: string] : Model};

export type NullableMatrix = math.Matrix | null;
