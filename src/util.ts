import * as math from 'mathjs';

/**
 * Retrieve the array key corresponding to the largest element in the array.
 *
 * @param {Array.<number>} array Input array
 * @return {number} Index of array element with largest value
 */
const argmax = (array: number[]) => {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

/**
 * Compute the expected improvement (EI) acquisition function.
 *
 * @param {number} bestObjective Best obtained objective value so far.
 * @param {math.Matrix} mean Mean values of the probability distribution over a given domain.
 * @param {math.Matrix} std Standard deviation values of the probability distribution over a given domain.
 * @return {number[]} Values of the expected improvement for all points of the mean and std.
 */
const expectedImprovement = (bestObjective: number, mean: math.Matrix, std: math.Matrix) => {
    const gamma = math.dotDivide(math.subtract(mean, bestObjective), std);
    // @ts-ignore
    const pdf: any = math.dotDivide(math.exp(math.dotDivide(math.square(gamma), -2)), math.sqrt(2 * 3.14159));
    // @ts-ignore
    const cdf = math.dotDivide(math.add(math.erf(math.dotDivide(gamma, math.sqrt(2))), 1), 2);

    const expectedImprovement = math.dotMultiply(std, math.add(math.dotMultiply(gamma, cdf), pdf)) as math.Matrix;
    return expectedImprovement.toArray() as number[];
}

/**
 * Compute the upper confidence bound (UCB) acquisition function.
 *
 * @param {math.Matrix} mean Mean values of the probability distribution over a given domain.
 * @param {math.Matrix} std Standard deviation values of the probability distribution over a given domain.
 * @return {number[]} Values of the upper confidence bound for all points of the mean and std.
 */
const upperConfidenceBound = (mean: math.Matrix, std: math.Matrix) => {
    const ucb = math.add(mean, math.dotMultiply(std, 0.75)) as math.Matrix;

    return ucb.toArray() as number[];
}

const equal = (a: number, b: string): boolean => {
    return (b === a.toString());
}

export { argmax, expectedImprovement, upperConfidenceBound, equal };