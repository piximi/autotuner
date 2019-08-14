import { Category, Image } from '@piximi/types';
import * as ImageJS from 'image-js';
import * as tensorflow from '@tensorflow/tfjs';

const VALIDATIONSET_RATIO = 0.2;
const TESTSET_RATIO = 0.2;

export const createTestDataset = async (
  categories: Category[],
  images: Image[]
) => {
  const trainData = images.filter(
    (image: Image) =>
      image.categoryIdentifier !== '00000000-0000-0000-0000-000000000000'
  );

  // shuffle the dataset
  tensorflow.util.shuffle(trainData);

  // create validation dataset
  const numSamplesValidation = Math.max(
    1,
    Math.round(trainData.length * VALIDATIONSET_RATIO)
  );
  const validationData = trainData.splice(0, numSamplesValidation);

  // create test dataset
  const numSamplesTest = Math.max(
    1,
    Math.round(trainData.length * TESTSET_RATIO)
  );
  const testData = trainData.splice(0, numSamplesTest);

  const validationDataSet = await createLabledTensorflowDataSet(
    testData,
    categories
  );
  const testDataSet = await createLabledTensorflowDataSet(
    validationData,
    categories
  );
  const trainDataSet = await createLabledTensorflowDataSet(
    trainData,
    categories
  );

  return { trainData: trainDataSet, testData: testDataSet, validationData: validationDataSet, numberOfCategories:  categories.length - 1};
};

const findCategoryIndex = (
  categories: Category[],
  identifier: string
): number => {
  return categories.findIndex(
    (category: Category) => category.identifier === identifier
  );
};

const tensorImageData = async (image: Image) => {
  const data = await ImageJS.Image.load(image.data);

  var tensorImage = tensorflow.browser
      .fromPixels(data.getCanvas())
      .toFloat()
      .sub(tensorflow.scalar(127.5))
      .div(tensorflow.scalar(127.5))

  return tensorImage.reshape([1, 28, 28, 3]);
};

const createLabledTensorflowDataSet = async (
  labledData: Image[],
  categories: Category[]
) => {
  let tensorData: tensorflow.Tensor<tensorflow.Rank>[] = [];
  let tensorLables: number[] = [];

  for (const image of labledData) {
    tensorData.push(await tensorImageData(image));
    tensorLables.push(
      findCategoryIndex(categories, image.categoryIdentifier) - 1
    );
  }

  //return { data: concatenatedTensorData, lables: concatenatedLableData };
  return { data: tensorData, lables: tensorLables };
};