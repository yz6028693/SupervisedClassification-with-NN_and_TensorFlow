import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import append
from matplotlib import colors, animation
from random import shuffle, randint
from numba import jit
from math import pi, sin
import os


class ArchimedeanSpiralMeetTensorFlow(object):

    def __init__(self):

        # Variables listed here is kind of like global variables which you can use them in other parts of the script.

        self.ModelOutputPath = r'/Users/yuanzhen/Desktop/TensorFlow/DemoModelOutput' # Neural Network output weights and bias, please change it to your folder
        self.imAnimate = self.RasterList()
        self.Losses = self.LossList()
        self.fig, self.axes = plt.subplots(ncols=2, figsize=(15, 7))  # Output plot size
        self.ani = animation.FuncAnimation(self.fig, self.updatefig, interval=300, repeat=False, blit=True)
        self.iteration0, self.trainloss0, self.testloss0 = np.zeros(1), np.ones(1), np.ones(1)

    # Function to transfer polar coordinate system to Cartesian coordinate system. （regular coordinate system）
    def pol2cart(self, rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, -y)

    # Function to change number of category to onehot numpy array. (eg. we have 4 class and class 1 can change
    # to [1, 0, 0, 0])
    def OneHot(self, Class, length):
        OneHot = [0] * length
        OneHot[Class - 1] = 1
        return np.asarray(OneHot)

    # Function to normalize input array value to the range between 0 to 1.
    def Normalize(self, Array):
        min = float(np.amin(Array))
        max = float(np.amax(Array))
        if min == max:
            return Array
        else:
            return (Array - min)/(max - min)

    # Initialize neural network weight, this is why every time the training process looks different, because we randomly
    #  initialize it.
    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name = name)

    # Initialize neural network bias (constant array), just like the weight above, this is why every time the training
    # process looks different, because we randomly initialize it.
    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name = name)

    # Randomly create small batches of inputs for training process.
    def BatchCreater(self, BatchNum, TrainArray, LabelArray):
        TrainBatch = []
        LabelBatch = []
        for _ in range(BatchNum):
            num = randint(0, len(TrainArray) - 1)
            TrainBatch.append(TrainArray[num])
            LabelBatch.append(LabelArray[num])
        return np.asarray(TrainBatch), np.asarray(LabelBatch)

    # Regroup the data structure for ANN training.
    def ReGroupForTraining(self, ArrayList):
        ReGroupForTrainingArray = []
        index = list(range(len(ArrayList[0])))
        for i in index:
            ReGroupForTrainingArray.append([array[i] for array in ArrayList])
        return ReGroupForTrainingArray

    # Separate previous two two groups one for training, the other for testing our training result.
    def SeparateTrainingTesting(self, Array, TrainingRatio):
        RandomlizeArray = []
        index = list(range(len(Array)))
        shuffle(index)
        for i in index:
            RandomlizeArray.append(Array[i])
        TrainingArray = RandomlizeArray[: int(len(RandomlizeArray) * TrainingRatio)]
        TestingArray =  RandomlizeArray[int(len(RandomlizeArray) * TrainingRatio):]

        TrainingInput = [i[: -1] for i in TrainingArray]
        TrainingLabel = [self.OneHot(i[-1], 2) for i in TrainingArray]
        TestingInput = [i[: -1] for i in TestingArray]
        TestingLabel = [self.OneHot(i[-1], 2) for i in TestingArray]
        return TrainingInput, TrainingLabel, TestingInput, TestingLabel


    # jit module in numba is to compile python script to C like script for better performance and running speed.
    @jit
    def outputArray(self, ClassificationArray):
        OutputArray = np.zeros((600, 600), dtype=np.int)
        for n in range(600):
            for m in range(600):
                classification = int(ClassificationArray[600 * n + m]) + 1
                OutputArray[n][m] = classification
        return OutputArray


    # Classify raster background image to numpy array for showing in the training animation.
    def ShowPlot(self, NormalizedRasterArray, W_1, W_2, b_1, b_2, sess):
        W1 = W_1
        b1 = b_1
        W2 = W_2
        b2 = b_2
        #scale2 = tf.Variable(tf.ones([2]))
        #shift2 = tf.Variable(tf.zeros([2]))
        x = tf.placeholder(tf.float32, shape=[None, 7])
        #sess.run(tf.global_variables_initializer())
        y1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
        y2 = tf.matmul(y1, W2) + b2
        #mean2, var2 = tf.nn.moments(y2, [0])
        #y2_BN = tf.nn.batch_normalization(y2, mean2, var2, shift2, scale2, 0.001)
        y3 = tf.nn.softmax(y2)
        classification = tf.argmax(y3, 1)
        ClassificationArray = sess.run(classification, feed_dict={x: NormalizedRasterArray})
        OutputArray = self.outputArray(ClassificationArray)
        return OutputArray
        # scipy.misc.toimage(OutputArray, cmin=0, cmax=255).save(r'/Users/yuanzhen/Desktop/TensorFlow/try2.tif')


    # Apply Tensorflow to build a one hidden layer ANN to train our inputed points. (The example here used 6 hidden
    # nodes and 15 as the batch size and our stop point is 99% testing accuracy.)
    def TensorFlowTraingWithAccuracy(self, NormalizedRasterArray, TrainArray, TrainLabelArray, TestArray,
                                     TestLabelArray, HiddenLayerNodes, BatchSize,  AccuracyRequirement):
        sess = tf.InteractiveSession()
        x = tf.placeholder(tf.float32, shape=[None, 7])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])
        #keep_prob = tf.placeholder(tf.float32)
        W1 = self.weight_variable([7, HiddenLayerNodes], 'W1')
        b1 = self.bias_variable([HiddenLayerNodes], 'b1')
        W2 = self.weight_variable([HiddenLayerNodes, 2], 'W2')
        b2 = self.bias_variable([2], 'b2')
        #scale2 = tf.Variable(tf.ones([2]))
        #shift2 = tf.Variable(tf.zeros([2]))
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        y1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
        #y1_drop = tf.nn.dropout(y1, keep_prob)
        y2 = tf.matmul(y1, W2) + b2
        #mean2, var2 = tf.nn.moments(y2, [0])
        #y2_BN = tf.nn.batch_normalization(y2, mean2, var2, shift2, scale2, 0.001)
        y3 = tf.nn.softmax(y2)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y2))
        mean_squared_error = tf.losses.mean_squared_error(y_, y3)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y3, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        n = 0
        result = 0
        ims = []
        testingloss = []
        trainingloss = []

        while result < AccuracyRequirement:
            n += 1
            if n % 300 == 0:
                result = accuracy.eval(feed_dict={x: TestArray, y_: TestLabelArray})  # HERE
                result1 = accuracy.eval(feed_dict={x: TrainArray, y_: TrainLabelArray})
                W_1 = W1.eval()
                W_2 = W2.eval()
                b_1 = b1.eval()
                b_2 = b2.eval()
                OutputArray = self.ShowPlot(NormalizedRasterArray, W_1, W_2, b_1, b_2, sess)
                ims.append(OutputArray)
                testingloss.append(1.0 - result)
                trainingloss.append(1.0 - result1)
                print("Testing Accracy: ", result)
                print("Training Accracy: ", result1)
                print("Training iteration:", n)
            TrainBatch, LabelBatch = self.BatchCreater(BatchSize, TrainArray, TrainLabelArray)
            train_step.run(feed_dict={x: TrainBatch, y_: LabelBatch})
        save_path = saver.save(sess, os.path.join(self.ModelOutputPath, 'OutputModel.ckpt'))
        print("Model weights saved in file: %s" % save_path)
        print("Final Testing Accracy: ", result)
        print("Final Training Accracy: ", result1)
        return ims, testingloss, trainingloss

    # Function to update plots in our output animation.
    def updatefig(self, *args):
        im0 = next(self.imAnimate)
        iteration_0, trainloss_0, testloss_0 = next(self.Losses)
        self.iteration0 = append(self.iteration0, iteration_0)
        self.trainloss0 = append(self.trainloss0, trainloss_0)
        self.testloss0 = append(self.testloss0, testloss_0)
        self.scat1.set_offsets(self.data1)
        self.scat2.set_offsets(self.data2)
        self.im.set_array(im0)
        self.traininglossline.set_data(self.iteration0, self.trainloss0)
        self.testinglossline.set_data(self.iteration0, self.testloss0)
        return [self.im, self.scat1, self.scat2, self.traininglossline, self.testinglossline, ]

    # Output raster plot size is 600 X 600 pixels.
    def PlotingRasterArray(self):
        NormalizedRasterArray = []
        Range = 600.0
        for y in list(range(-300, 300)):
            for x in list(range(-300, 300)):
                NormalizedRasterArray.append(
                    [(x + 300.0) / Range, (-y + 300.0) / Range, ((x + 300.0) / Range) ** 2, ((-y + 300.0) / Range) ** 2,
                     ((x + 300.0) * (-y + 300.0)) / (Range ** 2), (sin(x / 50.0) + 1) / 2, (sin(-y / 50.0) + 1) / 2])

        return NormalizedRasterArray

    def ForScatteroffset(self, x, y):
        data = []
        for i in range(len(x)):
            data.append([x[i], y[i]])
        return data

    def RasterList(self):
        for im in self.ims:
            yield im

    def LossList(self):
        for i in range(len(self.testingloss)):
            yield (i + 1) * 300, self.trainingloss[i], self.testingloss[i]

    # The workflow to train our network and to show the output animation.
    def InputforTraining(self, accuracy_needed, TrainingRatio):
        phi = [2 * i * (2 * pi / 360) for i in range(270)]
        rho1 = np.multiply(phi, 0.5)
        rho2 = np.multiply(phi, -0.5)
        x_1, y_1 = self.pol2cart(rho1, phi)
        x_2, y_2 = self.pol2cart(rho2, phi)
        Label1 = np.ones((len(phi)), dtype=np.int)
        Label2 = np.multiply(Label1, 2)

        InputClass1 = [(x_1 + 6) / 12.0, (y_1 + 6) / 12.0, ((x_1 + 6) / 12.0) ** 2, ((y_1 + 6) / 12.0) ** 2,
                       ((x_1 + 6) * (y_1 + 6)) / 144.0, (np.sin(x_1) + 1) / 2.0, (np.sin(y_1) + 1) / 2.0, Label1]
        InputClass2 = [(x_2 + 6) / 12.0, (y_2 + 6) / 12.0, ((x_2 + 6) / 12.0) ** 2, ((y_2 + 6) / 12.0) ** 2,
                       ((x_2 + 6) * (y_2 + 6)) / 144.0, (np.sin(x_2) + 1) / 2.0, (np.sin(y_2) + 1) / 2.0, Label2]

        ReGroupForTrainingArray1 = self.ReGroupForTraining(InputClass1)
        ReGroupForTrainingArray2 = self.ReGroupForTraining(InputClass2)
        NormalizedRasterArray = self.PlotingRasterArray()

        TrainingInput, TrainingLabel, TestingInput, TestingLabel = \
            self.SeparateTrainingTesting(ReGroupForTrainingArray1 + ReGroupForTrainingArray2, TrainingRatio)

        iteration0 = 0
        trainloss0 = 1
        testloss0 = 1
        self.ims, self.testingloss, self.trainingloss = self.TensorFlowTraingWithAccuracy(NormalizedRasterArray, TrainingInput,
                                                                      TrainingLabel, TestingInput, TestingLabel, 6, 15,
                                                                                          accuracy_needed)
        ax1 = self.axes[0]
        ax2 = self.axes[1]
        ax1.set_title('ANN Classification Result')
        ax2.set_title('Loss Plot')
        ax2.grid(True)
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Loss")
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, 300 * (len(self.testingloss) + 1))
        self.traininglossline, = ax2.plot(iteration0, trainloss0, 'b-', label="trainingloss")
        self.testinglossline, = ax2.plot(iteration0, testloss0, 'r-', label="testingloss")
        ax2.legend([self.traininglossline, self.testinglossline], [self.traininglossline.get_label(), self.testinglossline.get_label()])
        cmap = colors.ListedColormap(['blue', 'orange'])
        self.data1 = self.ForScatteroffset(x_1, y_1)
        self.data2 = self.ForScatteroffset(x_2, y_2)
        self.im = ax1.imshow(self.ims[0], extent=[-6, 6, -6, 6], cmap=cmap, animated=True)
        self.scat1 = ax1.scatter([], [], s=10, c='b', edgecolors='white', linewidth='0.5', animated=True, zorder=10)
        self.scat2 = ax1.scatter([], [], s=10, c='coral', edgecolors='white', linewidth='0.5', animated=True, zorder=10)
        plt.show()

if __name__ == '__main__':
    tool = ArchimedeanSpiralMeetTensorFlow()
    # Accuracy Requirement: 99%, Training and testing data radio: 8 to 2.
    tool.InputforTraining(0.99, 0.8)








