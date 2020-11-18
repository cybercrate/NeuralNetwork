using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        public Topology Topology { get; }
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public Neuron FeedForward(params float[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        public float Learn(float[] expected, float[,] inputs, int epoch)
        {
            var signals = Normalize(inputs);

            var error = 0.0f;
            for (var i = 0; i < epoch; i++)
            {
                for (var j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(signals, j);

                    error += Backpropagation(output, input);
                }
            }

            return error / epoch;
        }

        public static float[] GetRow(float[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new float[columns];

            for (var i = 0; i < columns; ++i)
            {
                array[i] = matrix[row, i];
            }

            return array;
        }

        private float[,] Scalling(float[,] inputs)
        {
            var result = new float[inputs.GetLength(0), inputs.GetLength(1)];

            for (var column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];

                for (var row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, column];

                    if (item < min)
                    {
                        min = item;
                    }
                    else if (item > max)
                    {
                        max = item;
                    }
                }

                var divider = max - min;
                for (var row = 1; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - min) / divider;
                }
            }

            return result;
        }

        private float[,] Normalize(float[,] inputs)
        {
            var result = new float[inputs.GetLength(0), inputs.GetLength(1)];

            for (var column = 0; column < inputs.GetLength(1); column++)
            {
                // Average neuron signal.
                var sum = 0.0f;

                for (var row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }

                var average = sum / inputs.GetLength(0);

                // Neuron standard deviation.
                var error = 0.0f;

                for (var row = 0; row < inputs.GetLength(0); row++)
                {
                    error += MathF.Pow((inputs[row, column] - average), 2);
                }

                var standardError = MathF.Sqrt(error / inputs.GetLength(0));

                for (var row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standardError;
                }
            }

            return result;
        }

        private float Backpropagation(float expected, params float[] inputs)
        {
            var actualValue = FeedForward(inputs).Output;
            var difference = actualValue - expected;

            foreach (Neuron neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }

            for (var i = Layers.Count - 2; i >= 0; i--)
            {
                var layer = Layers[i];
                var previousLayer = Layers[i + 1];

                for (var j = 0; j < layer.NeuronCount; j++)
                {
                    var neuron = layer.Neurons[j];

                    for (var k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;

                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }

            return difference * difference;
        }

        public void FeedForwardAllLayersAfterInput()
        {
            for (var i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();

                foreach (Neuron neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        public void SendSignalsToInputNeurons(params float[] inputSignals)
        {
            for (var i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<float>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();

            for (var i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }

            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();

            for (var i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }

            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateHiddenLayers()
        {
            for (var i = 0; i < Topology.HiddenLayers.Count; i++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();

                for (var j = 0; j < Topology.HiddenLayers[i]; j++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }

                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }
    }
}
