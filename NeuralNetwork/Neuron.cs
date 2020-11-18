using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public enum NeuronType
    {
        Input,
        Hidden,
        Output
    }

    public class Neuron
    {
        public List<float> Weights { get; }
        public List<float> Inputs { get; }
        public NeuronType NeuronType { get; }
        public float Output { get; private set; }
        public float Delta { get; private set; }

        public Neuron(int inputCount, NeuronType neuronType = NeuronType.Hidden)
        {
            NeuronType = neuronType;
            Weights = new List<float>();
            Inputs = new List<float>();

            InitWeightsRandomValues(inputCount);
        }

        public void InitWeightsRandomValues(int inputCount)
        {
            var rand = new Random();

            for (var i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add((float)rand.NextDouble());
                }

                Inputs.Add(0.0f);
            }
        }

        public float FeedForward(List<float> inputs)
        {
            for (var i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }
            
            float sum = default;

            for (var i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }

            return Output = Sigmoid(sum);
        }

        private float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));

        private float SigmoidDx(float x)
        {
            var sigmoid = Sigmoid(x);
            return sigmoid / (1.0f - sigmoid);
        }

        public void Learn(float error, float learningRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);

            for (var i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
            }
        }

        public override string ToString() => Output.ToString();
    }
}
