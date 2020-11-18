using System.Collections.Generic;

namespace NeuralNetwork
{
    public class Layer
    {
        public List<Neuron> Neurons { get; }
        public int NeuronCount => Neurons?.Count ?? 0;
        public NeuronType NeuronType;

        public Layer(List<Neuron> neurons, NeuronType neuronType = NeuronType.Hidden)
        {
            // TODO: check all input neurons for type matching.

            Neurons = neurons;
            NeuronType = neuronType;
        }

        public List<float> GetSignals()
        {
            var result = new List<float>();

            foreach (Neuron neuron in Neurons)
            {
                result.Add(neuron.Output);
            }

            return result;
        }

        public override string ToString() => NeuronType.ToString();
    }
}
