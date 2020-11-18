using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NeuralNetwork.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var outputs = new float[]
            {
                0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f
            };

            var inputs = new float[,]
            {
                // Result - the patient is sick    : 1
                //        - the patient is healthy : 0

                // T - Bad temperature
                // A - Good age
                // S - Smoking
                // F - Proper nutrition
                //   T     A     S     F
                { 0.0f, 0.0f, 0.0f, 0.0f },
                { 0.0f, 0.0f, 0.0f, 1.0f },
                { 0.0f, 0.0f, 1.0f, 0.0f },
                { 0.0f, 0.0f, 1.0f, 1.0f },
                { 0.0f, 1.0f, 0.0f, 0.0f },
                { 0.0f, 1.0f, 0.0f, 1.0f },
                { 0.0f, 1.0f, 1.0f, 0.0f },
                { 0.0f, 1.0f, 1.0f, 1.0f },
                { 1.0f, 0.0f, 0.0f, 0.0f },
                { 1.0f, 0.0f, 0.0f, 1.0f },
                { 1.0f, 0.0f, 1.0f, 0.0f },
                { 1.0f, 0.0f, 1.0f, 1.0f },
                { 1.0f, 1.0f, 0.0f, 0.0f },
                { 1.0f, 1.0f, 0.0f, 1.0f },
                { 1.0f, 1.0f, 1.0f, 0.0f },
                { 1.0f, 1.0f, 1.0f, 1.0f }
            };

            var topology = new Topology(4, 1, 0.1f, 2);
            var neuralNetwork = new NeuralNetwork(topology);
            var difference = neuralNetwork.Learn(outputs, inputs, 10000);

            var results = new List<float>();

            for (var i = 0; i < outputs.Length; i++)
            {
                var row = NeuralNetwork.GetRow(inputs, i);
                var res = neuralNetwork.FeedForward(row).Output;
                results.Add(res);
            }

            for (var i = 0; i < results.Count; i++)
            {
                var expected = MathF.Round(outputs[i], 2);
                var actual = MathF.Round(results[i], 2);

                Assert.AreEqual(expected, actual);
            }
        }

        [TestMethod()]
        public void DatasetTest()
        {
            var outputs = new List<float>();
            var inputs = new List<float[]>();

            using (var streamReader = new StreamReader(@"Resources\heart.data"))
            {
                var header = streamReader.ReadLine();

                while (!streamReader.EndOfStream)
                {
                    var row = streamReader.ReadLine();
                    var values = row.Split(',').Select(v => Convert.ToSingle(v.Replace('.', ','))).ToList();
                    var output = values.Last();
                    var input = values.Take(values.Count - 1).ToArray();

                    outputs.Add(output);
                    inputs.Add(input);
                }

                var inputSignals = new float[inputs.Count, inputs[0].Length];
                for (var i = 0; i < inputSignals.GetLength(0); i++)
                {
                    for (var j = 0; j < inputSignals.GetLength(1); j++)
                    {
                        inputSignals[i, j] = inputs[i][j];
                    }
                }

                var topology = new Topology(outputs.Count, 1, 1.0f, outputs.Count / 2);
                var neuralNetwork = new NeuralNetwork(topology);
                var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 10);

                var results = new List<float>();

                for (var i = 0; i < outputs.Count; i++)
                {
                    var res = neuralNetwork.FeedForward(inputs[i]).Output;
                    results.Add(res);
                }

                for (var i = 0; i < results.Count; i++)
                {
                    var expected = MathF.Round(outputs[i], 2);
                    var actual = MathF.Round(results[i], 2);

                    Assert.AreEqual(expected, actual);
                }
            }
        }
    }
}
