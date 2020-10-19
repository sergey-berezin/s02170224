using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Linq;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.IO;
using System.IO.Enumeration;
using System.Threading;

namespace RecognitionLibrary
{
    public class PredictionResult
    {
        public string Path;
        public string Label;
        public double Confidence;
        public PredictionResult(string path, string label, double confidence)
        {
            this.Path = path;
            this.Label = label;
            this.Confidence = confidence;
        }
    };
    public class Model
    {
        private static ManualResetEvent _stopSignal = new ManualResetEvent(false);
        private string _img_path;
        private InferenceSession session;
        private ConcurrentQueue<string> filenames;


        public delegate void PredictionHandler(PredictionResult Result);
        public event PredictionHandler ResultEvent;

        public delegate void ErrorHandler(string errMessage);
        public event ErrorHandler ErrMessage;

        public delegate void InfoHandler(string infoMessage);
        public event InfoHandler InfoMessage;

        public delegate void Output(string msg);

        
        public Model(
            string model_path = "./../../../../resnet50-v2-7.onnx",
            string img_path = "./../../../../dog.jpeg"
            )
        {
            this._img_path = img_path;
            this.session = new InferenceSession(model_path);
        }
        
        private DenseTensor<float> ImageToTensor(string single_img_path)
        {
            using var image = Image.Load<Rgb24>(single_img_path);
            const int TargetWidth = 224;
            const int TargetHeight = 224;

            // Изменяем размер картинки до 224 x 224
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(TargetWidth, TargetHeight),
                    Mode = ResizeMode.Crop // Сохраняем пропорции обрезая лишнее
                });
            });

            // Перевод пикселов в тензор и нормализация
            var input = new DenseTensor<float>(new[] { 1, 3, TargetHeight, TargetWidth });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < TargetHeight; y++)
            {
                Span<Rgb24> pixelSpan = image.GetPixelRowSpan(y);
                for (int x = 0; x < TargetWidth; x++)
                {
                    input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                    input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                    input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                }
            }
            return input;
        }
        private PredictionResult Predict(DenseTensor<float> input, string single_image_path)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("data", input)
            };
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Получаем 1000 выходов и считаем для них softmax
            var output = results.First().AsEnumerable<float>().ToArray();
            var sum = output.Sum(x => (float)Math.Exp(x));
            var softmax = output.Select(x => (float)Math.Exp(x) / sum);

            var confidence = softmax.Max();
            var class_idx = softmax.ToList().IndexOf(confidence);


            return new PredictionResult(single_image_path, LabelMap.classLabels[class_idx], confidence);
        }
        public void Stop() => _stopSignal.Set();
        private void worker()
        {
            string name;
            while (filenames.TryDequeue(out name))
            {
                if (_stopSignal.WaitOne(0))
                {
                    ErrMessage?.Invoke("Stopping Thread by signal");
                    return;
                }
                ResultEvent?.Invoke(Predict(ImageToTensor(name), name));
               
            }
            InfoMessage?.Invoke("Stopping thread normally");
        }
        public void Work()
        {
            try
            {
                filenames = new ConcurrentQueue<string>(Directory.GetFiles(_img_path, "*.jpg"));
            }
            catch (DirectoryNotFoundException exc)
            {
                ErrMessage?.Invoke("Directory doesn't exist!");
                return;
            }

            var max_proc_count = Environment.ProcessorCount;
            Thread[] threads = new Thread[max_proc_count];
            for (int i = 0; i < max_proc_count; ++i)
            {
                InfoMessage?.Invoke("Statring thread");
                threads[i] = new Thread(worker);
                threads[i].Start();
            }

            for (int i = 0; i < max_proc_count; ++i)
            {
                threads[i].Join();
                
            }
            InfoMessage?.Invoke("Done!");
        }
    }
}
