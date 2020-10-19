using RecognitionLibrary;
using System;
using System.Linq;


namespace ImageRecognition
{
    class Program

    {
        private static void OutResultFunc(PredictionResult predictionResult)
        {
            var res = String.Format("Path: {0}, Label: {1}, Confidence: {2}", predictionResult.Path, predictionResult.Label, predictionResult.Confidence);
            Console.WriteLine(res);
        }

        private static void OutErrFunc(string err_msg)
        {
            Console.WriteLine(err_msg);
        }
        private static void OutInfoFunc(string info_msg)
        {
            Console.WriteLine(info_msg);
        }

        static void Main(string[] args)
        {
            

            string imgPath = args.FirstOrDefault() ?? "./../../../../dataset/";
            string modelFilePath = "./../../../../resnet50-v2-7.onnx";

            Model model = new Model(modelFilePath, imgPath);

            model.ResultEvent += OutResultFunc;
            model.ErrMessage += OutErrFunc;
            model.InfoMessage += OutInfoFunc;

            Console.CancelKeyPress += (sender, eArgs) => {
                model.Stop();
                eArgs.Cancel = true;
            };

            model.Work();
        }
    }
}

