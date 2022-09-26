using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OpenVinoSharpPPTinyPose
{
    internal class PicoDet
    {
        private Core predictor = null;
        private string input_node_name = "image";
        private string output_node_name_1 = "concat_8.tmp_0";
        private string output_node_name_2 = "transpose_8.tmp_0";
        private float scale_factor;

        public PicoDet(string mode_path, string device_name) {
            predictor = new Core(mode_path, device_name);
        }


        public void predict(Mat image, Size size) 
        {
            // 设置图片输入
            // 配置图片数据            
            // 将图片放在矩形背景下
            // 图片数据解码
            byte[] input_image_data = image.ImEncode(".bmp");
            // 数据长度
            ulong input_image_length = Convert.ToUInt64(input_image_data.Length);
            // 设置图片输入
            predictor.load_input_data(input_node_name, input_image_data, input_image_length, 0);

            // 模型推理
            predictor.infer();

            // 读取模型输出
            // 2125 765
            float[] results2 = predictor.read_infer_result<float>(output_node_name_2, 765);
            int index = 0;
            float confindence = max_index<float>(results2, ref index);
            Console.WriteLine(confindence);
            Console.WriteLine(index);
            float[] results = predictor.read_infer_result<float>(output_node_name_1, 4*765);
            for (int i = 0; i < 4; i++)
            {
                Console.WriteLine(results[index*4+i]);
            }
            Mat out_image = new Mat();
            Cv2.Resize(image, out_image, new Size(192, 192));
            Cv2.Rectangle(out_image, new Point(results[index * 4 + 0], results[index * 4 + 1]),
                new Point( results[index * 4 + 2], + results[index * 4 + 3]),
                new Scalar(0, 0, 255),2);
            Cv2.ImShow("out_image", out_image);
            Cv2.WaitKey(0);
        }


        private T max_index<T>(T[] data, ref int index) where T : IComparable<T>
        {
            var max = data[0];
            index = 0;
            for (int l = 0; l < data.Length; l++) {
                var temp = data[l];
                if (temp.CompareTo(max) > 0)
                {
                    max = data[l];
                    index = l;
                }
            
            }
            return max;
        }

        private Mat process_resule(Mat image, float[] result)
        {
            Mat result_image = image.Clone();

            Mat result_data = new Mat(25200, 85, MatType.CV_32F, result);

            // 存放结果list
            List<Rect> position_boxes = new List<Rect>();
            List<int> class_ids = new List<int>();
            List<float> confidences = new List<float>();
            // 预处理输出结果
            for (int i = 0; i < result_data.Rows; i++)
            {
                // 获取置信值
                float confidence = result_data.At<float>(i, 4);
                if (confidence < 0.2)
                {
                    continue;
                }
                Console.WriteLine(confidence);

                Mat classes_scores = result_data.Row(i).ColRange(5, 85);//GetArray(i, 5, classes_scores);
                Point max_classId_point, min_classId_point;
                double max_score, min_score;
                // 获取一组数据中最大值及其位置
                Cv2.MinMaxLoc(classes_scores, out min_score, out max_score,
                    out min_classId_point, out max_classId_point);
                // 置信度 0～1之间
                // 获取识别框信息
                if (max_score > 0.25)
                {
                    float cx = result_data.At<float>(i, 0);
                    float cy = result_data.At<float>(i, 1);
                    float ow = result_data.At<float>(i, 2);
                    float oh = result_data.At<float>(i, 3);
                    int x = (int)((cx - 0.5 * ow) * scale_factor);
                    int y = (int)((cy - 0.5 * oh) * scale_factor);
                    int width = (int)(ow * scale_factor);
                    int height = (int)(oh * scale_factor);
                    Rect box = new Rect();
                    box.X = x;
                    box.Y = y;
                    box.Width = width;
                    box.Height = height;

                    position_boxes.Add(box);
                    class_ids.Add(max_classId_point.X);
                    confidences.Add((float)max_score);
                }
            }

            // NMS非极大值抑制
            int[] indexes = new int[position_boxes.Count];
            CvDnn.NMSBoxes(position_boxes, confidences, 0.25f, 0.45f, out indexes);
            // 将识别结果绘制到图片上
            for (int i = 0; i < indexes.Length; i++)
            {
                int index = indexes[i];
                int idx = class_ids[index];
                Cv2.Rectangle(result_image, position_boxes[index], new Scalar(0, 0, 255), 2, LineTypes.Link8);
                Cv2.Rectangle(result_image, new Point(position_boxes[index].TopLeft.X, position_boxes[index].TopLeft.Y - 20),
                    new Point(position_boxes[index].BottomRight.X, position_boxes[index].TopLeft.Y), new Scalar(0, 255, 255), -1);
                
            }

            Cv2.ImShow("C# + TensorRT + Yolov5 推理结果", result_image);
            Cv2.WaitKey();
            result_data.Dispose();
            return result_image;

        }
    }
}
