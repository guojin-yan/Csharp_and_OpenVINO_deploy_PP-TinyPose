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
        private Size input_size = new Size(0,0);
        private int output_length = 0;

        public PicoDet(string mode_path, string device_name) {
            predictor = new Core(mode_path, device_name);
        }

        public void set_shape(Size input_image_size, int output_length) 
        { 
            this.input_size = input_image_size;
            this.output_length = output_length;
        }


        public List<Rect> predict(Mat image)
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
            // 读取置信值结果
            float[] results_con = predictor.read_infer_result<float>(output_node_name_2, output_length);
            // 读取预测框
            float[] result_box = predictor.read_infer_result<float>(output_node_name_1, 4 * output_length);


            // 求取缩放大小
            double scale_x = (double)image.Width / (double)this.input_size.Width;
            double scale_y = (double)image.Height / (double)this.input_size.Height;

            // 处理预测结果
            List<float> confidences = new List<float>();
            List<Rect> boxes = new List<Rect>();
            for (int c = 0; c < output_length; c++)
            {
                Rect rect = new Rect((int)(result_box[4 * c] * scale_x), (int)(result_box[4 * c + 1] * scale_y),
                    (int)((result_box[4 * c + 2] - result_box[4 * c]) * scale_x),
                    (int)((result_box[4 * c + 3] - result_box[4 * c + 1]) * scale_y));
                boxes.Add(rect);
                confidences.Add(results_con[c]);
            }
            // 非极大值抑制获取结果候选框
            int[] indexes = new int[boxes.Count];
            CvDnn.NMSBoxes(boxes, confidences, 0.5f, 0.5f, out indexes);
            List<Rect> boxes_result = new List<Rect>();
            for (int i = 0; i < indexes.Length; i++) 
            {
                boxes_result.Add(boxes[indexes[i]]);
            }
            Console.WriteLine(indexes[0]);

            return boxes_result;
        }


    }
}
