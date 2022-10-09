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
        // 成员变量
        private Core predictor; // 模型推理器
        private string input_node_name = "image"; // 模型输入节点名称
        private string output_node_name_1 = "concat_8.tmp_0"; // 模型预测框输出节点名
        private string output_node_name_2 = "transpose_8.tmp_0"; // 模型预测置信值输出节点
        private Size input_size = new Size(0,0); // 模型输入节点形状
        private int output_length = 0; // 模型输出数据长度

        public PicoDet(string mode_path, string device_name) {
            predictor = new Core(mode_path, device_name);
        }

        /// <summary>
        /// 设置模型输入输出形状
        /// </summary>
        /// <param name="input_image_size">输入形状</param>
        /// <param name="output_length">输出尺寸</param>
        public void set_shape(Size input_image_size, int output_length) 
        { 
            this.input_size = input_image_size;
            this.output_length = output_length;
        }


        public List<Rect> predict(Mat image)
        {
            // 设置图片输入
            // 图片数据解码
            byte[] input_image_data = image.ImEncode(".bmp");
            // 数据长度
            ulong input_image_length = Convert.ToUInt64(input_image_data.Length);
            // 设置图片输入
            predictor.load_input_data(input_node_name, input_image_data, input_image_length, 0);

            // 求取缩放大小
            double scale_x = (double)image.Width / (double)this.input_size.Width;
            double scale_y = (double)image.Height / (double)this.input_size.Height;
            Point2d scale_factor = new Point2d(scale_x, scale_y);
            // 模型推理
            predictor.infer();

            // 读取模型输出
            // 2125 765
            // 读取置信值结果
            float[] results_con = predictor.read_infer_result<float>(output_node_name_2, output_length);
            // 读取预测框
            float[] result_box = predictor.read_infer_result<float>(output_node_name_1, 4 * output_length);
            // 处理模型推理数据
            List<Rect> boxes_result = process_result(results_con, result_box, scale_factor);
            return boxes_result;
        }

        private List<Rect> process_result(float[] results_con, float[] result_box, Point2d scale_factor)
        {

            // 处理预测结果
            List<float> confidences = new List<float>();
            List<Rect> boxes = new List<Rect>();
            for (int c = 0; c < output_length; c++)
            {
                Rect rect = new Rect((int)(result_box[4 * c] * scale_factor.X), (int)(result_box[4 * c + 1] * scale_factor.Y),
                    (int)((result_box[4 * c + 2] - result_box[4 * c]) * scale_factor.X),
                    (int)((result_box[4 * c + 3] - result_box[4 * c + 1]) * scale_factor.Y));
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
