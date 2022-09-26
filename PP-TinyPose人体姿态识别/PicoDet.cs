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
        private Size input_image_size = new Size(0,0);
        private int output_length = 0;

        public PicoDet(string mode_path, string device_name) {
            predictor = new Core(mode_path, device_name);
        }

        public void set_shape(Size input_image_size, int output_length) 
        { 
            this.input_image_size = input_image_size;
            this.output_length = output_length;
        }


        public List<Mat> predict(Mat image)
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
            float[] results_con = predictor.read_infer_result<float>(output_node_name_2, 765);
            // 读取预测框
            float[] result_box = predictor.read_infer_result<float>(output_node_name_1, 4 * output_length);

            // 处理预测结果
            List<float> confidences = new List<float>();
            List<Rect> boxes = new List<Rect>();
            for (int c = 0; c < output_length; c++) 
            {
                Rect rect = new Rect((int)result_box[4 * c], (int)result_box[4 * c + 1],
                    (int)result_box[4 * c + 2] - (int)result_box[4 * c],
                    (int)result_box[4 * c + 3] - (int)result_box[4 * c + 1]);
                boxes.Add(rect);
                confidences.Add(results_con[c]);
            }
            // 非极大值抑制获取结果候选框
            int[] indexes = new int[boxes.Count];
            CvDnn.NMSBoxes(boxes, confidences, 0.25f, 0.45f, out indexes);
            // 裁剪指定区域
            List<Mat> out_roi = new List<Mat>();
            Mat temp_mat = new Mat();
            Cv2.Resize(image, temp_mat, input_image_size);
            for (int c = 0; c < indexes.Length; c++)
            {
                Mat roi = new Mat(temp_mat, boxes[indexes[c]]);
                out_roi.Add(roi);
            }
            return out_roi;
        }


    }
}
