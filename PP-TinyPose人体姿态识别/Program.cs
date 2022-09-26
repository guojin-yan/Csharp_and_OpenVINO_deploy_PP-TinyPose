using System;
using OpenCvSharp;

namespace OpenVinoSharpPPTinyPose
{
    internal class Program
    {
        static void Main(string[] args)
        {
            tiny_pose_image_192();
        }

        public static void tiny_pose_image_192() 
        {
            //-------------------一、引入模型相关信息------------------//
            // 行人检测模型
            // ONNX格式
            string mode_path_det = @"E:\Text_Model\TinyPose\picodet_v2_s_192_pedestrian\picodet_s_192_lcnet_pedestrian.onnx";


            // 关键点检测模型
            // paddle格式
            string mode_path_pose = @"E:\Text_Model\TinyPose\tinypose_128_96\paddle\model.pdmodel";

            // 设备名称
            string device_name = "CPU";


            // 测试图片
            string image_path = @"E:\Git_space\基于Csharp和OpenVINO部署PP-TinyPose\image\demo_4.jpg";

            Mat image = Cv2.ImRead(image_path);


            //---------------------二、行人检测--------------------//

            PicoDet picoDet = new PicoDet(mode_path_det, device_name);

            Size size_det = new Size(192, 192);
            picoDet.set_shape(size_det, 765);
            List<Mat> image_dets = picoDet.predict(image);
            for (int i = 0; i < image_dets.Count; i++) {
                Cv2.ImShow(i.ToString(), image_dets[i]);
            }
            Cv2.WaitKey(0);


            Console.WriteLine("qwert");

        }
    }
}
