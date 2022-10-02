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
            string mode_path_pose = @"E:\Text_Model\TinyPose\tinypose_128_96\tinypose_128_96.onnx";
            //string mode_path_pose = @"E:\Text_Model\TinyPose\tinypose_256_192\tinypose_256x192.onnx";

            // 设备名称
            string device_name = "CPU";


            // 测试图片
            string image_path = @"E:\Git_space\基于Csharp和OpenVINO部署PP-TinyPose\image\demo_6.jpg";

            Mat image = Cv2.ImRead(image_path);


            //---------------------二、行人检测--------------------//

            PicoDet pico_det = new PicoDet(mode_path_det, device_name);

            Size size_det = new Size(192, 192);
            pico_det.set_shape(size_det, 765);
            List<Mat> image_dets = pico_det.predict(image);
            //for (int i = 0; i < image_dets.Count; i++)
            //{
            //    Cv2.ImShow(i.ToString(), image_dets[i]);
            //}
            //Cv2.WaitKey(0);

            PPTinyPose tiny_pose = new PPTinyPose(mode_path_pose, device_name);
            Size size_pose = new Size(128, 96);
            // Size size_pose = new Size(256, 192);
            tiny_pose.set_shape(size_pose);
            tiny_pose.predict(image);
            Console.WriteLine("qwert");

        }
    }
}
