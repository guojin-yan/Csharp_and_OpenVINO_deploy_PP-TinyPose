using System;
using OpenCvSharp;

namespace OpenVinoSharpPPTinyPose
{
    internal class Program
    {
        static void Main(string[] args)
        {
            tiny_pose_image();
            // Time_text.test_time();
        }

        public static void tiny_pose_image() 
        {
            //-------------------一、引入模型相关信息------------------//
            // 行人检测模型
            // ONNX格式
            // string mode_path_det = @"E:\Text_Model\TinyPose\picodet_v2_s_192_pedestrian\picodet_s_192_lcnet_pedestrian.onnx";
            string mode_path_det = @"E:\Text_Model\TinyPose\picodet_v2_s_320_pedestrian\ir\picodet_s_320_lcnet_pedestrian.xml";

            // 关键点检测模型
            // onnx格式
            // string mode_path_pose = @"E:\Text_Model\TinyPose\tinypose_128_96\tinypose_128_96.onnx";
            string mode_path_pose = @"E:\Text_Model\TinyPose\tinypose_256_192\tinypose_256_192.onnx";

            // 设备名称
            string device_name = "CPU";


            // 测试图片
            string image_path = @"E:\Git_space\基于Csharp和OpenVINO部署PP-TinyPose\image\demo_7.jpg";

            Mat image = Cv2.ImRead(image_path);


            //---------------------二、行人区域检测--------------------//

            PicoDet pico_det = new PicoDet(mode_path_det, device_name);

            //Size size_det = new Size(192, 192);
            //pico_det.set_shape(size_det, 765);
            Size size_det = new Size(320, 320);
            pico_det.set_shape(size_det, 2125);
            List<Rect> result_rect = pico_det.predict(image);


            //Cv2.Rectangle(image, result_rect[0], new Scalar(255, 0, 0), 2);

            //Cv2.ImShow("result", image);
            //Cv2.WaitKey(0);
            ////---------------------三、人体姿势检测--------------------//
            PPTinyPose tiny_pose = new PPTinyPose(mode_path_pose, device_name);
            // Size size_pose = new Size(128, 96);
            Size size_pose = new Size(256, 192);
            tiny_pose.set_shape(size_pose);

            List<Rect> point_rects;
            List<Mat> person_rois = tiny_pose.get_point_roi(image, result_rect, out point_rects);

            for (int p = 0; p < person_rois.Count; p++)
            {
                // 关键点识别
                float[,] person_point = tiny_pose.predict(person_rois[p]);

                tiny_pose.draw_poses(person_point, point_rects[p], ref image);
            }

            for (int i = 0; i < result_rect.Count; i++)
            {
                Cv2.Rectangle(image, result_rect[i], new Scalar(255, 0, 0), 2);
            }
            Cv2.ImShow("result", image);
            Cv2.WaitKey(0);

        }
    }
}
