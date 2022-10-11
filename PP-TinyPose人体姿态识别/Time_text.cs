using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace OpenVinoSharpPPTinyPose
{
    internal class Time_text
    {
        public static void test_time()
        {
            //-------------------一、引入模型相关信息------------------//
            // 行人检测模型
            // ONNX格式
            // string mode_path_det = @"E:\Text_Model\TinyPose\picodet_v2_s_192_pedestrian\picodet_s_192_lcnet_pedestrian.onnx";
            string mode_path_det = @"E:\Text_Model\TinyPose\picodet_v2_s_320_pedestrian\picodet_s_320_lcnet_pedestrian.onnx";

            // 关键点检测模型
            // onnx格式
            //string mode_path_pose = @"E:\Text_Model\TinyPose\tinypose_128_96\tinypose_128_96.onnx";
            string mode_path_pose = @"E:\Text_Model\TinyPose\tinypose_256_192\tinypose_256x192.onnx";

            // 设备名称
            string device_name = "CPU";


            // 测试图片
            string image_path = @"E:\Git_space\基于Csharp和OpenVINO部署PP-TinyPose\image\demo_2.jpg";

            Mat image = Cv2.ImRead(image_path);

            //------------------一、行人识别-------------------------------//

            DateTime begin = DateTime.Now;

            Core predictor_det = new Core(mode_path_det, device_name); // 模型推理器
            DateTime end = DateTime.Now;
            TimeSpan oTime = end.Subtract(begin); //求时间差的函数  

            //输出运行时间。  
            Console.WriteLine("模型加载运行时间：{0} 毫秒", oTime.TotalMilliseconds);


            string input_node_name_det = "image"; // 模型输入节点名称
            string output_node_name_1_det = "concat_8.tmp_0"; // 模型预测框输出节点名
            string output_node_name_2_det = "transpose_8.tmp_0"; // 模型预测置信值输出节点
            Size input_size_det = new Size(320, 320); // 模型输入节点形状
            int output_length_det = 2125;


            begin = DateTime.Now;
            byte[] input_image_data_det = image.ImEncode(".bmp");
            // 数据长度
            ulong input_image_length_det = Convert.ToUInt64(input_image_data_det.Length);
            // 设置图片输入
            predictor_det.load_input_data(input_node_name_det, input_image_data_det, input_image_length_det, 0);
            end = DateTime.Now;
            //输出运行时间。  
            
            oTime = end.Subtract(begin); //求时间差的函数  
            Console.WriteLine("数据加载运行时间：{0} 毫秒", oTime.TotalMilliseconds);
            // 求取缩放大小
            double scale_x = (double)image.Width / (double)input_size_det.Width;
            double scale_y = (double)image.Height / (double)input_size_det.Height;
            Point2d scale_factor = new Point2d(scale_x, scale_y);
            begin = DateTime.Now;
            // 模型推理
            predictor_det.infer();
            end = DateTime.Now;      
            oTime = end.Subtract(begin); //求时间差的函数  
            Console.WriteLine("模型推理运行时间：{0} 毫秒", oTime.TotalMilliseconds);
            begin = DateTime.Now;
            // 读取模型输出
            // 2125 765
            // 读取置信值结果
            float[] results_con = predictor_det.read_infer_result<float>(output_node_name_2_det, output_length_det);
            // 读取预测框
            float[] result_box = predictor_det.read_infer_result<float>(output_node_name_1_det, 4 * output_length_det);
            // 处理模型推理数据
            List<Rect> boxes_result = process_result_det(results_con, result_box, scale_factor);
            end = DateTime.Now;
            oTime = end.Subtract(begin); //求时间差的函数  
            Console.WriteLine("结果处理运行时间：{0} 毫秒", oTime.TotalMilliseconds);

            //------------------二、姿态识别-------------------------------//

            begin = DateTime.Now;
            // 成员变量
            Core predictor = new Core(mode_path_pose, device_name); // 模型推理器
            end = DateTime.Now;
            oTime = end.Subtract(begin); //求时间差的函数  
            Console.WriteLine("模型加载运行时间：{0} 毫秒", oTime.TotalMilliseconds);


            string input_node_name = "image"; // 模型输入节点名称
            string output_node_name_1 = "conv2d_441.tmp_1"; // 模型输出节点名称
            string output_node_name_2 = "argmax_0.tmp_0"; // 模型输出节点名称
            Size input_size = new Size(256, 192); // 模型输入节点形状
            Size output_size = new Size(input_size.Width / 4, input_size.Height / 4); // 模型输出节点形状
            Size image_size = new Size(0, 0); // 待推理图片形状

            image_size.Width = image.Cols;
            image_size.Height = image.Rows;
            // 设置输入形状
            ulong[] input_size_a = new ulong[] { 1, 3, (ulong)(input_size.Width), (ulong)(input_size.Height) };
            predictor.set_input_sharp(input_node_name, input_size_a);

            begin = DateTime.Now;
            // 设置图片输入
            // 配置图片数据            
            // 将图片放在矩形背景下
            // 图片数据解码
            byte[] input_image_data = image.ImEncode(".bmp");
            // 数据长度
            ulong input_image_length = Convert.ToUInt64(input_image_data.Length);
            // 设置图片输入
            predictor.load_input_data(input_node_name, input_image_data, input_image_length, 2);
            end = DateTime.Now;
            oTime = end.Subtract(begin); //求时间差的函数  
            Console.WriteLine("数据加载运行时间：{0} 毫秒", oTime.TotalMilliseconds);
            begin = DateTime.Now;
            // 模型推理
            predictor.infer();
            end = DateTime.Now;
            oTime = end.Subtract(begin); //求时间差的函数  
            Console.WriteLine("模型推理运行时间：{0} 毫秒", oTime.TotalMilliseconds);
            begin = DateTime.Now;
            // 读取模型输出
            //// 2125 765
            //// 读取模型位置输出
            long[] result_pos = predictor.read_infer_result<long>(output_node_name_2, 17);
            // 单个预测点数据长度
            int point_size = output_size.Width * output_size.Height;
            // 读取预测结果
            float[] result = predictor.read_infer_result<float>(output_node_name_1, 17 * point_size);
            // 处理模型输出结果
            float[,] points = process_result(result, output_size, image_size);
            // 绘制人体姿态
            draw_poses(points, ref image);
            end = DateTime.Now;
            oTime = end.Subtract(begin); //求时间差的函数  
            Console.WriteLine("数据处理运行时间：{0} 毫秒", oTime.TotalMilliseconds);


        }




        static List<Rect> process_result_det(float[] results_con, float[] result_box, Point2d scale_factor)
        {
            // 处理预测结果
            List<float> confidences = new List<float>();
            List<Rect> boxes = new List<Rect>();
            for (int c = 0; c < 2125; c++)
            {   // 重新构建
                Rect rect = new Rect((int)(result_box[4 * c] * scale_factor.X), (int)(result_box[4 * c + 1] * scale_factor.Y),
                    (int)((result_box[4 * c + 2] - result_box[4 * c]) * scale_factor.X),
                    (int)((result_box[4 * c + 3] - result_box[4 * c + 1]) * scale_factor.Y));
                boxes.Add(rect);
                confidences.Add(results_con[c]);
            }
            // 非极大值抑制获取结果候选框
            int[] indexes = new int[boxes.Count];
            CvDnn.NMSBoxes(boxes, confidences, 0.5f, 0.5f, out indexes);
            // 提取合格的预测框
            List<Rect> boxes_result = new List<Rect>();
            for (int i = 0; i < indexes.Length; i++)
            {
                boxes_result.Add(boxes[indexes[i]]);
            }
            return boxes_result;
        }



        /// <summary>
        /// 处理关键点预测结果
        /// </summary>
        /// <param name="het_map"></param>
        /// <param name="image"></param>
        /// <returns>预测点(x, y, confindence)</returns>
        private static float[,] process_result(float[] het_map, Size output_size, Size image_size)
        {
            float[,] point_meses = new float[17, 3];

            for (int p = 0; p < 17; p++)
            {
                // 提取一个点结果图像
                float[,] map = new float[output_size.Width, output_size.Height];
                for (int h = 0; h < output_size.Width; h++)
                {
                    for (int w = 0; w < output_size.Height; w++)
                    {
                        map[h, w] = het_map[p * output_size.Width * output_size.Height + h * output_size.Height + w];
                    }
                }
                // 通过获取最大值获得点的粗略位置
                float maxval = 0;
                int[] index_int = get_max_point(map, ref maxval);
                // 保存关键点的信息
                point_meses[p, 0] = index_int[0];
                point_meses[p, 1] = index_int[1];
                point_meses[p, 2] = maxval;
                // 高斯滤波细化点位置
                Mat gaussianblur = Mat.Zeros(output_size.Width + 2, output_size.Height + 2, MatType.CV_32FC1); // 高斯图像背景
                Mat roi = new Mat(new List<int>() { output_size.Width, output_size.Height }, MatType.CV_32FC1, map); // 将点结果转为Mat数据
                Rect rect = new Rect(1, 1, output_size.Height, output_size.Width);
                roi.CopyTo(new Mat(gaussianblur, rect)); // 将点结果放在背景上
                Cv2.GaussianBlur(gaussianblur, gaussianblur, new Size(3, 3), 0); // 高斯滤波
                gaussianblur = new Mat(gaussianblur, rect); // 提取高斯滤波结果
                double max_temp = 0;
                double min_temp = 0;
                Cv2.MinMaxIdx(gaussianblur, out min_temp, out max_temp); // 获取高斯滤波后的最大值
                Mat mat = new Mat(output_size.Width, output_size.Height, MatType.CV_32FC1, maxval / max_temp);
                gaussianblur = gaussianblur.Mul(mat); // 滤波结果乘滤波前后最大值的比值
                // 将数据小于1e-10去掉，并取对数结果
                float[,] process_map = new float[output_size.Width, output_size.Height];
                for (int h = 0; h < output_size.Width; h++)
                {
                    for (int w = 0; w < output_size.Height; w++)
                    {
                        float temp = gaussianblur.At<float>(h, w);
                        if (temp < 1e-10)
                        {
                            temp = (float)1e-10;
                        }
                        temp = (float)Math.Log(temp);
                        process_map[h, w] = temp;

                    }
                }

                // 基于泰勒展开的坐标解码
                int py = index_int[1];
                int px = index_int[0];
                if ((2 < py) && (py < output_size.Width - 2) && (2 < px) && (px < output_size.Height - 2))
                {
                    // 求导数和偏导数
                    float dx = 0.5f * (process_map[py, px + 1] - process_map[py, px - 1]);
                    float dy = 0.5f * (process_map[py + 1, px] - process_map[py - 1, px]);
                    float dxx = 0.25f * (process_map[py, px + 2] - 2 * process_map[py, px] + process_map[py, px - 2]);
                    float dxy = 0.25f * (process_map[py + 1, px + 1] - process_map[py - 1, px + 1]
                        - process_map[py + 1, px - 1] + process_map[py - 1, px - 1]);
                    float dyy = 0.25f * (process_map[py + 2 * 1, px] - 2 * process_map[py, px] + process_map[py - 2 * 1, px]);
                    // 构建相应的倒数矩阵
                    Mat derivative = new Mat(2, 2, MatType.CV_32FC1, new float[] { dx, 0, dy, 0 });
                    Mat hessian = new Mat(2, 2, MatType.CV_32FC1, new float[] { dxx, dxy, dxy, dyy });
                    if (dxx * dyy - dxy * dxy != 0)
                    {
                        Mat hessianinv = new Mat();
                        Cv2.Invert(hessian, hessianinv); // 矩阵求逆
                        mat = new Mat(2, 2, MatType.CV_32FC1, -1);
                        hessianinv = hessianinv.Mul(mat); // 矩阵取－
                        Mat offset = new Mat();
                        Cv2.Multiply(hessianinv, derivative, offset); // 矩阵相乘
                        offset = offset.T(); // 矩阵转置
                        // 获取定位偏差
                        double error_x = offset.At<Vec2d>(0)[0];
                        double error_y = offset.At<Vec2d>(0)[1];
                        // 修正横纵坐标
                        point_meses[p, 0] = px + (float)error_x;
                        point_meses[p, 1] = py + (float)error_y;

                    }
                }
            }

            // 获取反向变换矩阵
            Point center = new Point(image_size.Width / 2, image_size.Height / 2); // 变换中心点
            Size input_size = new Size(image_size.Width, image_size.Height); // 输入尺寸
            int rot = 0; // 旋转角度
            Size output_size_1 = new Size(output_size.Height, output_size.Width); // 输出尺寸
            Mat trans = get_affine_transform(center, input_size, rot, output_size_1, true); // 变换矩阵
            // 获取变换结果
            double scale_x_1 = trans.At<Vec3d>(0)[0];
            double scale_x_2 = trans.At<Vec3d>(0)[1];
            double scale_x_3 = trans.At<Vec3d>(0)[2];
            double scale_y_1 = trans.At<Vec3d>(1)[0];
            double scale_y_2 = trans.At<Vec3d>(1)[1];
            double scale_y_3 = trans.At<Vec3d>(1)[2];
            // 变换预测点的位置
            for (int p = 0; p < 17; p++)
            {
                point_meses[p, 0] = point_meses[p, 0] * (float)scale_x_1 + point_meses[p, 1] * (float)scale_x_2 + 1.0f * (float)scale_x_3;
                point_meses[p, 1] = point_meses[p, 0] * (float)scale_y_1 + point_meses[p, 1] * (float)scale_y_2 + 1.0f * (float)scale_y_3;

            }
            return point_meses;
        }



        /// <summary>
        /// 获取模型输出最大点位置
        /// </summary>
        /// <param name="map"></param>
        /// <param name="maxval"></param>
        /// <returns></returns>
        private static int[] get_max_point(float[,] map, ref float maxval)
        {
            int height = map.GetLength(0);
            int width = map.GetLength(1);
            int[] index = new int[2];
            int[] index_h = new int[height];
            float[] maxval_h = new float[height];
            for (int h = 0; h < height; h++)
            {
                float val = map[h, 0];
                for (int w = 0; w < width; w++)
                {
                    if (val < map[h, w])
                    {
                        val = map[h, w];
                        maxval_h[h] = val;
                        index_h[h] = w;
                    }
                }
            }
            float maxval_temp = maxval_h[0];
            for (int h = 0; h < height; h++)
            {
                if (maxval_temp < maxval_h[h])
                {
                    maxval_temp = maxval_h[h];
                    index[1] = h;
                    index[0] = index_h[h];
                    maxval = maxval_temp;
                }
            }
            return index;
        }


        /// <summary>
        /// 获取仿射变换矩阵
        /// </summary>
        /// <param name="center">变换中心</param>
        /// <param name="input_size">输入尺寸</param>
        /// <param name="rot">旋转角度</param>
        /// <param name="output_size">输出尺寸</param>
        /// <param name="inv">是否反向</param>
        /// <returns>变换矩阵</returns>
        static Mat get_affine_transform(Point center, Size input_size, int rot, Size output_size, bool inv = false)
        {
            Point2f shift = new Point2f(0.0f, 0.0f);
            // 输入尺寸宽度
            int src_w = input_size.Width;

            // 输出尺寸
            int dst_w = output_size.Width;
            int dst_h = output_size.Height;

            // 旋转角度
            double rot_rad = 3.1715926 * rot / 180.0;
            int pt = (int)(src_w * -0.5);
            double sn = Math.Sin(rot_rad);
            double cs = Math.Cos(rot_rad);

            Point2f src_dir = new Point2f((float)(-1.0 * pt * sn), (float)(pt * cs));
            Point2f dst_dir = new Point2f(0.0f, (float)(dst_w * -0.5));
            Point2f[] src = new Point2f[3];
            src[0] = new Point2f((float)(center.X + input_size.Width * shift.X), (float)(center.Y + input_size.Height * shift.Y));
            src[1] = new Point2f(center.X + src_dir.X + input_size.Width * shift.X, center.Y + src_dir.Y + input_size.Height * shift.Y);
            Point2f direction = src[0] - src[1];
            src[2] = new Point2f(src[1].X - direction.Y, src[1].Y - direction.X);

            Point2f[] dst = new Point2f[3];
            dst[0] = new Point2f((float)(dst_w * 0.5), (float)(dst_h * 0.5));
            dst[1] = new Point2f((float)(dst_w * 0.5 + dst_dir.X), (float)(dst_h * 0.5 + dst_dir.Y));
            direction = dst[0] - dst[1];
            dst[2] = new Point2f(dst[1].X - direction.Y, dst[1].Y - direction.X);

            // 是否为反向
            if (inv) { return Cv2.GetAffineTransform(dst, src); }
            else { return Cv2.GetAffineTransform(src, dst); }
        }
        /// <summary>
        /// 绘制预测结果
        /// </summary>
        /// <param name="points"></param>
        /// <param name="image"></param>
        static void draw_poses(float[,] points, ref Mat image)
        {
            // 连接点关系
            int[,] edgs = new int[17, 2] { { 0, 1 }, { 0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7}, {6, 8},
                 {7, 9}, {8, 10}, {5, 11}, {6, 12}, {11, 13}, {12, 14},{13, 15 }, {14, 16 }, {11, 12 } };
            // 颜色库
            Scalar[] colors = new Scalar[18] { new Scalar(255, 0, 0), new Scalar(255, 85, 0), new Scalar(255, 170, 0),
                new Scalar(255, 255, 0), new Scalar(170, 255, 0), new Scalar(85, 255, 0), new Scalar(0, 255, 0),
                new Scalar(0, 255, 85), new Scalar(0, 255, 170), new Scalar(0, 255, 255), new Scalar(0, 170, 255),
                new Scalar(0, 85, 255), new Scalar(0, 0, 255), new Scalar(85, 0, 255), new Scalar(170, 0, 255),
                new Scalar(255, 0, 255), new Scalar(255, 0, 170), new Scalar(255, 0, 85) };
            // 绘制阈值
            double visual_thresh = 0.4;
            // 绘制关键点
            for (int p = 0; p < 17; p++)
            {
                if (points[p, 2] < visual_thresh)
                {
                    continue;
                }
                Point point = new Point((int)points[p, 0], (int)points[p, 1]);
                Cv2.Circle(image, point, 2, colors[p], -1);
            }
            // 绘制
            for (int p = 0; p < 17; p++)
            {
                if (points[edgs[p, 0], 2] < visual_thresh || points[edgs[p, 1], 2] < visual_thresh)
                {
                    continue;
                }

                float[] point_x = new float[] { points[edgs[p, 0], 0], points[edgs[p, 1], 0] };
                float[] point_y = new float[] { points[edgs[p, 0], 1], points[edgs[p, 1], 1] };

                Point center_point = new Point((int)((point_x[0] + point_x[1]) / 2), (int)((point_y[0] + point_y[1]) / 2));
                double length = Math.Sqrt(Math.Pow((double)(point_x[0] - point_x[1]), 2.0) + Math.Pow((double)(point_y[0] - point_y[1]), 2.0));
                int stick_width = 2;
                Size axis = new Size(length / 2, stick_width);
                double angle = (Math.Atan2((double)(point_y[0] - point_y[1]), (double)(point_x[0] - point_x[1]))) * 180 / Math.PI;
                Point[] polygon = Cv2.Ellipse2Poly(center_point, axis, (int)angle, 0, 360, 1);
                Cv2.FillConvexPoly(image, polygon, colors[p]);

            }
        }

    }
}

