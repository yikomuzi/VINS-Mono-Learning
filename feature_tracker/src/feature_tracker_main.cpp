#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

//ros::Publisher pub_img, pub_match;
//ros::Publisher pub_restart;

//每个相机都有一个FeatureTracker实例，即trackerData[i]
FeatureTracker trackerData[NUM_OF_CAM];

double first_image_time;
int pub_count = 1;  // 时间间隔内关键帧的数量，用于判断发布频率
bool first_image_flag = true;
double last_image_time = 0;//上一帧相机的时间戳
bool init_pub = 0;
vector<sensor_msgs::PointCloudPtr> v_feature_points;  // 创建一个数组存储原本应该发布的信息

/**
 * @brief   ROS的回调函数，对新来的图像进行特征点追踪，发布
 * @Description readImage()函数对新来的图像使用光流法进行特征点跟踪
 *              追踪的特征点封装成feature_points发布到pub_img的话题下，
 *              图像封装成ptr发布在pub_match下
 * @param[in]   img_msg 输入的图像
 * @return      void
*/
void img_callback(const sensor_msgs::ImageConstPtr &img_msg) {
    cout << "[img_callback] 运行一次img_callback函数, 图片时间戳："
         << img_msg->header.stamp.sec << " " << img_msg->header.stamp.nsec << endl;
    //判断是否是第一帧
    if (first_image_flag) {
//        sleep(60 * 2);
        cout << "[img_callback] 第一帧图片" << endl;
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();//记录第一个图像帧的时间
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }

//    // 通过时间间隔判断相机数据流是否稳定，有问题则restart
//    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
//    {
//        ROS_WARN("image discontinue! reset the feature tracker!");
//        first_image_flag = true;
//        last_image_time = 0;
//        pub_count = 1;
//        std_msgs::Bool restart_flag;
//        restart_flag.data = true;
//        pub_restart.publish(restart_flag);
//        return;
//    }
    last_image_time = img_msg->header.stamp.toSec();

    // 发布频率控制
    // 并不是每读入一帧图像，就要发布特征点
    // 判断间隔时间内的发布次数
    cout << "[img_callback] pub_count= " << pub_count << endl;
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <=
        FREQ)  // FREQ=10 (视频帧率为20，发布帧率为10)
    {
        cout << "[img_callback] PUB_THIS_FRAME=true " << endl;
        PUB_THIS_FRAME = true;
        // 时间间隔内的发布频率十分接近设定频率时，更新时间间隔起始时刻，并将数据发布次数置0
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ) {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    } else {
        cout << "[img_callback] PUB_THIS_FRAME=false " << endl;
        PUB_THIS_FRAME = false;
    }

    cv_bridge::CvImageConstPtr ptr;

    //将图像编码8UC1转换为mono8
    if (img_msg->encoding == "8UC1") {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    } else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat show_img = ptr->image;

    TicToc t_r;

    for (int i = 0; i < NUM_OF_CAM; i++) {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)//单目
            //readImage()函数读取图像数据进行处理
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        else//双目
        {
            if (EQUALIZE) {
                //自适应直方图均衡化处理
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            } else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    //更新全局ID
    for (unsigned int i = 0;; i++) {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

    //1、将特征点id，矫正后归一化平面的3D点(x,y,z=1)，像素2D点(u,v)，像素的速度(vx,vy)，
    //封装成sensor_msgs::PointCloudPtr类型的feature_points实例中,发布到pub_img;
    //2、将图像封装到cv_bridge::cvtColor类型的ptr实例中发布到pub_match
    if (PUB_THIS_FRAME) {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++) {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++) {
                if (trackerData[i].track_cnt[j] > 1) {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());

        // skip the first image; since no optical speed on frist image
        if (!init_pub)//第一帧不发布
        {
            init_pub = 1;
        } else {
//            pub_img.publish(feature_points);
            v_feature_points.push_back(feature_points);
        }


//        if (SHOW_TRACK) {
//            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
//            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
//            cv::Mat stereo_img = ptr->image;
//
//            for (int i = 0; i < NUM_OF_CAM; i++) {
//                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
//                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);
//                //显示追踪状态，越红越好，越蓝越不行
//                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++) {
//                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
//                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
//                    //draw speed line
//                    /*
//                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
//                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
//                    Vector3d tmp_prev_un_pts;
//                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
//                    tmp_prev_un_pts.z() = 1;
//                    Vector2d tmp_prev_uv;
//                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
//                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
//                    */
//                    //char name[10];
//                    //sprintf(name, "%d", trackerData[i].ids[j]);
//                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
//                }
//            }
//            //cv::imshow("vis", stereo_img);
//            //cv::waitKey(5);
//            pub_match.publish(ptr->toImageMsg());
//        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv) {

    /// 读取yaml中的一些配置参数
    std::string config_file = "/home/ubuntu/Desktop/VINS-Mono-Learning_study/catkin_ws/src/VINS-Mono-Learning/config/euroc/euroc_config.yaml";
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    fsSettings["image_topic"] >> IMAGE_TOPIC;
    fsSettings["imu_topic"] >> IMU_TOPIC;
    MAX_CNT = fsSettings["max_cnt"];
    MIN_DIST = fsSettings["min_dist"];
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    FREQ = fsSettings["freq"];
    F_THRESHOLD = fsSettings["F_threshold"];
    SHOW_TRACK = fsSettings["show_track"];
    EQUALIZE = fsSettings["equalize"];
    FISHEYE = fsSettings["fisheye"];
    CAM_NAMES.push_back(config_file);

    WINDOW_SIZE = 20;
    STEREO_TRACK = false;
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    fsSettings.release();

    /// 读取每个相机实例对应的相机内参
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    ifstream cam_file("/remote-home/2132917/Desktop/EuRoC_MAV_Dataset/MH_01_easy/mav0/cam0/data.csv");
    if (cam_file.is_open()) {
        string file_line;
        int count_line = 0;
        while (getline(cam_file, file_line)) {
            count_line++;
            if (count_line == 1) {
                continue;
            }

//            // 测试用
//            if (count_line == 30) {
//                break;
//            }

            istringstream line(file_line);
            string img_stamp, img_name;
            getline(line, img_stamp, ',');
            getline(line, img_name, '\r');
            uint64_t stamp;
            uint32_t sec, nsec;
            stamp = stoul(img_stamp);
            sec = stamp / 1000000000;
            nsec = stamp % 1000000000;

            string img_path = "/remote-home/2132917/Desktop/EuRoC_MAV_Dataset/MH_01_easy/mav0/cam0/data/" + img_name;
            cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
            cv_bridge::CvImage img_bridge;
            sensor_msgs::Image img_msg; // >> message to be sent
            std_msgs::Header header; // empty header
            header.seq = count_line - 2; // user defined counter
            header.stamp = ros::Time(sec, nsec); // time
            img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, img);
            img_bridge.toImageMsg(img_msg); // from cv_bridge to sensor_msgs::Image

            sensor_msgs::ImageConstPtr image(new sensor_msgs::Image(img_msg));
            img_callback(image);
        }
        cam_file.close();
    }

    cv::FileStorage fs("feature_tracker.yml", cv::FileStorage::WRITE);
    for (sensor_msgs::PointCloudPtr feature_points: v_feature_points) {
        cout << "[main] 存储v_feature_points信息到文件 " << endl;

        uint32_t seq = feature_points->header.seq;
        uint64_t stamp = feature_points->header.stamp.toNSec();
        string frame_id = feature_points->header.frame_id;
        fs << "stamp " + to_string(stamp) << "{";  // 【bug】不能以数字开头
        fs << "header" << "{";
        fs << "seq" << to_string(seq);
        fs << "stamp" << to_string(stamp);
        fs << "frame_id" << frame_id;
        fs << "}";

        fs << "points" << "{";
        auto points = feature_points->points;
        for (int i = 0; i <= points.size() - 1; i++) {
            auto point = points[i];
            float x = point.x;
            float y = point.y;
            float z = point.z;
            fs << "point" + to_string(i) << "{";
            fs << "x" << x;
            fs << "y" << y;
            fs << "z" << z;
            fs << "}";
        }
        fs << "}";

        fs << "channels" << "{";
        auto channels = feature_points->channels;
        for (int i = 0; i <= channels.size() - 1; i++) {
            fs << "channel" + to_string(i) << "{";
            auto channel = channels[i];
            string name = channel.name;
            fs << "name" << name;
            auto channel_values = channel.values;
            fs << "channel_values" << "{";
            for (int j = 0; j <= channel_values.size() - 1; j++) {
                auto channel_value = channel_values[j];
                fs << "channel_value" + to_string(j) << channel_value;
            }
            fs << "}";
            fs << "}";
        }
        fs << "}";

        fs << "}";
    }

    fs.release();

    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?