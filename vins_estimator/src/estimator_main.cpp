#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

//std::condition_variable con;//条件变量
double current_time = -1;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;

int sum_of_wait = 0;

//互斥量
//std::mutex m_buf;
//std::mutex m_state;
//std::mutex i_buf;
//std::mutex m_estimator;

double latest_time;

//IMU项[P,Q,B,Ba,Bg,a,g]
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

//从IMU测量值imu_msg和上一个PVQ递推得到下一个tmp_Q，tmp_P，tmp_V，中值积分
void predict(const sensor_msgs::ImuConstPtr &imu_msg) {
    double t = imu_msg->header.stamp.toSec();

    //init_imu=1表示第一个IMU数据
    if (init_imu) {
        latest_time = t;
        init_imu = 0;
        return;
    }

    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

//从估计器中得到滑动窗口当前图像帧的imu更新项[P,Q,V,ba,bg,a,g]
//对imu_buf中剩余的imu_msg进行PVQ递推
void update() {
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

/**
 * @brief   对imu和图像数据进行对齐并组合
 * @Description     img:    i -------- j  -  -------- k
 *                  imu:    - jjjjjjjj - j/k kkkkkkkk -
 *                  直到把缓存中的图像特征数据或者IMU数据取完，才能够跳出此函数，并返回数据
 * @return  vector<std::pair<vector<ImuConstPtr>, PointCloudConstPtr>> (IMUs, img_msg)s
*/
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements() {
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true) {
        cout << "[getMeasurements] imu_buf size:" << imu_buf.size() << endl;
        cout << "[getMeasurements] feature_buf size:" << feature_buf.size() << endl;
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        //对齐标准：IMU最后一个数据的时间要大于第一个图像特征数据的时间(td表示imu和相机时间戳的误差间隔，本数据集td为0)
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td)) {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        //对齐标准：IMU第一个数据的时间要小于第一个图像特征数据的时间
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td)) {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }

        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;

        //图像数据(img_msg)，对应多组在时间戳内的imu数据,然后塞入measurements
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td) {
            //emplace_back相比push_back能更好地避免内存的拷贝与移动
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        //这里把下一个imu_msg也放进去了,但没有pop，因此当前图像帧和下一图像帧会共用这个imu_msg
        IMUs.emplace_back(imu_buf.front());

        if (IMUs.empty())
            ROS_WARN("no imu between two image");

        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

// imu回调函数，将imu_msg保存到imu_buf，IMU状态递推并发布[P,Q,V,header]
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg) {
    //判断时间间隔是否为正
    if (imu_msg->header.stamp.toSec() <= last_imu_t) {
        //ROS_WARN("imu message ins disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();


//    m_buf.lock();
    imu_buf.push(imu_msg);
//    m_buf.unlock();

//    con.notify_one();//唤醒作用于process线程中的获取观测值数据的函数

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        //构造互斥锁m_state，析构时解锁
//        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);//递推得到IMU的PQV
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";

        //发布最新的由IMU直接递推得到的PQV
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

//feature回调函数，将feature_msg放入feature_buf
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg) {
    if (!init_feature) {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }

//    m_buf.lock();
    feature_buf.push(feature_msg);
//    m_buf.unlock();

//    con.notify_one();
}

////restart回调函数，收到restart时清空feature_buf和imu_buf，估计器重置，时间重置
//void restart_callback(const std_msgs::BoolConstPtr &restart_msg) {
//    if (restart_msg->data == true) {
//        ROS_WARN("restart the estimator!");
//
//        m_buf.lock();
//        while (!feature_buf.empty())
//            feature_buf.pop();
//        while (!imu_buf.empty())
//            imu_buf.pop();
//        m_buf.unlock();
//
//        m_estimator.lock();
//        estimator.clearState();
//        estimator.setParameter();
//        m_estimator.unlock();
//
//        current_time = -1;
//        last_imu_t = 0;
//    }
//    return;
//}

////relocalization回调函数，将points_msg放入relo_buf
//void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg) {
//    //printf("relocalization callback! \n");
//    m_buf.lock();
//    relo_buf.push(points_msg);
//    m_buf.unlock();
//}

/**
 * @brief   VIO的主线程
 * @Description 等待并获取measurements：(IMUs, img_msg)s，计算dt
 *              estimator.processIMU()进行IMU预积分
 *              estimator.setReloFrame()设置重定位帧
 *              estimator.processImage()处理图像帧：初始化，紧耦合的非线性优化
 * @return      void
*/
void process() {
//    while (true) {
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

//        std::unique_lock<std::mutex> lk(m_buf);

    //等待上面两个接收数据完成就会被唤醒
    //在提取measurements时互斥锁m_buf会锁住，此时无法接收数据
//        con.wait(lk, [&] {
//            return (measurements = getMeasurements()).size() != 0;
//        });
//        lk.unlock();

//        m_estimator.lock();

    measurements = getMeasurements();

    int count_measurement = 0;
    for (auto &measurement: measurements) {
        cout << "[process] measurements遍历次数: " << count_measurement++ << endl;
        //对应这段的img data
        auto img_msg = measurement.second;
        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
        for (auto &imu_msg: measurement.first) {
            double t = imu_msg->header.stamp.toSec();
            double img_t = img_msg->header.stamp.toSec() + estimator.td;

            //发送IMU数据进行预积分
            if (t <= img_t) {
                if (current_time < 0)
                    current_time = t;
                double dt = t - current_time;
                ROS_ASSERT(dt >= 0);
                current_time = t;
                dx = imu_msg->linear_acceleration.x;
                dy = imu_msg->linear_acceleration.y;
                dz = imu_msg->linear_acceleration.z;
                rx = imu_msg->angular_velocity.x;
                ry = imu_msg->angular_velocity.y;
                rz = imu_msg->angular_velocity.z;
                //imu预积分
                estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

            } else {
                double dt_1 = img_t - current_time;
                double dt_2 = t - img_t;
                current_time = img_t;
                ROS_ASSERT(dt_1 >= 0);
                ROS_ASSERT(dt_2 >= 0);
                ROS_ASSERT(dt_1 + dt_2 > 0);
                double w1 = dt_2 / (dt_1 + dt_2);
                double w2 = dt_1 / (dt_1 + dt_2);
                dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
            }
        }

/*        // set relocalization frame
        sensor_msgs::PointCloudConstPtr relo_msg = NULL;

        //取出最后一个重定位帧
        while (!relo_buf.empty()) {
            relo_msg = relo_buf.front();
            relo_buf.pop();
        }

        if (relo_msg != NULL) {
            vector<Vector3d> match_points;
            double frame_stamp = relo_msg->header.stamp.toSec();
            for (unsigned int i = 0; i < relo_msg->points.size(); i++) {
                Vector3d u_v_id;
                u_v_id.x() = relo_msg->points[i].x;
                u_v_id.y() = relo_msg->points[i].y;
                u_v_id.z() = relo_msg->points[i].z;
                match_points.push_back(u_v_id);
            }
            Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1],
                            relo_msg->channels[0].values[2]);
            Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4],
                               relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
            Matrix3d relo_r = relo_q.toRotationMatrix();
            int frame_index;
            frame_index = relo_msg->channels[0].values[7];

            estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
        }*/

        ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

        TicToc t_s;

        //建立每个特征点的(camera_id,[x,y,z,u,v,vx,vy])s的map，索引为feature_id
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
        for (unsigned int i = 0; i < img_msg->points.size(); i++) {
//            int v = img_msg->channels[0].values[i] + 0.5;  // 【问题】没看明白为什么要加0.5
            int v = img_msg->channels[0].values[i];
            int feature_id = v / NUM_OF_CAM;
            int camera_id = v % NUM_OF_CAM;
            double x = img_msg->points[i].x;
            double y = img_msg->points[i].y;
            double z = img_msg->points[i].z;
            double p_u = img_msg->channels[1].values[i];
            double p_v = img_msg->channels[2].values[i];
            double velocity_x = img_msg->channels[3].values[i];
            double velocity_y = img_msg->channels[4].values[i];
            ROS_ASSERT(z == 1);
            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            image[feature_id].emplace_back(camera_id, xyz_uv_velocity);
        }

        //【核心代码】处理图像特征
        estimator.processImage(image, img_msg->header);

        double whole_t = t_s.toc();
        printStatistics(estimator, whole_t);
        std_msgs::Header header = img_msg->header;
        header.frame_id = "world";

        //给RVIZ发送topic
        pubOdometry(estimator, header);//"odometry" 里程计信息PQV
        pubKeyPoses(estimator, header);//"key_poses" 关键点三维坐标
        pubCameraPose(estimator, header);//"camera_pose" 相机位姿
        pubPointCloud(estimator, header);//"history_cloud" 点云信息
        pubTF(estimator, header);//"extrinsic" 相机到IMU的外参
        pubKeyframe(estimator);//"keyframe_point"、"keyframe_pose" 关键帧位姿和点云

//        if (relo_msg != NULL)
//            pubRelocalization(estimator);//"relo_relative_pose" 重定位位姿
        //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());

        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();//更新IMU参数[P,Q,V,ba,bg,a,g]

    }
//        m_estimator.unlock();

//        m_buf.lock();
//        m_state.lock();
//    if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
//        update();//更新IMU参数[P,Q,V,ba,bg,a,g]
//        m_state.unlock();
//        m_buf.unlock();
//    }
}

int main(int argc, char **argv) {
    //ROS初始化，设置句柄n
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    /// 读取参数，设置估计器参数
    {
        std::string config_file;
        config_file = "/home/ubuntu/Desktop/VINS-Mono-Learning_study/catkin_ws/src/VINS-Mono-Learning/config/euroc/euroc_config.yaml";
        cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
        if (!fsSettings.isOpened()) {
            std::cerr << "ERROR: Wrong path to settings" << std::endl;
        }

        fsSettings["imu_topic"] >> IMU_TOPIC;

        SOLVER_TIME = fsSettings["max_solver_time"];
        NUM_ITERATIONS = fsSettings["max_num_iterations"];
        MIN_PARALLAX = fsSettings["keyframe_parallax"];
        MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

        std::string OUTPUT_PATH;
        fsSettings["output_path"] >> OUTPUT_PATH;
        VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv";
        std::cout << "result path " << VINS_RESULT_PATH << std::endl;
        std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
        fout.close();

        ACC_N = fsSettings["acc_n"];
        ACC_W = fsSettings["acc_w"];
        GYR_N = fsSettings["gyr_n"];
        GYR_W = fsSettings["gyr_w"];
        G.z() = fsSettings["g_norm"];
        ROW = fsSettings["image_height"];
        COL = fsSettings["image_width"];
        ROS_INFO("ROW: %f COL: %f ", ROW, COL);

        //IMU和CAM的外参是否提供
        ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
        if (ESTIMATE_EXTRINSIC == 2)//不提供
        {
            ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
            RIC.push_back(Eigen::Matrix3d::Identity());
            TIC.push_back(Eigen::Vector3d::Zero());
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";

        } else {
            if (ESTIMATE_EXTRINSIC == 1)//不准确
            {
                ROS_WARN(" Optimize extrinsic param around initial guess!");
                EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv";
            }
            if (ESTIMATE_EXTRINSIC == 0)//准确
                ROS_WARN(" fix extrinsic param ");

            //读取初始R,t存入各自vector
            cv::Mat cv_R, cv_T;
            fsSettings["extrinsicRotation"] >> cv_R;
            fsSettings["extrinsicTranslation"] >> cv_T;
            Eigen::Matrix3d eigen_R;
            Eigen::Vector3d eigen_T;
            cv::cv2eigen(cv_R, eigen_R);
            cv::cv2eigen(cv_T, eigen_T);
            Eigen::Quaterniond Q(eigen_R);
            eigen_R = Q.normalized();//归一化
            RIC.push_back(eigen_R);
            TIC.push_back(eigen_T);
            ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
            ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
        }

        INIT_DEPTH = 5.0;
        BIAS_ACC_THRESHOLD = 0.1;
        BIAS_GYR_THRESHOLD = 0.1;

        //IMU和cam时间校准
        TD = fsSettings["td"];
        ESTIMATE_TD = fsSettings["estimate_td"];
        if (ESTIMATE_TD)
            ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
        else
            ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

        ROLLING_SHUTTER = fsSettings["rolling_shutter"];
        if (ROLLING_SHUTTER) {
            TR = fsSettings["rolling_shutter_tr"];
            ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
        } else {
            TR = 0;
        }

        fsSettings.release();
    }

    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");


    //用于RVIZ显示的Topic
    registerPub(n);


    /// 读取并处理imu数据
    ifstream imu_file("/remote-home/2132917/Desktop/EuRoC_MAV_Dataset/MH_01_easy/mav0/imu0/data.csv");
    if (imu_file.is_open()) {
        string file_line;
        int count_line = 0;
        while (getline(imu_file, file_line)) {
            count_line++;
            if (count_line == 1) {
                continue;
            }

//            // 测试用
//            if (count_line == 20 * 5*10) {
//                break;
//            }

            istringstream line(file_line);
            string img_stamp, img_name;
            string s_imu_timestamp;
            string s_w_RS_S_x, s_w_RS_S_y, s_w_RS_S_z, s_a_RS_S_x, s_a_RS_S_y, s_a_RS_S_z;
            getline(line, s_imu_timestamp, ',');
            getline(line, s_w_RS_S_x, ',');
            getline(line, s_w_RS_S_y, ',');
            getline(line, s_w_RS_S_z, ',');
            getline(line, s_a_RS_S_x, ',');
            getline(line, s_a_RS_S_y, ',');
            getline(line, s_a_RS_S_z, '\r');

            uint64_t stamp = stoull(s_imu_timestamp);
            uint32_t sec, nsec;
            sec = stamp / 1000000000;
            nsec = stamp % 1000000000;
            double w_RS_S_x, w_RS_S_y, w_RS_S_z, a_RS_S_x, a_RS_S_y, a_RS_S_z;
            w_RS_S_x = stod(s_w_RS_S_x);
            w_RS_S_y = stod(s_w_RS_S_y);
            w_RS_S_z = stod(s_w_RS_S_z);
            a_RS_S_x = stod(s_a_RS_S_x);
            a_RS_S_y = stod(s_a_RS_S_y);
            a_RS_S_z = stod(s_a_RS_S_z);


            sensor_msgs::Imu imu_msg; // >> message to be sent
            imu_msg.header.seq = count_line - 2; // user defined counter
            imu_msg.header.stamp = ros::Time(sec, nsec); // time
            imu_msg.angular_velocity.x = w_RS_S_x;
            imu_msg.angular_velocity.y = w_RS_S_y;
            imu_msg.angular_velocity.z = w_RS_S_z;
            imu_msg.linear_acceleration.x = a_RS_S_x;
            imu_msg.linear_acceleration.y = a_RS_S_y;
            imu_msg.linear_acceleration.z = a_RS_S_z;

            sensor_msgs::ImuConstPtr imu(new sensor_msgs::Imu(imu_msg));
            imu_callback(imu);
//            imu_buf.push(imu);
        }
        imu_file.close();
    }


    /// 读取并保持feature track的数据(读取yml文件内容，转换为对象)
    vector<sensor_msgs::PointCloudPtr> v_feature_points_yml;  // 创建一个数组存储原本应该发布的信息
    cv::FileStorage fs_read;
    fs_read.open(
            "/home/ubuntu/Desktop/VINS-Mono-Learning_study/catkin_ws/src/cmake-build-debug/devel/lib/feature_tracker/feature_tracker.yml",
            cv::FileStorage::READ);
    cv::FileNode fileNodes = fs_read["feature_points"];
    for (auto fileNode: fileNodes) {
        sensor_msgs::PointCloudPtr p_pointcloud(new sensor_msgs::PointCloud());

        string seq = fileNode["header"]["seq"];
        string stamp = fileNode["header"]["stamp"];
        string frame_id = fileNode["header"]["frame_id"];
        p_pointcloud->header.seq = stoul(seq);
        p_pointcloud->header.stamp = ros::Time(stoull(stamp) / 1000000000, stoull(stamp) % 1000000000);
        p_pointcloud->header.frame_id = frame_id;

        cv::FileNode points = fileNode["points"];
        for (auto point: points) {
            geometry_msgs::Point32 msgs_point32;

            float x = point["x"];
            float y = point["y"];
            float z = point["z"];
            msgs_point32.x = x;
            msgs_point32.y = y;
            msgs_point32.z = z;
            p_pointcloud->points.push_back(msgs_point32);
        }

        cv::FileNode channels = fileNode["channels"];
        int count_channel = 0;
        for (auto channel: channels) {
            sensor_msgs::ChannelFloat32 msgs_channel;

            string name = channel["name"];
            msgs_channel.name = name;
            cv::FileNode channel_value = channel["channel_values"];
            for (auto value: channel_value) {
                msgs_channel.values.push_back(value);
            }
            p_pointcloud->channels.push_back(msgs_channel);
            count_channel++;
        }

        v_feature_points_yml.push_back(p_pointcloud);
    }

    for (auto i: v_feature_points_yml) {
//        feature_buf.push(i);
        feature_callback(i);
    }


//    //订阅IMU、feature、restart、match_points的topic,执行各自回调函数
//    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
//    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
//    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
//    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    //创建VIO主线程
//    std::thread measurement_process{process};
    process();

    ros::spin();

    return 0;
}
