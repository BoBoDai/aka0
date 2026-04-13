#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <inttypes.h>
#include <opencv2/opencv.hpp>
#include "cviruntime.h"
#include "motor.hpp"
#include "arm.hpp"

// VI related headers - must include base types first
#include "vi_capture.hpp"
#include "logger.hpp"

// 控制宏定义
//#define ENABLE_DEBUG_OUTPUT 0 // Replaced by LOGD
#define ENABLE_DRAW_BBOX 1    // 是否画框并保存图片
#define ENABLE_SAVE_IMAGE 0   // 是否保存检测结果图片

typedef struct {
    float x, y, w, h;
} box;

typedef struct {
    box bbox;
    int cls;
    float score;
    int batch_idx;
} detection;

static const char* tennis_names[] = {"tennis"}; // 单类别网球检测


static void usage(char** argv) {
    LOGI("Usage:");
    LOGI("   %s cvimodel [vi_channel]", argv[0]);
    LOGI("   Example: %s model.cvimodel 0", argv[0]);
    LOGI("   This will capture video from VI channel (default: 0)");
}

template <typename T> int argmax(const T* data, size_t len, size_t stride = 1) {
    int maxIndex = 0;
    for (size_t i = stride; i < len; i += stride) {
        if (data[maxIndex] < data[i]) {
            maxIndex = i;
        }
    }
    return maxIndex;
}

float calIou(box a, box b) {
    float area1 = a.w * a.h;
    float area2 = b.w * b.h;
    float wi = std::min((a.x + a.w / 2), (b.x + b.w / 2)) - std::max((a.x - a.w / 2), (b.x - b.w / 2));
    float hi = std::min((a.y + a.h / 2), (b.y + b.h / 2)) - std::max((a.y - a.h / 2), (b.y - b.h / 2));
    float area_i = std::max(wi, 0.0f) * std::max(hi, 0.0f);
    return area_i / (area1 + area2 - area_i);
}

static void NMS(std::vector<detection>& dets, int* total, float thresh) {
    if (*total) {
        std::sort(dets.begin(), dets.end(), [](detection& a, detection& b) { return b.score < a.score; });
        int new_count = *total;
        for (int i = 0; i < *total; ++i) {
            detection& a = dets[i];
            if (a.score == 0)
                continue;
            for (int j = i + 1; j < *total; ++j) {
                detection& b = dets[j];
                if (dets[i].batch_idx == dets[j].batch_idx && b.score != 0 && dets[i].cls == dets[j].cls &&
                    calIou(a.bbox, b.bbox) > thresh) {
                    b.score = 0;
                    new_count--;
                }
            }
        }
        std::vector<detection>::iterator it = dets.begin();
        while (it != dets.end()) {
            if (it->score == 0) {
                dets.erase(it);
            } else {
                it++;
            }
        }
        *total = new_count;
    }
}

void correctYoloBoxes(std::vector<detection>& dets, int det_num, int image_h, int image_w, int input_height,
                      int input_width) {
    // 计算缩放比例和padding，与Python代码保持一致
    float scale = std::min((float)input_width / image_w, (float)input_height / image_h);
    int new_h = (int)(image_h * scale);
    int new_w = (int)(image_w * scale);
    int pad_top = (input_height - new_h) / 2;
    int pad_left = (input_width - new_w) / 2;

    LOGD("=== Coordinate correction ===");
    LOGD("Original image: %dx%d, Input size: %dx%d", image_w, image_h, input_width, input_height);
    LOGD("Scale: %.3f, New size: %dx%d, Padding: left=%d, top=%d", scale, new_w, new_h, pad_left, pad_top);

    for (int i = 0; i < det_num; ++i) {
        // YOLOv8输出的是中心点坐标(cx,cy)和宽高(w,h)
        float cx = dets[i].bbox.x;
        float cy = dets[i].bbox.y;
        float w = dets[i].bbox.w;
        float h = dets[i].bbox.h;

        // 转换为左上角和右下角坐标(相对于640x640输入图像)
        float x1 = cx - 0.5f * w;
        float y1 = cy - 0.5f * h;
        float x2 = cx + 0.5f * w;
        float y2 = cy + 0.5f * h;

        // 去除padding并缩放回原图尺寸
        x1 = std::max(0.0f, (x1 - pad_left) / scale);
        y1 = std::max(0.0f, (y1 - pad_top) / scale);
        x2 = std::min((float)image_w, (x2 - pad_left) / scale);
        y2 = std::min((float)image_h, (y2 - pad_top) / scale);

        // 转换回中心点坐标和宽高格式
        dets[i].bbox.x = (x1 + x2) / 2.0f; // 中心点x
        dets[i].bbox.y = (y1 + y2) / 2.0f; // 中心点y
        dets[i].bbox.w = x2 - x1;          // 宽度
        dets[i].bbox.h = y2 - y1;          // 高度

        LOGD("Det[%d]: input_bbox(%.1f,%.1f,%.1f,%.1f) -> "
             "output_bbox(%.1f,%.1f,%.1f,%.1f)",
             i, cx, cy, w, h, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
    }
}

/**
 * @brief
 * @param output
 * @note scores_shape : [batch , class_num, det_num, 1]
 * @note des_shape: [batch, 1, 4, det_num]
 * @return int
 */
int getDetections(CVI_TENSOR* output, int32_t input_height, int32_t input_width, int classes_num,
                  CVI_SHAPE output_shape, float conf_thresh, std::vector<detection>& dets) {
    // 添加调试信息：打印输出tensor信息
    LOGD("=== DEBUG: Output tensor information ===");
    LOGD("Output shape: [%d, %d, %d, %d]", output_shape.dim[0], output_shape.dim[1], output_shape.dim[2],
           output_shape.dim[3]);

    // 检查是否有足够的输出tensor
    if (output == nullptr) {
        LOGE("ERROR: output tensor is null");
        return 0;
    }

    float* output_ptr = (float*)CVI_NN_TensorPtr(&output[0]);

    // 检查指针是否有效
    if (output_ptr == nullptr) {
        LOGE("ERROR: tensor pointer is null");
        return 0;
    }

    float stride[3] = {8, 16, 32};
    int count = 0;
    int batch = output_shape.dim[0];
    int channels = output_shape.dim[1];      // 应该是4(bbox) + 1(objectness) + classes_num
    int total_anchors = output_shape.dim[2]; // 27600

    LOGD("Batch: %d, Channels: %d, Total_anchors: %d", batch, channels, total_anchors);

    // 计算每个stride层的anchor数量
    int anchor_counts[3];
    for (int i = 0; i < 3; i++) {
        int nh = input_height / stride[i];
        int nw = input_width / stride[i];
        anchor_counts[i] = nh * nw;
        LOGD("Stride[%d]: %f, grid: %dx%d, anchors: %d", i, stride[i], nh, nw, anchor_counts[i]);
    }

    int anchor_offset = 0;
    for (int b = 0; b < batch; b++) {
        anchor_offset = 0;
        for (int stride_idx = 0; stride_idx < 3; stride_idx++) {
            int nh = input_height / stride[stride_idx];
            int nw = input_width / stride[stride_idx];
            int current_anchors = anchor_counts[stride_idx];

            for (int anchor_idx = 0; anchor_idx < current_anchors; anchor_idx++) {
                int total_anchor_idx = anchor_offset + anchor_idx;

                // 获取objectness/confidence (第5个通道)
                float objectness = output_ptr[4 * total_anchors + total_anchor_idx];

                // 添加调试输出：打印前几个anchor的原始数据
                if (objectness > 0.1 || total_anchor_idx < 3) { // Reduced debug spam condition
                    LOGD("Anchor[%d]: raw values = [%.6f, %.6f, %.6f, %.6f, %.6f]", total_anchor_idx,
                           output_ptr[0 * total_anchors + total_anchor_idx],
                           output_ptr[1 * total_anchors + total_anchor_idx],
                           output_ptr[2 * total_anchors + total_anchor_idx],
                           output_ptr[3 * total_anchors + total_anchor_idx],
                           output_ptr[4 * total_anchors + total_anchor_idx]);
                }

                if (objectness <= conf_thresh) {
                    continue;
                }

                // 获取bbox坐标 (前4个通道)
                // YOLOv8输出格式：cx, cy, w, h (相对于640x640输入图像的绝对像素坐标)
                float cx = output_ptr[0 * total_anchors + total_anchor_idx];
                float cy = output_ptr[1 * total_anchors + total_anchor_idx];
                float w = output_ptr[2 * total_anchors + total_anchor_idx];
                float h = output_ptr[3 * total_anchors + total_anchor_idx];

                detection det;
                det.score = objectness;
                det.cls = 0; // 单类别网球检测
                det.batch_idx = b;

                // 直接使用模型输出的中心点坐标和宽高，无需grid计算
                det.bbox.x = cx; // 中心点x坐标
                det.bbox.y = cy; // 中心点y坐标
                det.bbox.w = w;  // 宽度
                det.bbox.h = h;  // 高度

                LOGD("Detection[%d]: conf=%.3f, bbox_center=(%.1f,%.1f), "
                       "size=(%.1f,%.1f)",
                       count, objectness, cx, cy, w, h);

                count++;
                dets.emplace_back(det);
            }

            anchor_offset += current_anchors;
        }
    }
    return count;
}

// ============ 状态机 (移植自 AKA-00 tennis_hunter.py) ============

enum RobotStatus {
    STATUS_CHASE_TENNIS,     // 追球
    STATUS_POSITION_TENNIS,  // 精确对准
    STATUS_GRAB_TENNIS,      // 确认并抓取
};

static const char* status_name(RobotStatus s) {
    switch (s) {
        case STATUS_CHASE_TENNIS:    return "chase_tennis";
        case STATUS_POSITION_TENNIS: return "position_tennis";
        case STATUS_GRAB_TENNIS:     return "grab_tennis";
    }
    return "unknown";
}

// P 控制参数（AKA-00 的 MAX_SPEED=240 对应 PWM 范围，我们的 Motor 是 0-100）
static const int FRAME_WIDTH       = 640;
static const int X_LEFT_GRAB       = 258;
static const int X_RIGHT_GRAB      = 298;   // X_LEFT_GRAB + 40
static const int TENNIS_WIDTH_FAR  = 320;
static const int TENNIS_WIDTH_NEAR = 380;
static const int MAX_SPEED         = 100;    // Motor 范围 0-100
static const int MIN_SPEED         = MAX_SPEED / 6;  // ~16
static const int IDLE_SPEED        = MAX_SPEED / 3;  // ~33
static const float WHEEL_BASE      = 10.0f;
static const float Kp_dist         = 0.8f;
static const float Kp_angle        = 0.02f;
static const int GRAB_CONFIRM_THRESHOLD = 10;

struct RobotState {
    RobotStatus status;
    int box_x;          // bbox 左上角 x
    int box_w;          // bbox 宽度
    int grab_confirm_count;

    RobotState() : status(STATUS_CHASE_TENNIS), box_x(0), box_w(0), grab_confirm_count(0) {}

    void update_status() {
        if (status == STATUS_CHASE_TENNIS) {
            if (box_w >= TENNIS_WIDTH_FAR && box_w <= TENNIS_WIDTH_NEAR) {
                status = STATUS_POSITION_TENNIS;
                LOGI("[STATE] chase -> position (w=%d)", box_w);
            }
        } else if (status == STATUS_POSITION_TENNIS) {
            if (box_w < TENNIS_WIDTH_FAR || box_w > TENNIS_WIDTH_NEAR) {
                status = STATUS_CHASE_TENNIS;
                LOGI("[STATE] position -> chase (w=%d out of range)", box_w);
            } else if (box_x >= X_LEFT_GRAB && box_x <= X_RIGHT_GRAB) {
                status = STATUS_GRAB_TENNIS;
                LOGI("[STATE] position -> grab (x=%d centered)", box_x);
            }
        }
    }

    // 计算差速 PWM，返回 left, right（范围 [-MAX_SPEED, MAX_SPEED]）
    void calc_motor_speed(int& left_pwm, int& right_pwm) {
        int TARGET_X = FRAME_WIDTH / 2;
        int TARGET_W = static_cast<int>(TENNIS_WIDTH_FAR * 0.6f + TENNIS_WIDTH_NEAR * 0.4f);

        float error_x = (box_x + box_w / 2.0f) - TARGET_X;
        float error_w = box_w - TARGET_W;

        float raw_v = -Kp_dist * error_w;
        float raw_omega = -Kp_angle * error_x;

        // 动态限速：急转弯时降速
        float turn_factor = fabs(error_x) / (FRAME_WIDTH / 2.0f);
        float max_v = (turn_factor > 0.8f) ? MIN_SPEED * 0.3f : MAX_SPEED;

        float v = std::max(-max_v, std::min(max_v, raw_v));
        if (fabs(v) < MIN_SPEED && fabs(v) > 0)
            v = (v > 0) ? MIN_SPEED : -MIN_SPEED;

        float diff_speed = raw_omega * WHEEL_BASE;
        float lf = v + diff_speed;
        float rf = v - diff_speed;

        // 限幅
        lf = std::max((float)-MAX_SPEED, std::min((float)MAX_SPEED, lf));
        rf = std::max((float)-MAX_SPEED, std::min((float)MAX_SPEED, rf));

        if (fabs(lf) < MIN_SPEED && lf != 0) lf = (lf > 0) ? MIN_SPEED : -MIN_SPEED;
        if (fabs(rf) < MIN_SPEED && rf != 0) rf = (rf > 0) ? MIN_SPEED : -MIN_SPEED;

        left_pwm = static_cast<int>(lf);
        right_pwm = static_cast<int>(rf);
    }
};




int main(int argc, char** argv) {
    int ret = 0;
    CVI_MODEL_HANDLE model;

    if (argc < 2 || argc > 3) {
        usage(argv);
        exit(-1);
    }

    CVI_U8 vi_channel = 0; // Default VI channel
    if (argc == 3) {
        vi_channel = atoi(argv[2]);
    }

    LOGI("Using VI channel: %d", vi_channel);

    // Initialize VI system
    VICapture vi_capture;
    LOGI("Initializing VI system...");
    if (vi_capture.init() != CVI_SUCCESS) {
        LOGE("Failed to initialize VI system");
        exit(-1);
    }
    LOGI("VI system initialized successfully");

    // Wait for sensor to stabilize
    usleep(500 * 1000);

#if USE_VPSS_RESIZE
    // Initialize VPSS for hardware resize (2560x1440 -> 640x640)
    LOGI("Initializing VPSS for hardware resize...");
    if (vi_capture.initVpssResize(2560, 1440, 640, 640) != CVI_SUCCESS) {
        LOGE("Failed to initialize VPSS");
        vi_capture.deinit();
        exit(-1);
    }
    LOGI("VPSS initialized successfully");
#endif

    // 初始化电机、机械臂和状态机
    Motor motor;
    Arm arm;
    RobotState robot;
    arm.grab_pos(); // 机械臂回到待抓取位置
    CVI_TENSOR* input;
    CVI_TENSOR* output;
    CVI_TENSOR* input_tensors;
    CVI_TENSOR* output_tensors;
    int32_t input_num;
    int32_t output_num;
    CVI_SHAPE input_shape;
    CVI_SHAPE* output_shape;
    int32_t height;
    int32_t width;
    // int bbox_len = 5; // 1 class + 4 bbox
    int classes_num = 1;
    float conf_thresh = 0.5;
    float iou_thresh = 0.5;
    ret = CVI_NN_RegisterModel(argv[1], &model);
    if (ret != CVI_RC_SUCCESS) {
        LOGE("CVI_NN_RegisterModel failed, err %d", ret);
        exit(1);
    }
    LOGI("CVI_NN_RegisterModel succeeded");

    // get input output tensors
    CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors, &output_num);

    LOGI("=== Model information ===");
    LOGI("Input number: %d, Output number: %d", input_num, output_num);

    input = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors, input_num);
    assert(input);
    output = output_tensors;
    output_shape = reinterpret_cast<CVI_SHAPE*>(calloc(output_num, sizeof(CVI_SHAPE)));
    for (int i = 0; i < output_num; i++) {
        output_shape[i] = CVI_NN_TensorShape(&output[i]);
        LOGI("Output[%d] shape: [%d, %d, %d, %d]", i, output_shape[i].dim[0], output_shape[i].dim[1],
               output_shape[i].dim[2], output_shape[i].dim[3]);
    }

    // nchw
    input_shape = CVI_NN_TensorShape(input);
    height = input_shape.dim[2];
    width = input_shape.dim[3];
    assert(height % 32 == 0 && width % 32 == 0);

    // 循环处理摄像头帧
    int frame_idx = 0;
    struct timeval start_time, end_time;
    struct timeval t1, t2;
    long total_time_us = 0;
    int frame_count = 0;

    cv::setNumThreads(1);

    while (true) {
        gettimeofday(&start_time, NULL);

        frame_idx++;
        LOGI("\n[Frame %d]", frame_idx);

        // Get YUV frame from VI and convert to BGR
        gettimeofday(&t1, NULL);
        cv::Mat image;
        if (vi_capture.getFrameAsBGR(vi_channel, image) != CVI_SUCCESS) {
            LOGW("Failed to get frame from VI channel %d", vi_channel);
            usleep(100000); // 休眠0.1秒
            continue;
        }

        if (!image.data) {
            LOGW("Empty image data");
            usleep(100000);
            continue;
        }

        cv::Mat cloned = image.clone();
        gettimeofday(&t2, NULL);
        long read_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

        // Image is already 640x640 from vi_get_frame_as_bgr, no need to resize again
        gettimeofday(&t1, NULL);
        
        // Convert BGR to RGB
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // Packed2Planar
        cv::Mat channels[3];
        for (int i = 0; i < 3; i++) {
            channels[i] = cv::Mat(image.rows, image.cols, CV_8SC1);
        }
        cv::split(image, channels);

        // fill data
        int8_t* ptr = (int8_t*)CVI_NN_TensorPtr(input);
        int channel_size = height * width;
        for (int i = 0; i < 3; ++i) {
            memcpy(ptr + i * channel_size, channels[i].data, channel_size);
        }
        gettimeofday(&t2, NULL);
        long preprocess_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

        // run inference
        gettimeofday(&t1, NULL);
        CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
        gettimeofday(&t2, NULL);
        long inference_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        // do post proprocess
        gettimeofday(&t1, NULL);
        int det_num = 0;
        std::vector<detection> dets;

        det_num = getDetections(output, height, width, classes_num, output_shape[0], conf_thresh, dets);
        // correct box with origin image size
        NMS(dets, &det_num, iou_thresh);
        correctYoloBoxes(dets, det_num, cloned.rows, cloned.cols, height, width);
        gettimeofday(&t2, NULL);
        long postprocess_time = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
        // ============ 状态机控制逻辑 ============
        if (det_num > 0) {
            // 选择最大的球（最近的）
            int best_idx = 0;
            for (int i = 1; i < det_num; i++) {
                if (dets[i].bbox.w > dets[best_idx].bbox.w)
                    best_idx = i;
            }

            box b = dets[best_idx].bbox;
            // bbox 中心坐标 → 左上角坐标 + 宽度（状态机用的是 x, w 格式）
            int box_x = static_cast<int>(b.x - b.w / 2);
            int box_w = static_cast<int>(b.w);
            robot.box_x = box_x;
            robot.box_w = box_w;

            LOGI("[DETECT] Ball: x=%d, w=%d, conf=%.3f, status=%s",
                 box_x, box_w, dets[best_idx].score, status_name(robot.status));

            // 更新状态
            robot.update_status();

            if (robot.status == STATUS_GRAB_TENNIS) {
                // 抓取状态：刹车，确认后抓
                motor.brake();
                robot.grab_confirm_count++;
                LOGI("[GRAB] confirm %d/%d", robot.grab_confirm_count, GRAB_CONFIRM_THRESHOLD);

                if (robot.grab_confirm_count >= GRAB_CONFIRM_THRESHOLD) {
                    LOGI("[ARM] Grabbing!");
                    arm.grab();
                    usleep(1000 * 1000);
                    // 抓完回到待命
                    arm.release_pos();
                    robot.grab_confirm_count = 0;
                    robot.status = STATUS_CHASE_TENNIS;
                    arm.grab_pos();
                }
            } else if (robot.status == STATUS_POSITION_TENNIS) {
                // 精确对准：只做左右微调
                robot.grab_confirm_count = 0;
                if (box_x < X_LEFT_GRAB) {
                    LOGD("[MOTOR] position: turn left (x=%d < %d)", box_x, X_LEFT_GRAB);
                    motor.left(MIN_SPEED);
                } else if (box_x > X_RIGHT_GRAB) {
                    LOGD("[MOTOR] position: turn right (x=%d > %d)", box_x, X_RIGHT_GRAB);
                    motor.right(MIN_SPEED);
                }
            } else {
                // 追球：P 控制差速驱动
                robot.grab_confirm_count = 0;
                int left_pwm, right_pwm;
                robot.calc_motor_speed(left_pwm, right_pwm);
                LOGD("[MOTOR] chase: L=%d R=%d", left_pwm, right_pwm);
                motor.drive(left_pwm, right_pwm);
            }

#if ENABLE_DRAW_BBOX
            for (int i = 0; i < det_num; i++) {
                box b = dets[i].bbox;
                int x1 = (b.x - b.w / 2);
                int y1 = (b.y - b.h / 2);
                int x2 = (b.x + b.w / 2);
                int y2 = (b.y + b.h / 2);
                x1 = std::max(0, std::min(x1, cloned.cols - 1));
                y1 = std::max(0, std::min(y1, cloned.rows - 1));
                x2 = std::max(0, std::min(x2, cloned.cols - 1));
                y2 = std::max(0, std::min(y2, cloned.rows - 1));
                cv::rectangle(cloned, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255), 3, 8, 0);
                char content[100];
                sprintf(content, "%s %0.3f", tennis_names[dets[i].cls], dets[i].score);
                cv::putText(cloned, content, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            }
#endif

        } else {
            LOGI("[DETECT] No ball detected, idling (turn right)");
            robot.grab_confirm_count = 0;
            motor.right(IDLE_SPEED);  // 没看到球就原地右转找球
        }
        
#if ENABLE_SAVE_IMAGE
            // save picture with detection results (only when ball detected)
            char output_path[256];
            sprintf(output_path, "/boot/images/detected_%d.jpg", frame_idx);
            LOGD("[DEBUG] Saving image: %dx%d, channels: %d", cloned.cols, cloned.rows, cloned.channels());
            cv::imwrite(output_path, cloned);
            LOGI("[SAVE] %s", output_path);
#endif
        // 计算帧率
        gettimeofday(&end_time, NULL);
        long frame_time_us = (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec);
        float fps = 1000000.0f / frame_time_us;

        frame_count++;
        total_time_us += frame_time_us;
        float avg_fps = 1000000.0f * frame_count / total_time_us;

        LOGI("[FPS] Current: %.2f, Average: %.2f (total: %.2f ms)", fps, avg_fps, frame_time_us / 1000.0f);
        LOGD("[PROFILE] Read: %.2f ms, Preprocess: %.2f ms, Inference: %.2f ms, "
               "Postprocess: %.2f ms",
               read_time / 1000.0f, preprocess_time / 1000.0f, inference_time / 1000.0f, postprocess_time / 1000.0f);

        // 每处理完一张图片，休眠一段时间再处理下一张
        // usleep(500000); // 休眠0.5秒
        if (frame_idx >= 200) {
            // 处理200帧后退出
            break;
        }
    } // end while loop

    // Cleanup
    printf("\nCleaning up...\n");
#if USE_VPSS_RESIZE
    vi_capture.deinitVpssResize();
#endif
    vi_capture.deinit();
    CVI_NN_CleanupModel(model);
    printf("CVI_NN_CleanupModel succeeded\n");
    free(output_shape);
    return 0;
}