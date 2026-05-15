#ifndef USB_CAMERA_CAPTURE_HPP
#define USB_CAMERA_CAPTURE_HPP

#include <opencv2/opencv.hpp>
#include <linux/videodev2.h>

#define USB_CAMERA_DEVICE "/dev/video0"
#define USB_CAMERA_WIDTH 640
#define USB_CAMERA_HEIGHT 480

class UsbCameraCapture {
public:
    UsbCameraCapture();
    ~UsbCameraCapture();

    /**
     * @brief 初始化 USB 摄像头 (V4L2)
     * @param device 设备路径，默认 /dev/video0
     * @return 0 success, -1 failed
     */
    int init(const char* device = USB_CAMERA_DEVICE);

    /**
     * @brief 反初始化，释放资源
     */
    void deinit();

    /**
     * @brief 获取一帧并转换为 BGR 格式
     * @param chn 通道号 (兼容接口，V4L2 下忽略)
     * @param bgr_image 输出的 BGR 图像
     * @return 0 success, -1 failed
     */
    int getFrameAsBGR(int chn, cv::Mat& bgr_image);

private:
    int fd_;
    struct buffer {
        void *start;
        size_t length;
    };
    struct buffer buffers_[4];
    unsigned int n_buffers_;
    bool initialized_;
    int xioctl(int req, void* arg);
    int capture_one_frame(void** out_data, size_t* out_size);
    int start_capture();
    int stop_capture();
    int init_mmap();
    int init_device(const char* device);
};

#endif // USB_CAMERA_CAPTURE_HPP