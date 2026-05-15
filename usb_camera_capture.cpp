#include "usb_camera_capture.hpp"
#include "logger.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>

UsbCameraCapture::UsbCameraCapture()
    : fd_(-1), n_buffers_(0), initialized_(false) {
    memset(buffers_, 0, sizeof(buffers_));
}

UsbCameraCapture::~UsbCameraCapture() {
    deinit();
}

int UsbCameraCapture::xioctl(int req, void* arg) {
    int r;
    do {
        r = ioctl(fd_, req, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

int UsbCameraCapture::capture_one_frame(void** out_data, size_t* out_size) {
    fd_set fds;
    struct timeval tv = {2, 0};
    struct v4l2_buffer buf;

    FD_ZERO(&fds);
    FD_SET(fd_, &fds);

    int r = select(fd_ + 1, &fds, NULL, NULL, &tv);
    if (r < 0) {
        if (errno == EINTR) return -1;
        return -1;
    }
    if (r == 0) return -1;  // timeout

    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(VIDIOC_DQBUF, &buf) < 0) {
        return -1;
    }

    *out_data = buffers_[buf.index].start;
    *out_size = buf.bytesused;

    if (xioctl(VIDIOC_QBUF, &buf) < 0) {
        return -1;
    }

    return 0;
}

int UsbCameraCapture::start_capture() {
    for (unsigned i = 0; i < n_buffers_; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        xioctl(VIDIOC_QBUF, &buf);
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(VIDIOC_STREAMON, &type) < 0) {
        return -1;
    }
    return 0;
}

int UsbCameraCapture::stop_capture() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(VIDIOC_STREAMOFF, &type);
    return 0;
}

int UsbCameraCapture::init_mmap() {
    struct v4l2_requestbuffers req = {0};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(VIDIOC_REQBUFS, &req) < 0) {
        LOGE("VIDIOC_REQBUFS failed - maybe not supported");
        return -1;
    }

    if (req.count < 2) {
        LOGE("Insufficient buffer memory");
        return -1;
    }

    n_buffers_ = req.count;

    for (unsigned i = 0; i < n_buffers_; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (xioctl(VIDIOC_QUERYBUF, &buf) < 0) {
            LOGE("VIDIOC_QUERYBUF failed");
            return -1;
        }

        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(NULL, buf.length,
                PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);

        if (buffers_[i].start == MAP_FAILED) {
            LOGE("mmap failed");
            return -1;
        }
    }

    return 0;
}

int UsbCameraCapture::init_device(const char* device) {
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = USB_CAMERA_WIDTH;
    fmt.fmt.pix.height = USB_CAMERA_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    if (xioctl(VIDIOC_S_FMT, &fmt) < 0) {
        LOGE("VIDIOC_S_FMT failed");
        return -1;
    }

    if (xioctl(VIDIOC_G_FMT, &fmt) < 0) {
        LOGE("VIDIOC_G_FMT failed");
        return -1;
    }

    LOGI("USB Camera: %dx%d, fourcc: %c%c%c%c",
            fmt.fmt.pix.width, fmt.fmt.pix.height,
            fmt.fmt.pix.pixelformat & 0xff,
            (fmt.fmt.pix.pixelformat >> 8) & 0xff,
            (fmt.fmt.pix.pixelformat >> 16) & 0xff,
            (fmt.fmt.pix.pixelformat >> 24) & 0xff);

    if (init_mmap() < 0) {
        return -1;
    }

    return 0;
}

int UsbCameraCapture::init(const char* device) {
    if (initialized_) {
        LOGW("USB camera already initialized");
        return 0;
    }

    fd_ = open(device, O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        LOGE("Cannot open %s: %s", device, strerror(errno));
        return -1;
    }

    if (init_device(device) < 0) {
        close(fd_);
        fd_ = -1;
        return -1;
    }

    if (start_capture() < 0) {
        LOGE("start_capture failed");
        deinit();
        return -1;
    }

    // Skip first 2 frames (often corrupt/underexposed)
    LOGI("Warming up USB camera...");
    for (int i = 0; i < 2; i++) {
        void* data = NULL;
        size_t size = 0;
        capture_one_frame(&data, &size);
    }

    initialized_ = true;
    LOGI("USB camera initialized successfully");
    return 0;
}

void UsbCameraCapture::deinit() {
    if (!initialized_) return;

    stop_capture();

    for (unsigned i = 0; i < n_buffers_; i++) {
        if (buffers_[i].start) {
            munmap(buffers_[i].start, buffers_[i].length);
            buffers_[i].start = NULL;
        }
    }
    n_buffers_ = 0;

    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }

    initialized_ = false;
}

int UsbCameraCapture::getFrameAsBGR(int chn, cv::Mat& bgr_image) {
    (void)chn;  // unused, only one USB camera

    if (!initialized_) {
        LOGE("USB camera not initialized");
        return -1;
    }

    void* data = NULL;
    size_t size = 0;

    // Retry up to 3 times
    for (int retry = 0; retry < 3; retry++) {
        if (capture_one_frame(&data, &size) == 0) {
            break;
        }
        LOGW("Frame capture retry %d", retry + 1);
    }

    if (!data) {
        LOGE("Failed to capture frame");
        return -1;
    }

    // YUYV -> BGR using OpenCV
    cv::Mat yuyv(USB_CAMERA_HEIGHT, USB_CAMERA_WIDTH, CV_8UC2, data);
    cv::cvtColor(yuyv, bgr_image, cv::COLOR_YUV2BGR_YUY2);

    return 0;
}