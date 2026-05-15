#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <linux/videodev2.h>

#define DEVICE "/dev/video0"
#define WIDTH 640
#define HEIGHT 480

struct buffer {
    void   *start;
    size_t length;
};

static int fd = -1;
static struct buffer buffers[4];
static unsigned int n_buffers;

static void fatal(const char *msg) { perror(msg); exit(1); }

static int xioctl(int fd, int req, void *arg) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

static int capture_one_frame(void **out_data, size_t *out_size) {
    fd_set fds;
    struct timeval tv = {2, 0};
    struct v4l2_buffer buf;

    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    int r = select(fd + 1, &fds, NULL, NULL, &tv);
    if (r < 0) { if (errno == EINTR) return -1; fatal("select"); }
    if (r == 0) { fprintf(stderr, "Timeout waiting for frame\n"); return -1; }

    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_DQBUF, &buf) < 0) {
        fprintf(stderr, "DQBUF failed\n");
        return -1;
    }

    *out_data = buffers[buf.index].start;
    *out_size = buf.bytesused;

    if (xioctl(fd, VIDIOC_QBUF, &buf) < 0) {
        fprintf(stderr, "QBUF failed\n");
        return -1;
    }

    return 0;
}

static void stop_capture(void) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd, VIDIOC_STREAMOFF, &type);
}

static void start_capture(void) {
    for (unsigned i = 0; i < n_buffers; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        xioctl(fd, VIDIOC_QBUF, &buf);
    }
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) < 0) fatal("VIDIOC_STREAMON");
}

static void init_mmap(void) {
    struct v4l2_requestbuffers req = {0};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd, VIDIOC_REQBUFS, &req) < 0) fatal("VIDIOC_REQBUFS");
    if (req.count < 2) { fprintf(stderr, "Insufficient buffer memory\n"); exit(1); }

    for (n_buffers = 0; n_buffers < req.count; n_buffers++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = n_buffers;
        if (xioctl(fd, VIDIOC_QUERYBUF, &buf) < 0) fatal("VIDIOC_QUERYBUF");

        buffers[n_buffers].length = buf.length;
        buffers[n_buffers].start = mmap(NULL, buf.length,
                PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (buffers[n_buffers].start == MAP_FAILED) fatal("mmap");
    }
}

static void init_device(void) {
    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) < 0) fatal("VIDIOC_S_FMT");

    if (xioctl(fd, VIDIOC_G_FMT, &fmt) < 0) fatal("VIDIOC_G_FMT");
    printf("Camera: %dx%d, fourcc: %c%c%c%c\n",
            fmt.fmt.pix.width, fmt.fmt.pix.height,
            fmt.fmt.pix.pixelformat & 0xff,
            (fmt.fmt.pix.pixelformat >> 8) & 0xff,
            (fmt.fmt.pix.pixelformat >> 16) & 0xff,
            (fmt.fmt.pix.pixelformat >> 24) & 0xff);

    init_mmap();
}

int main(int argc, char *argv[]) {
    const char *dev = (argc > 1) ? argv[1] : DEVICE;
    int count = (argc > 2) ? atoi(argv[2]) : 10;

    printf("Opening: %s\n", dev);
    fd = open(dev, O_RDWR | O_NONBLOCK);
    if (fd < 0) fatal("open");

    init_device();
    start_capture();

    struct timeval t1, t2, t_total_start;
    long capture_us, total_us;
    long min_capture = 999999999, max_capture = 0;
    long total_capture = 0;

    printf("Capturing %d frames, measuring time...\n", count);

    gettimeofday(&t_total_start, NULL);

    struct timeval t_capture_end;
    for (int frame = 0; frame < count; frame++) {
        void *data = NULL;
        size_t size = 0;

        gettimeofday(&t1, NULL);
        int retry = 0;
        while (capture_one_frame(&data, &size) < 0) {
            printf("  Retrying frame %d...\n", frame);
            retry++;
            if (retry > 5) { printf("Failed after 5 retries\n"); break; }
        }
        gettimeofday(&t2, NULL);

        if (!data) continue;

        capture_us = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);

        if (capture_us < min_capture) min_capture = capture_us;
        if (capture_us > max_capture) max_capture = capture_us;
        total_capture += capture_us;

        printf("  Frame %d: %ld us (%ld ms), %zu bytes\n",
                frame, capture_us, capture_us / 1000, size);

        // Save last frame as raw YUYV
        if (frame == count - 1) {
            FILE *f = fopen("capture.yuyv", "wb");
            if (f) {
                fwrite(data, 1, size, f);
                fclose(f);
                printf("  Saved last frame to capture.yuyv\n");
            }
        }
    }

    gettimeofday(&t_total_start, NULL);
    total_us = (t_capture_end.tv_sec - t_total_start.tv_sec) * 1000000 +
               (t_capture_end.tv_usec - t_total_start.tv_usec);

    stop_capture();

    for (unsigned i = 0; i < n_buffers; i++) {
        munmap(buffers[i].start, buffers[i].length);
    }
    close(fd);

    printf("\n=== Summary ===\n");
    printf("Frames captured: %d\n", count);
    printf("Capture time per frame:\n");
    printf("  Min:  %ld us (%.2f ms)\n", min_capture, min_capture / 1000.0);
    printf("  Max:  %ld us (%.2f ms)\n", max_capture, max_capture / 1000.0);
    printf("  Avg:  %ld us (%.2f ms)\n", total_capture / count, total_capture / (float)count / 1000.0);
    printf("\nConvert on PC with:\n");
    printf("  ffmpeg -f rawvideo -pix_fmt yuyv422 -s %dx%d -i capture.yuyv capture.jpg\n", WIDTH, HEIGHT);

    return 0;
}