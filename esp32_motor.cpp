#include "esp32_motor.hpp"
#include "logger.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/select.h>

// Protocol defines
#define FRAME_HEADER0    0xAA
#define FRAME_HEADER1    0x55
#define CMD_INIT         0x01
#define CMD_CONFIG       0x02
#define CMD_SET_SPEED    0x10
#define CMD_STOP         0x11
#define CMD_BRAKE        0x12
#define CMD_SET_SPEEDS   0x13
#define CMD_GET_RPM      0x20
#define CMD_GET_STATUS   0x21
#define CMD_RESET        0xFF
#define CMD_ACK          0x80

// Protocol params
#define PPR       2496
#define PWM_FREQ  20000

Esp32Motor::Esp32Motor(const char* uart_dev, int baud)
    : uart_fd_(-1), uart_dev_(uart_dev), baud_(baud), initialized_(false) {
}

Esp32Motor::~Esp32Motor() {
    if (uart_fd_ >= 0) {
        close(uart_fd_);
        uart_fd_ = -1;
    }
}

bool Esp32Motor::init() {
    uart_fd_ = open(uart_dev_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (uart_fd_ < 0) {
        LOGE("Cannot open UART %s: %s", uart_dev_.c_str(), strerror(errno));
        return false;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(uart_fd_, &tty) != 0) {
        LOGE("tcgetattr failed: %s", strerror(errno));
        close(uart_fd_);
        uart_fd_ = -1;
        return false;
    }

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 5;

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(uart_fd_, TCSANOW, &tty) != 0) {
        LOGE("tcsetattr failed: %s", strerror(errno));
        close(uart_fd_);
        uart_fd_ = -1;
        return false;
    }

    // Send INIT
    if (!send_frame(CMD_INIT)) {
        LOGE("INIT cmd failed");
        return false;
    }
    if (!recv_ack(CMD_INIT, 500)) {
        LOGW("INIT ack not received");
    }

    usleep(100000);  // 100ms delay after INIT

    // Send CONFIG: PPR + PWM_FREQ (big-endian)
    uint8_t config_payload[4];
    config_payload[0] = (PPR >> 8) & 0xFF;
    config_payload[1] = PPR & 0xFF;
    config_payload[2] = (PWM_FREQ >> 8) & 0xFF;
    config_payload[3] = PWM_FREQ & 0xFF;

    if (!send_frame(CMD_CONFIG, config_payload, 4)) {
        LOGE("CONFIG cmd failed");
        return false;
    }
    if (!recv_ack(CMD_CONFIG, 500)) {
        LOGW("CONFIG ack not received");
    }

    usleep(100000);

    initialized_ = true;
    LOGI("Esp32Motor initialized on %s", uart_dev_.c_str());
    return true;
}

uint8_t Esp32Motor::calc_chk(uint8_t cmd, uint8_t len, const uint8_t* payload) {
    uint8_t chk = cmd ^ len;
    for (int i = 0; i < len; i++) {
        chk ^= payload[i];
    }
    return chk;
}

bool Esp32Motor::send_frame(uint8_t cmd, const uint8_t* payload, uint8_t len) {
    if (uart_fd_ < 0) return false;

    uint8_t chk = calc_chk(cmd, len, payload);

    uint8_t frame[64];
    frame[0] = FRAME_HEADER0;
    frame[1] = FRAME_HEADER1;
    frame[2] = cmd;
    frame[3] = len;
    for (int i = 0; i < len; i++) {
        frame[4 + i] = payload[i];
    }
    frame[4 + len] = chk;

    int frame_len = 5 + len;
    int written = write(uart_fd_, frame, frame_len);
    if (written != frame_len) {
        LOGE("UART write error: wrote %d, expected %d", written, frame_len);
        return false;
    }

    return true;
}

bool Esp32Motor::recv_ack(uint8_t expected_cmd, int timeout_ms) {
    if (uart_fd_ < 0) return false;

    uint8_t buf[32];
    fd_set rfds;
    struct timeval tv;

    int total_timeout_ms = timeout_ms;
    while (total_timeout_ms > 0) {
        FD_ZERO(&rfds);
        FD_SET(uart_fd_, &rfds);
        tv.tv_sec = 0;
        tv.tv_usec = total_timeout_ms * 1000;
        int ret = select(uart_fd_ + 1, &rfds, NULL, NULL, &tv);
        if (ret <= 0) break;

        int n = read(uart_fd_, buf, sizeof(buf));
        if (n <= 0) break;

        // Look for header AA 55
        for (int i = 0; i < n - 1; i++) {
            if (buf[i] == FRAME_HEADER0 && buf[i + 1] == FRAME_HEADER1) {
                if (i + 4 < n) {
                    uint8_t cmd = buf[i + 2];
                    uint8_t len = buf[i + 3];
                    if (cmd == CMD_ACK && len == 1 && i + 5 < n) {
                        if (buf[i + 4] == expected_cmd) {
                            return true;
                        }
                    }
                }
                // Continue searching
                break;
            }
        }

        total_timeout_ms -= 10;
    }
    return false;
}

void Esp32Motor::forward(int speed) {
    if (!initialized_) return;
    drive(speed, speed);
}

void Esp32Motor::backward(int speed) {
    if (!initialized_) return;
    drive(-speed, -speed);
}

void Esp32Motor::left(int speed) {
    if (!initialized_) return;
    drive(-speed, speed);
}

void Esp32Motor::right(int speed) {
    if (!initialized_) return;
    drive(speed, -speed);
}

void Esp32Motor::brake() {
    if (!initialized_) return;
    uint8_t p = 2;
    send_frame(CMD_BRAKE, &p, 1);  // p=2 一次刹两轮
}

void Esp32Motor::standby() {
    if (!initialized_) return;
    drive(0, 0);
}

void Esp32Motor::drive(int left_speed, int right_speed) {
    if (!initialized_) {
        LOGE("drive called but not initialized");
        return;
    }

    // Clamp to -255~255
    if (left_speed > 255) left_speed = 255;
    if (left_speed < -255) left_speed = -255;
    if (right_speed > 255) right_speed = 255;
    if (right_speed < -255) right_speed = -255;

    // Use SET_SPEEDS (0x13): M1=left_speed, M2=right_speed, big-endian int16
    uint8_t payload[4];
    int16_t m1 = (int16_t)left_speed;   // no negation - ESP32 handles direction internally
    int16_t m2 = (int16_t)right_speed;

    payload[0] = (m1 >> 8) & 0xFF;
    payload[1] = m1 & 0xFF;
    payload[2] = (m2 >> 8) & 0xFF;
    payload[3] = m2 & 0xFF;

    LOGI("[MOTOR] drive left=%d right=%d -> m1=%d m2=%d (M1=left)", left_speed, right_speed, m1, m2);
    send_frame(CMD_SET_SPEEDS, payload, 4);
}