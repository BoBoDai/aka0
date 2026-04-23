#include "arm.hpp"
#include "logger.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <iostream>

const float Arm::ANGLE_MAX = 270.0f;

// Default config path
const std::string DEFAULT_ARM_CONFIG = "/root/AKA-00/arm_angles.json";

Arm::Arm(const std::string& port, int baudrate) : fd_(-1),
    SERVO0_READY(138.0f), SERVO1_READY(129.0f),
    SERVO0_LIFT(132.0f), SERVO1_LIFT(219.0f),
    SERVO2_PREPARE(170.0f), SERVO2_APPROACH(101.0f),
    SERVO2_GRAB(188.0f), SERVO2_LIFT(177.0f) {
    load_config(DEFAULT_ARM_CONFIG);
    open_serial(port, baudrate);
}

Arm::Arm(const std::string& port, int baudrate, const std::string& config_path) : fd_(-1),
    SERVO0_READY(138.0f), SERVO1_READY(129.0f),
    SERVO0_LIFT(132.0f), SERVO1_LIFT(219.0f),
    SERVO2_PREPARE(170.0f), SERVO2_APPROACH(101.0f),
    SERVO2_GRAB(188.0f), SERVO2_LIFT(177.0f) {
    load_config(config_path);
    open_serial(port, baudrate);
}

int Arm::parse_int_from_json(const char* json_str, const char* key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* p = strstr(json_str, pattern);
    if (p) {
        p += strlen(pattern);
        while (*p == ' ') p++;
        return atoi(p);
    }
    return -1;
}

void Arm::load_config(const std::string& config_path) {
    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        LOGE("[ARM] Failed to open config %s, exiting", config_path.c_str());
        exit(1);
    }

    std::string json_str((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    ifs.close();

    int val;
    val = parse_int_from_json(json_str.c_str(), "servo0_prepare");
    if (val >= 0) SERVO0_READY = val;
    val = parse_int_from_json(json_str.c_str(), "servo1_prepare");
    if (val >= 0) SERVO1_READY = val;
    val = parse_int_from_json(json_str.c_str(), "servo0_lift");
    if (val >= 0) SERVO0_LIFT = val;
    val = parse_int_from_json(json_str.c_str(), "servo1_lift");
    if (val >= 0) SERVO1_LIFT = val;
    val = parse_int_from_json(json_str.c_str(), "servo2_prepare");
    if (val >= 0) SERVO2_PREPARE = val;
    val = parse_int_from_json(json_str.c_str(), "servo2_approach");
    if (val >= 0) SERVO2_APPROACH = val;
    val = parse_int_from_json(json_str.c_str(), "servo2_grab");
    if (val >= 0) SERVO2_GRAB = val;
    val = parse_int_from_json(json_str.c_str(), "servo2_lift");
    if (val >= 0) SERVO2_LIFT = val;

    std::cout << "[ARM] Loaded config: servo0_prepare=" << SERVO0_READY
              << ", servo1_prepare=" << SERVO1_READY
              << ", servo2_prepare=" << SERVO2_PREPARE
              << ", servo2_approach=" << SERVO2_APPROACH
              << ", servo2_grab=" << SERVO2_GRAB
              << ", servo0_lift=" << SERVO0_LIFT
              << ", servo1_lift=" << SERVO1_LIFT
              << ", servo2_lift=" << SERVO2_LIFT << std::endl;
}

Arm::~Arm() {
    close_serial();
}

void Arm::open_serial(const std::string& port, int baudrate) {
    fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd_ < 0) {
        LOGE("[ARM] Failed to open serial port %s: %s", port.c_str(), strerror(errno));
        return;
    }

    struct termios tty;
    memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd_, &tty) != 0) {
        LOGE("[ARM] tcgetattr failed: %s", strerror(errno));
        close_serial();
        return;
    }

    speed_t baud;
    switch (baudrate) {
        case 9600:   baud = B9600;   break;
        case 115200: baud = B115200; break;
        default:     baud = B115200; break;
    }
    cfsetispeed(&tty, baud);
    cfsetospeed(&tty, baud);

    // 8N1, no flow control
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CLOCAL | CREAD;

    // Raw mode
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(INLCR | ICRNL | IGNCR);
    tty.c_oflag &= ~OPOST;

    // 100ms timeout
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    tcflush(fd_, TCIFLUSH);
    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        LOGE("[ARM] tcsetattr failed: %s", strerror(errno));
        close_serial();
        return;
    }

    LOGI("[ARM] Serial port %s opened at %d baud", port.c_str(), baudrate);
}

void Arm::close_serial() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

void Arm::send_command(const std::string& cmd) {
    if (fd_ < 0) {
        LOGE("[ARM] Serial port not open, cannot send: %s", cmd.c_str());
        return;
    }
    write(fd_, cmd.c_str(), cmd.size());
    tcdrain(fd_);
}

int Arm::angle_to_pulse(float angle) const {
    int pulse = static_cast<int>(500 + (angle / ANGLE_MAX) * 2000);
    return std::max(PULSE_MIN, std::min(PULSE_MAX, pulse));
}

void Arm::set_angle(int servo_id, float angle, int time_ms) {
    if (angle < 0 || angle > ANGLE_MAX) {
        LOGW("[ARM] Angle %.1f out of range [0, %.0f], clamping", angle, ANGLE_MAX);
        angle = std::max(0.0f, std::min(ANGLE_MAX, angle));
    }
    int pulse = angle_to_pulse(angle);
    char buf[32];
    snprintf(buf, sizeof(buf), "#%03dP%04dT%d!", servo_id, pulse, time_ms);
    send_command(buf);
    LOGD("[ARM] servo %d -> angle %.1f (pulse %d, %dms)", servo_id, angle, pulse, time_ms);
}

void Arm::release_torque(int servo_id) {
    char buf[16];
    snprintf(buf, sizeof(buf), "#%03dPULK", servo_id);
    send_command(buf);
    LOGD("[ARM] servo %d torque released", servo_id);
}

void Arm::restore_torque(int servo_id) {
    char buf[16];
    snprintf(buf, sizeof(buf), "#%03dPULR", servo_id);
    send_command(buf);
    LOGD("[ARM] servo %d torque restored", servo_id);
}

// Grab sequence - 伸下去 -> 抬起来 两个状态
void Arm::grab() {
    LOGI("[ARM] Grab sequence start");

    // 1. 伸下去，爪子张开 (servo2_grab)
    set_angle(0, SERVO0_READY);
    set_angle(1, SERVO1_READY);
    set_angle(2, SERVO2_PREPARE);
    usleep(1500 * 1000);

    // 2. 夹爪闭合
    set_angle(2, SERVO2_GRAB);
    usleep(1000 * 1000);

    // 3. 抬起来
    set_angle(0, SERVO0_LIFT);
    set_angle(1, SERVO1_LIFT);
    usleep(1000 * 1000);

    LOGI("[ARM] Grab sequence done");
}

void Arm::release() {
    LOGI("[ARM] Releasing gripper");
    set_angle(0, SERVO0_LIFT);
    set_angle(1, SERVO1_LIFT);
    set_angle(2, SERVO2_PREPARE);
}

void Arm::grab_pos() {
    LOGI("[ARM] Moving to home/ready position");
    set_angle(0, SERVO0_LIFT);
    set_angle(1, SERVO1_LIFT);
    set_angle(2, SERVO2_LIFT);
}