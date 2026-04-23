#include "motor.hpp"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <cstring>

// Default config path - 使用相对路径，项目目录下
const std::string DEFAULT_PWM_CONFIG = "/root/AKA-00/pwm_channels.json";

Motor::Motor() : LEFT_WHEEL_BACKWARD(0), LEFT_WHEEL_FORWARD(1),
                 RIGHT_WHEEL_BACKWARD(2), RIGHT_WHEEL_FORWARD(3) {
    load_config(DEFAULT_PWM_CONFIG);
    init_pwm(LEFT_WHEEL_BACKWARD);
    init_pwm(LEFT_WHEEL_FORWARD);
    init_pwm(RIGHT_WHEEL_BACKWARD);
    init_pwm(RIGHT_WHEEL_FORWARD);
}

Motor::Motor(const std::string& config_path) : LEFT_WHEEL_BACKWARD(0), LEFT_WHEEL_FORWARD(1),
                 RIGHT_WHEEL_BACKWARD(2), RIGHT_WHEEL_FORWARD(3) {
    load_config(config_path);
    init_pwm(LEFT_WHEEL_BACKWARD);
    init_pwm(LEFT_WHEEL_FORWARD);
    init_pwm(RIGHT_WHEEL_BACKWARD);
    init_pwm(RIGHT_WHEEL_FORWARD);
}

int Motor::parse_int_from_json(const char* json_str, const char* key) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char* p = strstr(json_str, pattern);
    if (p) {
        // Move past the pattern to the start of the number
        p += strlen(pattern);
        // Skip whitespace
        while (*p == ' ') p++;
        return atoi(p);
    }
    return -1;
}

void Motor::load_config(const std::string& config_path) {
    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {
        std::cerr << "[Motor] Warning: cannot open config " << config_path
                  << ", using defaults (0,1,2,3)" << std::endl;
        return;
    }

    std::string json_str((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    ifs.close();

    int val;
    val = parse_int_from_json(json_str.c_str(), "left_ch1");
    if (val >= 0) LEFT_WHEEL_BACKWARD = val;
    val = parse_int_from_json(json_str.c_str(), "left_ch2");
    if (val >= 0) LEFT_WHEEL_FORWARD = val;
    val = parse_int_from_json(json_str.c_str(), "right_ch1");
    if (val >= 0) RIGHT_WHEEL_BACKWARD = val;
    val = parse_int_from_json(json_str.c_str(), "right_ch2");
    if (val >= 0) RIGHT_WHEEL_FORWARD = val;

    std::cout << "[Motor] Loaded PWM config: left_ch1=" << LEFT_WHEEL_BACKWARD
              << ", left_ch2=" << LEFT_WHEEL_FORWARD
              << ", right_ch1=" << RIGHT_WHEEL_BACKWARD
              << ", right_ch2=" << RIGHT_WHEEL_FORWARD << std::endl;
}

Motor::~Motor() {
    standby();
}

void Motor::init_pwm(int pwm_id) {
    std::ofstream ofs_export(PWM_PATH + "export");
    if (ofs_export.is_open()) {
        ofs_export << pwm_id;
        ofs_export.close();
    }

    std::string pwm_channel_path = PWM_PATH + "pwm" + std::to_string(pwm_id);

    std::ofstream ofs_period(pwm_channel_path + "/period");
    if (ofs_period.is_open()) {
        ofs_period << PERIOD;
        ofs_period.close();
    }
}

void Motor::set_pwm_duty_cycle(int pwm_id, int duty_cycle) {
    std::string duty_cycle_path = PWM_PATH + "pwm" + std::to_string(pwm_id) + "/duty_cycle";
    std::ofstream ofs(duty_cycle_path);
    if (ofs.is_open()) {
        ofs << duty_cycle;
        ofs.close();
    }
}

void Motor::set_pwm_enable(int pwm_id, bool enable) {
    std::string enable_path = PWM_PATH + "pwm" + std::to_string(pwm_id) + "/enable";
    std::ofstream ofs(enable_path);
    if (ofs.is_open()) {
        ofs << (enable ? "1" : "0");
        ofs.close();
    }
}

void Motor::set_speed(int pwm_id, int speed) {
    if (speed < 0)
        speed = 0;
    if (speed > 100)
        speed = 100;
    int duty_cycle = PERIOD - (speed / 100.0) * PERIOD;
    set_pwm_duty_cycle(pwm_id, duty_cycle);
}

void Motor::forward(int speed) {
    set_speed(LEFT_WHEEL_FORWARD, speed);
    set_pwm_enable(LEFT_WHEEL_FORWARD, true);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, false);

    set_speed(RIGHT_WHEEL_FORWARD, speed);
    set_pwm_enable(RIGHT_WHEEL_FORWARD, true);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, false);
}

void Motor::backward(int speed) {
    set_speed(LEFT_WHEEL_BACKWARD, speed);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, true);
    set_pwm_enable(LEFT_WHEEL_FORWARD, false);

    set_speed(RIGHT_WHEEL_BACKWARD, speed);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, true);
    set_pwm_enable(RIGHT_WHEEL_FORWARD, false);
}

void Motor::left(int speed) {
    set_speed(RIGHT_WHEEL_FORWARD, speed);
    set_pwm_enable(RIGHT_WHEEL_FORWARD, true);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, false);

    set_pwm_enable(LEFT_WHEEL_FORWARD, false);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, false);
}

void Motor::right(int speed) {
    set_speed(LEFT_WHEEL_FORWARD, speed);
    set_pwm_enable(LEFT_WHEEL_FORWARD, true);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, false);

    set_pwm_enable(RIGHT_WHEEL_FORWARD, false);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, false);
}

void Motor::brake() {
    set_pwm_enable(LEFT_WHEEL_FORWARD, true);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, true);
    set_pwm_enable(RIGHT_WHEEL_FORWARD, true);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, true);
}

void Motor::standby() {
    set_pwm_enable(LEFT_WHEEL_FORWARD, false);
    set_pwm_enable(LEFT_WHEEL_BACKWARD, false);
    set_pwm_enable(RIGHT_WHEEL_FORWARD, false);
    set_pwm_enable(RIGHT_WHEEL_BACKWARD, false);
}

void Motor::drive(int left_speed, int right_speed) {
    // 左轮
    if (left_speed > 0) {
        set_speed(LEFT_WHEEL_FORWARD, left_speed);
        set_pwm_enable(LEFT_WHEEL_FORWARD, true);
        set_pwm_enable(LEFT_WHEEL_BACKWARD, false);
    } else if (left_speed < 0) {
        set_speed(LEFT_WHEEL_BACKWARD, -left_speed);
        set_pwm_enable(LEFT_WHEEL_BACKWARD, true);
        set_pwm_enable(LEFT_WHEEL_FORWARD, false);
    } else {
        set_pwm_enable(LEFT_WHEEL_FORWARD, false);
        set_pwm_enable(LEFT_WHEEL_BACKWARD, false);
    }
    // 右轮
    if (right_speed > 0) {
        set_speed(RIGHT_WHEEL_FORWARD, right_speed + 2);
        set_pwm_enable(RIGHT_WHEEL_FORWARD, true);
        set_pwm_enable(RIGHT_WHEEL_BACKWARD, false);
    } else if (right_speed < 0) {
        set_speed(RIGHT_WHEEL_BACKWARD, -right_speed + 2);
        set_pwm_enable(RIGHT_WHEEL_BACKWARD, true);
        set_pwm_enable(RIGHT_WHEEL_FORWARD, false);
    } else {
        set_pwm_enable(RIGHT_WHEEL_FORWARD, false);
        set_pwm_enable(RIGHT_WHEEL_BACKWARD, false);
    }
}
